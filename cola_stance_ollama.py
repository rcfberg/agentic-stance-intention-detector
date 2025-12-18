#!/usr/bin/env python3
import json, argparse
import pandas as pd
from typing import Dict, Any
from tenacity import retry, wait_exponential, stop_after_attempt
from pydantic import BaseModel, ValidationError
import ollama
from tqdm import tqdm

# ---------------- Config ----------------
# Run Gemma models via Ollama
JUDGE_MODEL     = "gemma3:4b"
ANALYST_MODEL   = JUDGE_MODEL
ADVOCATE_MODEL  = JUDGE_MODEL

TEMP_ANALYST  = 0.0
TEMP_ADVOC    = 0.0
TEMP_JUDGE    = 0.0

# -------------- Schemas -----------------
class A1Schema(BaseModel):
    salient_phrases: list[str]
    stance_markers: Dict[str, Any]
    argumentative_patterns: list[str]
    polarity_hint: str

class A2Schema(BaseModel):
    entities: list[Dict[str, Any]]
    claims: list[str]
    policy_frames: list[str]
    domain_hint: str

class A3Schema(BaseModel):
    platform_cues: list[str]
    sarcasm_or_irony: bool
    quote_context_interpretation: str
    platform_hint: str
    target_relevance_0_1: float

class AdvocateSchema(BaseModel):
    stance: str
    reasoning_steps: list[str]
    evidence_spans: list[Dict[str, str]]
    confidence_0_1: float

class JudgeSchema(BaseModel):
    final_stance: str
    arbiter_confidence_0_1: float
    justification: str
    key_evidence_quotes: list[str]

# -------------- Helpers -----------------
def jdump(x):
    return json.dumps(x, ensure_ascii=False)

# --- Option A: Coercion shim for A2 payload ---
def _coerce_a2_payload(data: dict) -> dict:
    """Normalize A2 payload to match the schema:
    - entities -> list[dict], converting strings to {"name": str}
    - claims   -> list[str], extracting 'claim' from dicts when present
    """
    if not isinstance(data, dict):
        return data

    out = dict(data)

    # --- entities: ensure list[dict] ---
    ents = out.get("entities", [])
    coerced_ents = []
    if isinstance(ents, list):
        for e in ents:
            if isinstance(e, str):
                coerced_ents.append({"name": e})
            elif isinstance(e, dict):
                if "name" not in e:
                    name = e.get("entity") or e.get("text") or e.get("label")
                    if name:
                        e = {"name": name, **{k: v for k, v in e.items() if k not in ("entity", "text", "label")}}
                coerced_ents.append(e)
    out["entities"] = coerced_ents

    # --- claims: ensure list[str] ---
    cls = out.get("claims", [])
    coerced_claims = []
    if isinstance(cls, list):
        for c in cls:
            if isinstance(c, str):
                coerced_claims.append(c)
            elif isinstance(c, dict):
                if isinstance(c.get("claim"), str):
                    coerced_claims.append(c["claim"])
                else:
                    coerced_claims.append(json.dumps(c, ensure_ascii=False))
    out["claims"] = coerced_claims

    return out

def _fill_missing_fields_for_schema(schema, data: dict) -> dict:
    """Add sensible defaults for missing keys so validation succeeds even if
    the model returned partial or empty JSON. Preserves existing keys."""
    if not isinstance(data, dict):
        data = {}

    # Base copy
    out = dict(data)

    if schema.__name__ == "A1Schema":
        out.setdefault("salient_phrases", [])
        # stance_markers should be a dict with specific subkeys
        sm = out.get("stance_markers", {})
        if not isinstance(sm, dict):
            sm = {}
        sm.setdefault("negation", [])
        sm.setdefault("intensifiers", [])
        sm.setdefault("modals", [])
        # prefer boolean; if string, coerce
        irony = sm.get("irony", False)
        if isinstance(irony, str):
            irony = irony.strip().lower() in ("true", "yes", "1")
        sm["irony"] = bool(irony)
        out["stance_markers"] = sm

        out.setdefault("argumentative_patterns", [])
        # polarity_hint must be one of "favor"|"against"|"uncertain"
        ph = out.get("polarity_hint", "uncertain")
        if ph not in ("favor", "against", "uncertain"):
            ph = "uncertain"
        out["polarity_hint"] = ph

    elif schema.__name__ == "A2Schema":
        out.setdefault("entities", [])
        out.setdefault("claims", [])
        out.setdefault("policy_frames", [])
        dh = out.get("domain_hint", "uncertain")
        if dh not in ("favor", "against", "uncertain"):
            dh = "uncertain"
        out["domain_hint"] = dh

    elif schema.__name__ == "A3Schema":
        out.setdefault("platform_cues", [])
        soi = out.get("sarcasm_or_irony", False)
        if isinstance(soi, str):
            soi = soi.strip().lower() in ("true", "yes", "1")
        out["sarcasm_or_irony"] = bool(soi)

        qci = out.get("quote_context_interpretation", "")
        if not isinstance(qci, str):
            qci = str(qci) if qci is not None else ""
        out["quote_context_interpretation"] = qci

        ph = out.get("platform_hint", "uncertain")
        if ph not in ("favor", "against", "uncertain"):
            ph = "uncertain"
        out["platform_hint"] = ph

        tr = out.get("target_relevance_0_1", 0.0)
        try:
            tr = float(tr)
        except Exception:
            tr = 0.0
        # clamp 0..1
        tr = max(0.0, min(1.0, tr))
        out["target_relevance_0_1"] = tr

    elif schema.__name__ == "AdvocateSchema":
        st = out.get("stance", "neutral")
        if st not in ("favor", "against", "neutral"):
            st = "neutral"
        out["stance"] = st

        rs = out.get("reasoning_steps", [])
        if not isinstance(rs, list):
            rs = [str(rs)]
        out["reasoning_steps"] = rs

        ev = out.get("evidence_spans", [])
        if not isinstance(ev, list):
            ev = []
        # Coerce each entry to {"quote": str, "source": "linguistic"|"domain"|"platform"}
        coerced_ev = []
        for e in ev:
            if isinstance(e, dict):
                quote = e.get("quote", "")
                source = e.get("source", "")
                if source not in ("linguistic", "domain", "platform"):
                    source = "linguistic"
                coerced_ev.append({"quote": str(quote), "source": source})
            elif isinstance(e, str):
                coerced_ev.append({"quote": e, "source": "linguistic"})
        out["evidence_spans"] = coerced_ev

        cf = out.get("confidence_0_1", 0.0)
        try:
            cf = float(cf)
        except Exception:
            cf = 0.0
        cf = max(0.0, min(1.0, cf))
        out["confidence_0_1"] = cf

    elif schema.__name__ == "JudgeSchema":
        fs = out.get("final_stance", "irrelevant")
        if fs not in ("favor", "against", "neutral", "irrelevant"):
            fs = "irrelevant"
        out["final_stance"] = fs

        ac = out.get("arbiter_confidence_0_1", 0.0)
        try:
            ac = float(ac)
        except Exception:
            ac = 0.0
        ac = max(0.0, min(1.0, ac))
        out["arbiter_confidence_0_1"] = ac

        just = out.get("justification", "")
        if not isinstance(just, str):
            just = str(just) if just is not None else ""
        out["justification"] = just

        keq = out.get("key_evidence_quotes", [])
        if not isinstance(keq, list):
            keq = [str(keq)]
        # Ensure string items
        keq = [str(x) for x in keq if x is not None]
        out["key_evidence_quotes"] = keq

    return out

def chat_json(model: str, system: str, user: str, temperature: float, schema):
    """Ask Ollama (Gemma) for JSON and validate with pydantic. Retries on parse errors."""
    return _chat_json_retry(model, system, user, temperature, schema)

@retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3))
def _chat_json_retry(model: str, system: str, user: str, temperature: float, schema):
    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": user.strip()})

    # Use Ollama's chat API with Gemma; request JSON formatting from the model.
    resp = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
        format=None if schema.__name__ == "JudgeSchema" else "json"
    )
    raw = resp["message"]["content"]

    # Robust JSON extraction in case the model wraps output
    try:
        data = json.loads(raw)
    except Exception:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end >= 0:
            data = json.loads(raw[start:end+1])
        else:
            # As a last resort, set to empty dict for default filling
            data = {}

    if schema.__name__ == "A2Schema":
        data = _coerce_a2_payload(data)

    # Fill missing fields so validation doesn't fail on empty/partial JSON
    data = _fill_missing_fields_for_schema(schema, data)

    # Validate with Pydantic; if still failing, attempt a final default-only pass
    try:
        obj = schema.model_validate(data)
    except ValidationError:
        # Overwrite with pure defaults and validate
        obj = schema.model_validate(_fill_missing_fields_for_schema(schema, {}))

    return obj, data

# -------- Prompt templates (COLA) -------
def p_a1(text_sender, target_policy, account_sender, description_sender, actor_type_sender, tweet_type, account_receiver, description_receiver, receiver_party):
    sys = "You are a linguist specializing in stance markers and argumentation."
    usr = f"""
# CONTEXT
**Sender**
ACCOUNT_SENDER: {account_sender}
DESCRIPTION_SENDER: {description_sender}
ACTOR_TYPE: {actor_type_sender}
TWEET_TYPE: {tweet_type}

**Receiver**
ACCOUNT_RECEIVER: {account_receiver}
DESCRIPTION_RECEIVER: {description_receiver}
RECEIVER_PARTY: {receiver_party}

# TASK 
Analyze TEXT_SENDER: {text_sender} with respect to the policy TARGET_POLICY: {target_policy}. Use the sender-receiver context above as background information to help inform your detection of stance-relevant linguistic cues (negation, intensifiers, modals, irony/sarcasm, concessions, rhetorical questions) and argumentation roles.
Return ONLY valid JSON:
{{
  "salient_phrases": [],
  "stance_markers": {{"negation": [], "intensifiers": [], "modals": [], "irony": false, "concessions": [], "rhetorical_questions": []}},
  "argumentative_patterns": [],
  "polarity_hint": "favor"|"against"|"uncertain"
}}

Do nothing else.
"""
    return sys, usr

def p_a2(text_sender, target_policy, account_sender, description_sender, actor_type_sender, tweet_type, account_receiver, description_receiver, receiver_party):
    sys = "You are a political science specialist with expertise in policy analysis."
    usr = f"""
# CONTEXT
**Sender**
ACCOUNT_SENDER: {account_sender}
DESCRIPTION_SENDER: {description_sender}
ACTOR_TYPE: {actor_type_sender}
TWEET_TYPE: {tweet_type}

**Receiver**
ACCOUNT_RECEIVER: {account_receiver}
DESCRIPTION_RECEIVER: {description_receiver}
RECEIVER_PARTY: {receiver_party}

# TASK
Analyze TEXT_SENDER: {text_sender} with respect to the policy TARGET_POLICY: {target_policy}. Use the sender-receiver context above as background information to help identify entities/events, policy frames, and map claims to target stance.
Return ONLY valid JSON:
{{
  "entities": [],
  "claims": [],
  "policy_frames": [],
  "domain_hint": "favor"|"against"|"uncertain"
}}

Do nothing else.
"""
    return sys, usr

def p_a3(text_sender, target_policy, account_sender, description_sender, actor_type_sender, tweet_type, account_receiver, description_receiver, receiver_party):
    sys = "You are social media specialist with expertise in platform pragmatics (hashtags, quotes, irony)."
    usr = f"""
# CONTEXT
**Sender**
ACCOUNT_SENDER: {account_sender}
DESCRIPTION_SENDER: {description_sender}
ACTOR_TYPE: {actor_type_sender}
TWEET_TYPE: {tweet_type}

**Receiver**
ACCOUNT_RECEIVER: {account_receiver}
DESCRIPTION_RECEIVER: {description_receiver}
RECEIVER_PARTY: {receiver_party}

# TASK
Analyze TEXT_SENDER: {text_sender} with respect to the policy TARGET_POLICY: {target_policy}. Use the sender-receiver context above as background information to help interpret platform cues and whether stance is about TARGET_POLICY.

Return ONLY valid JSON:
{{
  "platform_cues": [],
  "sarcasm_or_irony": false,
  "quote_context_interpretation": "",
  "platform_hint": "favor"|"against"|"uncertain",
  "target_relevance_0_1": 0.0
}}

Do nothing else.
"""
    return sys, usr

def p_adv(text_sender, target_policy, A1, A2, A3, stance,
          account_sender, description_sender, actor_type_sender,
          tweet_type, account_receiver, description_receiver, receiver_party):
    sys = f"You are an advocate arguing the text is {stance} toward TARGET_POLICY."
    usr = f"""

# CONTEXT
**Sender**
ACCOUNT_SENDER: {account_sender}
DESCRIPTION_SENDER: {description_sender}
ACTOR_TYPE: {actor_type_sender}
TWEET_TYPE: {tweet_type}

**Receiver**
ACCOUNT_RECEIVER: {account_receiver}
DESCRIPTION_RECEIVER: {description_receiver}
RECEIVER_PARTY: {receiver_party}

# TEXT
Text to be analyzed: {text_sender}

# Evidence from analysts
EVIDENCE: {jdump({"linguistic": A1, "domain": A2, "platform": A3})}

TASK: Provide a short, explicit reasoning chain citing quotes/cues.  The quotes should come from the evidence above.  Use the context as necessary to decide how the evidence applies to the target policy.
Return ONLY valid JSON:
{{
  "stance": "favor"|"against"|"neutral",
  "reasoning_steps": [],
  "evidence_spans": [{{"quote":"", "source":"linguistic"|"domain"|"platform"}}],
  "confidence_0_1": 0.0
}}

Do nothing else.
"""
    return sys, usr

def p_judge(text_sender, target_policy, advocates,
            account_sender, description_sender, actor_type_sender,
            tweet_type, account_receiver, description_receiver, receiver_party):
    sys = "You are an impartial adjudicator. Choose stance toward TARGET_POLICY."
    usr = f"""
# CONTEXT
**Sender**
ACCOUNT_SENDER: {account_sender}
DESCRIPTION_SENDER: {description_sender}
ACTOR_TYPE: {actor_type_sender}
TWEET_TYPE: {tweet_type}

**Receiver**
ACCOUNT_RECEIVER: {account_receiver}
DESCRIPTION_RECEIVER: {description_receiver}
RECEIVER_PARTY: {receiver_party}

# TEXT & POLICY
TEXT_SENDER: {text_sender}
TARGET_POLICY: {target_policy}

# ADVOCATES
ADVOCATES: {jdump(advocates)}

RULES:
- Prioritize directly-targeted evidence; penalize off-target or sarcasm misreads
- Consider the sender/receiver context when assessing relevance.
- If stance toward TARGET is unclear, choose "irrelevant".
Return ONLY valid JSON:
{{
  "final_stance": "favor"|"against"|"neutral"|"irrelevant",
  "arbiter_confidence_0_1": 0.0,
  "justification": "",
  "key_evidence_quotes": []
}}

Do nothing else.
"""
    return sys, usr
def run_item(text_sender: str, target_policy: str, account_sender: str, description_sender: str, actor_type_sender: str, tweet_type: str, account_receiver: str, description_receiver: str, receiver_party: str):
    # Stage A (Analysts)
    sys, u = p_a1(text_sender, target_policy, account_sender, description_sender, actor_type_sender, tweet_type, account_receiver, description_receiver, receiver_party)
     # ---- DEBUG ----
    #print("\n=== P_A1 PROMPTS ===")
    #print("SYSTEM:", sys[:200])          # first 200 chars to keep it tidy
    #print("USER  :", u[:200])
    # ---- END DEBUG ----
    A1_obj, A1_raw = chat_json(ANALYST_MODEL, sys, u, TEMP_ANALYST, A1Schema)
    # ---- DEBUG ----
    #print("A1 RAW   :", A1_raw)
    #print("A1 OBJ   :", A1_obj)  # pydantic model instance
    # ---- END DEBUG ----

    sys, u = p_a2(text_sender, target_policy, account_sender, description_sender, actor_type_sender, tweet_type, account_receiver, description_receiver, receiver_party)
    #print("\n=== P_A2 PROMPTS ===")
    #print("SYSTEM:", sys[:200]); print("USER  :", u[:200])
    A2_obj, A2_raw = chat_json(ANALYST_MODEL, sys, u, TEMP_ANALYST, A2Schema)
    #print("A2 RAW   :", A2_raw)
    #print("A2 OBJ   :", A2_obj)

    sys, u = p_a3(text_sender, target_policy, account_sender, description_sender, actor_type_sender, tweet_type, account_receiver, description_receiver, receiver_party)
    #print("\n=== P_A3 PROMPTS ===")
    #print("SYSTEM:", sys[:200]); print("USER  :", u[:200])
    A3_obj, A3_raw = chat_json(ANALYST_MODEL, sys, u, TEMP_ANALYST, A3Schema)
    #print("A3 RAW   :", A3_raw)
    #print("A3 OBJ   :", A3_obj)

    # Stage B (Advocates)
    adv_raw = {}
    for stance in ["favor", "against", "neutral"]:
        sys, u = p_adv(text_sender, target_policy, A1_raw, A2_raw, A3_raw, stance,
            account_sender, description_sender, actor_type_sender,
            tweet_type, account_receiver, description_receiver, receiver_party)
        #print(f"\n=== P_ADV ({stance.upper()}) PROMPTS ===")
        #print("SYSTEM:", sys[:200]); print("USER  :", u[:200])
        adv_obj, adv_json = chat_json(ADVOCATE_MODEL, sys, u, TEMP_ADVOC, AdvocateSchema)
        #print(f"ADV_{stance.upper()} RAW :", adv_json)
        #print(f"ADV_{stance.upper()} OBJ :", adv_obj)
        adv_raw[stance] = adv_json

    # Stage C (Judge/Arbiter)
    sys, u = p_judge(text_sender, target_policy, adv_raw,
        account_sender, description_sender, actor_type_sender,
        tweet_type, account_receiver, description_receiver, receiver_party)
    #print("\n=== P_JUDGE PROMPTS ===")
    #print("SYSTEM:", sys[:200]); print("USER  :", u[:200])

    judge_obj, judge_raw = chat_json(JUDGE_MODEL, sys, u, TEMP_JUDGE, JudgeSchema)
    #print("JUDGE RAW :", judge_raw)
    #print("JUDGE OBJ :", judge_obj)

    return {
        "A1": A1_raw, "A2": A2_raw, "A3": A3_raw,
        "adv_favor": adv_raw["favor"],
        "adv_against": adv_raw["against"],
        "adv_neutral": adv_raw["neutral"],
        "final_stance": judge_obj.final_stance,
        "confidence": judge_obj.arbiter_confidence_0_1,
        "explanation": judge_obj.justification
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="CSV or Parquet with text_sender, target_policy, account_sender, description_sender, actor_type_sender, tweet_type, account_receiver, description_receiver, receiver_party")
    ap.add_argument("--outfile", required=True, help="CSV or Parquet")
    ap.add_argument("--bgctx", default="none", help="Optional background text for targets")
    args = ap.parse_args()

    df = pd.read_parquet(args.infile) if args.infile.endswith(".parquet") else pd.read_csv(args.infile)

    rows = []
    # <-- progress bar starts here -->
    for r in tqdm(df.itertuples(index=False), total=len(df), desc="Processing rows"):
        res = run_item(r.text_sender, r.target_policy, r.account_sender, r.description_sender, r.actor_type_sender, r.tweet_type, r.account_receiver, r.description_receiver, r.receiver_party)
        rows.append(
            {
                "id": getattr(r, "id", None),
                "text_sender": r.text_sender,
                "target_policy": r.target_policy,
                "account_sender": r.account_sender,
                "description_sender": r.description_sender,
                "actor_type_sender": r.actor_type_sender,
                "tweet_type": r.tweet_type,
                "account_receiver": r.account_receiver,
                "description_receiver": r.description_receiver,
                "receiver_party": r.receiver_party,
                **res,
            }
        )

    out = pd.DataFrame(rows)
    if args.outfile.endswith(".parquet"):
        out.to_parquet(args.outfile, index=False)
    else:
        out.to_csv(args.outfile, index=False)

if __name__ == "__main__":
    main()