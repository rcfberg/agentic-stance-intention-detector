# Repository Overview

This repository implements a complete computational pipeline for:

- **Topic-driven clustering** of Finnish social-media posts (BERTopic).
- **Agentic LLM pipelines** that output stance (trinary) and intention (six-class) labels using the **Gemma 3-4B** model via Ollama.
- **Narrative embeddings** that combine text, stance, intention, and explanatory text into a shared vector space.
- **Community detection & thematic analysis** to uncover distinct narrative clusters among elite actors surrounding the public-debt-limit policy debate in Finland.

The three files are self-contained scripts/notebooks that can be executed sequentially:

| # | File                     | Role in the pipeline                                                                 |
|---|--------------------------|--------------------------------------------------------------------------------------|
| 1 | `cola_stance_ollama.py`  | Stance detection – runs an agentic LLM (Gemma 3-4B) to assign a trinary stance (in-favor, against, irrelevant) to each tweet. |
| 2 | `sisu_intent_ollama.py`  | Intention detection – extends the same agentic architecture to produce one of six intention categories (e.g., public-education, target-policymaker, etc.). |
| 3 | `narrativeEmbeddings.ipynb` | Narrative embedding & community analysis – builds enriched tweet embeddings, runs cosine similarity + Louvain clustering to reveal narrative communities; includes visualisation and thematic annotation. |
