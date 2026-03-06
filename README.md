# 🔭 Universal LLM Observability Framework

> **Auto-log every OpenAI, Gemini, and Claude API call to MLflow on Databricks — zero changes to your inference code.**

A unified observability layer for multi-provider LLM systems. Built for a Canadian client's production deployment. Tracks latency, token usage, cost, prompts, and responses in a single MLflow dashboard across all three major LLM providers.

---

## Key Features

| Feature | Detail |
|---------|--------|
| **OpenAI tracking** | Native `mlflow.openai.autolog()` |
| **Gemini tracking** | Native `mlflow.gemini.autolog()` |
| **Claude tracking** | Custom `functools.wraps` monkey-patch (no native MLflow support) |
| **DSPy RAG pipeline** | 28,000+ document FAISS corpus with sentence-transformers (1024-dim) |
| **MLflow 3.x evaluation** | `RetrievalGroundedness`, `RelevanceToQuery`, `Safety` scorers |
| **Custom scorers** | Response latency SLA + LLM-as-judge for Canadian-region deployments |

---

## Architecture

```
Your LLM Code (unchanged)
        │
        ▼
┌──────────────────────────────────────────┐
│         Observability Layer              │
│                                          │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │  OpenAI  │  │  Gemini  │  │ Claude │ │
│  │ autolog  │  │ autolog  │  │ patch  │ │
│  └──────────┘  └──────────┘  └────────┘ │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│         MLflow on Databricks             │
│  - Latency per call                      │
│  - Token usage (prompt/completion/total) │
│  - Estimated cost                        │
│  - Full prompt & response logged         │
│  - Custom evaluation scorers             │
└──────────────────────────────────────────┘
```

---

## Repository Structure

```
llm-observability-framework/
├── tracking/
│   ├── tracking_patchv3withlatency.py          # Claude monkey-patch with latency
│   └── tracking_patchBuiltinAutologging.py     # OpenAI + Gemini native autolog
├── notebooks/
│   ├── MLFlow_v2_19thMay_20thJuneWithDatabricks.ipynb   # Core tracking setup
│   ├── DSPyWithMLFLOWWithDatabricks.ipynb               # DSPy RAG + MLflow
│   └── mlflow3_dspy_20250625.ipynb                      # MLflow 3.x trace evaluation
└── README.md
```

---

## Tech Stack

- **LLM Providers**: OpenAI, Google Gemini, Anthropic Claude
- **Tracking**: MLflow 3.x on Databricks
- **RAG**: DSPy 2.6, FAISS, sentence-transformers
- **Evaluation**: RetrievalGroundedness, RelevanceToQuery, Safety scorers
- **Data**: PySpark (for corpus processing at scale)
- **Language**: Python

---

## Quick Start

```bash
git clone https://github.com/bushramaryam/llm-observability-framework
cd llm-observability-framework
pip install mlflow openai google-generativeai anthropic dspy-ai faiss-cpu sentence-transformers
```

### Enable tracking for OpenAI + Gemini (built-in):
```python
import mlflow
mlflow.openai.autolog()
mlflow.gemini.autolog()

# Your existing code — no changes needed
response = openai_client.chat.completions.create(...)
```

### Enable tracking for Claude (custom patch):
```python
from tracking.tracking_patchv3withlatency import patch_claude
patch_claude(anthropic_client)

# Your existing code — no changes needed
response = anthropic_client.messages.create(...)
```

All calls are now automatically logged to MLflow with full metadata.

---

## The Claude Problem (& Solution)

MLflow has no native autolog support for Anthropic's Claude. This repo solves that with a **monkey-patch approach** using `functools.wraps`:

```python
import functools, time, mlflow

def patch_claude(client):
    original = client.messages.create
    @functools.wraps(original)
    def patched(*args, **kwargs):
        start = time.time()
        response = original(*args, **kwargs)
        latency = time.time() - start
        # Log to MLflow
        mlflow.log_metrics({
            "latency": latency,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        })
        mlflow.log_text(str(kwargs.get("messages","")), "prompt.txt")
        mlflow.log_text(response.content[0].text, "response.txt")
        return response
    client.messages.create = patched
```

This requires **zero changes** to your existing Claude inference code.

---

## DSPy RAG Pipeline

The `DSPyWithMLFLOWWithDatabricks.ipynb` notebook implements a full RAG pipeline:

- **Corpus**: 28,000+ documents ingested via PySpark
- **Embeddings**: sentence-transformers (1024-dim vectors)
- **Index**: FAISS for fast approximate nearest-neighbor search
- **Framework**: DSPy 2.6 with optimizable prompts
- **Evaluation**: MLflow 3.x trace-based scoring

---

## Author

**Bushra Maryam** — AI & Backend Engineer  
[LinkedIn](https://www.linkedin.com/in/bushramaryam) · [GitHub](https://github.com/bushramaryam) · [Portfolio](https://bushramaryam.github.io)
