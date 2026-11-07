# Multimodal AI Agent for Industrial Inspection & Maintenance Decision Support

An industrial-inspection assistant: upload an equipment image + ask a question in
natural language, and get an evidence-cited inspection report. It combines
**computer vision** (PaDiM anomaly detection with defect localisation),
**retrieval-augmented generation** (hybrid dense + BM25 search over technical
docs), and a **bounded tool-calling agent** that decides which tools to invoke.

> Decision support only — findings are hypotheses, not confirmed diagnoses, and
> the system never authorises operating/repairing equipment.

## Quick start

```bash
# 1. Environment (Python 3.12)
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
# torch: pick the wheel for your hardware — see "Torch install" below.

# 2. Download the dataset subset (MVTec AD, ~3 categories)
python -m scripts.download_mvtec --categories bottle cable metal_nut

# 3. Fit the PaDiM model (unsupervised, on defect-free train images)
python -m scripts.fit_padim --all

# 4. Seed the technical knowledge base + build the retrieval index
python -m scripts.seed_kb

# 5. Run the API + UI
uvicorn src.app.main:app --reload
# open http://127.0.0.1:8000  → upload an image, ask a question
```

## Torch install

`requirements.txt` intentionally omits torch so you select CPU vs CUDA once:

```bash
# CUDA (RTX, ~2.5 GB) — used in this project
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# CPU-only
pip install torch torchvision
```

## What runs where

| Component | File | Notes |
|---|---|---|
| CV model (PaDiM) | `src/app/vision/model.py` | pretrained ResNet-18 + per-patch Gaussians |
| Retrieval / RAG | `src/app/retrieval/` | ingest → chunks → hybrid dense+BM25 index |
| Agent + tools | `src/app/agent/` | bounded tool-caller, deterministic-testable |
| Reports | `src/app/reporting/` | deterministic by default; LLM optional via `.env` |
| API | `src/app/api/routes.py` | `/api/inspect`, `/api/kb/*`, `/api/reports` |
| UI | `frontend/index.html` | single page, served by FastAPI |

## Evaluation (real, measured)

```bash
python -m scripts.evaluate_cv --all          # image + pixel AUROC -> runs/eval_cv.json
python -m scripts.evaluate_retrieval         # Recall@k + MRR -> runs/eval_retrieval.json
pytest                                       # unit + integration tests
```

## Docs

- `docs/architecture.md` — component boundaries & data flow
- `docs/decisions.md` — domain/dataset/model selection & trade-offs
- `docs/dataset_card.md` — MVTec AD facts, license, leakage protocol
- `docs/model_card.md` — PaDiM method, limits, reproducibility
- `docs/evaluation.md` — measured results & limitations

## Independent contributions (for CV + NLP coursework)

- **Computer Vision** — unsupervised anomaly detection + pixel-level
  localisation with a real metric (AUROC) on a benchmark with ground truth.
- **NLP** — document ingestion, metadata-preserving chunking, embeddings +
  hybrid retrieval, source-attributed answers.
- **Agentic AI** — one bounded tool-calling agent that selects tools based on
  the request and observations, with a recorded trace.

## Environment

`.env.example` → `.env`. No API key is needed: the report generator falls back
to a deterministic, extractive mode. Set `APP_LLM_PROVIDER`/`OPENAI_API_KEY` to
enable an optional LLM summary (still grounded in the same cited evidence).
