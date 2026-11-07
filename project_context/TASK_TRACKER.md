# TASK_TRACKER.md

## Completed Tasks

- [x] **Project Discovery & Inspection**
  - Inspected repository structure, virtual environment, and dependencies.
  - Verified PyTorch 2.14.1 with CUDA acceleration, torchvision, timm, and sentence-transformers.
  - Verified existing PaDiM models (`bottle`, `cable`, `metal_nut`) in `data/index/vision/`.
  - Verified evaluation artifacts in `runs/eval_cv.json` (Bottle 1.0, Cable 0.8413, Metal Nut 0.9741 AUROC).
  - Verified retrieval evaluation in `runs/eval_retrieval.json` (Recall@5 = 1.000, MRR = 0.844).
  - Verified threshold calibration files (`threshold.json` for 3 categories, `runs/calibration.json`).
  - Verified test suite structure across unit and integration suites (10 test files).

- [x] **PaDiM Model Implementation & Optimization**
  - ResNet-18 multi-scale feature extractor with random feature selection.
  - Per-patch multivariate Gaussian fit on defect-free training images.
  - Vectorized Mahalanobis distance map calculation with cached batched matmul (`torch.bmm`).
  - Min-max score normalization and calibrated raw threshold verdict modes.
  - Connected component defect region contouring and JET colormap overlay.

- [x] **Hybrid RAG Pipeline**
  - Chunk ingestion supporting PDF, TXT, MD, and RST with paragraph boundaries.
  - Sentence-transformers MiniLM-L6-v2 dense embeddings.
  - BM25Plus lexical ranker.
  - Reciprocal Rank Fusion (RRF, $k=60$) search.
  - Disk persistence via `chunks.json` and `embeddings.npz`.

- [x] **Deterministic Inspection Agent**
  - Bounded 6-tool execution agent with observation harvesting.
  - Decision policy handling general queries, historical lookups, and comparative questions.
  - Structured evidence harvesting and classification.
  - Complete execution trace recording.

- [x] **Persistence & Reporting**
  - SQLite database storage with indexing on category and timestamps.
  - Deterministic report generator with cited evidence and disclaimer.
  - Optional LLM summarization hook with automatic fallback.

- [x] **FastAPI Application & API Surface**
  - Implemented endpoints: `/health`, `/categories`, `/inspect`, `/kb/query`, `/kb/ingest`, `/heatmap/{name}`, `/reports`, `/reports/{id}`.
  - Strict upload validation (extension, size, decode).

- [x] **Persistent Project Memory Establishment**
  - Created root `AGENTS.md` defining orientation and handoff protocols.
  - Created `project_context/MASTER_CONTEXT.md`.
  - Created `project_context/PROJECT_STATUS.md`.
  - Created `project_context/ARCHITECTURE.md`.
  - Created `project_context/DECISIONS.md`.
  - Created `project_context/TASK_TRACKER.md`.

---

## Current / Active Tasks

- [ ] **Complete Context Documentation:**
  - Create `project_context/BUGS_AND_ISSUES.md`.
  - Create `project_context/SESSION_LOG.md`.
- [ ] **Verify Test Execution:**
  - Complete execution and logging of full pytest test suite.

---

## Pending Tasks

- [ ] **Local Application End-to-End Verification:**
  - Launch FastAPI application on local port.
  - Execute live test upload using sample image from `data/raw/mvtec_ad/bottle/test/` to verify PaDiM inference, heatmap rendering, agent trace, and SQLite insertion through live API.
  - Verify Knowledge Base querying and document ingestion through live API.
- [ ] **Frontend Verification & Refinement:**
  - Verify full SaaS dashboard interactivity (Overview metrics, New Inspection upload, Anomaly Heatmap viewer, KB Search, History viewer, Report viewer, JSON export).
- [ ] **Deployment Strategy:**
  - Prepare production containerization (`Dockerfile`, `.dockerignore`).
  - Document deployment topology and execution steps.

