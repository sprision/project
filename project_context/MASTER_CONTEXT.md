# MASTER_CONTEXT.md

## Project Identity & High-Level Mission

- **Project Title:** Multimodal AI Agent for Industrial Inspection and Maintenance Decision Support
- **Project Root:** `C:\Users\ayush\OneDrive\Desktop\project`
- **Domain:** Industrial computer vision anomaly detection, technical knowledge retrieval (RAG), and bounded agent-driven decision support for maintenance workflows.
- **Goal:** Provide human inspection teams with an automated, defensible triage assistant that detects anomalies on equipment surfaces, localizes defect regions with heatmap overlays, retrieves governing technical documentation and maintenance procedures, queries historical inspection records, and generates cited, structured inspection reports.

---

## Core Technologies & Stack

### Vision & Anomaly Detection
- **Framework & Libraries:** PyTorch (`2.14.1+cu126`), `torchvision`, `timm` (1.0.15), OpenCV (`opencv-python-headless`), SciPy, PIL, NumPy.
- **Model Architecture:** PaDiM (Patch Distribution Modeling — Defard et al., ICPR 2021).
  - Pretrained ImageNet feature extractor: `resnet18` (via `timm`, features-only, extracting layers 1, 2, and 3 upsampled and concatenated).
  - Random dimensionality reduction: 100 features retained (seed fixed at 1024).
  - Unsupervised Gaussian patch modeling: Per-location multivariate Gaussian (mean and covariance matrix) computed across defect-free training images.
  - Anomaly map computation: Mahalanobis distance per patch, accelerated via cached batched matrix multiplication (`torch.bmm`), followed by 2D Gaussian smoothing ($\sigma=4$) and bilinear interpolation upsampled to input size (256x256).
  - Region segmentation: Connected-component analysis (`cv2.connectedComponentsWithStats`) extracting bounding boxes, pixel areas, centroids, and mean anomaly scores.

### Retrieval-Augmented Generation (RAG)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, normalized).
- **Lexical Search:** BM25Plus (`rank_bm25`), chosen to prevent negative IDF scores on small corpora.
- **Fusion:** Hybrid Reciprocal Rank Fusion (RRF, $k=60$).
- **Storage:** Standalone local disk storage (`data/index/chunks.json` for chunk metadata, `data/index/embeddings.npz` for numpy embeddings vector matrix).
- **Ingestion Pipeline:** `src/app/retrieval/ingest.py` supporting `.pdf`, `.txt`, `.md`, `.rst` with paragraph-aware chunking and section header extraction.

### Orchestration & Agent
- **Agent Architecture:** Single bounded tool-calling orchestrator (`src/app/agent/agent.py`).
- **Policy:** Deterministic/observation-driven tool selection (max iterations = 8). Inspects image -> retrieves technical specs based on category and defect extent -> searches historical trends if requested or relevant -> summarizes evidence -> compiles report.
- **Tools (6 deterministic tools):**
  1. `inspect_image`: Evaluates uploaded image with category-specific PaDiM model.
  2. `retrieve_technical_documents`: Hybrid search over technical KB.
  3. `search_inspection_history`: SQLite query by category/verdict.
  4. `compare_inspection_results`: Computes score deltas against prior inspection.
  5. `summarize_evidence`: Synthesizes visual observations and retrieved facts.
  6. `generate_inspection_report`: Compiles cited, structured report.

### Persistence & Storage
- **Database:** SQLite (`data/index/inspections.sqlite`) storing inspection runs, scores, verdicts, region counts, full report JSON, and timestamps.
- **Uploads & Heatmaps:** `data/uploads/` (temporary uploads) and `data/heatmaps/` (saved JET colormap overlays).

### Backend & API
- **Framework:** FastAPI (0.115.8), Uvicorn, Pydantic (2.10.6), Pydantic Settings (2.8.0).
- **Endpoints:**
  - `GET /api/health` — Service health, device, available models, KB chunks, LLM status.
  - `GET /api/categories` — Available fitted categories.
  - `POST /api/inspect` — Multipart upload (image + category + question) executing the agent pipeline.
  - `GET /api/kb/query` — Semantic search endpoint.
  - `POST /api/kb/ingest` — Document upload and indexing endpoint.
  - `GET /api/heatmap/{name}` — Heatmap PNG retrieval.
  - `GET /api/reports` — Historical inspections list.
  - `GET /api/reports/{id}` — Inspection report detail by ID.

### Frontend
- **Interface:** Single-page dashboard served directly by FastAPI (`frontend/index.html`).

---

## Complete Known Project History

1. **Foundational Implementation:**
   - ResNet-18 PaDiM pipeline constructed.
   - MVTec AD dataset downloaded for three categories: `bottle`, `cable`, `metal_nut` (via Hugging Face mirror `foersben/mvtec-ad`).
   - Models fitted on defect-free training sets and saved to `data/index/vision/{category}/padim.pkl`.
   - Technical documents authored and seeded into the hybrid RAG index: `mvtec_ad_dataset.txt`, `padim_method.txt`, `surface_inspection_guidance.txt`, `maintenance_decision_support.txt`.
   - Deterministic 6-tool agent orchestrator and SQLite history database implemented.
   - FastAPI server and initial HTML frontend created.

2. **Inference Optimization:**
   - In `src/app/vision/model.py`, Mahalanobis distance calculation was optimized from naive `torch.einsum` (~1ms per patch location, ~16s total) to cached batched matrix multiplication (`torch.bmm`) with contiguous `(HW, C, C)` tensor layout, reducing inference time to sub-second on GPU / fast CPU.

3. **Benchmarking & Evaluation:**
   - Computer vision evaluation run (`runs/eval_cv.json`):
     - `bottle` (83 test images, 63 defective): Image AUROC = 1.0000, Pixel AUROC = 0.9841.
     - `cable` (150 test images, 92 defective): Image AUROC = 0.8413, Pixel AUROC = 0.9516.
     - `metal_nut` (115 test images, 93 defective): Image AUROC = 0.9741, Pixel AUROC = 0.9630.
   - Retrieval evaluation run (`runs/eval_retrieval.json`):
     - 8 technical queries evaluated: Recall@1 = 0.75, Recall@3 = 0.875, Recall@5 = 1.000, MRR = 0.844.

4. **Threshold Calibration:**
   - Threshold calibration script executed (`scripts/calibrate_thresholds.py`), generating `data/index/vision/{category}/threshold.json` and `runs/calibration.json`:
     - `bottle`: raw threshold = 23.4613 (Balanced Accuracy = 1.0, FPR = 0.0, 50-seed stability mean = 0.9751).
     - `cable`: raw threshold = 30.4218 (Balanced Accuracy = 0.8306 on calibration, 0.7181 on held-out, FPR = 0.1724, 50-seed stability mean = 0.7373). Documented notable normal/defective score overlap.
     - `metal_nut`: raw threshold = 23.1698 (Balanced Accuracy = 0.8785 on calibration, 0.9787 on held-out, FPR = 0.0, 50-seed stability mean = 0.9109).
   - Support for `calibrated_raw` verdict mode integrated into `src/app/vision/model.py`, `src/app/services/services.py`, and `src/app/reporting/report.py`.

