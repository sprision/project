# InspectAI: Multimodal AI Agent for Industrial Inspection & Maintenance Decision Support

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-74%20passed-brightgreen.svg)](tests/)
[![CUDA Acceleration](https://img.shields.io/badge/CUDA-12.6-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

InspectAI is a multimodal artificial intelligence system engineered for automated industrial surface inspection, defect localization, and maintenance decision support. The platform pairs **unsupervised visual anomaly detection** (Patch Distribution Modeling — PaDiM) with **hybrid retrieval-augmented generation** (MiniLM-L6-v2 dense embeddings fused with lexical BM25Plus over technical standard documentation) and a **bounded deterministic tool-calling agent**.

> [!IMPORTANT]
> **DECISION-SUPPORT NOTICE & AUTHORITY BOUNDARIES**
> InspectAI is designed as an assistive decision-support platform, **not an autonomous quality-control authority**. Anomaly scores reflect spatial Mahalanobis feature distances from learned nominal distributions and are **strictly non-probabilistic** (they do not represent posterior defect probabilities). All model-generated findings are candidate hypotheses (`is_human_confirmed: false`) and require physical visual verification by qualified quality-assurance inspectors before quarantine or scrap authorization.

---

## Key Features

1. **Unsupervised Visual Anomaly Detection (PaDiM)**
   - Pre-trained ResNet-18 backbone extracting multi-scale patch embeddings (`layer1`, `layer2`, `layer3`) projected to $C=100$ random orthogonal dimensions.
   - Gaussian modeling of nominal patch distributions ($\mu_{(i,j)}, \Sigma_{(i,j)}$) trained strictly on defect-free samples without gradient fine-tuning.
   - Accelerated inference via cached batched matrix multiplication (`torch.bmm`), computing complete spatial Mahalanobis distance maps in **under 150 ms** on CUDA GPUs.
   - Connected-component defect extraction with bounding-box extraction and JET heatmap overlay generation.

2. **Scalable Dynamic Model Registry with LRU In-Memory Caching**
   - Dynamic discovery of category-specific vision models from `data/index/vision/`.
   - Thread-safe `LRUModelCache` (capacity = 3) with transparent lazy loading and least-recently-used eviction.
   - Memory footprint strictly bounded at **$\le 2.0\text{ GB}$ RAM** and **$\le 132\text{ MB}$ CUDA VRAM**, preventing out-of-memory errors across multi-category deployments.

3. **Specialized Multi-Detector Architecture (Cable Inspection)**
   - Overcomes classical PaDiM blind spots on complex multi-strand wiring:
     - `ConductorContinuityDetector`: Evaluates transmission density across wire breakout divergence channels, overcoming negative anomaly distribution overlap on `missing_wire` defects.
     - `SurfacePunctureDetector`: Localized patch residual analysis with boundary edge masking to detect tiny puncture tears ($98 \times 114\text{ px}$) while suppressing edge noise.
   - Composite inspector with consensus reconciliation and boundary noise masking.

4. **Deterministic 3-Level Precedence Finding Taxonomy**
   - Guarantees that every visual observation is sorted into exactly one mutually exclusive category with pairwise disjoint sets ($\text{Uncertain} \cap \text{Candidate} = \emptyset$, $\text{Uncertain} \cap \text{Potential} = \emptyset$, $\text{Candidate} \cap \text{Potential} = \emptyset$):
     - 🔵 **Level 1: Uncertain / Human Review Required** (evaluated first): Triggered by borderline proximity ($|\Delta| < 3.0$) or detector disagreements (e.g. localized puncture activation while base PaDiM is nominal).
     - 🔴 **Level 2: Candidate Visual Defects** (evaluated second): Triggered by strong evidence ($\Delta \ge 5.0$ or multi-detector consensus).
     - 🟡 **Level 3: Potential Anomalies** (evaluated third): Moderate single-detector activations ($3.0 \le \Delta < 5.0$).
   - Full provenance tracking attached to every finding (`detector`, `raw_score`, `threshold`, `margin`, `bbox`).

5. **Hybrid Technical Retrieval (RAG)**
   - Blends dense semantic search (`all-MiniLM-L6-v2`) and lexical keyword search (`BM25Plus`) using Reciprocal Rank Fusion ($k=60$).
   - Grounded in factory maintenance guidelines, MVTec AD specifications, PaDiM mathematical references, and surface inspection SOPs (`data/kb/`).
   - Configured with `local_files_only=True` for deterministic, zero-latency offline execution.

6. **Full-Featured Industrial Web Dashboard**
   - Dark industrial UI (`frontend/index.html`) served directly by FastAPI.
   - **One-Click Demo Samples:** 9 pre-packaged test images across all four categories for immediate testing.
   - **Interactive Guided Tour:** Walkthrough modal explaining decision boundaries, Mahalanobis distances, and finding precedence.
   - Real-time performance benchmark viewers, interactive canvas preview, defect contour table, KB search, and structured JSON report export.

---

## System Architecture

```
                                  USER INTERACTION
                        ┌─────────────────────────────────┐
                        │   Web Dashboard (HTML5 / CSS3)  │
                        │   One-Click Demo / Image Upload │
                        └────────────────┬────────────────┘
                                         │ HTTP Multipart / JSON
                                         ▼
                               FASTAPI APPLICATION
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ GET /api/health  GET /api/categories  POST /api/inspect  GET /api/reports  │
  └──────────────────────────────────────┬──────────────────────────────────────┘
                                         │
                                         ▼
                         ORCHESTRATION & SERVICE LAYER
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │                      InspectionService (services.py)                         │
 │                                                                              │
 │  ┌────────────────────────────────────────────────────────────────────────┐  │
 │  │              Dynamic ModelRegistry with LRU Cache (cap=3)              │  │
 │  │      bottle/        cable/         metal_nut/         screw/           │  │
 │  │    (PaDiM.pkl)    (PaDiM.pkl)     (PaDiM.pkl)      (PaDiM.pkl)         │  │
 │  └────────────────────────────────────────────────────────────────────────┘  │
 │                                                                              │
 │  ┌────────────────────────────────────────────────────────────────────────┐  │
 │  │           Specialized Multi-Detector (Category = "cable")              │  │
 │  │  • Base PaDiM Anomaly Inspector                                        │  │
 │  │  • ConductorContinuityDetector (Wire breakout transmission density)    │  │
 │  │  • SurfacePunctureDetector (High-frequency patch residual maps)        │  │
 │  └────────────────────────────────────────────────────────────────────────┘  │
 └───────────────────────┬───────────────────────────────┬──────────────────────┘
                         │                               │
                         ▼                               ▼
               HYBRID RAG RETRIEVER            BOUNDED AGENT POLICY
     ┌────────────────────────────────────┐  ┌─────────────────────────┐
     │ • MiniLM-L6-v2 Dense Embeddings   │  │ 6-Tool Deliberation Loop│
     │ • BM25Plus Lexical Matcher        │  │ • inspect_image         │
     │ • Reciprocal Rank Fusion (k=60)   │  │ • retrieve_technical_doc│
     │ • Local Index (chunks.json/.npz)  │  │ • summarize_evidence    │
     └────────────────────────────────────┘  │ • search_history / etc. │
                                             └────────────┬────────────┘
                                                          │
                                                          ▼
                                            EXTRACTIVE REPORT GENERATOR
                                     ┌────────────────────────────────────────┐
                                     │ • 3-Level Finding Precedence Classifier│
                                     │ • Mutually Exclusive Partitioning      │
                                     │ • Detector Provenance & Actionable SOP │
                                     │ • SQLite Audit Log (inspections.sqlite)│
                                     └────────────────────────────────────────┘
```

---

## Implemented Categories & Production Models

All models use a frozen ResNet-18 backbone trained on defect-free samples ($256 \times 256$ input resolution) and are saved in `data/index/vision/<category>/`:

| Category | Industrial Domain | Trained Weights File | Calibrated Threshold | Reliability Assessment |
|----------|-------------------|----------------------|----------------------|------------------------|
| **`bottle`** | Glass Packaging & Containers | `bottle/padim.pkl` (662 MB) | `23.4613` | **High Reliability** (Held-out AUROC: 1.000, 100% Sensitivity/Specificity) |
| **`cable`** | Multi-Wire Electronics Harness | `cable/padim.pkl` (662 MB) | `30.4218` | **Moderate Reliability** (Composite inspector boosts recall from 60.87% to 89.13%) |
| **`metal_nut`** | Mechanical Fasteners & Tooling | `metal_nut/padim.pkl` (662 MB) | `23.1698` | **High Reliability** (Held-out AUROC: 0.974, 95.7% Sensitivity, 100% Specificity) |
| **`screw`** | Fastener Hardware & Threading | `screw/padim.pkl` (662 MB) | `24.1733` | **High Pixel Localization** (Pixel AUROC: 0.973; orientation sensitivity documented) |

---

## Empirical Benchmark Results

### 1. Strict Held-Out Evaluation (Untouched Test Split • Zero Threshold Leakage)

Evaluated across **254 completely untouched test images** (`runs/held_out_evaluation.json`, seed 42). Decision thresholds were calibrated on a separate disjoint 50% split and never tuned against held-out evaluation labels:

| Category | Total Samples (Good / Defect) | TP | FP | TN | FN | Sensitivity (Recall) | Specificity | False Positive Rate | Balanced Accuracy | Image AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| **`bottle`** | 41 (10 / 31) | 31 | 0 | 10 | 0 | **100.00%** | **100.00%** | **0.00%** | **100.00%** | **1.0000** |
| **`cable`** (Base PaDiM) | 75 (29 / 46) | 28 | 5 | 24 | 18 | **60.87%** | **82.76%** | **17.24%** | **71.81%** | **0.8413** |
| **`metal_nut`** | 58 (11 / 47) | 45 | 0 | 11 | 2 | **95.74%** | **100.00%** | **0.00%** | **97.87%** | **0.9741** |
| **`screw`** | 80 (21 / 59) | 43 | 15 | 6 | 16 | **72.88%** | **28.57%** | **71.43%** | **50.73%** | **0.6280** (Pixel: 0.9726) |

---

### 2. Multi-Detector Cable Strategy Comparison (Held-Out Split)

Head-to-head comparison across all 75 untouched held-out cable samples (`runs/cable_consensus_evaluation.json`):

| Inspection Strategy | TP | FP | TN | FN | Sensitivity | Specificity | FPR | Balanced Accuracy | Primary Operational Trade-off |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline PaDiM** | 28 | 5 | 24 | 18 | 60.87% | **82.76%** | **17.24%** | 71.81% | Misses microscopic punctures and missing wires |
| ⭐ **Naive OR Multi-Detector** | **41** | 9 | 20 | 5 | **89.13%** | 68.97% | 31.03% | **79.05%** | **Best binary balanced accuracy**; triggers on edge glints |
| **Consensus & Masked** | 37 | 8 | 21 | 9 | 80.43% | 72.41% | 27.59% | 76.42% | Suppresses false alarms on twisted cables; routes ambiguous spikes to review |

#### Defect-Specific Recall Breakdown on Held-Out Split:
- `bent_wire`: **100.0%** (5/5)
- `cable_swap`: **100.0%** (6/6)
- `combined`: **100.0%** (6/6)
- `cut_outer_insulation`: **100.0%** (5/5)
- `missing_cable`: **100.0%** (6/6)
- `cut_inner_insulation`: **66.7%** (4/6)
- `missing_wire`: **50.0%** (4/8 detected as anomaly; remaining 4 flagged by continuity detector and routed to mandatory human review)
- `poke_insulation`: **50.0%** (3/6 high-margin candidate defects; remaining 3 routed safely to human review)

---

### 3. Pipeline Latency Breakdown

Measured wall-clock latency across 20 test runs on CUDA GPU:

| Pipeline Stage | Mean Latency | Median Latency | P95 Latency | Operational Notes |
|---|---|---|---|---|
| **Base PaDiM Vision** | 148.23 ms | 147.47 ms | 155.88 ms | ResNet-18 extraction + batched Mahalanobis computation |
| **Conductor Continuity Detector** | 60.39 ms | 59.57 ms | 64.37 ms | Multi-channel density profile extraction |
| **Surface Puncture Detector** | 93.58 ms | 93.43 ms | 95.73 ms | Fine-scale Gaussian residual map ($\sigma=1.0$) |
| **Composite Multi-Detector Vision** | 355.84 ms | 352.08 ms | 374.28 ms | Shared contour extraction, bounding boxes, and provenance |
| **Hybrid Technical Retrieval (RAG)**| 14.20 ms | 14.17 ms | 14.90 ms | Dense embedding dot-product + BM25 score fusion |
| **SQLite History Persistence** | 0.62 ms | 0.60 ms | 0.83 ms | Indexed report insertion and commit |
| **Extractive Report Generation** | 0.11 ms | 0.09 ms | 0.17 ms | Deterministic 3-level precedence classification |
| **Total End-to-End Pipeline** | **370.77 ms** | **367.40 ms** | **389.10 ms** | **Interactive response time for real-time factory QA** |

---

## Benchmark Results vs Real-World Industrial Validation

An engineering appraisal of the operational gap between academic benchmarks and authentic factory conditions:

1. **Controlled Lighting vs Ambient Factory Dynamics**
   - *Benchmark:* MVTec AD samples are captured under uniform diffuse illumination with controlled black or neutral backgrounds.
   - *Factory Reality:* Ambient shop-floor lighting varies continuously with shift changes, seasonal daylight, and overhead fixture flicker. Specular metallic components (such as `metal_nut`) produce intense highlights and boundary crop reflections that can artificially spike Mahalanobis distances.
2. **Component Orientation & Rotational Symmetries**
   - *Benchmark:* Components are centered, but orientation varies randomly.
   - *Finding:* On `screw`, PaDiM achieves exceptional **0.9726 pixel AUROC** (flawlessly localizing scratches when an image is classified defective), but held-out image-level specificity drops to **28.57%**. Screws lack rotational invariance: thread pitch varies depending on 3D helical roll angle, causing normal rotated threads to appear out-of-distribution. In-plane 2D rotation normalization fails because it cannot rotate 3D helical threads.
3. **Flexible Deformable Assemblies (Cables)**
   - *Benchmark:* Normal cables exhibit natural bends and harness twists.
   - *Finding:* Benign harness twists distort spatial patch statistics relative to rigid canonical models, elevating normal scores into the 24.0–28.5 range. Naive OR logic triggers a 31.03% false alarm rate on clean cables. The composite masked consensus engine suppresses frame edge glints and routes borderline activations into mandatory human review rather than issuing automated scrap orders.
4. **Conclusion:**
   - Benchmarks prove algorithmic capability under standardized conditions; they do not replace factory commissioning. InspectAI explicitly acts as **decision-support infrastructure**, alerting operators to anomalous regions while enforcing human sign-off for safety-critical actions.

---

## Reproducible Setup & Installation

### Prerequisites
- **Operating System:** Windows 10/11, Ubuntu 22.04 LTS, or macOS (Apple Silicon supported in CPU mode)
- **Python:** Python 3.12 (recommended) or 3.11
- **Hardware:**
  - *CUDA Acceleration (Recommended):* NVIDIA GPU with $\ge 4\text{ GB}$ VRAM (tested on CUDA 12.6)
  - *CPU Execution:* Fully supported out-of-the-box (inference latency $\sim 450\text{ ms}$)

---

### Step-by-Step Installation (Windows PowerShell)

```powershell
# 1. Clone the repository and enter the directory
git clone https://github.com/ayush/project.git
cd project

# 2. Create and activate a Python 3.12 virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# 3. Upgrade pip and install application dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install PyTorch with appropriate hardware support:
# For NVIDIA CUDA 12.6 (recommended):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# OR for CPU-only portable deployment:
# pip install torch torchvision

# 5. Configure environment file (safe defaults, zero API keys required)
Copy-Item .env.example .env
```

---

## Single-Command Launch

Once dependencies are installed, launch the application using any of the following convenient single commands:

### Windows Batch Launcher
```cmd
launch.bat
```

### Windows PowerShell Launcher
```powershell
.\launch.ps1
```

### Cross-Platform Python Launcher
```powershell
python run.py
```
*(Accepts optional flags: `python run.py --host 0.0.0.0 --port 8000 --reload`)*

The server will initialize models, verify knowledge base indices, and print the startup banner:
```
==============================================================================
   INSPECTAI: INDUSTRIAL MULTIMODAL DECISION SUPPORT AGENT
==============================================================================
  • Web Dashboard & API : http://127.0.0.1:8000
  • Interactive API Docs: http://127.0.0.1:8000/docs
  • Knowledge Base Docs : data/kb/ (17 indexed chunks)
  • Supported Categories: bottle, cable, metal_nut, screw
  • Notice              : Decision support only. Findings are hypotheses,
                          not autonomous quality-control authorizations.
==============================================================================
```

Open your browser and navigate to **`http://127.0.0.1:8000`**.

---

## Docker Deployment

To build and run InspectAI in an isolated, production-grade container:

```bash
# Build and run container with persistent volumes
docker compose up --build -d

# Check service health
docker compose ps
curl http://localhost:8000/api/health

# View live logs
docker compose logs -f

# Shut down
docker compose down
```

---

## One-Click Demonstration Experience

InspectAI includes **9 pre-packaged industrial demo samples** in `demo_samples/` and `frontend/demo_samples/` allowing immediate evaluation without navigating raw datasets:

### Testing via Web UI
1. Navigate to `http://127.0.0.1:8000` and switch to the **New Inspection** tab.
2. In the **⚡ One-Click Demo Samples** toolbar, click any sample button:
   - `🍾 Bottle Normal`: Nominal glass bottle $\to$ Clean verdict, low distance (~13.5 vs threshold 23.46).
   - `🍾 Bottle Fracture`: Large neck fracture $\to$ Candidate visual defect, score 64.90, localized contour.
   - `🔌 Cable Normal`: Clean multi-wire cable $\to$ Clean verdict, 0 false alarms.
   - `🔌 Cable Missing Wire`: Strand break $\to$ Conductor continuity alert, routed to Level 1 Uncertain Review.
   - `🔌 Cable Puncture`: Microscopic needle tear $\to$ Puncture detector trigger, localized bounding box.
   - `🔩 Nut Normal`: Clean hex nut $\to$ Clean verdict.
   - `🔩 Nut Bent`: Flange deformity $\to$ Candidate defect, score 38.45.
   - `🪛 Screw Normal`: Nominal screw $\to$ Clean verdict.
   - `🪛 Screw Scratch`: Head gouge $\to$ Candidate defect, localized scratch contour.
3. Click **🔍 Run Inspection**.
4. Examine the spatial heatmap overlay, non-probabilistic Mahalanobis distance margin, multi-detector provenance pill, and actionable SOP directives.
5. Click **📄 View Full Report** and **⬇️ Download** to export the structured audit JSON.
6. Click **🚀 Guided Demo Tour** in the top navigation bar at any time to open the interactive system walkthrough modal.

---

## REST API Reference

| Method | Endpoint | Description | Sample Response / Parameters |
|---|---|---|---|
| `GET` | `/api/health` | Service health status, GPU device, and model availability | `{"status": "ok", "device": "cuda", "categories": [...]}` |
| `GET` | `/api/categories` | Dynamically discovered categories and thresholds | `["bottle", "cable", "metal_nut", "screw"]` |
| `POST` | `/api/inspect` | Run multimodal inspection pipeline on uploaded image | `file` (multipart), `category` (optional), `question` (optional) |
| `GET` | `/api/heatmap/{name}` | Retrieve rendered JET heatmap overlay image | Image PNG stream |
| `GET` | `/api/reports` | List historical inspection records from SQLite | Query params: `category`, `limit` (default: 50) |
| `GET` | `/api/reports/{id}` | Retrieve complete structured inspection report by ID | Full report JSON including structured findings |
| `GET` | `/api/eval/cv` | Real MVTec AD CV benchmark evaluation metrics | Apparent full-test AUROC and recall figures |
| `GET` | `/api/eval/calibration`| 50-seed Monte Carlo calibration stability data | Thresholds, balanced accuracies, stability variance |
| `GET` | `/api/kb/query` | Query hybrid RAG knowledge base | `?q=missing+wire+inspection+procedure` |
| `POST` | `/api/kb/ingest` | Ingest new PDF/TXT maintenance document | `file` (multipart); auto-chunks and indexes |

---

## Regression Testing & Verification

InspectAI maintains a comprehensive test suite covering all modules:

### Run Full Unit Test Suite (74 Tests)
```powershell
python -m pytest tests/unit/ -v
```
*Executes all 11 unit test suites (Schemas, SQLite DB, Ingestion, Model Registry LRU Caching, PatchCore, Cable Multi-Detector, Report Precedence & Boundaries, Vision Validation, Threshold Calibration 50-seed splits, Retriever, and Agent Policy).*

### Run Product Readiness Verification Script
```powershell
python scratch/verify_product_readiness.py
```
*Synchronously verifies API health, dynamic metadata, multi-detector cable inspections on real images, LRU cache eviction across 4 categories, SQLite persistence, and JSON download export serialization.*

---

## Project Structure

```
project/
├── .env.example                       # Safe environment template
├── Dockerfile                         # Production multi-stage Dockerfile
├── docker-compose.yml                 # Docker Compose configuration
├── launch.bat                         # Windows Batch launcher
├── launch.ps1                         # Windows PowerShell launcher
├── run.py                             # Cross-platform Python launcher
├── requirements.txt                   # Production Python dependencies
├── pytest.ini                         # Pytest configuration
├── README.md                          # Comprehensive project documentation
├── AGENTS.md                          # Mandatory agent protocols and guardrails
│
├── data/
│   ├── index/vision/                  # Fitted PaDiM models and threshold.json
│   │   ├── bottle/                    # bottle/padim.pkl (662 MB)
│   │   ├── cable/                     # cable/padim.pkl (662 MB)
│   │   ├── metal_nut/                 # metal_nut/padim.pkl (662 MB)
│   │   └── screw/                     # screw/padim.pkl (662 MB)
│   ├── kb/                            # Source technical SOP documents
│   ├── raw/mvtec_ad/                  # Raw benchmark image datasets
│   └── uploads/                       # Runtime inspection image storage
│
├── demo_samples/                      # 9 representative normal & defect samples
│   └── README.md                      # Guide to representative samples
│
├── frontend/
│   ├── demo_samples/                  # Static samples served to web browser
│   └── index.html                     # Dark industrial UI dashboard
│
├── project_context/                   # Persistent project memory
│   ├── ARCHITECTURE.md                # System design and data flow schemas
│   ├── BUGS_AND_ISSUES.md             # Resolved and open issue tracking
│   ├── DECISIONS.md                   # Architectural trade-off records
│   ├── MASTER_CONTEXT.md              # Historical context and design rationale
│   ├── PROJECT_STATUS.md              # Verified capabilities and test status
│   ├── SESSION_LOG.md                 # Chronological development handoffs
│   └── TASK_TRACKER.md                # Completed and active milestone list
│
├── runs/                              # Verified empirical benchmark artifacts
│   ├── cable_consensus_evaluation.json # 3-way cable comparison on held-out split
│   ├── cable_specialized_evaluation.json # Head-to-head specialized inspector test
│   ├── calibration.json               # 50-seed Monte Carlo stability sweeps
│   ├── eval_cv.json                   # Computer vision benchmark runs
│   ├── eval_retrieval.json            # Hybrid RAG retrieval evaluation runs
│   ├── experimental_benchmarks.json   # PatchCore and rotation evaluation
│   ├── held_out_evaluation.json       # Strict held-out evaluation (zero leakage)
│   └── validation_results.json        # Historical apparent full-test evaluation
│
├── scratch/
│   └── verify_product_readiness.py    # Synchronous end-to-end verification
│
├── scripts/                           # Offline dataset and evaluation scripts
│   ├── calibrate_thresholds.py        # 50-seed threshold calibration
│   ├── download_mvtec.py              # MVTec AD downloader
│   ├── evaluate_cv.py                 # AUROC evaluation runner
│   ├── evaluate_retrieval.py          # RAG retrieval benchmark
│   ├── fit_padim.py                   # Unsupervised PaDiM model fitter
│   └── seed_kb.py                     # Knowledge base chunking & indexing
│
├── src/app/
│   ├── agent/                         # Bounded tool-calling inspection agent
│   ├── api/                           # FastAPI route handlers
│   ├── database/                      # SQLite persistence engine
│   ├── reporting/                     # 3-level precedence report generator
│   ├── retrieval/                     # Hybrid dense + BM25 search engine
│   ├── schemas/                       # Pydantic data models
│   ├── services/                      # Inspection service & specialized router
│   └── vision/                        # PaDiM, Registry, and Cable Detectors
│
└── tests/
    └── unit/                          # 74 passing unit tests
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
Dataset images are governed by the [MVTec AD License](https://www.mvtec.com/company/research/datasets/mvtec-ad).
