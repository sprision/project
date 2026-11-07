# ARCHITECTURE.md

## High-Level System Architecture

The system is organized into decoupled layers: a thin HTTP API presentation layer, a bounded tool-calling orchestrator agent, a vision anomaly detection core, a hybrid dense+lexical technical retriever, an extractive/deterministic report generator, and an SQLite persistence layer.

```
                    ┌──────────────────────────────────────────────┐
                    │               User / Browser                 │
                    │        (Modern Industrial Dashboard)         │
                    └──────────────────────┬───────────────────────┘
                                           │ HTTP / JSON / Multipart
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                FastAPI Backend (api/routes.py)                         │
│   Endpoints: /health, /categories, /inspect, /kb/query, /kb/ingest, /heatmap, /reports │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              InspectionService (services.py)                           │
│   - Lazy model management (per-category AnomalyInspector caches)                       │
│   - Upload storage (data/uploads/) & Heatmap management (data/heatmaps/)               │
│   - Coordinates ToolContext across retriever, db, inspectors, and agent                │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
      ┌─────────────────────────────┐             ┌─────────────────────────────┐
      │   Vision Anomaly Detector   │             │   Inspection Agent          │
      │   (src/app/vision/model.py) │             │   (src/app/agent/agent.py)  │
      ├─────────────────────────────┤             ├─────────────────────────────┤
      │ • Pretrained ResNet-18      │             │ • Bounded loop (max 8 iters)│
      │ • Gaussian patch modeling   │             │ • Tool dispatch & trace log │
      │ • Cached batched matmul     │             │ • Evidence harvesting       │
      │ • Connected-comp. regions   │             └──────────────┬──────────────┘
      │ • JET heatmap blending      │                            │
      └─────────────────────────────┘                            │ calls tools
                                                                 ▼
      ┌─────────────────────────────────────────────────────────────────────────┐
      │ Tools Registry (src/app/agent/tools.py)                                 │
      │  ├─ inspect_image                 ──▶ AnomalyInspector                  │
      │  ├─ retrieve_technical_documents  ──▶ Retriever (Dense + BM25)         │
      │  ├─ search_inspection_history     ──▶ InspectionDB (SQLite)             │
      │  ├─ compare_inspection_results    ──▶ Delta vs prior inspection         │
      │  ├─ summarize_evidence            ──▶ Evidence counter/synthesizer      │
      │  └─ generate_inspection_report    ──▶ ReportGenerator                   │
      └─────────────────────────────────────────────────────────────────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
      ┌─────────────────────────────┐             ┌─────────────────────────────┐
      │   Hybrid Knowledge Base     │             │   Inspection Database       │
      │   (src/app/retrieval/)      │             │   (src/app/database/db.py)  │
      ├─────────────────────────────┤             ├─────────────────────────────┤
      │ • Dense MiniLM-L6-v2        │             │ • SQLite database           │
      │ • BM25Plus lexical ranker   │             │ • inspections table         │
      │ • Reciprocal Rank Fusion    │             │ • Historical queries & deltas│
      │ • chunks.json + .npz disk   │             │ • Serialized JSON reports   │
      └─────────────────────────────┘             └─────────────────────────────┘
```

---

## Component Boundaries & Contracts

### 1. Vision Layer (`src/app/vision/`)
- **Key Modules:**
  - `model.py` (`PaDiM`, `AnomalyInspector`, `regions_from_map`, `save_overlay`, `validate_image`, `load_calibrated_threshold`). Production model architecture.
  - `registry.py` (`ModelRegistry`, `LRUModelCache`, `CategoryMetadata`). Manages lazy model loading, bounded RAM/VRAM footprint, and dynamic discovery.
  - `patchcore.py` (`PatchCore`). Experimental alternative model architecture implementing position-free nearest-neighbor memory banks for rotationally unconstrained or non-rigid industrial categories.
  - `cable_inspector.py` (`SpecializedCableInspector`, `ConductorContinuityDetector`, `SurfacePunctureDetector`, `CableSpecializedEvidence`). Multi-detector composite inspection engine addressing negative anomalies (missing conductors) and microscopic punctures (poke insulation).
- **Contract:** Isolates PyTorch, TorchVision, and Timm entirely from the API and agent. Produces a validated Pydantic `VisionResult` including `is_anomalous`, `threshold`, `verdict_mode` (`calibrated_raw` or `normalised`), `image_anomaly_score`, `raw_anomaly_score`, `regions` (list of `DefectRegion`), `notes`, and `heatmap_path`.

### 2. Retrieval Layer (`src/app/retrieval/`)
- **Key Modules:** `retriever.py` (`Retriever`), `ingest.py` (`ingest_documents`).
- **Contract:** Takes arbitrary textual queries and returns `RetrievalResult` containing ranked `RetrievedChunk` items, retrieval mode, and attribution metadata. No external vector database daemon required.

### 3. Database Layer (`src/app/database/`)
- **Key Modules:** `db.py` (`InspectionDB`).
- **Contract:** Exposes `insert(report: InspectionReport) -> int`, `search(category, limit) -> list[dict]`, `get(insp_id) -> dict`, and `latest(category) -> dict`. Manages thread-safe connections with row factories.

### 4. Agent Layer (`src/app/agent/`)
- **Key Modules:** `agent.py` (`InspectionAgent`), `tools.py` (`Tool`, `ToolContext`, `ToolResult`, `build_tools`, `run_tool`).
- **Contract:** Receives `image_path`, `question`, optional `vision`, and optional `category`. Dispatches validated calls against tool registry. Harvests `EvidenceItem` records tagged by `EvidenceKind`. Generates `InspectionReport` with execution trace. Never raises uncaught exceptions.

### 5. Reporting Layer (`src/app/reporting/`)
- **Key Modules:** `report.py` (`ReportGenerator`).
- **Contract:** Takes accumulated evidence and structured findings, produces an evidence-grounded `InspectionReport` with distinct findings, recommendations, limitations, tripartite categorization (`confirmed_visual_defects`, `potential_anomalies`, `uncertain_human_review`), and mandatory safety disclaimer.

### 6. Presentation & API Layer (`src/app/api/`, `src/app/main.py`)
- **Key Modules:** `routes.py`, `main.py`.
- **Contract:** Thin FastAPI router. Validates incoming multipart streams (mime/extension/size limit). Translates exceptions into clean HTTP status codes (400, 404, 413, 415, 500). Mounts static assets for the frontend.

---

## Data Flow: Primary Inspection Lifecycle

1. **Client Request:** Browser or HTTP client submits `POST /api/inspect` with image binary, target `category` (optional), and user `question`.
2. **Input Validation:** API checks file extension against allowed whitelist and enforces 10 MB limit.
3. **Execution Routing:**
   - Image saved to a unique UUID path in `data/uploads/`.
   - Category-specific `AnomalyInspector` loaded lazily from cached weights in `data/index/vision/{category}/`.
   - Visual inspection executed: feature extraction -> patch covariance comparison -> anomaly score map -> defect region contouring -> JET heatmap image saved.
4. **Agent Orchestration:**
   - Agent receives `VisionResult` and begins deliberation loop.
   - Dispatches `retrieve_technical_documents` enriched with visual findings (category, defect area fraction).
   - If question contains historical or comparative tokens (or prior records exist), agent dispatches `search_inspection_history` and `compare_inspection_results`.
   - Summarizes evidence into visual vs technical findings.
   - Calls `generate_inspection_report` to format evidence.
5. **Persistence & Return:**
   - Final `InspectionReport` is inserted into SQLite database.
   - Complete report JSON returned to client.

---

## Failure Modes & Resilience Patterns

- **Unfitted Category Model:** Returns a structured `VisionResult` or limitation notice without crashing; agent reports model absence in limitations.
- **Corrupt / Invalid Upload:** Caught at validation stage (`validate_image`) returning clear HTTP 400 or HTTP 415.
- **Empty Knowledge Base:** Retriever returns `note="no documents indexed"`; agent reports limitation that recommendations lack technical grounding.
- **LLM Unavailable / Failed:** Automatic silent fallback to deterministic extractive report generator; limitation is recorded in report stating fallback reason.

---

## Multi-Category Scaling & Model Registry Architecture (8–10 Category Expansion Plan)

To transition from the initial 3-category baseline (`bottle`, `cable`, `metal_nut`) to an enterprise-grade 10-category platform without breaking existing models or core inspection logic:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Dynamic Model Registry (data/index/vision/)                     │
│                                                                                        │
│  bottle/          cable/           metal_nut/       capsule/        screw/     ...     │
│  ├ padim.pkl      ├ padim.pkl      ├ padim.pkl      ├ padim.pkl     ├ padim.pkl        │
│  └ threshold.json └ threshold.json └ threshold.json └ threshold.json └ threshold.json  │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                           InspectionService (services.py)                              │
│                                                                                        │
│  - Dynamic Category Discovery: Scans index directory for valid (padim.pkl, threshold)  │
│  - LRU Model Cache (Capacity = 3): Keeps max 3 models in memory (~1.98 GB cap)         │
│  - Transparent Lazy Loading: Loads requested category on-demand; evicts LRU            │
│  - API Integration: GET /api/categories exposes active categories to frontend          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Architectural Principles for Multi-Category Scaling:
1. **Zero Core Logic Modification:** Core modules (`model.py`, `agent.py`, `tools.py`, `routes.py`) operate on abstract category identifiers. Adding a category requires only writing its fitted pickle and threshold JSON to disk.
2. **LRU In-Memory Cache:** PaDiM models are ~662 MB each. An LRU cache with capacity 3 ensures memory consumption never exceeds ~2.0 GB RAM/VRAM regardless of how many models are stored on disk.
3. **Automated End-to-End Pipeline:** Each category uses identical execution scripts:
   - `scripts.download_mvtec` -> `scripts.fit_padim` -> `scripts.calibrate_thresholds` -> `scripts.evaluate_cv` -> `scripts.validate_anomaly_detection`.
4. **Knowledge Base Parity:** New categories register corresponding technical inspection guidelines into `data/kb/` so agent retrieval citations remain grounded.


