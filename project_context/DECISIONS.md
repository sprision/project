# DECISIONS.md

## Key Architectural and Technical Decisions

### 1. Dataset & Problem Formulation: Unsupervised Anomaly Detection on MVTec AD
- **Decision:** Use MVTec AD (`bottle`, `cable`, `metal_nut`) benchmark instead of supervised classification (e.g., NEU-DET) or synthetic defect generators.
- **Rationale:**
  - In real-world industrial environments, defect classes are highly imbalanced and unforeseen defect topologies arise continuously. Supervised classifiers trained on limited defective samples fail on novel anomalies.
  - MVTec AD provides the canonical industrial inspection benchmark, featuring pixel-precise ground truth masks for rigorous evaluation (image-level and pixel-level AUROC).
  - Training strictly on defect-free samples simulates authentic factory commissioning where only normal components are abundant.
- **Alternatives Considered & Rejected:**
  - *NEU-DET:* Defect classification with VOC XML bounding boxes, but licensing is ambiguous and it does not represent unsupervised surface triage.
  - *YOLO / Defect Detectors:* Require extensive labeled defect bounding boxes which contradicts the unsupervised triage requirement.

---

### 2. Anomaly Detection Architecture: Custom PaDiM over Anomalib or Trained Autoencoders
- **Decision:** Implement PaDiM (Patch Distribution Modeling) directly using a pretrained ResNet-18 feature extractor without gradient training.
- **Rationale:**
  - *No Gradient Training:* Eliminates expensive fine-tuning loops and convergence instabilities. Gaussian modeling of normal patch features is CPU/GPU feasible and fully deterministic.
  - *Interpretability & Localisation:* Produces dense spatial anomaly maps directly from Mahalanobis distances, allowing transparent contouring of defect regions.
  - *Framework Independence:* Implementing PaDiM directly in `src/app/vision/model.py` keeps dependencies light and transparent compared to heavy frameworks like Anomalib v2, which introduce complex abstractions and version churn.
- **Alternatives Considered & Rejected:**
  - *PatchCore:* Delivers slightly higher AUROC on certain textures but requires a massive memory-bank of patch features and greedy coreset subsampling, increasing memory overhead and inference latency.
  - *Reconstruction Autoencoders:* Suffer from reconstruction blurriness and hyperparameter sensitivity during reconstruction error thresholding.

---

### 3. Inference Acceleration: Batched Matrix Multiplication over Naive Einsum
- **Decision:** Replace naive location-by-location `torch.einsum` contraction with cached contiguous transposed covariance tensors `(HW, C, C)` evaluated via `torch.bmm`.
- **Rationale:**
  - The original einsum implementation incurred ~1ms latency per spatial patch location, summing to ~16 seconds per 256x256 image.
  - Batching the Mahalanobis quadratic form across all patch locations simultaneously drops inference latency to under 50ms on GPU and ~200ms on CPU, making interactive web inspection feasible.

---

### 4. Calibration Strategy: Raw-Score Threshold Calibration with 50-Seed Stability Sweep
- **Decision:** Implement threshold calibration on raw Mahalanobis scores (`threshold_raw`) maximizing balanced accuracy on a dedicated split, supported by a 50-seed Monte Carlo stability evaluation.
- **Rationale:**
  - Standard min-max score normalization to $[0, 1]$ depends on extreme outliers in the calibration set and varies across runs.
  - Calibrating on raw score preserves absolute distance from normality.
  - 50-seed evaluation quantifies metric variance across small held-out subsets (e.g., bottle: 10 good / 32 defect; cable: 29 good / 46 defect) rather than over-relying on a single split.
  - Revealed critical domain limitation: `cable` has inherent normal/defective score overlap (image AUROC 0.8413), which calibration mitigates but cannot eliminate.

---

### 5. Orchestration: Deterministic Bounded Tool-Calling Agent
- **Decision:** Implement a bounded deterministic agent policy with an explicit 6-tool registry rather than an unconstrained multi-agent framework or hardcoded linear pipeline.
- **Rationale:**
  - Avoids brittle, non-reproducible LLM agent loops during automated testing.
  - The agent genuinely deliberates: it selects tools dynamically based on visual evidence (e.g., enriching search queries with observed defect fraction) and user question keywords (historical vs technical vs comparative).
  - Can be tested completely offline without external LLM keys or network dependencies.
  - Allows drop-in replacement with an LLM planner in `_decide()` in the future without changing tool interfaces or schemas.

---

### 6. Retrieval Strategy: Hybrid Dense + Lexical Fusion (Local Disk)
- **Decision:** Pair dense MiniLM-L6-v2 embeddings with BM25Plus, fused via Reciprocal Rank Fusion ($k=60$), stored locally on disk as `.npz` and `.json`.
- **Rationale:**
  - Pure vector search often fails on exact alphanumeric part numbers, standard codes, or technical tolerances. Pure BM25 fails on semantic conceptual paraphrasing.
  - BM25Plus ensures real term matches never receive negative scores on small industrial corpora.
  - Persisting directly to disk eliminates the operational complexity of external vector databases (Milvus, Qdrant, Pinecone) while keeping chunk attribution intact.

---

### 7. Reporting: Extractive Deterministic-First with Evidence Grounding
- **Decision:** Base report generator is deterministic and extractive; LLM summarization is strictly optional and falls back gracefully.
- **Rationale:**
  - Industrial inspection reports cannot tolerate hallucinations or fabricated claims.
  - Every single finding item is explicitly typed (`visual_observation`, `retrieved_fact`, `inspection_history`, `hypothesis`, `recommendation`, `limitation`) with source attribution and confidence.
  - Mandatory disclaimer is hardcoded into every report schema: "Decision support only. Not an authorisation to operate, repair or shut down equipment."

---

### 8. Non-Probabilistic Score Presentation & Authority Boundaries in UI
- **Decision:** Present anomaly scores strictly as unnormalized spatial Mahalanobis distance metrics with distance margins relative to calibrated cut-offs, rather than converting them into pseudo-probabilities (percentages). Enforce persistent authority boundary notices across UI and report artifacts.
- **Rationale:**
  - Mahalanobis distances under Gaussian assumptions on patch feature activations do not represent calibrated posterior defect probabilities $P(\text{defect} \mid x)$. Presenting a score of e.g. "0.85" or "85%" deceives operators into perceiving statistical certainty.
  - Category-specific physical variances (e.g., twisted normal cables or crop-boundary specular glints on metal nuts) produce high distances without true physical failure.
  - Adding an explicit "Borderline Score" alert when $|raw - threshold| < 3.0$ and persistent "Decision Support Notice: Not an autonomous quality-control authority" establishes unambiguous operational accountability.

---

### 9. Multi-Category Scaling & Dynamic Model Registry (10-Category Target)
- **Decision:** Scale the inspection platform to 10 MVTec AD categories by reusing the existing ResNet-18 PaDiM pipeline with dynamic category discovery and LRU model caching, preserving existing models (`bottle`, `cable`, `metal_nut`) and all benchmark records.
- **Target Selection:** Add 7 high-value industrial categories across pharmaceuticals (`capsule`, `pill`), mechanical fasteners (`screw`), electronics (`transistor`), surface materials (`tile`), mesh filtration (`grid`), and industrial closures (`zipper`).
- **Rationale:**
  - *Pipeline Determinism:* The existing PaDiM architecture (patch Gaussians + batched matmul) is completely domain-agnostic and trains without gradient descent.
  - *Storage & Memory Bound:* 10 models require 6.62 GB on disk. Implementing an LRU cache (capacity = 3) bounds active GPU/RAM footprint to $\approx 1.98\text{ GB}$.
  - *Zero Downtime / Zero Code Changes:* Because `InspectionService.available_categories()` dynamically interrogates `data/index/vision/`, new models become immediately discoverable by the API and dashboard upon disk write.



