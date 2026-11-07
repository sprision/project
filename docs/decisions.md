# Engineering decisions

## 1. Domain & dataset selection (Phase One)

Goal: a real, evaluable industrial CV task feasible for one student on a single
GPU. Candidates evaluated against dataset access, annotation quality, task fit,
licensing and compute.

| Candidate | Task | Access | Licensing | Verdict |
|---|---|---|---|---|
| **MVTec AD** | Unsupervised anomaly detection + pixel-level segmentation | Official mydrive links 404 (dead); **full HF mirror `foersben/mvtec-ad`** w/ ground-truth masks | CC-BY-NC-SA-4.0 (non-commercial) | **Chosen** |
| NEU-DET (steel) | 6-class defect detection / classification | HF mirror `KeenForgeAI/NEU-DET-corrected` (VOC XML) | unclear ("other") | Backup |
| VisA | Anomaly detection | HF mirrors exist | similar to MVTec | Backup (larger) |
| Solar panel EL | Defect classification | no reliable public source found quickly | — | Rejected |
| Pipeline / electrical | — | no public benchmark with masks found | — | Rejected |

**Why MVTec AD:**
- The canonical industrial-inspection benchmark — immediately recognisable.
- The task is *visual anomaly detection*, the honest fit for industrial
  inspection where defective examples are scarce. We do **not** force a
  classifier onto data that isn't labelled for it.
- Pixel-precise ground truth → genuine segmentation-style evaluation
  (image- and pixel-level AUROC, plus region overlays).
- Unsupervised: trains on *defect-free* images only, matching the real
  maintenance scenario. No fabricated labels.
- CC-BY-NC-SA-4.0 is fine for a college/portfolio project (non-commercial).

**Scope:** 3 categories to start — `bottle`, `cable`, `metal_nut` (mix of object
and texture). The downloader supports all 15; more can be added without code
changes.

## 2. Model: PaDiM (localisation) instead of training a classifier

We implement **PaDiM** (Defard et al., 2020) directly rather than pull in the
full `anomalib` framework:

- It is a pretrained-backbone (ResNet-18 via `timm`) + multivariate-Gaussian
  patch model — **no gradient training**, CPU-feasible, deterministic, and easy
  to explain and test. That is a "best defensible pretrained baseline".
- `anomalib` v2.x is a large framework; a focused PaDiM keeps the code
  understandable, dependency-light, and avoids silent framework magic.
- Training a defect classifier or a detector (YOLO) is rejected for the MVP:
  it would need either more defect data than unsupervised AD assumes, or a
  training pipeline + tuning budget that doesn't fit a one-student MVP. We note
  it as a limitation / future work.

Alternatives considered: PatchCore (slightly better AUROC, needs a memory bank
of patch features — more memory, similar effort), autoencoder (needs training).
PaDiM is the simplest defensible choice.

## 3. Agent: single bounded tool-caller, deterministic-testable

One orchestrating agent. No multi-agent framework. Tool selection is
rule-based/keyword-driven by default (`APP_AGENT_DETERMINISTIC=1` makes it fully
deterministic), which keeps the core testable without an LLM key and avoids
misrepresenting a fixed pipeline as autonomous — the agent genuinely *chooses*
tools based on the request and observations, within bounded iterations.

## 4. Report generation: deterministic-first

No LLM API key is present in the environment, so the default report generator is
a deterministic, extractive template that assembles cited evidence. An LLM
provider can be enabled via `.env` later without changing the agent/tool code.

## 5. Retrieval: hybrid, local

Dense embeddings (`all-MiniLM-L6-v2`, `sentence-transformers`) + BM25
(`rank_bm25`), fused by reciprocal-rank. Persists embeddings to disk (numpy +
JSON) — a real vector index without a heavy external DB; Chroma/FAISS can be
swapped in later behind the same retriever interface.
