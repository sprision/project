# BUGS_AND_ISSUES.md

## Known Bugs, Model Limitations & Operational Issues

### 1. `cable` Category Score Overlap (Model Limitation)
- **Component:** `src/app/vision/model.py` / `data/index/vision/cable/padim.pkl`
- **Description:** PaDiM anomaly scores for defective and defect-free cable samples exhibit noticeable overlap, yielding an image AUROC of 0.8413 (compared to 1.0000 for bottle and 0.9741 for metal_nut).
- **Validation Findings (Full 150 Test Images):**
  - **Clean Samples (58):** 52 correctly classified (TN), 6 false alarms (FP) on twisted/bent cables (`007.png`, `008.png`, `016.png`, `019.png`, `036.png`, `041.png`) scoring 31.42 to 35.80 vs threshold 30.4218. Specificity is 89.66%, FPR is 10.34%.
  - **Defective Samples (92):** 60 detected (TP), 32 missed (FN). Coarse defects (`cut_outer_insulation`, `bent_wire`, `combined`, `missing_cable`) achieve 85–100% recall. Low-patch-activation defects (`missing_wire` missing 9/10, `poke_insulation` missing 10/10) score between 15.8 and 27.9 due to Gaussian smoothing over small spatial disruptions and missing conductor features matching normal background.
- **Impact:** Any single threshold produces either false alarms on bent/twisted normal cables or misses subtle insulation cracks/missing wires.
- **Mitigation:** The system explicitly classifies findings as hypotheses rather than definitive diagnoses and surfaces candidate defect regions for manual human visual inspection.

---

### 2. Crop-Boundary False Positive on Normal Metal Nut (Sample `006.png`)
- **Component:** `src/app/vision/model.py` / `metal_nut`
- **Description:** During full 115-sample test validation, 21 of 22 normal metal_nut images correctly scored below calibrated threshold 23.1698 (specificity 95.45%). One sample (`metal_nut/test/good/006.png`) scored 30.0865.
- **Root Cause:** Connected component analysis localized a single defect region at `y=0` (`bbox=(42, 0, 21, 18)`), corresponding to a bright specular edge reflection touching the top crop boundary.
- **Status:** Documented model edge characteristic. Consistent with calibration stability log (`runs/calibration.json` 50-seed eval FPR mean = 10.36%).

---

### 2. Hugging Face Hub Online Check Latency in Sandboxed/Isolated Environments
- **Component:** `src/app/retrieval/retriever.py`
- **Description:** Initializing `SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')` defaults to checking Hugging Face Hub online for new commit revisions. In network-isolated or strictly sandboxed runtimes, HTTP socket connection attempts may wait for socket timeouts before falling back to local cached snapshots.
- **Impact:** Initial retriever initialization or unit test discovery can experience startup latency (~30-60s) unless `HF_HUB_OFFLINE=1` or `local_files_only=True` is enforced.
- **Recommended Remediation:** Ensure `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set in `.env` or passed explicitly in environment configuration when running offline.

---

### 3. LLM API Outage / Provider Failures (Handled by Fallback)
- **Component:** `src/app/reporting/report.py`
- **Description:** Historical coding sessions recorded external API provider errors when invoking remote LLM endpoints.
- **Impact:** When external LLM APIs fail or when API keys are absent, non-deterministic report rewriting cannot complete.
- **Status:** **Resolved by Design.** The application implements an extractive deterministic report generator as the default (`APP_LLM_PROVIDER=none`). If an LLM provider is requested but fails, it catches the exception and appends a limitation notice: `"LLM report generation failed; used deterministic generator instead."`

---

### 4. Report Generation Data-Contract Mismatch (Resolved)
- **Component:** `src/app/agent/tools.py` (`_t_report`) & `src/app/reporting/report.py` (`generate()`, `_deterministic()`)
- **Description:** When the inspection agent deliberated and executed `generate_inspection_report`, it serialized the retrieval result as a dict (`retrieval.model_dump() if retrieval else None`). The tool implementation `_t_report` forwarded this dictionary directly to `report_fn` without deserializing. When `_deterministic()` attempted `if retrieval and not retrieval.chunks:`, it failed with:
  `AttributeError: 'dict' object has no attribute 'chunks'`.
- **Status:** **Resolved.**
  1. Updated `_t_report` in `src/app/agent/tools.py` to deserialize dictionary `retrieval` arguments into `RetrievalResult(**ret_arg)` and ensure `evidence` items are coerced to `EvidenceItem`.
  2. Updated `generate()` and `_deterministic()` in `src/app/reporting/report.py` to defensively coerce dictionary inputs to `RetrievalResult(**retrieval)` and use safe `getattr(retrieval, "chunks", None)` / `.get("chunks", [])` chunk extraction.
  3. Added regression tests in `tests/unit/test_report.py` (`test_regression_retrieval_as_dict`, `test_regression_retrieval_as_dict_empty_chunks`) and `tests/unit/test_agent.py` (`test_regression_generate_report_with_dict_retrieval`).
  4. Verified all 43 unit tests passing.

---

### 5. Git Large File Storage (LFS) for PaDiM Pickles
- **Component:** `data/index/vision/{category}/padim.pkl`
- **Description:** Each fitted PaDiM model file is approximately 662 MB (combined total ~1.98 GB across bottle, cable, and metal_nut).
- **Impact:** Exceeds standard Git 100 MB single-file push limits on platforms like GitHub.
- **Status:** Models are locally preserved. Any remote repository sync will require Git LFS tracking or an external model artifact registry (e.g., S3, Hugging Face Hub, or GCS release asset).

---

### 6. Windows PowerShell Syntax Sensitivity
- **Component:** Terminal tooling
- **Description:** Using Linux/POSIX command chaining syntax (e.g. `&&`) in standard Windows PowerShell triggers parser errors (`The token '&&' is not a valid statement separator in this version`).
- **Mitigation:** Use PowerShell statement separator `;` or run individual commands.

