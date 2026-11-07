# BUGS_AND_ISSUES.md

## Known Bugs, Model Limitations & Operational Issues

### 1. `cable` Category Score Overlap & Sensitivity Limitations (Detailed Analysis)
- **Component:** `src/app/vision/model.py` / `data/index/vision/cable/padim.pkl`
- **Description:** PaDiM anomaly scores for defective and defect-free cable samples exhibit noticeable overlap, yielding an image AUROC of 0.8413 (apparent full test) / 0.7961 (strict held-out evaluation), and a held-out sensitivity of **60.87%** (28 detected, 18 missed).
- **Defect Breakdown & Root Cause Findings (Evaluated Across 150 Test Images):**
  - **Coarse Surface & Geometry Defects (High Recall):**
    - `cut_outer_insulation`: 10/10 detected (**100% recall**, mean score 42.14 vs threshold 30.4218)
    - `missing_cable`: 11/12 detected (**91.7% recall**, mean score 33.55)
    - `combined`: 10/11 detected (**90.9% recall**, mean score 41.86)
    - `bent_wire`: 11/13 detected (**84.6% recall**, mean score 42.24)
    - `cable_swap`: 9/12 detected (**75.0% recall**, mean score 34.02)
    - `cut_inner_insulation`: 8/14 detected (**57.1% recall**, mean score 33.37)
  - **Failure Modes Driving Low Sensitivity:**
    - **`missing_wire` (1/10 detected, 90.0% missed!):** Score range $[15.84, 32.78]$, **mean score = 21.74**.
      *Root Cause (Negative Anomaly Problem):* When an inner colored wire is missing, the image exposes the plain light-grey background surface. Because cables are flexible, training set cables had slight geometric bend variations, so the background was frequently observed at patch $(i, j)$ during normal training. Uniform background features produce tiny Mahalanobis distances that match the background Gaussian distribution (scoring lower than clean cables: 21.74 vs 22.77).
    - **`poke_insulation` (0/10 detected, 100.0% missed!):** Score range $[21.22, 27.92]$, **mean score = 24.87**.
      *Root Cause (Resolution & Gaussian Smoothing):* Punctures are sub-millimeter pinholes in the cable jacket. Layer3 features downsample by $16\times$, and PaDiM's post-processing applies `gaussian_filter(score_map, sigma=4)`. High-frequency microscopic disruptions are diluted by the surrounding black rubber jacket.
  - **Clean Cable Variance & Threshold Compression:**
    - Normal twisted cables produce geometric strain scores up to 35.80 (mean 22.77). Setting the threshold at 30.4218 was required during calibration to control false alarms (specificity 89.66%), but this high threshold compresses the decision boundary and guarantees missing low-activation defects.
- **Mitigation:** The system flags cable inspections as requiring visual verification on conductor counts and surfaces candidate defect regions. For critical cable QA, specialized object-detection models (counting wires) or higher-resolution patch models (e.g., $1024\times 1024$ without Gaussian smoothing) are recommended.

---

### 2. Crop-Boundary False Positive on Normal Metal Nut (Sample `006.png`)
- **Component:** `src/app/vision/model.py` / `metal_nut`
- **Description:** During full 115-sample test validation, 21 of 22 normal metal_nut images correctly scored below calibrated threshold 23.1698 (specificity 95.45%). One sample (`metal_nut/test/good/006.png`) scored 30.0865.
- **Root Cause:** Connected component analysis localized a single defect region at `y=0` (`bbox=(42, 0, 21, 18)`), corresponding to a bright specular edge reflection touching the top crop boundary.
- **Status:** Documented model edge characteristic. Consistent with calibration stability log (`runs/calibration.json` 50-seed eval FPR mean = 10.36%).

---

### 3. `screw` Category Thread Rotation Score Variance & Evaluation Disparity (Domain Characteristic)
- **Component:** `src/app/vision/model.py` / `screw` PaDiM model
- **Description:** On the MVTec AD screw test set, the model achieves a pixel-level AUROC of **0.9726**, confirming defect region localization (identifying manipulated fronts, damaged threads, and head scratches). However, image-level AUROC is **0.6280** (apparent full test) / **0.5843** (strict held-out), with a false-positive rate of **71.43%** on untouched held-out clean screws (specificity **28.57%**, balanced accuracy **50.73%**).
- **Empirical Investigation & Root Cause Findings:**
  1. **Score Overlap Across Subdirectories:** Clean test screws exhibit raw max scores averaging **25.12** (range 17.41 to 35.27), which already exceeds the calibrated threshold of **24.1733**. At the same time, subtle defect types have max scores that completely overlap with clean screws: `scratch_neck` (avg 24.62), `thread_side` (avg 26.14), `scratch_head` (avg 26.54), `thread_top` (avg 27.56). Only coarse defects like `manipulated_front` (avg 32.70) reliably exceed all clean screws.
  2. **Pooling Aggregation Analysis:** Testing alternative pooling aggregations on the anomaly maps proved that pooling is not the root cause:
     - Global Max Pooling Image AUROC: `0.6280`
     - Top-10 Mean Pooling Image AUROC: `0.6270`
     - Top-50 Mean Pooling Image AUROC: `0.6259`
     - 99.5th Percentile Image AUROC: `0.6186`
     - 99.0th Percentile Image AUROC: `0.6182`
  3. **Architectural Mechanism (Spatial Registration vs Helical Rotation):**
     PaDiM assumes spatial registration: at each spatial patch position $(i, j)$, it fits a multivariate Gaussian $\mathcal{N}(\mu_{ij}, \Sigma_{ij})$ across normal training images.
     Screws are 3D cylindrical objects with helical thread ridges, photographed at arbitrary rotation angles ($0^\circ$ to $360^\circ$). Because the 320 normal training screws cannot cover all continuous 3D rotations, unseen rotation angles shift helical crests into positions where the model learned valleys. This produces patch-level Mahalanobis distance spikes (25 to 35) on normal thread perimeters and specular metallic boundaries.
  4. **Why Pixel AUROC is 0.9726 while Image AUROC is 0.6280:**
     Across 8 million total test pixels, ~99.7% of pixels are non-defective (dark interior or background) with low scores ($\approx 8$), and defect pixels are higher ($\approx 30$). Pixel AUROC evaluates the ranking across all 8 million pixels, where background vastly outnumbers thread perimeter artifacts. But Image AUROC relies on the single worst patch per image: because rotational thread shifts generate localized peaks of 25–35 on clean screws, clean image max scores breach the decision threshold just like real defects.
- **Mitigation & Operational Protocol:**
  - The system alerts operators: `"Screw category exhibits high pixel AUROC (0.973) for defect localization, but arbitrary screw thread rotation angles can elevate image-level anomaly scores on normal screws; verify thread defects against localized candidate regions."`
  - High human-review priority is assigned to screw alerts; operators inspect localized contour heatmaps rather than trusting binary image verdicts.

---

### 4. Calibration-Overlap Contamination in Historical Validation Reports (Classified & Resolved)
- **Component:** `scripts/validate_anomaly_detection.py` / `runs/validation_results.json`
- **Audit Findings:**
  1. `scripts/calibrate_thresholds.py` partitions the MVTec AD test set into a 50% calibration split (`cal`) and a 50% evaluation split (`ev`) using seed 42. `pick_threshold` maximizes balanced accuracy solely on `cal`.
  2. However, `scripts/validate_anomaly_detection.py` originally iterated over all folders in `raw_root / <category> / test`, evaluating the entire test set (`cal` + `ev`).
  3. Consequently, 50% of the samples evaluated in `runs/validation_results.json` were present in the pool used to pick the decision threshold.
  4. Formally classified historical full-test confusion matrices as **threshold-selection-contaminated (apparent metrics)**.
- **Resolution & Strict Separation Protocol:**
  - Updated `scripts/validate_anomaly_detection.py` with explicit `--split` modes:
    - `--split held_out`: (Default) Evaluates strictly untouched `ev` split. Zero threshold selection leakage. Results saved to `runs/held_out_evaluation.json`.
    - `--split full`: Evaluates all test samples, but explicitly marks the report as `is_contaminated = True` with a contamination notice. Results saved to `runs/validation_results.json`.
    - `--split calibration`: Evaluates only `cal` split.
  - Added unit tests in `tests/unit/test_threshold_calibration.py` verifying that calibration and held-out paths are strictly disjoint across all categories: `set(cal).isdisjoint(set(ev))`.
  - Recalculated strict held-out metrics across all 4 categories. Historical benchmark files preserved intact.

---

### 4. Hugging Face Hub Online Check Latency in Sandboxed/Isolated Environments
- **Component:** `src/app/retrieval/retriever.py`
- **Description:** Initializing `SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')` defaults to checking Hugging Face Hub online for new commit revisions. In network-isolated or strictly sandboxed runtimes, HTTP socket connection attempts may wait for socket timeouts before falling back to local cached snapshots.
- **Impact:** Initial retriever initialization or unit test discovery can experience startup latency (~30-60s) unless `HF_HUB_OFFLINE=1` or `local_files_only=True` is enforced.
- **Recommended Remediation:** Ensure `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set in `.env` or passed explicitly in environment configuration when running offline.

---

### 5. LLM API Outage / Provider Failures (Handled by Fallback)
- **Component:** `src/app/reporting/report.py`
- **Description:** Historical coding sessions recorded external API provider errors when invoking remote LLM endpoints.
- **Impact:** When external LLM APIs fail or when API keys are absent, non-deterministic report rewriting cannot complete.
- **Status:** **Resolved by Design.** The application implements an extractive deterministic report generator as the default (`APP_LLM_PROVIDER=none`). If an LLM provider is requested but fails, it catches the exception and appends a limitation notice: `"LLM report generation failed; used deterministic generator instead."`

---

### 6. Report Generation Data-Contract Mismatch (Resolved)
- **Component:** `src/app/agent/tools.py` (`_t_report`) & `src/app/reporting/report.py` (`generate()`, `_deterministic()`)
- **Description:** When the inspection agent deliberated and executed `generate_inspection_report`, it serialized the retrieval result as a dict (`retrieval.model_dump() if retrieval else None`). The tool implementation `_t_report` forwarded this dictionary directly to `report_fn` without deserializing. When `_deterministic()` attempted `if retrieval and not retrieval.chunks:`, it failed with:
  `AttributeError: 'dict' object has no attribute 'chunks'`.
- **Status:** **Resolved.**
  1. Updated `_t_report` in `src/app/agent/tools.py` to deserialize dictionary `retrieval` arguments into `RetrievalResult(**ret_arg)` and ensure `evidence` items are coerced to `EvidenceItem`.
  2. Updated `generate()` and `_deterministic()` in `src/app/reporting/report.py` to defensively coerce dictionary inputs to `RetrievalResult(**retrieval)` and use safe `getattr(retrieval, "chunks", None)` / `.get("chunks", [])` chunk extraction.
  3. Added regression tests in `tests/unit/test_report.py` (`test_regression_retrieval_as_dict`, `test_regression_retrieval_as_dict_empty_chunks`) and `tests/unit/test_agent.py` (`test_regression_generate_report_with_dict_retrieval`).
  4. Verified all 43 unit tests passing.

---

### 7. Git Large File Storage (LFS) for PaDiM Pickles
- **Component:** `data/index/vision/{category}/padim.pkl`
- **Description:** Each fitted PaDiM model file is approximately 662 MB (combined total ~2.65 GB across bottle, cable, metal_nut, and screw).
- **Impact:** Exceeds standard Git 100 MB single-file push limits on platforms like GitHub.
- **Status:** Models are locally preserved. Any remote repository sync will require Git LFS tracking or an external model artifact registry (e.g., S3, Hugging Face Hub, or GCS release asset).

---

### 8. Windows PowerShell Syntax Sensitivity
- **Component:** Terminal tooling
- **Description:** Using Linux/POSIX command chaining syntax (e.g. `&&`) in standard Windows PowerShell triggers parser errors (`The token '&&' is not a valid statement separator in this version`).
- **Mitigation:** Use PowerShell statement separator `;` or run individual commands.

