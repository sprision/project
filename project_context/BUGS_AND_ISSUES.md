# BUGS_AND_ISSUES.md

## Known Bugs, Model Limitations & Operational Issues

### 1. `cable` Category Score Overlap (Model Limitation)
- **Component:** `src/app/vision/model.py` / `data/index/vision/cable/padim.pkl`
- **Description:** PaDiM anomaly scores for defective and defect-free cable samples exhibit noticeable overlap, yielding an image AUROC of 0.8413 (compared to 1.0000 for bottle and 0.9741 for metal_nut).
- **Impact:** Any single threshold produces either false alarms on bent/twisted normal cables or misses subtle insulation cracks. In `runs/calibration.json`, the 50-seed Monte Carlo sweep shows an evaluation false positive rate of ~17.2% to 21.2%.
- **Mitigation:** The system explicitly classifies findings as hypotheses rather than definitive diagnoses and surfaces candidate defect regions for manual human visual inspection.

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

### 4. Git Large File Storage (LFS) for PaDiM Pickles
- **Component:** `data/index/vision/{category}/padim.pkl`
- **Description:** Each fitted PaDiM model file is approximately 662 MB (combined total ~1.98 GB across bottle, cable, and metal_nut).
- **Impact:** Exceeds standard Git 100 MB single-file push limits on platforms like GitHub.
- **Status:** Models are locally preserved. Any remote repository sync will require Git LFS tracking or an external model artifact registry (e.g., S3, Hugging Face Hub, or GCS release asset).

---

### 5. Windows PowerShell Syntax Sensitivity
- **Component:** Terminal tooling
- **Description:** Using Linux/POSIX command chaining syntax (e.g. `&&`) in standard Windows PowerShell triggers parser errors (`The token '&&' is not a valid statement separator in this version`).
- **Mitigation:** Use PowerShell statement separator `;` or run individual commands.

