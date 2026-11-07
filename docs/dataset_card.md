# Dataset card — MVTec AD

| | |
|---|---|
| **Dataset** | MVTec AD (MVTec Anomaly Detection) |
| **Paper** | Bergmann, Fauser, Sattlegger, Steger. *"MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection."* CVPR 2019. |
| **Task** | Unsupervised anomaly detection + pixel-level localisation (NOT supervised classification) |
| **Size** | 5,354 images, 15 categories, 70+ defect types |
| **Annotations** | pixel-precise ground-truth masks for every defective test image |
| **License** | **CC-BY-NC-SA-4.0** — non-commercial use only. Appropriate for this college/portfolio project; **not** for commercial deployment. |

## Access

Official mydrive download links are **dead (HTTP 404)** as of Oct 2026.
This project uses the HuggingFace mirror **`foersben/mvtec-ad`**, which mirrors
the complete dataset (train/good, test/&lt;defect&gt;, ground_truth masks) under
the same license. Download is handled by `scripts/download_mvtec.py`
(selective per-category to limit footprint).

## Why this dataset fits the project

- The canonical industrial-inspection benchmark — immediately recognisable.
- The task is *anomaly detection*: only defect-free images are used for fitting,
  matching real maintenance where defects are scarce. No fabricated labels.
- Pixel-precise masks → genuine segmentation-level evaluation (image- and
  pixel-AUROC), and real defect-localisation overlays for the UI.
- Unsupervised + pretrained backbone → feasible on a single 6 GB GPU / CPU.

## Categories used in the MVP

`bottle`, `cable`, `metal_nut` — a mix of object and texture categories.
The downloader supports all 15; additional categories require only re-running
`download_mvtec.py` + `fit_padim.py`, no code changes.

## Splits & leakage prevention

- **Train** (`train/good/`): only defect-free images → PaDiM Gaussian fitting.
- **Test** (`test/`): held-out defective + defect-free images → evaluation only.
- No test image (or its label/mask) is used during fitting — the PaDiM model is
  fit exclusively on `train/good`, so there is **no train/test leakage**.
- Normalisation (min/max of train-good anomaly scores) is also computed on the
  training split only.

## Known limitations

- Anomaly detection does **not** identify the *type* or *severity* of a defect —
  a high score is a hypothesis of abnormality, not a diagnosis.
- Scores are relative to the fitted normal distribution and are **not**
  calibrated probabilities.
- Non-commercial license: fine for coursework, not for production/commercial use.
