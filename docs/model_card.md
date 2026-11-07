# Model card — PaDiM

| | |
|---|---|
| **Model** | PaDiM — Patch Distribution Modeling |
| **Paper** | Defard, Setkov, Loesch, Audigier. *"PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization."* ICPR 2021 (DIPA workshop). |
| **Type** | Pretrained-backbone feature extractor + per-patch multivariate Gaussian (unsupervised, **no gradient training**) |
| **Backbone** | ResNet-18 (ImageNet-pretrained, via `timm`), first three feature levels |
| **Input** | RGB image → resize + center-crop to 256×256, ImageNet normalisation |
| **Output** | anomaly map (per-location Mahalanobis distance) + image-level score |
| **Inference source** | pretrained feature extractor + Gaussian fitting (documented as such — not a fine-tuned or trained-from-scratch classifier) |

## How it works

1. Extract multi-scale patch features from a defect-free training set using the
   pretrained backbone (layers 1–3, upsampled and concatenated → 448-dim, then
   randomly reduced to `padim_n_features`).
2. Model each patch location as a multivariate Gaussian (mean + covariance,
   `0.01·I` regularisation) estimated across the training set.
3. At inference, the Mahalanobis distance of each test patch from its Gaussian
   gives an anomaly map; Gaussian smoothing (σ=4) and upsampling produce the
   final map. The image score is the map maximum.
4. A min/max normaliser (fit on train-good scores) maps the raw score to [0,1]
   for thresholding and display.

## What it does NOT do

- It does **not** classify the defect *type* (scratches vs cracks vs dents are
  all just "anomalous").
- It does **not** produce a calibrated defect probability — the score is a
  distance from learned normality.
- It does **not** measure severity or structural consequence.

## Why PaDiM (not a trained classifier / detector)

- Matches the unsupervised task the dataset actually defines.
- Deterministic, CPU-feasible, no training budget — a defensible pretrained
  baseline rather than an under-trained network.
- Produces a real anomaly map → genuine defect-localisation overlays.

## Reproducibility

- `scripts/fit_padim.py` refits the model from `train/good` deterministically
  (fixed `torch.Generator` seed 1024 for feature selection; no other
  randomness).
- `scripts/evaluate_cv.py` reports measured image- and pixel-AUROC on the held-
  out test split. Numbers in `docs/evaluation.md` come from running it.

## Threshold

`anomaly_threshold` (default 0.5 on the normalised score) is a triage aid, not
an automated pass/fail authority. It should be re-validated against the
operating point you actually want; AUROC is threshold-free.
