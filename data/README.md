# Data layout

Raw datasets and indices are **not** committed to git (see `.gitignore`). Run the
scripts below to populate this directory.

```
data/
  raw/mvtec_ad/         MVTec AD subset (downloaded by scripts/download_mvtec.py)
    <category>/train/good/*.png
    <category>/test/<defect|good>/*.png
    <category>/ground_truth/<defect>/*_mask.png
  index/                retrieval index + inspection-history DB (generated)
  kb/                   knowledge-base documents (PDF/text) to ingest
```

## Source

MVTec AD (Bergmann et al., 2019), mirrored on HuggingFace as
`foersben/mvtec-ad` (CC-BY-NC-SA-4.0 — non-commercial use only, appropriate for
this student project; not for commercial deployment). See `docs/dataset_card.md`.
