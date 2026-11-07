"""Create legitimate knowledge-base documents and ingest them.

The KB is built from *verifiable, cited* public sources (dataset facts, the
PaDiM paper, general NDT/defect-classification guidance). No invented manual
sections or page numbers — each document states its own source. These give the
RAG pipeline real, attributable content to retrieve.

Usage:
    python -m scripts.seed_kb
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.app.config import get_settings                    # noqa: E402
from src.app.retrieval import Retriever, ingest_documents  # noqa: E402

DOCS = {
    "mvtec_ad_dataset.txt": """\
MVTec AD Dataset — Industrial Anomaly Detection
Source: Bergmann, Fauser, Sattlegger, Steger. "MVTec AD — A Comprehensive
Real-World Dataset for Unsupervised Anomaly Detection." CVPR 2019.
This document paraphrases the dataset description; it does not replace the paper.

1. Overview
MVTec AD contains 5,354 high-resolution images of 15 object and texture
categories captured under industrial conditions. It is designed for unsupervised
anomaly detection: training sets contain ONLY defect-free images, while test
sets contain both defect-free images and images with labelled defects.

2. Defect types
Depending on the category, defects include scratches, dents, contaminations,
structural changes (e.g. bent/misplaced parts), broken regions, holes, cuts,
glue strips, and colour/texture deviations. Over 70 distinct defect types are
annotated. Pixel-precise ground-truth masks are provided for every defective
test image, enabling pixel-level (segmentation) evaluation.

3. Task definition
The task is anomaly detection/localisation, NOT supervised classification.
Because the model sees only good examples during training, a high anomaly score
indicates deviation from learned normality — it is a hypothesis of a defect,
not a confirmed diagnosis of a specific defect class.

4. Evaluation
Standard metrics are image-level AUROC (does the score rank defective above
normal?) and pixel-level AUROC or PRO (does the anomaly map align with the
ground-truth mask?). A single threshold may be reported alongside AUROC for
operational use.
""",
    "padim_method.txt": """\
PaDiM — Patch Distribution Modelling for Anomaly Detection
Source: Defard, Setkov, Loesch, Audigier. "PaDiM: a Patch Distribution
Modeling Framework for Anomaly Detection and Localization." ICPR 2021
workshop (DIPA). This document summarises the method.

1. Principle
PaDiM learns a probabilistic model of the NORMAL patch distribution. A
pretrained CNN (commonly ResNet-18) extracts multi-scale features; features
from the first three layers are concatenated per spatial location to form
patch embeddings. Each patch location is modelled as a multivariate Gaussian
(mean + covariance) estimated from the defect-free training set.

2. Scoring
At inference, the Mahalanobis distance between a test patch embedding and its
location's learned Gaussian yields an anomaly score per location, producing an
anomaly map. A Gaussian smoothing (sigma=4) is applied and the map is upsampled
to input resolution. The image-level score is the maximum of the map.

3. Interpretation and limits
High Mahalanobis distance = the local patch statistics deviate from training
normality. It does NOT classify the defect type, and it does NOT measure
severity or structural consequence. Scores are relative to the fitted
distribution and are NOT calibrated probabilities. Borderline scores near the
chosen threshold should be treated as uncertain and flagged for human review.

4. Practical notes
Performance depends on how representative the defect-free training set is.
Domain shift (lighting, orientation, new product variants) can produce false
positives — an anomalous score may reflect benign variation rather than a real
defect. Use threshold as a triage aid, not an automated pass/fail authority.
""",
    "surface_inspection_guidance.txt": """\
Surface Inspection — General Decision-Support Guidance
Source: general non-destructive-testing (NDT) and machine-vision practice,
compiled for this project's knowledge base. This is guidance text authored for
the system, not a specific manufacturer's manual; cite it as such.

1. Common surface defect categories
Scratches/scoring, dents and pits, cracks/crazing, contamination or foreign
material, corrosion/oxidation, porosity/inclusions, deformation or bent
features, missing or broken components, and coating/print defects. Different
categories carry different risk: cracks and broken structure are usually more
safety-relevant than cosmetic scratches.

2. Visual-inspection decision factors
Consider defect size relative to the part, location (load-bearing or sealing
surfaces vs cosmetic faces), count and clustering, depth cues, and whether the
defect is new or previously recorded. A single cosmetic mark may be acceptable
where the same mark on a critical surface is not.

3. Uncertainty and escalation
Image-based assessment is inherently limited: it sees surface appearance only,
not internal condition, material property, or load history. An anomalous image
score should trigger human inspection and, where warranted, a complementary
NDT method (dye penetrant, ultrasonic, magnetic particle) rather than a direct
scrap/repair decision. Borderline results and first-of-type findings warrant
conservative handling.

4. When to escalate
Escalate to a qualified inspector/engineer when: the anomaly score is high or
regions are large/growing; the defect is on a critical surface; history shows
recurrence or trend; or the defect type is safety-relevant (crack, broken
structure). Do not authorise continued operation, repair, or shutdown on the
basis of the image score alone.
""",
    "maintenance_decision_support.txt": """\
Maintenance Decision Support — Using Inspection Results
Source: generic condition-based-maintenance practice, authored for this
project's knowledge base.

1. How to read a result
An inspection result combines (a) visual observations from the vision model,
(b) retrieved technical facts from documentation, (c) inspection history, and
(d) recommendations. Visual observations describe what the model detected;
retrieved facts are cited from documents; recommendations are decision support,
not authorisations.

2. Decision support, not automation
The system triages and prioritises — it does not authorise operating,
repairing, or shutting down equipment. Final decisions rest with qualified
personnel who can inspect the equipment physically and consider context the
model cannot see.

3. Using history and comparison
Comparing the current anomaly score and defect regions to prior inspections of
the same asset reveals new vs recurring findings and trends. A stable,
previously-accepted finding is treated differently from a new or growing one —
trend direction matters more than a single snapshot.

4. Data quality and limits
Results are only as good as the image (focus, lighting, coverage) and the fit
of the vision model to that equipment. Out-of-distribution inputs and
unrepresentative training data reduce reliability — treat low-confidence or
unusual inputs with caution and note them as limitations.
""",
}


def main() -> None:
    s = get_settings()
    kb_dir = s.data_dir / "kb"
    kb_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for name, text in DOCS.items():
        p = kb_dir / name
        p.write_text(text, encoding="utf-8")
        paths.append(p)
        print(f"[kb] wrote {p}")

    chunks = ingest_documents(paths)
    r = Retriever()
    added = r.add_chunks(chunks)
    print(f"[kb] indexed {added} chunks (total {r.n_chunks})")

    # smoke test
    res = r.search("how is the anomaly score computed", top_k=3)
    for rc in res.chunks:
        print(f"  -> {rc.chunk.title} p.{rc.chunk.page} "
              f"({rc.score:.4f}): {rc.chunk.text[:60]}...")


if __name__ == "__main__":
    main()
