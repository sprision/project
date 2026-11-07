# Evaluation

All numbers below come from running the project scripts — no invented results.
Re-run to reproduce; outputs land in `runs/`.

## A. Computer vision — PaDiM on MVTec AD

Measured by `scripts/evaluate_cv.py` on the held-out test split. Reported:
image-level AUROC (does the score rank defective above normal?) and pixel-level
AUROC (does the anomaly map align with ground-truth masks?). Reference: PaDiM
paper reports ~97.9 mean image-AUROC on MVTec AD (ResNet-18).

| Category | n test | image-AUROC | pixel-AUROC |
|---|---|---|---|
| *(fill in after running)* | | | |

## B. Retrieval quality

Measured by `scripts/evaluate_retrieval.py` on an 8-query documented set over
the seeded KB. Reported: Recall@1/3/5 and MRR.

| Metric | Value |
|---|---|
| Recall@1 / @3 / @5 | *(fill in)* |
| MRR | *(fill in)* |

## C. Agent behaviour

Covered by `tests/unit/test_agent.py`: tool selection by question type, arg
validation, tool-failure handling, bounded iterations, end-to-end report
completion. Deterministic policy — no LLM needed.

## D. End-to-end grounding & source correctness

Each `InspectionReport` distinguishes visual observations, retrieved facts,
inspection history, hypotheses and recommendations, and carries a per-finding
`source`. Verified by the report tests and by inspecting `rep.trace`.

## E. Latency

`VisionResult.inference_ms` records model latency per image. Timings recorded
in `runs/eval_cv.json` (`elapsed_s` per category).

## Known limitations

- Anomaly score ≠ calibrated defect probability; does not identify defect type.
- Deterministic report generator is extractive, not fluent prose (LLM optional).
- Retrieval quality depends on the small seeded KB; expands by ingesting docs.
- Dataset is non-commercial (CC-BY-NC-SA-4.0).
