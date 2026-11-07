"""Evaluate retrieval quality on a small documented query set.

Each query has an expected relevant doc_id/keyword; we report Recall@k and MRR.
Run after seeding the KB. Results -> runs/eval_retrieval.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.app.config import get_settings       # noqa: E402
from src.app.retrieval import Retriever       # noqa: E402

# Documented eval set: query -> substring that must appear in a relevant chunk.
EVAL_SET = [
    ("does the dataset contain defect-free training images only",
     "defect-free"),
    ("what metric is used to evaluate anomaly detection", "AUROC"),
    ("is an anomaly score a confirmed diagnosis", "hypothesis"),
    ("how is the anomaly score computed from patch features", "Mahalanobis"),
    ("are anomaly scores calibrated probabilities", "calibrated"),
    ("when should a finding be escalated to an engineer", "escalat"),
    ("what does an image-based inspection not capture", "internal condition"),
    ("what matters more than a single snapshot", "trend"),
]


def recall_at_k(res, keyword, k) -> bool:
    return any(keyword.lower() in rc.chunk.text.lower()
               for rc in res.chunks[:k])


def main() -> None:
    s = get_settings()
    r = Retriever()
    if r.n_chunks == 0:
        print("KB empty — run scripts/seed_kb.py first"); sys.exit(1)

    K = [1, 3, 5]
    hits = {k: 0 for k in K}
    mrr = 0.0
    per_query = []
    for q, kw in EVAL_SET:
        res = r.search(q, top_k=max(K))
        first_rel = None
        for rc in res.chunks:
            if kw.lower() in rc.chunk.text.lower():
                first_rel = rc.rank
                break
        if first_rel:
            mrr += 1.0 / first_rel
        for k in K:
            if first_rel and first_rel <= k:
                hits[k] += 1
        per_query.append({"query": q, "keyword": kw,
                          "first_relevant_rank": first_rel})

    n = len(EVAL_SET)
    report = {
        "n_queries": n,
        "recall_at_k": {str(k): round(v / n, 3) for k, v in hits.items()},
        "mrr": round(mrr / n, 3),
        "per_query": per_query,
    }
    out = s.runs_dir / "eval_retrieval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
