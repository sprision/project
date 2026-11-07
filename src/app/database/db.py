"""SQLite persistence for inspections — powers search/compare history tools."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..config import get_settings
from ..schemas import InspectionReport

_SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id TEXT NOT NULL,
    category TEXT,
    question TEXT,
    image_anomaly_score REAL,
    is_anomalous INTEGER,
    n_regions INTEGER,
    generator TEXT,
    report_json TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_insp_cat ON inspections(category, created_at DESC);
"""


class InspectionDB:
    def __init__(self, path: Path | None = None):
        s = get_settings()
        self.path = Path(path or (s.index_dir / "inspections.sqlite"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def insert(self, report: InspectionReport) -> int:
        v = report.vision
        cur = self._conn.execute(
            "INSERT INTO inspections (image_id, category, question,"
            " image_anomaly_score, is_anomalous, n_regions, generator,"
            " report_json, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (report.image_id,
             v.category if v else None,
             report.question,
             v.image_anomaly_score if v else None,
             int(v.is_anomalous) if v else 0,
             len(v.regions) if v else 0,
             report.generator,
             report.model_dump_json(),
             datetime.now(timezone.utc).isoformat()))
        self._conn.commit()
        return int(cur.lastrowid)

    def search(self, category: str | None = None, limit: int = 10) -> list[dict]:
        sql = "SELECT id,image_id,category,image_anomaly_score,is_anomalous," \
              "n_regions,created_at FROM inspections"
        params: list = []
        if category:
            sql += " WHERE category=?"
            params.append(category)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def get(self, insp_id: int) -> dict | None:
        r = self._conn.execute(
            "SELECT id,image_id,category,question,image_anomaly_score,"
            "is_anomalous,n_regions,generator,report_json,created_at "
            "FROM inspections WHERE id=?",
            (insp_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        raw = d.pop("report_json", None)
        if raw:
            try:
                d["report"] = json.loads(raw)
            except Exception:
                d["report"] = None
        return d

    def latest(self, category: str | None = None) -> dict | None:
        rows = self.search(category=category, limit=1)
        return rows[0] if rows else None

    def close(self) -> None:
        self._conn.close()
