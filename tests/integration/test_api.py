"""API tests — upload validation & workflow. Uses injected vision stub so no
real model or dataset is required; end-to-end PaDiM path is tested separately."""
import io

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("sentence_transformers")

from fastapi.testclient import TestClient

from src.app.main import create_app


def _png_bytes(size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, (140, 140, 140)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture()
def client(tmp_workspace, monkeypatch):
    # stub the model so /inspect runs without a fitted PaDiM
    from src.app.schemas import ImageQuality, VisionResult

    def fake_inspect(self, image_path):
        return VisionResult(
            image_id=image_path.stem, category="bottle", model_name="PaDiM",
            backbone="resnet18", image_anomaly_score=0.7, raw_anomaly_score=3.0,
            is_anomalous=True, threshold=0.5, regions=[],
            quality=ImageQuality(is_valid=True, width=64, height=64))
    monkeypatch.setattr(
        "src.app.services.services.InspectionService.inspector_for",
        lambda self, c: None)  # no fitted model -> agent inspects via tool path
    return TestClient(create_app())


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_reject_bad_extension(client):
    r = client.post("/api/inspect",
                    files={"file": ("x.exe", b"MZ...", "application/octet-stream")})
    assert r.status_code == 415


def test_reject_empty(client):
    r = client.post("/api/inspect",
                    files={"file": ("x.png", b"", "image/png")})
    assert r.status_code == 400


def test_inspect_no_model_returns_report(client):
    # without a fitted model the report should still return with a limitation
    r = client.post("/api/inspect",
                    files={"file": ("x.png", _png_bytes(), "image/png")},
                    data={"question": "defect?"})
    assert r.status_code == 200
    rep = r.json()
    assert "summary" in rep and "disclaimer" in rep


def test_kb_query_empty(client):
    r = client.get("/api/kb/query", params={"q": "   "})
    assert r.status_code == 400
