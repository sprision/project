"""Image-validation unit tests — no model or dataset needed."""
from pathlib import Path

from src.app.vision.model import validate_image


def test_missing_file(tmp_path):
    q = validate_image(tmp_path / "nope.png", max_mb=10)
    assert not q.is_valid and q.reason == "file not found"


def test_oversize(tmp_path):
    p = tmp_path / "big.png"
    p.write_bytes(b"x" * (11 * 1024 * 1024))
    q = validate_image(p, max_mb=10)
    assert not q.is_valid and "MB" in q.reason


def test_corrupt(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"not an image at all")
    q = validate_image(p, max_mb=10)
    assert not q.is_valid and "cannot decode" in q.reason


def test_too_small(tmp_path):
    from PIL import Image
    p = tmp_path / "tiny.png"
    Image.new("RGB", (8, 8)).save(p)
    q = validate_image(p, max_mb=10)
    assert not q.is_valid and "too small" in q.reason


def test_valid(sample_image):
    q = validate_image(sample_image, max_mb=10)
    assert q.is_valid and q.width == 64
