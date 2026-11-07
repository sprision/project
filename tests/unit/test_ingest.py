"""Ingestion tests: chunking, metadata preservation, unsupported types."""
from pathlib import Path

from src.app.retrieval.ingest import _chunk_page, ingest_documents


def test_txt_chunking_and_metadata(tmp_path):
    f = tmp_path / "manual.txt"
    f.write_text("1. Intro\nSome intro text here. " * 40, encoding="utf-8")
    chunks = ingest_documents([f], chunk_size=400, overlap=50)
    assert chunks
    c = chunks[0]
    assert c.doc_id == "manual" and c.source == "manual.txt"
    assert c.chunk_id == "manual:0"
    assert c.title == "Manual"


def test_chunk_size_respected(tmp_path):
    text = "word " * 2000
    chunks = _chunk_page(text, chunk_size=500, overlap=40)
    assert all(len(c) <= 500 for c in chunks)
    assert "".join(chunks)  # non-empty


def test_overlap_carried(tmp_path):
    text = "A" * 300 + "\n\n" + "B" * 300
    chunks = _chunk_page(text, chunk_size=350, overlap=30)
    assert len(chunks) >= 2


def test_unsupported_type_skipped(tmp_path):
    f = tmp_path / "x.xyz"
    f.write_text("hello", encoding="utf-8")
    assert ingest_documents([f]) == []


def test_section_guess(tmp_path):
    f = tmp_path / "d.txt"
    f.write_text("2.3 Acceptance Criteria\nThe limit is 5mm. " * 10,
                 encoding="utf-8")
    chunks = ingest_documents([f], chunk_size=500, overlap=20)
    assert chunks[0].section is not None
