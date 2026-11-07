"""Retriever tests — real embeddings on a tiny corpus (model downloads once)."""
import pytest

pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")

from src.app.retrieval import Retriever, ingest_documents


def test_empty_index_returns_note(tmp_workspace):
    r = Retriever()
    res = r.search("anything")
    assert res.chunks == [] and res.note == "no documents indexed"


def test_index_and_retrieve(tmp_workspace, kb_docs):
    r = Retriever()
    n = r.add_chunks(ingest_documents(kb_docs))
    assert n > 0 and r.n_chunks == n

    res = r.search("what metric evaluates anomaly detection", top_k=3)
    assert res.chunks, "expected at least one retrieved chunk"
    top = res.chunks[0]
    assert top.rank == 1 and top.chunk.title and top.chunk.text
    # the AUROC passage should be within the top-3 for this query
    assert any("AUROC" in rc.chunk.text for rc in res.chunks)


def test_bm25_mode(tmp_workspace, kb_docs):
    r = Retriever()
    r.add_chunks(ingest_documents(kb_docs))
    res = r.search("Mahalanobis distance", top_k=2, mode="bm25")
    assert res.mode == "bm25"
    assert res.chunks, "BM25 should return the matching chunk"
    assert "Mahalanobis" in res.chunks[0].chunk.text


def test_persistence(tmp_workspace, kb_docs):
    r1 = Retriever()
    r1.add_chunks(ingest_documents(kb_docs))
    n = r1.n_chunks
    # new instance loads the saved index
    r2 = Retriever()
    assert r2.n_chunks == n
