"""Hybrid retriever: dense embeddings + BM25 fused by reciprocal rank.

Embeddings are computed with sentence-transformers and persisted alongside the
chunk store as a .npz + .json — a real vector index without an external DB.
The public surface is the `Retriever` class so a different backend
(Chroma/FAISS) can be swapped in later without touching the agent.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Plus

from ..config import get_settings
from ..schemas import DocumentChunk, RetrievalResult, RetrievedChunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _rrf(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


class Retriever:
    """Hybrid dense+BM25 retriever over ingested DocumentChunks."""

    def __init__(self, index_dir: Path | None = None):
        s = get_settings()
        self.index_dir = Path(index_dir or s.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embed_model_name = s.embed_model
        self.mode = s.retriever_mode
        self._embedder = None  # lazy

        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None
        self._bm25: BM25Plus | None = None
        self._load_index()

    # ---- persistence ----
    def _load_index(self) -> None:
        meta = self.index_dir / "chunks.json"
        emb = self.index_dir / "embeddings.npz"
        if meta.exists():
            data = json.loads(meta.read_text(encoding="utf-8"))
            self.chunks = [DocumentChunk(**c) for c in data["chunks"]]
            self._rebuild_bm25()
        if emb.exists():
            self.embeddings = np.load(emb)["embeddings"]

    def _save_index(self) -> None:
        (self.index_dir / "chunks.json").write_text(
            json.dumps({"chunks": [c.model_dump() for c in self.chunks]},
                       ensure_ascii=False, indent=0),
            encoding="utf-8")
        if self.embeddings is not None:
            np.savez_compressed(self.index_dir / "embeddings.npz",
                                embeddings=self.embeddings)

    def _rebuild_bm25(self) -> None:
        # BM25Plus floors IDF at epsilon so a real term match never scores
        # below a non-match (BM25Okapi yields negative IDF on small corpora).
        self._bm25 = BM25Plus([_tokenize(c.text) for c in self.chunks]) \
            if self.chunks else None

    # ---- embedding ----
    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder

    def _embed(self, texts: list[str]) -> np.ndarray:
        emb = self._get_embedder().encode(
            texts, normalize_embeddings=True, convert_to_numpy=True)
        return emb.astype(np.float32)

    # ---- indexing ----
    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Index new chunks. Returns number added."""
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        emb = self._embed(texts)
        self.chunks.extend(chunks)
        self.embeddings = emb if self.embeddings is None else \
            np.vstack([self.embeddings, emb])
        self._rebuild_bm25()
        self._save_index()
        return len(chunks)

    # ---- search ----
    def _dense_rank(self, query: str) -> list[tuple[int, float]]:
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        q = self._embed([query])[0]
        sims = self.embeddings @ q  # cosine (embeddings normalised)
        order = np.argsort(-sims)
        return [(int(i), float(sims[i])) for i in order]

    def _bm25_rank(self, query: str) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        order = np.argsort(-scores)
        return [(int(i), float(scores[i])) for i in order]

    def search(self, query: str, top_k: int | None = None,
               mode: str | None = None) -> RetrievalResult:
        s = get_settings()
        top_k = top_k or s.retriever_top_k
        mode = mode or self.mode
        if not self.chunks:
            return RetrievalResult(query=query, chunks=[], mode=mode,
                                   note="no documents indexed")

        dense = self._dense_rank(query) if mode in {"dense", "hybrid"} else []
        bm25 = self._bm25_rank(query) if mode in {"bm25", "hybrid"} else []

        if mode == "dense":
            ranked = dense
        elif mode == "bm25":
            ranked = bm25
        else:  # hybrid: reciprocal-rank fusion
            score: dict[int, float] = {}
            for r, (i, _) in enumerate(dense):
                score[i] = score.get(i, 0.0) + _rrf(r)
            for r, (i, _) in enumerate(bm25):
                score[i] = score.get(i, 0.0) + _rrf(r)
            ranked = sorted(score.items(), key=lambda kv: -kv[1])

        out: list[RetrievedChunk] = []
        for rank, (i, sc) in enumerate(ranked[:top_k], start=1):
            out.append(RetrievedChunk(chunk=self.chunks[i], score=round(sc, 6),
                                      rank=rank))
        return RetrievalResult(query=query, chunks=out, mode=mode)

    # ---- stats ----
    @property
    def n_chunks(self) -> int:
        return len(self.chunks)
