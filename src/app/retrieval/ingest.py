"""Document ingestion: PDF/text -> metadata-preserving chunks.

Chunks keep doc title, page number (when available), a coarse section heading
guess, and a stable chunk_id for source attribution. Uploaded/ingested content
is treated as *untrusted data*: it is stored as evidence text, never executed or
passed to the model as instructions.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from ..schemas import DocumentChunk

_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+([A-Z][^\n]{3,80})$")


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", " ")
    # collapse >2 newlines and trailing spaces
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pdf(path: Path) -> list[tuple[int | None, str]]:
    """Return [(page_number, page_text)]. Page numbers are 1-based."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append((i + 1, page.extract_text() or ""))
        except Exception:
            pages.append((i + 1, ""))
    return pages


def _extract_text(path: Path) -> list[tuple[int | None, str]]:
    return [(None, path.read_text(encoding="utf-8", errors="replace"))]


def _guess_section(block: str) -> str | None:
    """Grab the first heading-like line to label the chunk's section."""
    for line in block.split("\n")[:6]:
        m = _HEADING_RE.match(line.strip())
        if m:
            return f"{m.group(1)} {m.group(2).strip()}"
        if line.strip().isupper() and 4 <= len(line.strip()) <= 80:
            return line.strip().title()
    return None


def _chunk_page(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding-window chunk on characters, preferring paragraph boundaries."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= chunk_size:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            # carry overlap from the tail of buf
            tail = buf[-overlap:] if overlap > 0 and buf else ""
            buf = (tail + "\n\n" + p).strip() if tail else p
    if buf:
        chunks.append(buf)
    # split any still-oversized chunks hard
    final: list[str] = []
    for c in chunks:
        while len(c) > chunk_size:
            final.append(c[:chunk_size])
            c = c[chunk_size - overlap:]
        if c:
            final.append(c)
    return [c for c in final if c.strip()]


def ingest_documents(paths: list[Path], chunk_size: int = 900,
                     overlap: int = 120) -> list[DocumentChunk]:
    """Ingest files into chunks with provenance metadata."""
    chunks: list[DocumentChunk] = []
    for path in paths:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            page_texts = _extract_pdf(path)
        elif suffix in {".txt", ".md", ".rst"}:
            page_texts = _extract_text(path)
        else:
            continue  # unsupported type; caller filters

        doc_id = path.stem
        title = doc_id.replace("_", " ").replace("-", " ").title()
        idx = 0
        for page_no, text in page_texts:
            text = _clean(text)
            if not text:
                continue
            for block in _chunk_page(text, chunk_size, overlap):
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc_id}:{idx}",
                    doc_id=doc_id,
                    title=title,
                    text=block,
                    page=page_no,
                    section=_guess_section(block),
                    source=path.name,
                ))
                idx += 1
    return chunks
