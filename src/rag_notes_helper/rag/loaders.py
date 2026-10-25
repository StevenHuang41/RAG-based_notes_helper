from collections.abc import Iterator
from pathlib import Path

import pymupdf

from rag_notes_helper.rag.chunking import Chunk, chunk_lines


def load_pdf_file(path: Path, doc_id: str, source: str, **kws) -> Iterator[Chunk]:
    def iter_pdf_text(path: Path) -> Iterator[str]:
        with pymupdf.open(path) as doc:
            for page in doc:
                for line in str(page.get_text()).splitlines():
                    yield line

    lines = iter_pdf_text(path)
    yield from chunk_lines(
        lines=lines,
        doc_id=doc_id,
        source=source,
        **kws,
    )


def load_text_file(path: Path, doc_id: str, source: str, **kws) -> Iterator[Chunk]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        yield from chunk_lines(
            lines=(line.rstrip("\r\n") for line in f),
            doc_id=doc_id,
            source=source,
            **kws,
        )
