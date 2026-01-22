from pathlib import Path
from io import StringIO

import pymupdf

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag.chunking import chunk_text
from rag_notes_helper.utils.logger import get_logger

logger = get_logger("loaders")

def load_pdf_file(path, doc_id, source):
    settings = get_settings()
    chunk_size=settings.chunk_size
    overlap=settings.chunk_overlap

    buffer = ""
    buffer_len = 0
    with pymupdf.open(path) as doc:
        for page in doc:
            page_text = str(page.get_text())
            page_len = len(page_text)

            if buffer_len + page_len >= chunk_size and buffer:
                yield from chunk_text(
                    file_object=StringIO(buffer),
                    doc_id=doc_id,
                    source=source,
                )
                buffer = buffer[:-overlap]

            buffer += page_text
            buffer_len = len(buffer)

    if buffer:
        yield from chunk_text(
            file_object=StringIO(buffer),
            doc_id=doc_id,
            source=source,
        )


def load_text_file(path, doc_id, source):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        yield from chunk_text(
            file_object=f,
            doc_id=doc_id,
            source=source,
        )
