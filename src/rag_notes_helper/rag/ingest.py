from pathlib import Path
import hashlib

from rag_notes_helper.core.config import settings
from rag_notes_helper.rag.chunking import Chunk, chunk_text
from rag_notes_helper.rag.loaders import load_text_file


def _stable_doc_id(path: Path) -> str:
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()[:16]

    return f"{path.stem}-{digest}"


def load_notes(notes_dir: Path | None = None) -> list[Chunk]:
    notes_dir = notes_dir or settings.NOTES_DIR

    if not notes_dir.exists():
        raise FileNotFoundError(f"Notes directory not found: {notes_dir}")

    all_chunks: list[Chunk] = []

    for file_path in sorted(notes_dir.rglob("*")):
        if not file_path.is_file():
            continue

        text = load_text_file(file_path)
        if not text:
            continue

        doc_id = _stable_doc_id(file_path)

        chunks = chunk_text(
            text=text,
            doc_id=doc_id,
            source=str(file_path.relative_to(settings.PROJECT_ROOT)),
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP,
        )

        all_chunks.extend(chunks)

    return all_chunks

