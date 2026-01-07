from pathlib import Path
import hashlib

from rag_notes_helper.core.config import settings
from rag_notes_helper.rag.chunking import Chunk, chunk_text
from rag_notes_helper.rag.loaders import is_text_file


def _stable_doc_id(
    path: Path,
    *,
    chunk_bytes: int = 8192,
    hash_len: int = 16,
) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as f:
        while (chunk:= f.read(chunk_bytes)) != b'':
            hasher.update(chunk)

    return hasher.hexdigest()[:hash_len]



def load_notes(notes_dir: Path | None = None) -> list[Chunk]:
    notes_dir = notes_dir or settings.NOTES_DIR

    if not notes_dir.exists():
        raise FileNotFoundError(f"Notes directory not found: {notes_dir}")

    all_chunks: list[Chunk] = []

    for file_path in sorted(notes_dir.rglob("*")):
        if not file_path.is_file():
            continue

        if not is_text_file(file_path):
            continue

        doc_id = _stable_doc_id(file_path)

        source = str(file_path.relative_to(settings.NOTES_DIR))

        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            for chunk in chunk_text(
                file_object=f,
                doc_id=doc_id,
                source=source,
                chunk_size=settings.CHUNK_SIZE,
                overlap=settings.CHUNK_OVERLAP,
            ):
                all_chunks.append(chunk)

    return all_chunks

