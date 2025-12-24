from pathlib import Path
import hashlib

from rag_notes_helper.core.config import settings
from rag_notes_helper.rag.chunking import Chunk, chunk_text
from rag_notes_helper.rag.loaders import load_text_file


def _stable_doc_id(
    path: Path,
    *,
    head_bytes: int = 1024,
    hash_len: int = 16,
) -> str:
    hasher = hashlib.sha256()

    time_stamp = str(path.stat().st_mtime_ns).encode("utf-8")

    with path.open("rb") as f:
        content = f.read(head_bytes)

    hasher.update(time_stamp)
    hasher.update((content))

    digest = hasher.hexdigest()[:hash_len]

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

        source = str(file_path.relative_to(settings.PROJECT_ROOT))
        source = source[len("data/"):]

        chunks = chunk_text(
            text=text,
            doc_id=doc_id,
            source=source,
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP,
        )

        all_chunks.extend(chunks)

    return all_chunks

