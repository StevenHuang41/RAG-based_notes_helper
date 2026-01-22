from pathlib import Path
import hashlib
import mmap
from typing import Iterator

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag.chunking import Chunk
from rag_notes_helper.rag.loaders import load_pdf_file, load_text_file
from rag_notes_helper.utils.logger import get_logger
from rag_notes_helper.utils.timer import deco_time_block


logger = get_logger("ingest")


def get_stable_doc_id(path: Path, *, hash_len: int = 16) -> str:
    hasher = hashlib.blake2b(digest_size=32)

    # include file name in doc_id
    hasher.update(path.name.encode("utf-8"))

    # guard empty files
    if path.stat().st_size == 0:
        return hasher.hexdigest()[:hash_len]

    with path.open("rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            hasher.update(mm)

    return hasher.hexdigest()[:hash_len]


def is_supported_file(path: Path) -> bool:
    return path.suffix.lower() in [".txt", ".md", ".pdf", ".py"]


@deco_time_block
def load_notes(
    notes_dir: Path | None = None,
) -> Iterator[Chunk]:
    settings = get_settings()
    notes_dir = notes_dir or settings.notes_dir

    for file_path in sorted(notes_dir.rglob("*")):
        if not file_path.is_file() or not is_supported_file(file_path):
            continue

        doc_id = get_stable_doc_id(file_path)
        source = str(file_path.relative_to(notes_dir))

        if file_path.suffix.lower() == ".pdf":
            yield from load_pdf_file(file_path, doc_id, source)

        else :
            yield from load_text_file(file_path, doc_id, source)


@deco_time_block
def get_changed_doc_ids(
    old_doc_ids: set[str],
    notes_dir: Path | None = None
):
    notes_dir = notes_dir or get_settings().notes_dir

    changed_ids = []
    unchanged_ids = set()

    for path in notes_dir.rglob("*"):
        if path.is_file() and is_supported_file(path):
            doc_id = get_stable_doc_id(path)

            if doc_id in old_doc_ids:
                unchanged_ids.add(doc_id)
            else :
                changed_ids.append((doc_id, path))

    return changed_ids, unchanged_ids



