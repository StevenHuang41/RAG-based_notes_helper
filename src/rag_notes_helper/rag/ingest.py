from inspect import currentframe
from pathlib import Path
import hashlib
import mmap
from typing import Iterable

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag.chunking import Chunk, chunk_text
from rag_notes_helper.rag.loaders import is_text_file
from rag_notes_helper.utils.logger import get_logger


logger = get_logger("cli")

def get_stable_doc_id(path: Path, *, hash_len: int = 16) -> str:
    hasher = hashlib.blake2b(digest_size=32)

    with path.open("rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            hasher.update(mm)

    hv = hasher.hexdigest()[:hash_len]
    logger.debug(f"create doc_id: {path} - {hv}")
    return hv


def load_notes(
    notes_dir: Path | None = None,
    # doc_caches: set[str] | None = None,
) -> Iterable[Chunk]:
    settings = get_settings()
    notes_dir = notes_dir or settings.notes_dir

    # doc_caches = doc_caches or set()

    for file_path in sorted(notes_dir.rglob("*")):
        if not file_path.is_file() or not is_text_file(file_path):
            continue

        doc_id = get_stable_doc_id(file_path)
        # if doc_id in doc_caches:
        #     logger.info(f"skipping unchanged file: {file_path}")
        #     continue

        source = str(file_path.relative_to(notes_dir))

        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            yield from chunk_text(
                file_object=f,
                doc_id=doc_id,
                source=source,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )


def get_changed_doc_ids(old_doc_ids: set[str]):
    notes_dir = get_settings().notes_dir

    changed_ids = []
    unchanged_ids = set()

    current_files = {}
    for path in notes_dir.rglob("*"):
        if path.is_file() and is_text_file(path):
            doc_id = get_stable_doc_id(path)
            current_files[doc_id] = path

            if doc_id in old_doc_ids:
                unchanged_ids.add(doc_id)
            else :
                changed_ids.append((doc_id, path))

    return changed_ids, unchanged_ids



