from pathlib import Path
import hashlib
import mmap
import os
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


def _resolve_runtime_notes_root(notes_dir: Path | None = None) -> Path:
    settings = get_settings()
    project_root = settings.project_root.resolve()
    approved_data_root = (settings.project_root / "data").resolve()
    configured_root = settings.notes_dir.resolve()

    if not approved_data_root.is_relative_to(project_root):
        raise ValueError(
            f"Runtime data root resolves outside the project root: {approved_data_root}"
        )
    if not configured_root.is_relative_to(approved_data_root):
        raise ValueError(
            f"Configured notes directory {configured_root} is outside the approved "
            f"data root {approved_data_root}"
        )

    requested_root = Path(notes_dir or configured_root).resolve()
    if not requested_root.is_relative_to(approved_data_root):
        raise ValueError(
            f"Notes directory {requested_root} is outside the approved data root "
            f"{approved_data_root}"
        )
    if not requested_root.is_dir():
        raise ValueError(f"Notes directory does not exist: {requested_root}")

    return requested_root


def _iter_supported_note_files(
    notes_root: Path,
) -> Iterator[Path]:
    for current_root, directories, filenames in os.walk(
        notes_root,
        followlinks=False,
    ):
        current_path = Path(current_root)
        directories[:] = sorted(
            directory
            for directory in directories
            if not (current_path / directory).is_symlink()
        )

        for filename in sorted(filenames):
            path = current_path / filename
            if not is_supported_file(path):
                continue

            try:
                resolved_path = path.resolve(strict=True)
            except (OSError, RuntimeError):
                continue

            if not resolved_path.is_relative_to(notes_root):
                continue
            if resolved_path.is_file():
                yield path


@deco_time_block
def load_notes(
    notes_dir: Path | None = None,
) -> Iterator[Chunk]:
    notes_dir = _resolve_runtime_notes_root(notes_dir)

    for file_path in _iter_supported_note_files(notes_dir):
        doc_id = get_stable_doc_id(file_path)
        source = str(file_path.relative_to(notes_dir))

        if file_path.suffix.lower() == ".pdf":
            yield from load_pdf_file(file_path, doc_id, source)

        else :
            yield from load_text_file(file_path, doc_id, source)


@deco_time_block
def load_notes_from_isolated_corpus(corpus_root: Path) -> Iterator[Chunk]:
    """Load an explicit isolated corpus, such as the reviewed evaluation fixture."""
    corpus_root = Path(corpus_root).resolve()
    if not corpus_root.is_dir():
        raise ValueError(f"Isolated corpus directory does not exist: {corpus_root}")

    for file_path in _iter_supported_note_files(corpus_root):
        doc_id = get_stable_doc_id(file_path)
        source = str(file_path.relative_to(corpus_root))

        if file_path.suffix.lower() == ".pdf":
            yield from load_pdf_file(file_path, doc_id, source)
        else:
            yield from load_text_file(file_path, doc_id, source)


@deco_time_block
def get_changed_doc_ids(
    old_doc_ids: set[str],
    notes_dir: Path | None = None
):
    notes_dir = _resolve_runtime_notes_root(notes_dir)

    changed_ids = []
    unchanged_ids = set()

    for path in _iter_supported_note_files(notes_dir):
        doc_id = get_stable_doc_id(path)

        if doc_id in old_doc_ids:
            unchanged_ids.add(doc_id)
        else:
            changed_ids.append((doc_id, path))

    return changed_ids, unchanged_ids



