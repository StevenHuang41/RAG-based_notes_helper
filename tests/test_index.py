from unittest.mock import MagicMock, PropertyMock
import pytest

import numpy as np

from rag_notes_helper.rag.index import (
    build_and_save_rag,
    build_index,
    RagIndex,
    rebuild_index,
)
from rag_notes_helper.rag.chunking import Chunk
from rag_notes_helper.rag.meta_store import MetaStore
from rag_notes_helper.rag.retrieval import retrieve
from rag_notes_helper.core.config import get_settings


def test_build_index_no_chunk():
    with pytest.raises(ValueError, match="No chunks to index"):
        build_index([])


def test_build_index_creates_faiss_index(monkeypatch, tmp_path):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array(
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]
    ).astype("float32")

    monkeypatch.setattr(
        RagIndex,
        "embed_model",
        PropertyMock(return_value=mock_model)
    )

    chunks = [
        Chunk(doc_id="d1", chunk_id=0, source="note.md", text="test0"),
        Chunk(doc_id="d1", chunk_id=1, source="note.md", text="test1"),
    ]

    rag = build_index(chunks)

    assert rag.index.ntotal == 2


def test_build_index_persists_chunk_line_ranges(monkeypatch):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0]]).astype("float32")
    monkeypatch.setattr(
        RagIndex,
        "embed_model",
        PropertyMock(return_value=mock_model),
    )

    build_index([
        Chunk(
            doc_id="d1",
            chunk_id=0,
            source="note.md",
            text="first\nsecond",
            start_line=8,
            end_line=9,
        )
    ])

    with MetaStore(get_settings().storage_dir) as meta_store:
        record = meta_store.get(0)

    assert record["start_line"] == 8
    assert record["end_line"] == 9


@pytest.mark.parametrize("remove_note", ["delete", "exclude"])
def test_smart_rebuild_removes_deleted_or_excluded_notes_from_search(
    monkeypatch,
    remove_note,
):
    notes_dir = get_settings().notes_dir
    kept_note = notes_dir / "kept.md"
    removed_note = notes_dir / "removed.md"
    kept_note.write_text("The kept project note.", encoding="utf-8")
    removed_note.write_text("The removed project note.", encoding="utf-8")

    mock_model = MagicMock()
    mock_model.encode.side_effect = lambda texts, **_kwargs: np.tile(
        np.array([[1.0, 0.0, 0.0]], dtype="float32"),
        (len(texts), 1),
    )
    monkeypatch.setattr(
        RagIndex,
        "embed_model",
        PropertyMock(return_value=mock_model),
    )

    build_and_save_rag()
    if remove_note == "delete":
        removed_note.unlink()
    else:
        removed_note.rename(removed_note.with_suffix(".bin"))

    rebuilt = rebuild_index()

    with MetaStore() as meta_store:
        results = retrieve(
            rebuilt,
            meta_store,
            query="removed project note",
            top_k=10,
        )

    assert {result["source"] for result in results} == {"kept.md"}


def test_smart_rebuild_of_empty_corpus_makes_search_return_no_results(
    monkeypatch,
):
    notes_dir = get_settings().notes_dir
    note = notes_dir / "only-note.md"
    note.write_text("The only indexed note.", encoding="utf-8")

    mock_model = MagicMock()
    mock_model.encode.side_effect = lambda texts, **_kwargs: np.tile(
        np.array([[1.0, 0.0, 0.0]], dtype="float32"),
        (len(texts), 1),
    )
    monkeypatch.setattr(
        RagIndex,
        "embed_model",
        PropertyMock(return_value=mock_model),
    )

    build_and_save_rag()
    note.unlink()

    rebuilt = rebuild_index()

    with MetaStore() as meta_store:
        results = retrieve(
            rebuilt,
            meta_store,
            query="only indexed note",
            top_k=10,
        )

    assert len(results) == 0
