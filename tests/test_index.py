from unittest.mock import MagicMock, PropertyMock
import pytest

import numpy as np

from rag_notes_helper.rag.index import build_index, RagIndex
from rag_notes_helper.rag.chunking import Chunk
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
