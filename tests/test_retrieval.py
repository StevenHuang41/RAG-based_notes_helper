from unittest.mock import MagicMock, PropertyMock
import pytest

import numpy as np
import faiss

from rag_notes_helper.rag.retrieval import retrieve
from rag_notes_helper.rag.index import RagIndex
from rag_notes_helper.core.config import get_settings

@pytest.fixture
def mock_meta_store():
    class DummyMetaStore:
        def get(self, faiss_id: int) -> dict:
            data = [{"text": "high"},{"text": "low"}]
            return data[faiss_id]

    return DummyMetaStore()


def test_retrieve_filters_by_score(monkeypatch, mock_meta_store):
    monkeypatch.setenv("MIN_RETRIEVAL_SCORE", "0.5")

    dim = 3
    index = faiss.IndexFlatIP(dim)
    vectors = np.array([[1, 0, 0], [0.1, 0, 0]]).astype("float32")

    index.add(vectors)

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array(
        [[0.9, 0.1, 0.1]]
    ).astype("float32")

    monkeypatch.setattr(
        RagIndex,
        "embed_model",
        PropertyMock(return_value=mock_model),
    )

    rag = RagIndex(index=index)

    results = retrieve(
        query="any query will lead to the mock return value",
        rag=rag,
        top_k=2,
        meta_store=mock_meta_store,
    )

    assert len(results) == 1
    assert results[0]["text"] == "high"
    assert results[0]["score"] > 0.5

