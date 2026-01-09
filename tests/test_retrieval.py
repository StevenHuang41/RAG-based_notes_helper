import numpy as np
import faiss

from rag_notes_helper.rag.retrieval import retrieve
from rag_notes_helper.rag.index import RagIndex
from rag_notes_helper.core.config import get_settings


def test_retrieve_filters_by_score(monkeypatch):
    dim = 3
    index = faiss.IndexFlatIP(dim)
    vectors = np.array([[1, 0, 0], [0.1, 0, 0]])

    index.add(vectors)

    rag = RagIndex(index=index)

    class DummyMetaStore:
        def get(self, faiss_id: int) -> dict:
            data = [
                {"text": "first row", "source": "a", "chunk_id": 0},
                {"text": "second row", "source": "b", "chunk_id": 1},
            ]
            return data[faiss_id]

    meta_store = DummyMetaStore()

    def mock_encode(self, texts, **kws):
        return np.array([[0.9, 0, 0]])

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer.encode",
        mock_encode,
    )

    monkeypatch.setenv("MIN_RETRIEVAL_SCORE", "0.5")

    get_settings.cache_clear()

    results = retrieve(
        query="first",
        rag=rag,
        top_k=2,
        meta_store=meta_store,
    )

    assert len(results) == 1
    assert results[0]["text"] == "first row"

