import numpy as np
import faiss

from rag_notes_helper.rag.retrieval import retrieve
from rag_notes_helper.rag.index import RagIndex


def test_retrieve_filters_by_score(monkeypatch):
    dim = 3
    index = faiss.IndexFlatIP(dim)

    vectors = np.array([[1, 0, 0], [0.1, 0, 0]])
    index.add(vectors)

    rag = RagIndex(
        index=index,
        meta=[
            {"text": "first row", "source": "a", "chunk_id": 0},
            {"text": "second row", "source": "b", "chunk_id": 1},
        ],
    )

    def mock_encode(self, texts, **kws):
        return np.array([[0.9, 0, 0]])

    monkeypatch.setattr(
        "rag_notes_helper.rag.retrieval.settings.MIN_RETRIEVAL_SCORE",
        0.5,
    )
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer.encode",
        mock_encode,
    )

    results = retrieve(query="first", rag=rag, top_k=2)

    assert len(results) == 1
    assert results[0]["text"] == "first row"

