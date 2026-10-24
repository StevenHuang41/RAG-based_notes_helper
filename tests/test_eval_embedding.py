import sys
from types import ModuleType, SimpleNamespace

from rag_notes_helper.eval import embedding


def test_ragas_embedding_uses_configured_local_model(monkeypatch):
    constructed = []
    ragas_package = ModuleType("ragas")
    ragas_package.__path__ = []
    embeddings_module = ModuleType("ragas.embeddings")

    class FakeHuggingFaceEmbeddings:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

    embeddings_module.HuggingFaceEmbeddings = FakeHuggingFaceEmbeddings
    monkeypatch.setitem(sys.modules, "ragas", ragas_package)
    monkeypatch.setitem(sys.modules, "ragas.embeddings", embeddings_module)
    monkeypatch.setattr(
        "rag_notes_helper.core.config.get_settings",
        lambda: SimpleNamespace(embed_model_name="sentence-transformers/all-MiniLM-L6-v2"),
    )
    monkeypatch.setattr(embedding, "_embedding", None)

    first = embedding.get_embeddings()
    second = embedding.get_embeddings()

    assert first is second
    assert constructed == [
        {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "normalize_embeddings": True,
        }
    ]
