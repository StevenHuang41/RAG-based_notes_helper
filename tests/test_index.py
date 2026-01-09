from rag_notes_helper.rag.index import build_index
from rag_notes_helper.rag.chunking import Chunk
from rag_notes_helper.core.config import get_settings


def test_build_index_no_chunk():
    try :
        build_index([])
        assert False
    except ValueError:
        pass


def test_build_index_creates_faiss_index(tmp_path, monkeypatch):

    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))

    get_settings.cache_clear()

    chunks = [
        Chunk(doc_id="d1", chunk_id=0, source="note.md", text="test0"),
        Chunk(doc_id="d1", chunk_id=1, source="note.md", text="test1"),
    ]

    rag = build_index(chunks, batch_size=1)

    assert rag.index.ntotal == 2
