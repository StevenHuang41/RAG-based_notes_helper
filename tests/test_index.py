from rag_notes_helper.rag.index import build_index, list_indexed_sources
from rag_notes_helper.rag.chunking import Chunk


def test_build_index_no_chunk():
    try :
        build_index([])
        assert False
    except ValueError:
        pass


def test_list_indexed_sources():

    chunks = [
        Chunk("doc1", 0, "a file content", "a"),
        Chunk("doc2", 0, "b.md file content", "b.md"),
    ]

    rag = build_index(chunks)

    sources = list_indexed_sources(rag)
    assert sources == ["a", "b.md"]

