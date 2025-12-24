from io import StringIO

from rag_notes_helper.rag.chunking import chunk_text


def test_chunk_text_basic():
    f = StringIO("abcdefghij" * 50)

    chunks = list(
        chunk_text(
            file_object=f,
            doc_id="doc1",
            source="note.md",
            chunk_size=50,
            overlap=10,
        )
    )

    assert len(chunks) > 1
    assert chunks[0].chunk_id == 0
    assert chunks[1].chunk_id == 1

    for c in chunks:
        assert len(c.text) <= 50
        assert c.doc_id == "doc1"
        assert c.source == "note.md"
