from io import StringIO
import pytest

from rag_notes_helper.rag.chunking import chunk_text


def test_chunk_ids():
    f = StringIO("abcdefghij\n" * 50)

    chunk_gen = chunk_text(
        file_object=f,
        doc_id="doc1",
        source="note.md",
        chunk_size=30,
        overlap=10,
    )

    c1 = next(chunk_gen)
    c2 = next(chunk_gen)

    assert c1.chunk_id == 0
    assert c2.chunk_id == 1

    for c in chunk_gen:
        assert len(c.text) <= 50
        assert c.doc_id == "doc1"
        assert c.source == "note.md"




def test_chunk_text_logic():
    f = StringIO("12345\n678\n"  * 10)

    chunks = list(
        chunk_text(
            file_object=f,
            doc_id="doc",
            source="src",
            chunk_size=10,
            overlap=2,
        )
    )

    assert chunks[0].text == "12345, 678"
    assert chunks[1].text == "678, 12345"


def test_invalid_chunk_params():
    f = StringIO("test")

    with pytest.raises(ValueError, match="chunk_size must be > 0"):
        list(chunk_text(
            file_object=f,
            doc_id="doc",
            source="src",
            chunk_size=0,
            overlap=10,
        ))

    with pytest.raises(ValueError, match="overlap must be >= 0"):
        list(chunk_text(
            file_object=f,
            doc_id="doc",
            source="src",
            chunk_size=10,
            overlap=-1,
        ))

    with pytest.raises(
        ValueError,
        match="overlap must be smaller than chunk_size"
    ):
        list(chunk_text(
            file_object=f,
            doc_id="doc",
            source="src",
            chunk_size=10,
            overlap=10,
        ))

