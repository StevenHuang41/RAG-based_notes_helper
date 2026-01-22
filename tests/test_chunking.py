import pytest

from rag_notes_helper.rag.chunking import chunk_lines

def mock_iter():
    for _ in range(10):
        yield "123"
        yield "4567"
        yield "890"


def test_chunk_ids():
    chunks = list(chunk_lines(
        lines=mock_iter(),
        doc_id="doc_id1",
        source="note.md",
        chunk_size=20,
        overlap=2,
    ))

    assert len(chunks) > 1
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))

    assert all(c.doc_id == "doc_id1" for c in chunks)
    assert all(c.source == "note.md" for c in chunks)
    assert all(c.text for c in chunks)

    prev = chunks[0].text
    curr = chunks[1].text
    assert curr.startswith(prev.split(", ")[-1])
