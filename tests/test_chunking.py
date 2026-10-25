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

    prev = chunks[0].text.splitlines()
    curr = chunks[1].text.splitlines()
    assert set(prev) & set(curr)


def test_chunks_keep_inclusive_source_line_ranges_and_overlaps():
    chunks = list(chunk_lines(
        lines=iter(["line 1", "line 2", "line 3"]),
        doc_id="doc_id1",
        source="note.md",
        chunk_size=15,
        overlap=7,
    ))

    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [
        (1, 2),
        (2, 3),
    ]
    assert [chunk.text for chunk in chunks] == ["line 1\nline 2", "line 2\nline 3"]
