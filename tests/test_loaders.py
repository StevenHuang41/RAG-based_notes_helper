from pathlib import Path

import pymupdf

from rag_notes_helper.rag.chunking import Chunk
from rag_notes_helper.rag.loaders import load_text_file, load_pdf_file


def test_load_pdf_file(tmp_path):
    pdf_path = tmp_path / "test.pdf"

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (50, 50),
        "pdf line 1\npdf line 2\npdf line 3\npdf line 4\npdf line 5\npdf line 6",
    )
    doc.save(pdf_path)
    doc.close()

    chunks = list(load_pdf_file(
        pdf_path,
        doc_id="doc_id1",
        source="test.pdf",
        chunk_size=30,
        overlap=5,
    ))

    assert chunks
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.doc_id == "doc_id1" for c in chunks)
    assert all(c.source == "test.pdf" for c in chunks)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))


def test_load_text_file(tmp_path: Path):
    text_path = tmp_path / "test.txt"
    text_path.write_text(
        "text line 1\ntext line 2\ntext line 3\ntext line 4\ntext line 5\n",
        encoding="utf-8",
    )

    chunks = list(load_text_file(
        text_path,
        doc_id="doc_id2",
        source="test.txt",
        chunk_size=30,
        overlap=5,
    ))

    assert chunks
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.doc_id == "doc_id2" for c in chunks)
    assert all(c.source == "test.txt" for c in chunks)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
