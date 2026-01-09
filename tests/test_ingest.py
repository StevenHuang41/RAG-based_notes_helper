from pathlib import Path

from rag_notes_helper.rag.ingest import load_notes


def test_load_notes_generates_chunks(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    note1 = data_dir / "note1.md"
    note1.write_text(
        "This is a test note1.\n",
        encoding="utf-8",
    )
    note2 = data_dir / "note2.md"
    note2.write_text(
        "This is a test note2.\n",
        encoding="utf-8",
    )


    chunks = list(load_notes(data_dir))

    assert len(chunks) == 2

    c1, c2 = chunks

    assert c1.doc_id != c2.doc_id
    assert c1.chunk_id  == 0
    assert c2.chunk_id  == 0
    assert "test note1" in c1.text
    assert "test note2" in c2.text
    assert c1.source == "note1.md"
    assert c2.source == "note2.md"
