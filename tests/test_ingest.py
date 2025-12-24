from pathlib import Path

from rag_notes_helper.rag.ingest import load_notes
from rag_notes_helper.core.config import settings


def test_load_notes_generates_chunks(tmp_path):
    temp_data_dir = tmp_path / "data"
    temp_data_dir.mkdir()

    note = temp_data_dir / "note.md"
    note.write_text(
        "This is a test note.\n" * 20,
        encoding="utf-8",
    )

    origin_data_dir, settings.NOTES_DIR = settings.NOTES_DIR, temp_data_dir

    try :
        chunks = load_notes()
    finally :
        settings.NOTES_DIR = origin_data_dir

    assert len(chunks) > 0
    assert all("test note" in c.text.lower() for c in chunks)

