from pathlib import Path
import pytest

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag.ingest import (
    get_changed_doc_ids,
    get_stable_doc_id,
    load_notes,
)

@pytest.fixture
def mock_notes_dir(tmp_path):
    data_dir = tmp_path / "data"

    # test file: note1
    note1 = data_dir / "note1.md"
    note1.write_text("This is a test note1.\n", encoding="utf-8",)

    # test file: note2
    note2 = data_dir / "note2.md"
    note2.write_text("This is a test note2.\n", encoding="utf-8",)

    return data_dir


def test_get_stable_doc_id(tmp_path):
    test1_path = tmp_path / "test1.md"
    test2_path = tmp_path / "test2.md"

    test1_path.write_text("test1")
    test2_path.write_text("test2")

    doc_id0 = get_stable_doc_id(test1_path)
    doc_id1 = get_stable_doc_id(test1_path)

    assert doc_id0 == doc_id1

    doc_id2 = get_stable_doc_id(test2_path)

    assert doc_id0 != doc_id2


def test_load_notes_generates_chunks(mock_notes_dir):
    chunks = list(load_notes(mock_notes_dir))

    assert len(chunks) == 2

    sources = {c.source for c in chunks}

    assert sources == {"note1.md", "note2.md"}

    doc_ids = {c.doc_id for c in chunks}

    assert len(doc_ids) == 2

    c1, c2 = chunks
    assert c1.chunk_id  == 0
    assert c2.chunk_id  == 0
    assert "test note1" in c1.text
    assert "test note2" in c2.text


def test_load_notes_skips_symlinks_resolving_outside_the_approved_root(
    monkeypatch,
    tmp_path,
):
    notes_dir = tmp_path / "data"
    notes_dir.mkdir(exist_ok=True)
    (notes_dir / "allowed.md").write_text("Allowed note", encoding="utf-8")

    outside_note = tmp_path / "private.md"
    outside_note.write_text("Outside note", encoding="utf-8")
    (notes_dir / "linked.md").symlink_to(outside_note)
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    (outside_dir / "private.md").write_text("Private directory note", encoding="utf-8")
    (notes_dir / "linked-dir").symlink_to(outside_dir, target_is_directory=True)

    monkeypatch.setenv("NOTES_DIR", str(notes_dir))
    get_settings.cache_clear()

    chunks = list(load_notes())

    assert {chunk.source for chunk in chunks} == {"allowed.md"}
    assert all("Outside note" not in chunk.text for chunk in chunks)
    assert all("Private directory note" not in chunk.text for chunk in chunks)


def test_load_notes_rejects_a_root_outside_the_approved_notes_directory(
    monkeypatch,
    tmp_path,
):
    notes_dir = tmp_path / "data"
    notes_dir.mkdir(exist_ok=True)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "private.md").write_text("Private note", encoding="utf-8")

    monkeypatch.setenv("NOTES_DIR", str(notes_dir))
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="outside the approved data root"):
        list(load_notes(outside_dir))


def test_load_notes_rejects_notes_dir_override_outside_data_root(
    monkeypatch,
    tmp_path,
):
    trusted_root = tmp_path / "data"
    outside_dir = tmp_path / "external-notes"
    trusted_root.mkdir(exist_ok=True)
    outside_dir.mkdir()
    (outside_dir / "private.md").write_text("Private note", encoding="utf-8")

    monkeypatch.setenv("NOTES_DIR", str(outside_dir))
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="outside the approved data root"):
        list(load_notes())


def test_load_notes_rejects_a_data_root_symlink_outside_the_project(
    monkeypatch,
    tmp_path,
):
    project_root = tmp_path / "project"
    project_root.mkdir()
    external_root = tmp_path / "external-data"
    external_root.mkdir()
    (external_root / "private.md").write_text("Private note", encoding="utf-8")
    (project_root / "data").symlink_to(external_root, target_is_directory=True)

    monkeypatch.setenv("PROJECT_ROOT", str(project_root))
    monkeypatch.setenv("NOTES_DIR", str(project_root / "data"))
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="Runtime data root resolves outside the project root"):
        list(load_notes())


def test_get_changed_doc_ids(mock_notes_dir):
    c1, c2 = list(load_notes(mock_notes_dir))

    c1_doc_id = c1.doc_id
    c2_doc_id = c2.doc_id
    c3_doc_id = "xxxxxxxxx"

    changed_ids, unchanged_ids = get_changed_doc_ids(
        {c1_doc_id, c3_doc_id},
        mock_notes_dir,
    )

    assert changed_ids[0][0] == c2_doc_id

    assert unchanged_ids == {c1.doc_id}
