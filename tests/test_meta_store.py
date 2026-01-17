import json
import struct
import pytest

from rag_notes_helper.rag.meta_store import MetaStore


@pytest.fixture
def create_mock_meta_files(tmp_path):
    meta_path = tmp_path / "meta.jsonl"
    idx_path = tmp_path / "meta.idx"

    records = [
        {"doc_id": "d1", "chunk_id": 0, "source": "note1.md", "text": "d1c0"},
        {"doc_id": "d1", "chunk_id": 1, "source": "note1.md", "text": "d1c1"},

        {"doc_id": "d2", "chunk_id": 0, "source": "note2.md", "text": "d2c0"},
    ]

    with meta_path.open("wb") as meta_f, idx_path.open("wb") as idx_f:
        for record in records:
            offset = meta_f.tell()
            meta_f.write(json.dumps(record).encode("utf-8") + b"\n")
            idx_f.write(struct.pack("Q", offset))

    return tmp_path

def test_meta_store_get(create_mock_meta_files):
    with MetaStore(create_mock_meta_files) as meta_store:
        assert meta_store.get(0)["text"] == "d1c0"
        assert meta_store.get(1)["chunk_id"] == 1
        assert meta_store.get(2)["doc_id"] == "d2"

def test_list_sources_sorted_and_unique(create_mock_meta_files):
    with MetaStore(create_mock_meta_files) as meta_store:
        assert meta_store.list_indexed_sources() == ["note1.md", "note2.md"]

def test_list_sources_from_cache(create_mock_meta_files):
    with MetaStore(create_mock_meta_files) as meta_store:
        first_call = meta_store.list_indexed_sources()

        # remove meta.jsonl
        (create_mock_meta_files / "meta.jsonl").unlink()

        # test list_indexed_sources() loads from cache
        assert first_call == meta_store.list_indexed_sources()

def test_get_all_doc_ids(create_mock_meta_files):
    with MetaStore(create_mock_meta_files) as meta_store:
        assert meta_store.get_all_doc_id() == {"d1", "d2"}
