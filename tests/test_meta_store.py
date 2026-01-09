import json
import struct

from rag_notes_helper.rag.meta_store import MetaStore


def test_meta_store_get(tmp_path):
    meta_path = tmp_path / "meta.jsonl"
    idx_path = tmp_path / "meta.idx"

    records = [
        {"doc_id": "d1", "chunk_id": 0, "source": "note.md", "text": "test0"},
        {"doc_id": "d1", "chunk_id": 1, "source": "note.md", "text": "test1"},
    ]

    with meta_path.open("wb") as meta_f, idx_path.open("wb") as idx_f:
        for record in records:
            offset = meta_f.tell()
            meta_f.write(json.dumps(record).encode("utf-8") + b"\n")
            idx_f.write(struct.pack("Q", offset))


    with MetaStore(tmp_path) as meta_store:
        assert meta_store.get(0)["text"] == "test0"
        assert meta_store.get(1)["text"] == "test1"

    meta_store.close()

def test_list_indexed_sources(tmp_path):
    meta_path = tmp_path / "meta.jsonl"
    idx_path = tmp_path / "meta.idx"

    records = [
        {"doc_id": "d1", "chunk_id": 0, "source": "a", "text": "test0"},
        {"doc_id": "d2", "chunk_id": 0, "source": "b.md", "text": "test1"},
    ]

    with meta_path.open("wb") as meta_f, idx_path.open("wb") as idx_f:
        for record in records:
            offset = meta_f.tell()
            meta_f.write(json.dumps(record).encode("utf-8") + b"\n")
            idx_f.write(struct.pack("Q", offset))

    with MetaStore(tmp_path) as meta_store:
        assert meta_store.list_indexed_sources() == ["a", "b.md"]

