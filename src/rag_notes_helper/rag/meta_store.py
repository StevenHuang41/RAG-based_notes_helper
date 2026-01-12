import json
import struct
from pathlib import Path

from rag_notes_helper.core.config import get_settings

class MetaStore:
    def __init__(self, storage_dir: Path | None = None):
        storage_dir = storage_dir or get_settings().STORAGE_DIR
        self.meta_f = (storage_dir / "meta.jsonl").open("rb")
        self.idx_f = (storage_dir / "meta.idx").open("rb")

        self._unpacker = struct.Struct("Q")
        self._cached_sources = None

    def get(self, faiss_id: int) -> dict:
        self.idx_f.seek(faiss_id * 8)
        raw = self.idx_f.read(8)

        if len(raw) != 8:
            raise IndexError(f"Invalid faiss_id: {faiss_id}")

        offset = self._unpacker.unpack(raw)[0]

        self.meta_f.seek(offset)
        return json.loads(self.meta_f.readline().decode("utf-8"))


    def list_indexed_sources(self) -> list[str]:
        if self._cached_sources is not None:
            return self._cached_sources

        position = self.meta_f.tell()
        try :
            sources = set()
            self.meta_f.seek(0)
            for line in self.meta_f:
                record = json.loads(line)
                sources.add(record["source"])

            self._cached_sources = sorted(sources)

        finally:
            self.meta_f.seek(position)

        return self._cached_sources


    def close(self) -> None:
        self.meta_f.close()
        self.idx_f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


