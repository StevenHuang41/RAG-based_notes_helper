import json
import struct
from pathlib import Path

class MetaStore:
    def __init__(self, storage_dir: Path):
        self.meta_f = (storage_dir / "meta.jsonl").open("rb")
        self.idx_f = (storage_dir / "meta.idx").open("rb")

    def get(self, faiss_id: int) -> dict:
        self.idx_f.seek(faiss_id * 8)
        offset = struct.unpack("Q", self.idx_f.read(8))[0]

        self.meta_f.seek(offset)
        return json.loads(self.meta_f.readline().decode("utf-8"))

    # def last_query_citations(self) -> list[str]:


    def list_indexed_sources(self) -> list[str]:
        position = self.meta_f.tell()

        try :
            sources = set()
            self.meta_f.seek(0)
            for line in self.meta_f:
                record = json.loads(line)
                sources.add(record["source"])

            return sorted(sources)

        finally:
            self.meta_f.seek(position)


    def close(self) -> None:
        self.meta_f.close()
        self.idx_f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


