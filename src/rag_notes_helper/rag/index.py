from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json
from tqdm import tqdm

import faiss
from sentence_transformers import SentenceTransformer

from rag_notes_helper.core.config import settings
from rag_notes_helper.rag.chunking import Chunk


@dataclass
class RagIndex:
    index: faiss.Index
    meta: dict[int, dict[str, Any]]


def _ensure_storage_dir() -> Path:
    settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return settings.STORAGE_DIR


def build_index(chunks: Iterable[Chunk], batch_size: int = 32) -> RagIndex:
    model = SentenceTransformer(settings.EMBEDDING_MODEL)

    index = None
    meta: dict[int, dict] = {}
    batch: list[Chunk] = []
    faiss_id = 0

    for chunk in tqdm(chunks, desc="Indexing chunks"):
        batch.append(chunk)
        if len(batch) < batch_size:
            continue

        embeddings = model.encode(
            [c.text for c in batch],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        if index is None:
            index = faiss.IndexFlatIP(embeddings.shape[1])

        index.add(embeddings)

        for chunk in batch:
            meta[faiss_id] = {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "text": chunk.text,
            }
            faiss_id += 1

        batch.clear()

    if batch:
        embeddings = model.encode(
            [c.text for c in batch],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        if index is None:
            index = faiss.IndexFlatIP(embeddings.shape[1])

        index.add(embeddings)

        for chunk in batch:
            meta[faiss_id] = {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "text": chunk.text,
            }
            faiss_id += 1

    if index is None:
        raise ValueError("No chunks to index")

    return RagIndex(index=index, meta=meta)


def save_index(rag: RagIndex) -> None:
    storage = _ensure_storage_dir()

    index_path = storage / "faiss.index"
    meta_path = storage / "meta.jsonl"

    faiss.write_index(rag.index, str(index_path))

    with meta_path.open("w", encoding="utf-8") as f:
        for faiss_id, row in rag.meta.items():
            record = {"faiss_id": faiss_id, **row}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_index() -> RagIndex:
    storage = _ensure_storage_dir()

    index_path = storage / "faiss.index"
    meta_path = storage / "meta.jsonl"

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Index not found. Build it first.")

    index = faiss.read_index(str(index_path))

    meta: dict[int, dict[str, Any]] = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            faiss_id = record.pop("faiss_id")
            meta[faiss_id] = record

    return RagIndex(index=index, meta=meta)

def list_indexed_sources(rag: RagIndex) -> list[str]:
    return sorted({row["source"] for row in rag.meta.values()})


