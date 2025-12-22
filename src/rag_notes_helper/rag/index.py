from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import faiss
from sentence_transformers import SentenceTransformer

from rag_notes_helper.core.config import settings
from rag_notes_helper.rag.chunking import Chunk


@dataclass
class RagIndex:
    index: faiss.Index
    meta: list[dict[str, Any]]


def _ensure_storage_dir() -> Path:
    settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return settings.STORAGE_DIR


def build_index(chunks: list[Chunk]) -> RagIndex:
    if not chunks:
        raise ValueError("No chunks to index")

    model = SentenceTransformer(settings.EMBEDDING_MODEL)

    embeddings = model.encode(
        [c.text for c in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    meta = [
        {
            "doc_id": c.doc_id,
            "chunk_id": c.chunk_id,
            "source": c.source,
            "text": c.text,
        }
        for c in chunks
    ]

    return RagIndex(index=index, meta=meta)


def save_index(rag: RagIndex) -> None:
    storage = _ensure_storage_dir()

    index_path = storage / "faiss.index"
    meta_path = storage / "meta.jsonl"

    faiss.write_index(rag.index, str(index_path))

    with meta_path.open("w", encoding="utf-8") as f:
        for row in rag.meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_index() -> RagIndex:
    storage = _ensure_storage_dir()

    index_path = storage / "faiss.index"
    meta_path = storage / "meta.jsonl"

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Index not found. Build it first.")

    index = faiss.read_index(str(index_path))

    meta: list[dict[str, Any]] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))

    return RagIndex(index=index, meta=meta)

