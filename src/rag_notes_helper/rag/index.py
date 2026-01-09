from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import struct

import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag.chunking import Chunk


@dataclass
class RagIndex:
    index: faiss.Index


def build_index(chunks: Iterable[Chunk], batch_size: int = 32) -> RagIndex:
    settings = get_settings()
    model = SentenceTransformer(settings.EMBEDDING_MODEL)

    index = None
    batch: list[Chunk] = []

    storage: Path = settings.STORAGE_DIR
    idx_f = (storage / "meta.idx").open("wb")
    meta_f = (storage / "meta.jsonl").open("wb")

    try :
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
                offset = meta_f.tell()

                meta_f.write(
                    json.dumps(
                        {
                            "doc_id": chunk.doc_id,
                            "chunk_id": chunk.chunk_id,
                            "source": chunk.source,
                            "text": chunk.text,

                        },
                        ensure_ascii=False,
                    ).encode("utf-8") + b"\n"
                )
                idx_f.write(struct.pack("Q", offset))

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
                offset = meta_f.tell()

                meta_f.write(
                    json.dumps(
                        {
                            "doc_id": chunk.doc_id,
                            "chunk_id": chunk.chunk_id,
                            "source": chunk.source,
                            "text": chunk.text,

                        },
                        ensure_ascii=False,
                    ).encode("utf-8") + b"\n"
                )
                idx_f.write(struct.pack("Q", offset))

    finally:
        idx_f.close()
        meta_f.close()

    if index is None:
        raise ValueError("No chunks to index")

    return RagIndex(index=index)


def save_index(rag: RagIndex) -> None:
    storage = get_settings().STORAGE_DIR
    index_path = storage / "faiss.index"
    faiss.write_index(rag.index, str(index_path))

def load_index() -> RagIndex:
    storage = get_settings().STORAGE_DIR
    index_path = storage / "faiss.index"

    if not index_path.exists():
        raise FileNotFoundError("Index not found. Build it first.")

    index = faiss.read_index(str(index_path))

    return RagIndex(index=index)



