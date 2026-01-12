from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import struct

import faiss
from tqdm import tqdm

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag.chunking import Chunk


@dataclass
class RagIndex:
    index: faiss.Index
    _model =  None

    @property
    def embed_model(self):
        return self.get_model()

    @classmethod
    def get_model(cls):
        if cls._model is None:
            from sentence_transformers import SentenceTransformer
            settings = get_settings()
            print(f"Loading {settings.EMBEDDING_MODEL}...")
            cls._model = SentenceTransformer(settings.EMBEDDING_MODEL)

        return cls._model


def build_index(chunks: Iterable[Chunk], batch_size: int = 1024) -> RagIndex:
    settings = get_settings()
    storage: Path = settings.STORAGE_DIR

    model = RagIndex.get_model()

    index = None
    batch: list[Chunk] = []

    offset_packer = struct.Struct("Q")

    with (
        (storage / "meta.idx").open("wb") as idx_f,
        (storage / "meta.jsonl").open("wb") as meta_f
    ):
        for chunk in tqdm(chunks, desc="Indexing chunks"):
            batch.append(chunk)

            if len(batch) >= batch_size:
                index = _process_batch(batch, model, index,
                                       meta_f, idx_f, offset_packer)
                batch.clear()

        if batch:
            index = _process_batch(batch, model, index,
                                   meta_f, idx_f, offset_packer)

    if index is None:
        raise ValueError("No chunks to index")

    return RagIndex(index=index)


def _process_batch(
    batch: list[Chunk],
    model,
    index,
    meta_f,
    idx_f,
    packer: struct.Struct,
) -> faiss.Index:
    if not batch:
        return index

    # 1. generate embedding vectors
    embeddings = model.encode(
        [c.text for c in batch],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    # 2. initialize or update faiss index
    if index is None:
        index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)

    # 3. write meta and offset
    for chunk in batch:
        offset = meta_f.tell()

        record = {
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "text": chunk.text,
        }
        # write meta
        meta_f.write(
            json.dumps(record, ensure_ascii=False).encode("utf-8")
            + b"\n"
        )
        # write offset
        idx_f.write(packer.pack(offset))

    return index


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



