from pathlib import Path
from typing import Iterable
import json
import struct

import numpy as np
import faiss
from tqdm import tqdm

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag.chunking import Chunk, chunk_text
from rag_notes_helper.rag.ingest import get_changed_doc_ids, load_notes
from rag_notes_helper.rag.meta_store import MetaStore
from rag_notes_helper.utils.logger import get_logger
from rag_notes_helper.utils.timer import LapTimer


logger = get_logger("cli")

class RagIndex:
    _model =  None

    def __init__(self, index: faiss.Index | None) -> None:
        self.index = index

    @property
    def embed_model(self):
        if RagIndex._model is None:
            from sentence_transformers import SentenceTransformer
            embed_model_name = get_settings().embed_model_name
            logger.info(f"Loading {embed_model_name}...")
            RagIndex._model = SentenceTransformer(embed_model_name)

        return RagIndex._model


def build_index(
    chunks: Iterable[Chunk],
    batch_size: int = 1024,
) -> RagIndex:

    storage: Path = get_settings().storage_dir

    model = RagIndex(None).embed_model

    index = None
    batch: list[Chunk] = []
    packer = struct.Struct("Q")

    with (
        (storage / "meta.idx").open("wb") as idx_f,
        (storage / "meta.jsonl").open("wb") as meta_f
    ):
        for chunk in tqdm(chunks, desc="Indexing chunks"):
            batch.append(chunk)

            if len(batch) >= batch_size:
                index = _process_batch(
                    batch,
                    model,
                    index,
                    meta_f,
                    idx_f,
                    packer
                )
                batch.clear()

        if batch:
            index = _process_batch(
                batch,
                model,
                index,
                meta_f,
                idx_f,
                packer
            )

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


    # 3. write meta_f and offset_f
    for chunk in batch:
        offset = meta_f.tell()

        record = {
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "text": chunk.text,
        }

        # write meta_f
        meta_f.write(
            json.dumps(record, ensure_ascii=False).encode("utf-8")
            + b"\n"
        )
        # write offset_f
        idx_f.write(packer.pack(offset))

    return index


def smart_rebuild(
    changed_ids: list[tuple[str, Path]],
    unchanged_ids: set[str],
    batch_size = 1024,
) -> RagIndex:

    settings = get_settings()
    storage = settings.storage_dir
    notes_dir = settings.notes_dir

    tmp_meta_path = storage / "meta.json.tmp"
    tmp_idx_path = storage / "meta.idx.tmp"

    old_rag = load_index()
    model = old_rag.embed_model

    new_index = None
    embeddings = []
    batch: list[Chunk] = []
    packer = struct.Struct("Q")

    storage.mkdir(parents=True, exist_ok=True)

    with (
        tmp_meta_path.open("wb") as meta_f,
        tmp_idx_path.open("wb") as idx_f,
    ):
        # 1. migrate unchanged files' chunks
        with MetaStore() as meta_store:
            for i in tqdm(
                range(old_rag.index.ntotal),
                desc="Migrating existing chunks ..."
            ):
                chunk_dict = meta_store.get(i)
                if chunk_dict["doc_id"] in unchanged_ids:
                    embed_vec = old_rag.index.reconstruct(i)
                    embeddings.append(embed_vec)
                    batch.append(Chunk(**chunk_dict))

                # write unchanged chunks into meta and idx
                if len(batch) >= batch_size:
                    new_index = _smart_process_chunks(
                        batch,
                        new_index,
                        meta_f,
                        idx_f,
                        packer,
                        embeddings=embeddings,
                    )
                    embeddings.clear()
                    batch.clear()

            if batch:
                new_index = _smart_process_chunks(
                    batch,
                    new_index,
                    meta_f,
                    idx_f,
                    packer,
                    embeddings=embeddings,
                )
                embeddings.clear()
                batch.clear()

        # 2. get chunks from changed files
        for doc_id, path in tqdm(changed_ids, desc="Embedding new chunks ..."):
            source = str(path.relative_to(notes_dir))

            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for chunk in chunk_text(
                    file_object=f,
                    doc_id=doc_id,
                    source=source,
                    chunk_size=settings.chunk_size,
                    overlap=settings.chunk_overlap
                ):
                    batch.append(chunk)

                    #  write new chunks into meta and idx
                    if len(batch) >= batch_size:
                        new_index = _smart_process_chunks(
                            batch,
                            new_index,
                            meta_f,
                            idx_f,
                            packer,
                            has_new_chunk=True,
                            model=model,
                        )
                        batch.clear()

        if batch:
            new_index = _smart_process_chunks(
                batch,
                new_index,
                meta_f,
                idx_f,
                packer,
                has_new_chunk=True,
                model=model,
            )

    if new_index is None:
        raise ValueError("No notes left after smart rebuild")

    # 3. swap temp with original file meta and idx
    tmp_meta_path.replace(storage / "meta.jsonl")
    tmp_idx_path.replace(storage / "meta.idx")

    return RagIndex(index=new_index)


def _smart_process_chunks(
    batch: list[Chunk],
    index,
    meta_f,
    idx_f,
    packer: struct.Struct,
    *,
    embeddings: list[np.ndarray] | None = None,
    has_new_chunk: bool = False,
    model = None,
):
    if has_new_chunk:
        embeddings = model.encode(
            [c.text for c in batch],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
    else :
        embeddings = np.array(embeddings).astype("float32")

    if index is None:
        index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)

    for chunk in batch:
        offset = meta_f.tell()
        record = {
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "text": chunk.text,
        }

        # write meta_f
        meta_f.write(
            json.dumps(record, ensure_ascii=False).encode("utf-8")
            + b"\n"
        )

        # write offset_f
        idx_f.write(packer.pack(offset))

    return index


def save_index(rag: RagIndex) -> None:
    storage = get_settings().storage_dir
    index_path = storage / "faiss.index"
    faiss.write_index(rag.index, str(index_path))


def load_index() -> RagIndex:
    storage = get_settings().storage_dir
    index_path = storage / "faiss.index"

    if not index_path.exists():
        raise FileNotFoundError("Index not found. Build it first.")

    index = faiss.read_index(str(index_path))

    return RagIndex(index=index)


def build_and_save_rag() -> RagIndex:
    """ full building rag pipeline """
    chunks = load_notes()
    rag = build_index(chunks)
    save_index(rag)

    return rag

    # logger.info(f"load_notes latency={timer.lap():.2f} ms")
    # logger.info(f"build_index latency={timer.lap():.2f} ms")
    # logger.info(f"save_index latency={timer.lap():.2f} ms")

def load_or_build_index():
    """ run when system start """
    timer = LapTimer()
    try:
        timer.start()
        rag = load_index()
        logger.info(f"load_index latency={timer.lap():.2f} ms")

    except FileNotFoundError:
        logger.info("index not exists")
        print("\nIndex not found. Building index from notes ...")

        timer.start()
        rag = build_and_save_rag()
        logger.info(f"build_and_save_rag latency={timer.lap():.2f} ms")
        print("Index built and saved.\n")

    return rag


def rebuild_index(force: bool = False):
    """ two rebuild mode: smart and force """
    logger.info("---- rebuild_index()")
    print(f"\n{'Force' if force else 'Smart'} rebuilding index from notes ...")

    timer = LapTimer()

    # force rebuild
    if force:
        timer.start()
        rag = build_and_save_rag()
        logger.info(f"force_build latency={timer.lap():.2f} ms")
        print("Index built and saved")
        return rag

    # smart rebuild
    # 1. get current files' hash value
    try :
        timer.start()
        with MetaStore() as meta_store:
            logger.info(f"load MetaStore latency={timer.lap():.2f} ms")

            old_doc_ids = meta_store.get_all_doc_id()
            logger.info(f"get_all_doc_id latency={timer.lap():.2f} ms")
    except Exception:
        logger.info("No existing meta, rebuilding full index")

        timer.start()
        rag = build_and_save_rag()
        logger.info(f"full_rebuild latency={timer.lap():.2f} ms")
        return rag

    # 2. get the changed and unchanged file
    timer.start()
    changed_ids, unchanged_ids = get_changed_doc_ids(old_doc_ids)
    logger.info(f"get_changed_doc_ids latency={timer.lap():.2f} ms")

    # 3. rebuild only the changed files' chunk
    if changed_ids:
        timer.start()
        rag = smart_rebuild(changed_ids, unchanged_ids)
        logger.info(f"smart_rebuild latency={timer.lap():.2f} ms")

        save_index(rag)
        logger.info(f"save_index latency={timer.lap():.2f} ms")

        print("Index built and saved")
    else :
        rag = load_index()
        print("Index is already up to date")

    logger.info("rebuild_index() ----")
    return rag

