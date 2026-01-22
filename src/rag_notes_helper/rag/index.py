from itertools import chain
from pathlib import Path
from typing import Iterator
import json
import struct

import numpy as np
import faiss
from tqdm import tqdm

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag.chunking import Chunk
from rag_notes_helper.rag.ingest import get_changed_doc_ids, load_notes
from rag_notes_helper.rag.loaders import load_pdf_file, load_text_file
from rag_notes_helper.rag.meta_store import MetaStore
from rag_notes_helper.utils.logger import get_logger
from rag_notes_helper.utils.timer import time_block, deco_time_block


logger = get_logger("index")

class RagIndex:
    _model =  None

    def __init__(self, index: faiss.Index | None) -> None:
        self.index = index

    @property
    def embed_model(self):
        if RagIndex._model is None:
            with time_block("loading embedding model"):
                from sentence_transformers import SentenceTransformer
                embed_model_name = get_settings().embed_model_name
                RagIndex._model = SentenceTransformer(embed_model_name)

        return RagIndex._model


def build_index(
    chunks: Iterator[Chunk],
    batch_size: int = 1024,
) -> RagIndex:
    # check chunk is not empty
    chunks = iter(chunks)

    try :
        first = next(chunks)
    except StopIteration:
        raise ValueError("No chunks to index")

    chunks = chain([first], chunks)

    storage: Path = get_settings().storage_dir

    model = RagIndex(None).embed_model
    index = None
    batch: list[Chunk] = []
    packer = struct.Struct("Q")

    with time_block("processing chunks"):
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

    index.add(embeddings) # type: ignore


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

    with time_block("smart process chunks"):
        with (
            tmp_meta_path.open("wb") as meta_f,
            tmp_idx_path.open("wb") as idx_f,
        ):
            # 1. migrate unchanged files' chunks
            with MetaStore() as meta_store:
                for i in tqdm(
                    range(old_rag.index.ntotal), # type: ignore
                    desc="Migrating existing chunks ..."
                ):
                    chunk_dict = meta_store.get(i)
                    if chunk_dict["doc_id"] in unchanged_ids:
                        embeddings.append(old_rag.index.reconstruct(i)) # type: ignore
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

                if source.endswith(".pdf"):
                    chunk_iter = load_pdf_file(
                        path,
                        doc_id=doc_id,
                        source=source,
                    )
                else :
                    chunk_iter = load_text_file(
                        path,
                        doc_id=doc_id,
                        source=source,
                    )

                for chunk in chunk_iter:
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
        embeddings = model.encode( # type: ignore
            [c.text for c in batch],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
    else :
        embeddings = np.asarray(embeddings).astype("float32") # type: ignore

    if index is None:
        index = faiss.IndexFlatIP(embeddings.shape[1]) # type: ignore

    index.add(embeddings) # type: ignore

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


@deco_time_block
def save_index(rag: RagIndex) -> None:
    store_path = get_settings().storage_dir / "faiss.index"
    faiss.write_index(rag.index, str(store_path))


@deco_time_block
def load_index() -> RagIndex:
    store_path = get_settings().storage_dir / "faiss.index"

    if not store_path.exists():
        raise FileNotFoundError("Index not found")

    index = faiss.read_index(str(store_path))

    return RagIndex(index=index)


def build_and_save_rag() -> RagIndex:
    """ full building rag pipeline """
    rag = build_index(load_notes())
    save_index(rag)
    print("Index built and saved")

    return rag


def load_or_build_index():
    """ run when system start """
    try:
        rag = load_index()

    except FileNotFoundError:
        print("\nIndex not found. Building index from notes ...")
        rag = build_and_save_rag()
        print("Index built and saved.\n")

    return rag


def rebuild_index(force: bool = False):
    """ two rebuild mode: smart and force """
    print(f"\n{'Force' if force else 'Smart'} rebuilding index from notes ...")

    # force rebuild
    if force:
        return build_and_save_rag()

    # smart rebuild
    try :
        # 1. get current files' hash value
        with MetaStore() as meta_store:
            old_doc_ids = meta_store.get_all_doc_id()

        # 2. get the changed and unchanged file
        changed_ids, unchanged_ids = get_changed_doc_ids(old_doc_ids)

        if not changed_ids:
            print("Index is already up to date")
            return load_index()

        rag = smart_rebuild(changed_ids, unchanged_ids)
        save_index(rag)
        print("Index updated and saved")
        return rag

    except Exception as e:
        logger.warning(
            f"Smart rebuild failed ({e}), fall back to full rebuild index"
        )
        return build_and_save_rag()
