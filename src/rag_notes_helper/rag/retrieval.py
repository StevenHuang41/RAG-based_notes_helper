from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag.index import RagIndex
from rag_notes_helper.rag.meta_store import MetaStore

def retrieve(
    rag: RagIndex,
    meta_store: MetaStore,
    *,
    query: str,
    top_k: int | None = None,
) -> list[dict]:
    settings = get_settings()
    top_k = top_k or settings.top_k

    model = rag.embed_model

    q_emb = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, indices = rag.index.search(q_emb, top_k)

    results: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        if score < settings.min_retrieval_score:
            continue

        item = meta_store.get(idx)
        item["score"] = float(score)
        results.append(item)

    return results

