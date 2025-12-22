from sentence_transformers import SentenceTransformer

from rag_notes_helper.core.config import settings
from rag_notes_helper.rag.index import RagIndex


def retrieve(
    *,
    query: str,
    rag: RagIndex,
    top_k: int | None = None,
) -> list[dict]:
    top_k = top_k or settings.TOP_K

    model = SentenceTransformer(settings.EMBEDDING_MODEL)

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

        if score < settings.MIN_RETRIEVAL_SCORE:
            continue

        item = dict(rag.meta[idx])
        item["score"] = float(score)
        results.append(item)

    return results

