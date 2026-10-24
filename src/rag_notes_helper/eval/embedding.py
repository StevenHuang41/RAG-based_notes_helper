_embedding = None


def get_embeddings():
    global _embedding

    if _embedding is None:
        from ragas.embeddings import HuggingFaceEmbeddings

        from rag_notes_helper.core.config import get_settings

        _embedding = HuggingFaceEmbeddings(
            model=get_settings().embed_model_name,
            normalize_embeddings=True,
        )

    return _embedding
