_embedding = None

def get_embeddings():
    global _embedding

    if _embedding is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        _embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    return _embedding
