from rag_notes_helper.rag.llm import get_llm
from rag_notes_helper.core.config import settings


def rag_answer(
    *,
    query: str,
    hits: list[dict]
) -> dict:

    if not hits:
        return {
            "answer": (
                "I could not find relevant information in your notes. \n"
                "Try rephrasing the question or add more notes."
            ),
            "citations": [],
        }

    hits = hits[: settings.LLM_MAX_CHUNKS]

    context_blocks = []
    citations = []

    for i, h in enumerate(hits):
        context_blocks.append(
            f"[{i+1}] {h['source']} (chunk {h['chunk_id']})\n{h['text']}"
        )
        citations.append(
            {
                "source": h["source"],
                "chunk_id": h["chunk_id"],
                "score": h["score"],
            }
        )

    context = "\n\n".join(context_blocks)

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful notes assistant"
                "Answer the question using ONLY the context below"

                "If the answer is not in the context, say you cannot find "
                "answers in the notes"

                "Do NOT include citation markers [1], (1) in answers"
                "Do NOT mention line numbers or chunk numbers"
                "Do NOT include asterisk if you find names with it"
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\nQuestion:{query}\nAnswer:"
        }
    ]

    llm = get_llm()
    answer_text = llm.generate(prompt)

    return {
        "answer": answer_text,
        "citations": citations,
    }
