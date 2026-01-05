from rag_notes_helper.rag.llm import get_llm
from rag_notes_helper.core.config import settings


def rag_answer(
    *,
    query: str,
    hits: list[dict],
    max_tokens: int | None = None,
    temperature: float | None = None,
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
                "You are a helpful notes assistant\n"
                "Answer the question using ONLY the context below\n"

                "If the answer is not in the context, say you cannot find the"
                "answers in the notes\n"

                "FORMAT RULES:\n"
                "- Use plain text only\n"
                "- Do NOT use markdown\n"
                "- Do NOT include citation markers [1], (1) in answers\n"
                "- Do NOT mention line numbers or chunk numbers"
                "- Do NOT use **, *, or _\n"
                "- Use numbered lists with plain numbers and periods only\n"
                "- a line should have maximum 70 characters\n"
                "- Do NOT use numbered list if the answer is brief"
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\nQuestion:{query}\nAnswer:"
        }
    ]

    llm = get_llm()
    answer_text = llm.generate(
        prompt,
        max_tokens or settings.LLM_MAX_TOKENS,
        temperature or settings.LLM_TEMPERATURE,
    )

    return {
        "answer": answer_text,
        "citations": citations,
    }
