from rag_notes_helper.rag.llm import get_llm
from rag_notes_helper.core.config import get_settings


def rag_answer(
    *,
    query: str,
    hits: list[dict],
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict:
    settings = get_settings()

    if not hits:
        return {
            "answer": (
                "I could not find relevant information in your notes. \n"
                "Try rephrasing the question or add more notes.\n"
            ),
            "citations": [],
        }

    hits = hits[: settings.LLM_MAX_CHUNKS]

    context_blocks = []
    citations = []

    for h in hits:
        context_blocks.append(
            f"file:{h['source']} \n{h['text']}"
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
            "content":
            """
                ROLE: You are "Notes Helper," a personal assistant.
                The person asking questions is the "Owner" of these notes.

                IDENTITY RULES:
                - If asked "Who are you?", answer: I am Notes Helper.
                - If asked "Who am I?", answer: You are the owner of these notes.

                GROUNDING RULES:
                - For all other questions: Use ONLY the provided Context.
                - If information is missing from Context, say: I cannot find
                  the answers in the notes
                - Do not use outside knowledge.

                OUTPUT FORMAT:
                - Plain text only. NO markdown, NO asterisks (*), NO bold (**).
                - Maximum 70 characters per line.
                - Do not include introductions like "Based on the notes..."
            """
        },
        {
            "role": "user",
            "content":
            f"""
                Context:
                {context}
                Question:
                {query}
            """
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
