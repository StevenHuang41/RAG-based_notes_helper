from collections.abc import Iterator
from typing import Any

from rag_notes_helper.rag.llm import get_llm
from rag_notes_helper.core.config import get_settings
from rag_notes_helper.utils.timer import deco_time_block


SYSTEM_PROMPT = """
ROLE: You are "Notes Helper," a personal assistant.
The person asking questions is the "Owner" of these notes.

IDENTITY RULES:
- If asked "Who are you?", answer ONLY: I am Notes Helper.
- If asked "Who am I?", answer ONLY: You are the owner of the
  notes in your data directory.
- If asked "What can you do?", answer ONLY: I help you search
  and reason across your personal notes.

GROUNDING & CROSS-CHUNK REASONING:
- For all other questions: Use ONLY the provided Context.
- MULTI-HOP REASONING: Treat all retrieved chunks as a single
  unified knowledge base. If information is split (e.g.,
  variables in one chunk, formula in another), you MUST
  combine them to answer the question.
- If information is missing, say: I cannot find the
  answers in the notes.
- Do not use outside knowledge.

OUTPUT FORMAT:
- Plain text only. NO markdown, NO asterisks (*), NO bold (**).
- Do not include introductions like "Based on the notes..."
""".strip()


@deco_time_block
def rag_answer(
    query: str,
    *,
    hits: list[dict],
) -> dict[str, Any]:
    settings = get_settings()

    if not hits:
        return {
            "answer": (
                "I could not find relevant information in your notes. \n"
                "Try rephrasing the question or add more notes."
            ),
            "citations": [],
        }

    hits = hits[: settings.llm.max_chunks]

    context_blocks = [f"file:{h['source']}\n{h['text']}" for h in hits]
    context = "\n\n".join(context_blocks)

    citations = [
        {
            "source": h["source"],
            "chunk_id": h["chunk_id"],
            "score": h["score"],
        } for h in hits
    ]

    prompt = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{query}"
        }
    ]

    llm = get_llm()

    if settings.stream:
        answer_text = llm.stream(
            prompt,
            line_width=settings.line_width,
        )
    else :
        answer_text = llm.generate(
            prompt,
            line_width=settings.line_width,
        )

    return {
        "answer": answer_text,
        "citations": citations,
    }
