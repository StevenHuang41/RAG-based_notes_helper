from typing import Any

from rag_notes_helper.rag.llm import get_llm
from rag_notes_helper.core.config import get_settings
from rag_notes_helper.utils.timer import deco_time_block


SYSTEM_PROMPT = """
ROLE: You are "Notes Helper," a personal note assistant.

IDENTITY:
- "Who are you", answer: I am Notes Helper. (Use context for details).
- "Who am I", answer: You are the owner of the notes in your data directory.
- "What can you do", answer: I search and reason across your personal notes.

CONSTRAINTS:
- START the answer immediately. NEVER use introductory phrases.
- NEVER include source citations, file names, or metadata.
- NEVER use markdown formatting (no **, no #, no lists with *).

GROUNDING:
- Use ONLY in the provided Context.
- REASONING: You MUST perform multi-step reasoning.
- Combine information across multiple chunks if necessary.
- If the answer is missing, answer ONLY: I cannot find the answers in the notes.
- Do NOT use outside knowledge

OUTPUT FORMAT:
- STRICT Plain text.
- Use only numbered lists (1. 2. 3.) if needed.
- THINK STEP_BY_STEP INTERNALLY to ensure answers are correct.
""".strip()


@deco_time_block
def rag_answer(
    query: str,
    *,
    hits: list[dict],
    stream: bool = False,
) -> dict[str, Any]:

    # if not hits:
    #     return {
    #         "answer": (
    #             "I could not find relevant information in your notes. \n"
    #                 "Try rephrasing the question or add more notes."
    #         ),
    #         "citations": [],
    #     }

    settings = get_settings()

    context = ""
    citations = []

    if hits:
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

    if not context.strip():
        context = "[No Context Provided]"

    USER_PROMPT = f"""
    Context: {context}
    USER Question: {query}
    """

    prompt = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": USER_PROMPT
        }
    ]

    llm = get_llm()

    if stream:
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
