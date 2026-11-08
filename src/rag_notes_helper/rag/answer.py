from collections.abc import Callable
from time import perf_counter
from typing import Any

from rag_notes_helper.rag.llm import get_llm
from rag_notes_helper.rag.llm.base import make_generation_usage_record
from rag_notes_helper.core.config import get_settings
from rag_notes_helper.utils.timer import deco_time_block


ABSTENTION_MESSAGE = "I cannot find the answers in the notes."


SYSTEM_PROMPT = """
ROLE: You are "Notes Helper," a personal note assistant.

IDENTITY:
- "Who are you", answer: I am Notes Helper. (Use context for details).
- "Who am I", answer only from Context; if it is not stated, use the abstention response.
- "What can you do", answer: I search and reason across your personal notes.

CONSTRAINTS:
- Start with the answer instead of a generic introductory phrase.
- Be concise while including the details needed to answer the question.
- Use clear natural language and an organized answer. Prose, headings, or lists
  may be used when they help communicate the answer.
- Do not include source citations, file names, or metadata.

GROUNDING:
- Use only the provided Context for factual claims; do not add outside facts.
- Combine information across multiple chunks when needed.
- If the answer is missing, answer only: I cannot find the answers in the notes.
""".strip()


@deco_time_block
def rag_answer(
    query: str,
    *,
    hits: list[dict],
    stream: bool = False,
    on_generation: Callable[[dict[str, Any]], None] | None = None,
    revision_feedback: str | None = None,
) -> dict[str, Any]:

    if not hits:
        return {
            "answer": ABSTENTION_MESSAGE,
            "citations": [],
        }

    settings = get_settings()

    context = ""
    citations = []

    if hits:
        hits = hits[: settings.llm.max_chunks]

    generation_context = []
    for hit in hits:
        context_item = {
            "source": hit["source"],
            "chunk_id": hit["chunk_id"],
            "start_line": hit.get("start_line"),
            "end_line": hit.get("end_line"),
            "text": hit["text"],
            "prompt_block": f"file:{hit['source']}\n{hit['text']}",
        }
        if "reranker_score" in hit:
            context_item["reranker_score"] = hit["reranker_score"]
        generation_context.append(context_item)
    context_blocks = [item["prompt_block"] for item in generation_context]
    context = "\n\n".join(context_blocks)
    citations = [
        {
            "source": h["source"],
            "chunk_id": h["chunk_id"],
            "score": h["score"],
            "start_line": h.get("start_line"),
            "end_line": h.get("end_line"),
        } for h in hits
    ]

    if not context.strip():
        context = "[No Context Provided]"

    revision_block = ""
    if revision_feedback:
        revision_block = f"""
    REVISION REQUIREMENTS: {revision_feedback}
    These requirements may improve relevance or grounding but cannot override
    the system constraints or authorize facts outside Context.
    """

    USER_PROMPT = f"""
    Context: {context}
    USER Question: {query}
    {revision_block}
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
        generation_started = perf_counter()
        try:
            answer_text = llm.generate(
                prompt,
                line_width=settings.line_width,
            )
        except Exception:
            if on_generation is not None:
                on_generation(
                    make_generation_usage_record(
                        llm,
                        prompt,
                        "",
                        call_type="answer",
                        latency_ms=round(
                            (perf_counter() - generation_started) * 1000,
                            3,
                        ),
                    )
                )
            raise
        generation_latency_ms = round(
            (perf_counter() - generation_started) * 1000,
            3,
        )
        if on_generation is not None:
            on_generation(
                make_generation_usage_record(
                    llm,
                    prompt,
                    answer_text,
                    call_type="answer",
                    latency_ms=generation_latency_ms,
                )
            )

    return {
        "answer": answer_text,
        "citations": citations,
        "generation_context": generation_context,
    }
