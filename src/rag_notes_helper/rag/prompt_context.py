"""Shared conversation-history and prompt-budget helpers."""

import json
from collections.abc import Callable, Sequence
from typing import TypeAlias


ConversationTurn: TypeAlias = tuple[str, str]
MAX_CONVERSATION_TURNS = 5


def estimate_prompt_tokens(messages: Sequence[dict[str, str]]) -> int:
    """Return a conservative token estimate for serialized chat messages.

    This deliberately avoids a provider-specific tokenizer dependency. UTF-8
    byte count is a conservative proxy for text tokens; framing overhead is
    added for each message and for the assistant reply prefix.
    """
    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return len(serialized.encode("utf-8")) + 4 * len(messages) + 2


def fit_conversation_history(
    history: Sequence[ConversationTurn],
    *,
    build_prompt: Callable[[Sequence[ConversationTurn]], Sequence[dict[str, str]]],
    context_window_tokens: int,
    output_tokens: int,
) -> list[ConversationTurn]:
    """Trim oldest whole turns until the serialized prompt fits its budget."""
    candidate = list(history[-MAX_CONVERSATION_TURNS:])
    while candidate:
        prompt = build_prompt(candidate)
        if (
            estimate_prompt_tokens(prompt) + output_tokens
            <= context_window_tokens
        ):
            return candidate
        candidate.pop(0)
    return []


def serialize_conversation_history(
    history: Sequence[ConversationTurn],
) -> list[dict[str, str]]:
    """Serialize user/assistant turns without merging them into note evidence."""
    return [
        {"user": user, "assistant": assistant}
        for user, assistant in history[-MAX_CONVERSATION_TURNS:]
    ]
