"""Shared conversation-history types and serialization."""

from collections.abc import Sequence
from typing import TypeAlias


ConversationTurn: TypeAlias = tuple[str, str]
MAX_CONVERSATION_TURNS = 5


def serialize_conversation_history(
    history: Sequence[ConversationTurn],
) -> list[dict[str, str]]:
    """Serialize user/assistant turns without merging them into note evidence."""
    return [
        {"user": user, "assistant": assistant}
        for user, assistant in history[-MAX_CONVERSATION_TURNS:]
    ]
