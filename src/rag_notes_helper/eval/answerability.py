"""Shared answer-availability rules for evaluation records and reports."""

def has_nonblank_answer(answer: object) -> bool:
    """Return whether an answer contains non-whitespace text."""
    return isinstance(answer, str) and bool(answer.strip())
