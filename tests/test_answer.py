from collections.abc import Iterable
from typing import Any

import pytest

from rag_notes_helper.rag.answer import rag_answer


@pytest.fixture
def mock_dependencies(monkeypatch):
    class DummyLLM:
        def generate(self, prompt, line_width):
            return "generate dummy answer"

        def stream(self, prompt, line_width) -> Iterable:
            yield "stream"
            yield "dummy"
            yield "answer"

    monkeypatch.setattr(
        "rag_notes_helper.rag.answer.get_llm",
        lambda: DummyLLM(),
    )


@pytest.fixture
def mock_hits():
    hits = [
        {
            "text": "hits content",
            "source": "note.md",
            "chunk_id": 0,
            "score": 0.9,
        }
    ]
    return hits


def test_rag_answer_no_hits():
    result = rag_answer(query="q", hits=[])

    assert "could not find relevant information" in result["answer"].lower()
    assert result["citations"] == []


def test_rag_answer_generate(monkeypatch, mock_dependencies, mock_hits):

    monkeypatch.setenv("STREAM", "false")

    result = rag_answer(query="q", hits=mock_hits)

    assert isinstance(result["answer"], str)
    assert result["answer"] == "generate dummy answer"
    assert result["citations"][0]["source"] == "note.md"

def test_rag_answer_stream(monkeypatch, mock_dependencies, mock_hits):

    monkeypatch.setenv("STREAM", "true")

    result = rag_answer(query="q", hits=mock_hits)

    assert isinstance(result["answer"], Iterable)
    assert " ".join(list(result["answer"])) == "stream dummy answer"
    assert result["citations"][0]["source"] == "note.md"
