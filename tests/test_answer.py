from rag_notes_helper.rag.answer import rag_answer


class DummyLLM:
    def generate(self, prompt):
        return "dummy answer"


def test_rag_answer_no_hits():
    result = rag_answer(query="q", hits=[])
    assert "could not find relevant information" in result["answer"].lower()


def test_rag_answer_hits(monkeypatch):

    monkeypatch.setattr(
        "rag_notes_helper.rag.answer.get_llm",
        lambda: DummyLLM(),
    )

    hits = [
        {
            "text": "hits content",
            "source": "note.md",
            "chunk_id": 0,
            "score": 0.9,
        }
    ]

    result = rag_answer(query="q", hits=hits)

    assert result["answer"] == "dummy answer"
    assert result["citations"][0]["source"] == "note.md"

