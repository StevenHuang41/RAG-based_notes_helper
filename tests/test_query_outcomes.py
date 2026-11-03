import sys
from types import SimpleNamespace

import pytest

from rag_notes_helper import cli
from rag_notes_helper.rag.answer import ABSTENTION_MESSAGE


@pytest.mark.parametrize("query", ["", "   \t\n"])
def test_classic_blank_query_returns_structured_invalid_outcome(
    monkeypatch,
    query,
):
    monkeypatch.setattr(
        cli,
        "retrieve",
        lambda *_args, **_kwargs: pytest.fail("blank input must not run Retrieval"),
    )
    monkeypatch.setattr(
        cli,
        "rag_answer",
        lambda *_args, **_kwargs: pytest.fail("blank input must not call the Answer model"),
    )

    result = cli.run_query(object(), object(), query=query)

    assert result["answer"].strip()
    assert result["reason_code"] == "invalid_query"
    assert result["response_origin"] == "deterministic"
    assert result["citations"] == []
    assert result["generation_context"] == []
    assert result["retrieved_evidence"] == []
    assert result["search_queries"] == []
    assert result["retrieval_stages"] == []


def test_classic_empty_retrieval_returns_structured_no_evidence_outcome(
    monkeypatch,
):
    queries = []

    def retrieve_no_candidates(_rag, _meta_store, *, query, **_kwargs):
        queries.append(query)
        return []

    monkeypatch.setattr(cli, "retrieve", retrieve_no_candidates)
    monkeypatch.setattr(
        cli,
        "rag_answer",
        lambda *_args, **_kwargs: pytest.fail(
            "empty Retrieval must not call the Answer model"
        ),
    )

    result = cli.run_query(object(), object(), query="Who owns the launch?")

    assert queries == ["Who owns the launch?"]
    assert result["answer"] == (
        "I could not find sufficient supporting evidence in the notes."
    )
    assert result["reason_code"] == "no_evidence_after_searches"
    assert result["response_origin"] == "deterministic"
    assert result["citations"] == []
    assert result["generation_context"] == []
    assert result["retrieved_evidence"] == []
    assert result["search_queries"] == ["Who owns the launch?"]
    assert result["retrieval_stages"] == []


def test_classic_grounded_answer_preserves_generation_evidence_and_metadata(
    monkeypatch,
):
    hit = {
        "source": "notes/launch.md",
        "chunk_id": 4,
        "score": 0.91,
        "text": "Mina owns the launch.",
    }
    context = [{
        "source": "notes/launch.md",
        "chunk_id": 4,
        "text": "Mina owns the launch.",
        "prompt_block": "file:notes/launch.md\nMina owns the launch.",
    }]
    answer_calls = []

    monkeypatch.setattr(cli, "retrieve", lambda *_args, **_kwargs: [hit])

    def answer_once(query, *, hits, stream=False, **_kwargs):
        answer_calls.append((query, hits, stream))
        return {
            "answer": "Mina owns the launch.",
            "citations": [{"source": "notes/launch.md", "chunk_id": 4}],
            "generation_context": context,
        }

    monkeypatch.setattr(cli, "rag_answer", answer_once)

    result = cli.run_query(object(), object(), query="Who owns the launch?")

    assert answer_calls == [("Who owns the launch?", [hit], False)]
    assert result["answer"] == "Mina owns the launch."
    assert result["reason_code"] == "answered"
    assert result["response_origin"] == "answer_model"
    assert result["citations"] == [
        {"source": "notes/launch.md", "chunk_id": 4}
    ]
    assert result["generation_context"] == context
    assert result["retrieved_evidence"] == [hit]
    assert result["search_queries"] == ["Who owns the launch?"]


def test_classic_model_insufficient_evidence_is_distinct_from_empty_retrieval(
    monkeypatch,
):
    hit = {
        "source": "notes/launch.md",
        "chunk_id": 4,
        "score": 0.91,
        "text": "The launch has an owner.",
    }
    monkeypatch.setattr(cli, "retrieve", lambda *_args, **_kwargs: [hit])
    monkeypatch.setattr(
        cli,
        "rag_answer",
        lambda *_args, **_kwargs: {
            "answer": ABSTENTION_MESSAGE,
            "citations": [{"source": "notes/launch.md", "chunk_id": 4}],
            "generation_context": [hit],
        },
    )

    result = cli.run_query(object(), object(), query="When is the launch?")

    assert result["answer"] == ABSTENTION_MESSAGE
    assert result["reason_code"] == "insufficient_evidence"
    assert result["response_origin"] == "answer_model"


def test_classic_empty_model_text_returns_non_empty_deterministic_outcome(
    monkeypatch,
):
    hit = {
        "source": "notes/launch.md",
        "chunk_id": 4,
        "score": 0.91,
        "text": "The launch has an owner.",
    }
    monkeypatch.setattr(cli, "retrieve", lambda *_args, **_kwargs: [hit])
    monkeypatch.setattr(
        cli,
        "rag_answer",
        lambda *_args, **_kwargs: {
            "answer": "  ",
            "citations": [{"source": "notes/launch.md", "chunk_id": 4}],
            "generation_context": [hit],
        },
    )

    result = cli.run_query(object(), object(), query="When is the launch?")

    assert result["answer"] == cli.NO_EVIDENCE_MESSAGE
    assert result["reason_code"] == "insufficient_evidence"
    assert result["response_origin"] == "deterministic"
    assert result["generation_context"] == [hit]


def test_classic_repl_displays_invalid_input_outcome(monkeypatch, capsys):
    inputs = iter(["   ", ":q"])

    class MetaStore:
        def close(self):
            pass

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    cli.repl(object(), MetaStore())

    output = capsys.readouterr().out
    assert cli.INVALID_QUERY_MESSAGE in output


def test_classic_one_shot_displays_invalid_input_outcome(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        cli,
        "retrieve",
        lambda *_args, **_kwargs: pytest.fail("blank input must not run Retrieval"),
    )

    cli.run_onetime(object(), object(), query="   ")

    output = capsys.readouterr().out
    assert cli.INVALID_QUERY_MESSAGE in output


def test_cli_routes_submitted_whitespace_to_classic_one_shot(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["rag-app", "   "])
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: SimpleNamespace(model_dump_json=lambda: "{}"),
    )
    monkeypatch.setattr(cli, "load_or_build_index", lambda: "index")
    monkeypatch.setattr(cli, "MetaStore", lambda: "meta-store")
    monkeypatch.setattr(
        cli,
        "run_onetime",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    cli.main()

    assert calls == [((), {
        "query": "",
        "citations": False,
        "rag": "index",
        "meta_store": "meta-store",
        "agentic": False,
    })]
