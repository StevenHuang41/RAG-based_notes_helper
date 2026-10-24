import sys

import pytest

from rag_notes_helper import cli
from rag_notes_helper.rag.agentic import AgenticWorkflowError


def test_parser_accepts_agentic_one_shot_flag():
    classic_args = cli.build_parser().parse_args(["What is in my notes?"])
    args = cli.build_parser().parse_args(["What is in my notes?", "--agentic"])

    assert classic_args.agentic is False
    assert args.agentic is True
    assert args.repl is False


def test_parser_accepts_agentic_repl_combination():
    args = cli.build_parser().parse_args(["--repl", "--agentic"])

    assert args.repl is True
    assert args.agentic is True


def test_run_onetime_dispatches_to_agentic_workflow(monkeypatch):
    answer = {"answer": "Agentic answer.", "citations": []}
    calls = []
    displayed = []

    def fake_agentic_answer(query, *, rag, meta_store, stream=False):
        calls.append((query, rag, meta_store))
        return answer

    monkeypatch.setattr(cli, "agentic_answer", fake_agentic_answer)
    monkeypatch.setattr(cli, "retrieve", lambda *args, **kwargs: pytest.fail(
        "Agentic one-shot should not dispatch to Classic retrieval directly"
    ))
    monkeypatch.setattr(cli, "display_ansewr", displayed.append)

    cli.run_onetime(
        rag="index",
        meta_store="metadata",
        query="Question",
        agentic=True,
    )

    assert calls == [("Question", "index", "metadata")]
    assert displayed == ["Agentic answer."]


def test_one_shot_reports_agentic_workflow_error(monkeypatch, capsys):
    class DummySettings:
        def model_dump_json(self):
            return "{}"

    monkeypatch.setattr(sys, "argv", ["rag-app", "Question", "--agentic"])
    monkeypatch.setattr(cli, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(cli, "load_or_build_index", lambda: "index")
    monkeypatch.setattr(cli, "MetaStore", lambda: "metadata")

    def fail_agentic(*args, **kwargs):
        raise AgenticWorkflowError("The Agentic action remained invalid after one retry.")

    monkeypatch.setattr(cli, "agentic_answer", fail_agentic)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert "Agentic workflow error" in capsys.readouterr().err


def test_agentic_repl_uses_selected_workflow_for_initial_and_followup_queries(
    monkeypatch,
):
    class DummySettings:
        stream = False

        def model_dump_json(self):
            return "{}"

    class DummyMetaStore:
        def close(self):
            pass

    meta_store = DummyMetaStore()
    queries = iter(["follow-up question", ":q"])
    calls = []
    displayed = []

    def fake_agentic_answer(query, *, rag, meta_store, stream=False):
        calls.append((query, rag, meta_store, stream))
        return {"answer": f"answer for {query}", "citations": []}

    monkeypatch.setattr(
        sys,
        "argv",
        ["rag-app", "--repl", "--agentic", "initial question"],
    )
    monkeypatch.setattr(cli, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(cli, "load_or_build_index", lambda: "index")
    monkeypatch.setattr(cli, "MetaStore", lambda: meta_store)
    monkeypatch.setattr(cli, "agentic_answer", fake_agentic_answer)
    monkeypatch.setattr(cli, "retrieve", lambda *args, **kwargs: pytest.fail(
        "Agentic REPL must not dispatch to Classic retrieval"
    ))
    monkeypatch.setattr(cli, "display_ansewr", displayed.append)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(queries))

    cli.main()

    assert calls == [
        ("initial question", "index", meta_store, False),
        ("follow-up question", "index", meta_store, False),
    ]
    assert displayed == [
        "answer for initial question",
        "answer for follow-up question",
    ]


def test_agentic_repl_reports_invalid_action_and_accepts_next_query(
    monkeypatch,
    capsys,
):
    class DummySettings:
        stream = False

    class DummyMetaStore:
        def close(self):
            pass

    meta_store = DummyMetaStore()
    queries = iter(["bad action", "valid follow-up", ":q"])
    calls = []

    def fake_agentic_answer(query, *, rag, meta_store, stream=False):
        calls.append(query)
        if query == "bad action":
            raise AgenticWorkflowError("The Agentic action remained invalid after one retry.")
        return {"answer": "valid answer", "citations": []}

    monkeypatch.setattr(cli, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(cli, "agentic_answer", fake_agentic_answer)
    monkeypatch.setattr(cli, "retrieve", lambda *args, **kwargs: pytest.fail(
        "Agentic REPL must not fall back to Classic retrieval"
    ))
    monkeypatch.setattr(cli, "display_ansewr", lambda answer: None)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(queries))

    cli.repl("index", meta_store, agentic=True)

    assert calls == ["bad action", "valid follow-up"]
    assert "Agentic workflow error" in capsys.readouterr().err


def test_repl_alone_keeps_classic_workflow(monkeypatch):
    class DummySettings:
        stream = False

        def model_dump_json(self):
            return "{}"

    class DummyMetaStore:
        def close(self):
            pass

    meta_store = DummyMetaStore()
    queries = iter(["classic question", ":q"])
    retrieval_queries = []
    answers = []

    monkeypatch.setattr(sys, "argv", ["rag-app", "--repl"])
    monkeypatch.setattr(cli, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(cli, "load_or_build_index", lambda: "index")
    monkeypatch.setattr(cli, "MetaStore", lambda: meta_store)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(queries))
    monkeypatch.setattr(
        cli,
        "retrieve",
        lambda rag, store, *, query: (
            retrieval_queries.append(query) or [{"source": "note.md"}]
        ),
    )
    monkeypatch.setattr(
        cli,
        "rag_answer",
        lambda query, *, hits, stream: (
            answers.append((query, hits, stream))
            or {"answer": "Classic answer", "citations": []}
        ),
    )
    monkeypatch.setattr(cli, "display_ansewr", lambda answer: None)

    cli.main()

    assert retrieval_queries == ["classic question"]
    assert answers == [
        ("classic question", [{"source": "note.md"}], False),
    ]


def test_eval_command_runs_benchmark_without_loading_user_index(monkeypatch):
    class DummySettings:
        def model_dump_json(self):
            return "{}"

    calls = []
    monkeypatch.setattr(sys, "argv", ["rag-app", "--eval"])
    monkeypatch.setattr(cli, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(cli, "run_evaluation", lambda: calls.append("benchmark"))
    monkeypatch.setattr(
        cli,
        "load_or_build_index",
        lambda: pytest.fail("--eval must use the fixed fixture corpus"),
    )

    cli.main()

    assert calls == ["benchmark"]


def test_eval_command_explains_its_source_checkout_requirement(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["rag-app", "--eval"])

    def missing_evaluator(_module_name):
        raise ModuleNotFoundError("No module named 'rag_notes_helper.eval'")

    monkeypatch.setattr(cli.importlib, "import_module", missing_evaluator)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert "source checkout" in capsys.readouterr().err


def test_eval_base_command_keeps_legacy_smoke_separate(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["rag-app", "--eval-base"])
    monkeypatch.setattr(
        cli,
        "run_base_smoke_evaluation",
        lambda: calls.append("legacy smoke"),
    )
    monkeypatch.setattr(
        cli,
        "load_or_build_index",
        lambda: pytest.fail("--eval-base manages its own user index"),
    )

    cli.main()

    assert calls == ["legacy smoke"]
