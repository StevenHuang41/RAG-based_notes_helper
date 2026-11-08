import json
import sys
from io import StringIO

import pytest

from rag_notes_helper import cli
from rag_notes_helper.rag import agentic as agentic_module
from rag_notes_helper.rag import answer as answer_module
from rag_notes_helper.rag.agentic import AgenticProgress, AgenticWorkflowError


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


def test_parser_accepts_evaluation_sample_and_seed():
    args = cli.build_parser().parse_args([
        "--agentic",
        "--eval",
        "--sample",
        "3",
        "--seed",
        "17",
    ])

    assert args.sample == 3
    assert args.seed == 17


@pytest.mark.parametrize(
    "argv",
    [
        ["rag-app", "--sample", "2"],
        ["rag-app", "--seed", "17"],
    ],
)
def test_sampling_options_require_fixture_evaluation(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2


def test_run_onetime_dispatches_to_agentic_workflow(monkeypatch):
    answer = {"answer": "Agentic answer.", "citations": []}
    calls = []
    displayed = []

    def fake_agentic_answer(
        query, *, rag, meta_store, stream=False, on_progress=None,
        on_progress_event=None,
        conversation_history=(),
    ):
        calls.append((query, rag, meta_store, tuple(conversation_history)))
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

    assert calls == [("Question", "index", "metadata", ())]
    assert displayed == ["Agentic answer."]


def test_classic_and_agentic_share_initial_search_and_answer_generation(
    monkeypatch,
):
    query = "Who owns the review?"
    hits = [
        {
            "source": "notes/review.md",
            "chunk_id": 0,
            "score": 0.9,
            "text": "Mina owns the review.",
        },
        {
            "source": "notes/archive.md",
            "chunk_id": 1,
            "score": 0.7,
            "text": "The former owner was Aria.",
        },
    ]
    retrieval_calls = []
    answer_prompts = []
    controller_prompts = []

    def classic_retrieve(
        rag,
        meta_store,
        *,
        query,
        cross_encoder=None,
        vector_min_score=None,
        bm25_min_score_ratio=None,
    ):
        retrieval_calls.append((
            "classic",
            query,
            (cross_encoder, vector_min_score, bm25_min_score_ratio),
        ))
        return hits

    def agentic_retrieve(
        rag,
        meta_store,
        *,
        query,
        cross_encoder=None,
        vector_min_score=None,
        bm25_min_score_ratio=None,
    ):
        retrieval_calls.append((
            "agentic",
            query,
            (cross_encoder, vector_min_score, bm25_min_score_ratio),
        ))
        return hits

    class AnswerSettings:
        line_width = 80

        class LLM:
            max_chunks = 1

        llm = LLM()

    class AnswerModel:
        def generate(self, prompt, line_width):
            answer_prompts.append(prompt)
            return "Mina owns the review."

    class Controller:
        provider_name = "fixture"

        def generate_raw(self, prompt):
            controller_prompts.append(prompt)
            payload = json.loads(prompt[-1]["content"])
            if "candidate_answer" in payload:
                return json.dumps({"action": "accept"})
            return json.dumps({
                "action": "answer",
                "evidence": [{"source": hits[0]["source"], "chunk_id": 0}],
            })

    monkeypatch.setattr(cli, "retrieve", classic_retrieve)
    monkeypatch.setattr(agentic_module, "retrieve", agentic_retrieve)
    monkeypatch.setattr(answer_module, "get_settings", AnswerSettings)
    monkeypatch.setattr(answer_module, "get_llm", AnswerModel)
    monkeypatch.setattr(agentic_module, "get_llm", Controller)

    classic = cli.run_query("index", "metadata", query=query)
    agentic = cli.run_query("index", "metadata", query=query, agentic=True)

    assert [call[1] for call in retrieval_calls] == [query, query]
    assert [call[0] for call in retrieval_calls] == ["classic", "agentic"]
    assert retrieval_calls[0][2] == retrieval_calls[1][2] == (None, None, None)
    assert classic["answer"] == agentic["answer"] == "Mina owns the review."
    assert classic["generation_context"] == agentic["generation_context"]
    assert len(classic["generation_context"]) == 1
    assert len(answer_prompts) == 2
    assert answer_prompts[0] == answer_prompts[1]
    assert len(controller_prompts) == 2


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

    def fake_agentic_answer(
        query, *, rag, meta_store, stream=False, on_progress=None,
        on_progress_event=None,
        conversation_history=(),
    ):
        calls.append((query, rag, meta_store, stream, list(conversation_history)))
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
        ("initial question", "index", meta_store, False, []),
        ("follow-up question", "index", meta_store, False, []),
    ]
    assert displayed == [
        "answer for initial question",
        "answer for follow-up question",
    ]


def test_agentic_repl_keeps_only_five_prior_turns_and_starts_fresh(
    monkeypatch,
):
    class DummySettings:
        stream = False

    class DummyMetaStore:
        def close(self):
            pass

    queries = iter([*(f"question {i}" for i in range(7)), ":q"])
    calls = []

    def fake_run_query(
        rag, meta_store, *, query, agentic, stream, on_progress_event,
        conversation_history=(),
    ):
        calls.append((query, list(conversation_history)))
        return {"answer": f"answer {query}", "citations": []}

    monkeypatch.setattr(cli, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(cli, "run_query", fake_run_query)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(queries))
    monkeypatch.setattr(cli, "display_ansewr", lambda _answer: None)

    cli.repl("index", DummyMetaStore(), agentic=True)

    assert calls[0] == ("question 0", [])
    assert calls[5][1] == [
        (f"question {i}", f"answer question {i}") for i in range(5)
    ]
    assert calls[6][1] == [
        (f"question {i}", f"answer question {i}") for i in range(1, 6)
    ]

    calls.clear()
    queries = iter(["new session question", ":q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(queries))
    cli.repl("index", DummyMetaStore(), agentic=True)
    assert calls == [("new session question", [])]


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

    def fake_agentic_answer(
        query, *, rag, meta_store, stream=False, on_progress=None,
        on_progress_event=None,
        conversation_history=(),
    ):
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


class _TerminalStream(StringIO):
    def __init__(self, *, terminal):
        super().__init__()
        self.terminal = terminal

    def isatty(self):
        return self.terminal


def test_agentic_one_shot_renders_safe_progress_only_on_terminal_stderr(
    monkeypatch, capsys
):
    note_text = "PRIVATE NOTE TEXT"
    private_prompt = "PRIVATE CONTROLLER PROMPT"
    rejected_answer = "REJECTED CANDIDATE ANSWER"
    stderr = _TerminalStream(terminal=True)

    def fake_agentic_answer(
        query, *, rag, meta_store, stream=False, on_progress_event=None
    ):
        assert on_progress_event is not None
        for event in [
            AgenticProgress(
                event="workflow_started",
                max_search_attempts=2,
                max_answer_attempts=1,
            ),
            AgenticProgress(
                event="search_started",
                max_search_attempts=2,
                max_answer_attempts=1,
                search_attempt=1,
                search_action="initial",
                search_query="focused search terms",
            ),
            AgenticProgress(
                event="search_completed",
                max_search_attempts=2,
                max_answer_attempts=1,
                search_attempt=1,
                result_count=2,
            ),
            AgenticProgress(
                event="controller_started",
                max_search_attempts=2,
                max_answer_attempts=1,
            ),
            AgenticProgress(
                event="action_selected",
                max_search_attempts=2,
                max_answer_attempts=1,
                decision="answer",
            ),
            AgenticProgress(
                event="answer_started",
                max_search_attempts=2,
                max_answer_attempts=1,
                answer_attempt=1,
            ),
            AgenticProgress(
                event="answer_review_started",
                max_search_attempts=2,
                max_answer_attempts=1,
                answer_attempt=1,
            ),
            AgenticProgress(
                event="answer_review_decided",
                max_search_attempts=2,
                max_answer_attempts=1,
                answer_attempt=1,
                decision="accept",
            ),
            AgenticProgress(
                event="terminal",
                max_search_attempts=2,
                max_answer_attempts=1,
                outcome="answered",
            ),
        ]:
            on_progress_event(event)
            if event.event == "controller_started":
                assert stderr.getvalue().splitlines()[-1] == (
                    "[Agentic] Controller is reviewing evidence"
                )
            elif event.event == "answer_started":
                assert stderr.getvalue().splitlines()[-1] == (
                    f"[Agentic] Generating answer {event.answer_attempt}/1"
                )
            elif event.event == "answer_review_started":
                assert stderr.getvalue().splitlines()[-1] == (
                    f"[Agentic] Reviewing answer {event.answer_attempt}/1"
                )
        assert private_prompt not in repr(event)
        assert rejected_answer not in repr(event)
        return {
            "answer": "Grounded answer.",
            "citations": [],
            "retrieved_evidence": [{"text": note_text}],
        }

    monkeypatch.setattr(cli.sys, "stderr", stderr)
    monkeypatch.setattr(cli, "agentic_answer", fake_agentic_answer)
    monkeypatch.setattr(cli, "display_ansewr", lambda answer: print(answer))

    cli.run_onetime(
        rag="index",
        meta_store="metadata",
        query="Question containing private details",
        agentic=True,
    )

    assert capsys.readouterr().out == "Grounded answer.\n"
    assert stderr.getvalue().splitlines() == [
        "[Agentic] Limits: Search 2, Answer 1",
        "[Agentic] Read-only Search 1/2 (initial): focused search terms",
        "[Agentic] Read-only Search 1/2 returned 2 chunks",
        "[Agentic] Controller is reviewing evidence",
        "[Agentic] Selected action: generate answer",
        "[Agentic] Generating answer 1/1",
        "[Agentic] Reviewing answer 1/1",
        "[Agentic] Answer review: accept",
        "[Agentic] Outcome: answered",
    ]
    assert note_text not in stderr.getvalue()
    assert "Question containing private details" not in stderr.getvalue()
    assert private_prompt not in stderr.getvalue()
    assert rejected_answer not in stderr.getvalue()


def test_agentic_one_shot_suppresses_progress_when_stderr_is_redirected(
    monkeypatch, capsys
):
    stderr = _TerminalStream(terminal=False)

    def fake_agentic_answer(
        query, *, rag, meta_store, stream=False, on_progress_event=None
    ):
        on_progress_event(
            AgenticProgress(
                event="search_started",
                max_search_attempts=3,
                max_answer_attempts=3,
                search_attempt=1,
                search_action="initial",
                search_query=query,
            )
        )
        on_progress_event(
            AgenticProgress(
                event="terminal",
                max_search_attempts=3,
                max_answer_attempts=3,
                outcome="insufficient_evidence",
            )
        )
        return {"answer": "I cannot find the answers in the notes.", "citations": []}

    monkeypatch.setattr(cli.sys, "stderr", stderr)
    monkeypatch.setattr(cli, "agentic_answer", fake_agentic_answer)
    monkeypatch.setattr(cli, "display_ansewr", lambda answer: print(answer))

    cli.run_onetime(
        rag="index",
        meta_store="metadata",
        query="Question",
        agentic=True,
    )

    assert capsys.readouterr().out == "I cannot find the answers in the notes.\n"
    assert stderr.getvalue() == ""


def test_agentic_repl_renders_progress_for_each_query(monkeypatch):
    class DummySettings:
        stream = False

    class DummyMetaStore:
        def close(self):
            pass

    stderr = _TerminalStream(terminal=True)
    answers = []
    queries = iter(["Question one", "Question two", ":q"])

    def fake_agentic_answer(
        query, *, rag, meta_store, stream=False, on_progress_event=None,
        conversation_history=(),
    ):
        if query == "Question one":
            events = [
                AgenticProgress(
                    event="workflow_started",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                ),
                AgenticProgress(
                    event="search_started",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                    search_attempt=1,
                    search_action="initial",
                    search_query="first selected query",
                ),
                AgenticProgress(
                    event="search_completed",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                    search_attempt=1,
                    result_count=3,
                ),
                AgenticProgress(
                    event="controller_started",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                ),
                AgenticProgress(
                    event="action_selected",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                    decision="search",
                ),
                AgenticProgress(
                    event="search_started",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                    search_attempt=2,
                    search_action="extend",
                    search_query="controller-selected query",
                ),
                AgenticProgress(
                    event="search_completed",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                    search_attempt=2,
                    result_count=1,
                ),
                AgenticProgress(
                    event="controller_started",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                ),
                AgenticProgress(
                    event="action_selected",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                    decision="answer",
                ),
                AgenticProgress(
                    event="answer_started",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                    answer_attempt=1,
                ),
                AgenticProgress(
                    event="answer_review_started",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                    answer_attempt=1,
                ),
                AgenticProgress(
                    event="answer_review_decided",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                    answer_attempt=1,
                    decision="revise",
                ),
                AgenticProgress(
                    event="answer_started",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                    answer_attempt=2,
                ),
                AgenticProgress(
                    event="answer_review_started",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                    answer_attempt=2,
                ),
                AgenticProgress(
                    event="answer_review_decided",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                    answer_attempt=2,
                    decision="search",
                ),
                AgenticProgress(
                    event="search_started",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                    search_attempt=3,
                    search_action="replace",
                    search_query="review-selected query",
                ),
                AgenticProgress(
                    event="search_completed",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                    search_attempt=3,
                    result_count=2,
                ),
                AgenticProgress(
                    event="controller_started",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                ),
                AgenticProgress(
                    event="action_selected",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                    decision="clarify",
                ),
                AgenticProgress(
                    event="clarification_requested",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                ),
                AgenticProgress(
                    event="terminal",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                    outcome="needs_clarification",
                ),
            ]
        else:
            events = [
                AgenticProgress(
                    event="clarification_requested",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                ),
                AgenticProgress(
                    event="terminal",
                    max_search_attempts=3,
                    max_answer_attempts=4,
                    outcome="needs_clarification",
                ),
            ]
        for event in events:
            on_progress_event(event)
            if event.event == "answer_started":
                assert stderr.getvalue().splitlines()[-1] == (
                    f"[Agentic] Generating answer {event.answer_attempt}/4"
                )
            elif event.event == "answer_review_started":
                assert stderr.getvalue().splitlines()[-1] == (
                    f"[Agentic] Reviewing answer {event.answer_attempt}/4"
                )
        return {
            "answer": f"Answer for {query}",
            "citations": [],
            "retrieved_evidence": [{"text": "PRIVATE NOTE TEXT"}],
        }

    monkeypatch.setattr(cli.sys, "stderr", stderr)
    monkeypatch.setattr(cli, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(cli, "agentic_answer", fake_agentic_answer)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(queries))
    monkeypatch.setattr(cli, "display_ansewr", answers.append)

    cli.repl("index", DummyMetaStore(), agentic=True)

    assert answers == ["Answer for Question one", "Answer for Question two"]
    assert stderr.getvalue().splitlines() == [
        "[Agentic] Limits: Search 3, Answer 4",
        "[Agentic] Read-only Search 1/3 (initial): first selected query",
        "[Agentic] Read-only Search 1/3 returned 3 chunks",
        "[Agentic] Controller is reviewing evidence",
        "[Agentic] Selected read-only action: adaptive Search",
        "[Agentic] Read-only Search 2/3 (adaptive extend): controller-selected query",
        "[Agentic] Read-only Search 2/3 returned 1 chunks",
        "[Agentic] Controller is reviewing evidence",
        "[Agentic] Selected action: generate answer",
        "[Agentic] Generating answer 1/4",
        "[Agentic] Reviewing answer 1/4",
        "[Agentic] Answer review: revise",
        "[Agentic] Generating answer 2/4",
        "[Agentic] Reviewing answer 2/4",
        "[Agentic] Answer review: search",
        "[Agentic] Read-only Search 3/3 (adaptive replace): review-selected query",
        "[Agentic] Read-only Search 3/3 returned 2 chunks",
        "[Agentic] Controller is reviewing evidence",
        "[Agentic] Selected action: clarify",
        "[Agentic] Asking for clarification",
        "[Agentic] Outcome: needs_clarification",
        "[Agentic] Asking for clarification",
        "[Agentic] Outcome: needs_clarification",
    ]
    assert "PRIVATE NOTE TEXT" not in stderr.getvalue()


def test_eval_command_runs_benchmark_without_loading_user_index(monkeypatch):
    class DummySettings:
        def model_dump_json(self):
            return "{}"

    calls = []
    monkeypatch.setattr(sys, "argv", ["rag-app", "--eval"])
    monkeypatch.setattr(cli, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(
        cli,
        "run_evaluation",
        lambda **kwargs: calls.append(("benchmark", kwargs)),
    )
    monkeypatch.setattr(
        cli,
        "load_or_build_index",
        lambda: pytest.fail("--eval must use the fixed fixture corpus"),
    )

    cli.main()

    assert calls == [("benchmark", {
        "agentic": False,
        "sample": None,
        "seed": None,
    })]


def test_eval_command_forwards_sample_and_seed(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["rag-app", "--agentic", "--eval", "--sample", "4", "--seed", "27"],
    )
    monkeypatch.setattr(
        cli,
        "run_evaluation",
        lambda **kwargs: calls.append(kwargs),
    )

    cli.main()

    assert calls == [{"agentic": True, "sample": 4, "seed": 27}]


@pytest.mark.parametrize("sample", ["0", "-1"])
def test_eval_command_rejects_nonpositive_sample(monkeypatch, sample):
    monkeypatch.setattr(
        sys,
        "argv",
        ["rag-app", "--eval", "--sample", sample],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2


def test_classic_query_reports_retrieval_latency(monkeypatch):
    retrieval_samples = iter([10.0, 10.025])
    monkeypatch.setattr(cli, "perf_counter", lambda: next(retrieval_samples))
    monkeypatch.setattr(cli, "retrieve", lambda *_args, **_kwargs: [{"source": "note.md"}])
    monkeypatch.setattr(
        cli,
        "rag_answer",
        lambda *_args, **_kwargs: {"answer": "answer", "citations": []},
    )
    retrieval_events = []

    result = cli.run_query(
        object(),
        object(),
        query="question",
        on_retrieval=retrieval_events.append,
    )

    assert result["answer"] == "answer"
    assert retrieval_events == [{
        "search_index": 1,
        "latency_ms": 25.0,
        "result_count": 1,
    }]


def test_agentic_eval_command_selects_agentic_fixture_workflow(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["rag-app", "--agentic", "--eval"])
    monkeypatch.setattr(
        cli,
        "run_evaluation",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        cli,
        "load_or_build_index",
        lambda: pytest.fail("--eval must use the fixed fixture corpus"),
    )

    cli.main()

    assert calls == [{"agentic": True, "sample": None, "seed": None}]


def test_run_evaluation_forwards_selected_workflow(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli,
        "_load_evaluation_function",
        lambda name: lambda **kwargs: calls.append((name, kwargs)),
    )

    cli.run_evaluation(agentic=True)

    assert calls == [("run_evaluation", {
        "agentic": True,
        "sample": None,
        "seed": None,
    })]


def test_eval_command_explains_its_source_checkout_requirement(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["rag-app", "--eval"])

    def missing_evaluator(_module_name):
        raise ModuleNotFoundError("No module named 'rag_notes_helper.eval'")

    monkeypatch.setattr(cli.importlib, "import_module", missing_evaluator)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert "source checkout" in capsys.readouterr().err


def test_parser_rejects_removed_eval_base_flag():
    with pytest.raises(SystemExit) as exc_info:
        cli.build_parser().parse_args(["--eval-base"])

    assert exc_info.value.code == 2
