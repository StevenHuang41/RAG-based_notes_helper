from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Literal

import pandas as pd
from tqdm import tqdm

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.eval.dataset_builder import build_dataset
from rag_notes_helper.eval.ragas_runner import METRIC_NAMES
from rag_notes_helper.rag.answer import ABSTENTION_MESSAGE
from rag_notes_helper.rag.agentic import MAX_SEARCHES
from rag_notes_helper.rag.index import RagIndex, load_or_build_index
from rag_notes_helper.rag.meta_store import MetaStore
from rag_notes_helper.utils.logger import get_logger


Workflow = Literal["classic", "agentic"]
# Minimum reviewed seed cases per category; the fixture may grow beyond this baseline.
BASELINE_CATEGORY_COUNTS = {
    "direct": 4,
    "multi_hop": 4,
    "unanswerable": 4,
    "ambiguous": 4,
}
SEED_CASE_IDS = frozenset(
    {
        "direct-01",
        "direct-02",
        "direct-03",
        "direct-04",
        "multi-hop-01",
        "multi-hop-02",
        "multi-hop-03",
        "multi-hop-04",
        "unanswerable-01",
        "unanswerable-02",
        "unanswerable-03",
        "unanswerable-04",
        "ambiguous-01",
        "ambiguous-02",
        "ambiguous-03",
        "ambiguous-04",
    }
)
NOTES_FIXTURE = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "notes"
QA_FIXTURE = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "notes_qa.json"


def _run_ragas_rows_isolated(
    rows: list[dict[str, Any]],
    *,
    usage_recorder: Callable[[dict[str, Any]], None],
) -> list[dict[str, Any]]:
    return _run_ragas_batches_isolated(
        [{"rows": rows, "metric_names": list(METRIC_NAMES)}],
        usage_recorder=usage_recorder,
    )[0]


def _run_ragas_batches_isolated(
    batches: list[dict[str, Any]],
    *,
    usage_recorder: Callable[[dict[str, Any]], None],
) -> list[list[dict[str, Any]]]:
    completed = subprocess.run(
        [sys.executable, "-m", "rag_notes_helper.eval.ragas_worker"],
        input=json.dumps({"batches": batches}),
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        outcome = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ragas worker returned an invalid result.") from exc
    if completed.returncode != 0:
        raise RuntimeError("Ragas worker exited unexpectedly.")
    for call in outcome.get("judge_calls", []):
        usage_recorder(call)
    if "error" in outcome:
        raise RuntimeError(f"Ragas worker failed ({outcome['error']}).")
    return outcome["scores"]


def _run_ragas_isolated(
    cases: list[dict[str, Any]],
    *,
    usage_recorder: Callable[[dict[str, Any]], None],
) -> list[dict[str, Any]]:
    scored_cases = [case for case in cases if case["expected_answerable"]]
    if not scored_cases:
        return [{} for _case in cases]

    retrieval_rows = [
        {
            "question": case["question"],
            "answer": case["answer"] or "",
            "contexts": case["contexts"],
        }
        for case in scored_cases
    ]
    answer_rows = [
        {
            "question": case["question"],
            "answer": case["answer"] or "",
            "contexts": case["answer_contexts"],
        }
        for case in scored_cases
    ]
    retrieval_metrics = [
        "answer_relevancy",
        "context_utilization",
    ]
    score_sets = _run_ragas_batches_isolated(
        [
            {"rows": retrieval_rows, "metric_names": retrieval_metrics},
            {"rows": answer_rows, "metric_names": ["faithfulness"]},
        ],
        usage_recorder=usage_recorder,
    )
    if any(len(scores) != len(scored_cases) for scores in score_sets):
        raise ValueError("Ragas returned a different number of scores than cases.")
    scored_results = [
        {**retrieval_scores, **faithfulness_scores}
        for retrieval_scores, faithfulness_scores in zip(*score_sets)
    ]
    scored_iter = iter(scored_results)
    return [
        next(scored_iter) if case["expected_answerable"] else {}
        for case in cases
    ]


def load_qa_fixture(path: Path = QA_FIXTURE) -> list[dict[str, Any]]:
    """Load reviewed QA cases while preserving the original seed coverage."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or any(
        not isinstance(record, dict) for record in records
    ):
        raise ValueError("Notes QA fixture must be a JSON list of case objects.")

    required = {
        "id",
        "category",
        "question",
        "answerable",
        "reference_answer",
        "expected_sources",
    }
    for record in records:
        missing = required - record.keys()
        if missing:
            raise ValueError(
                f"QA case {record.get('id', '<unknown>')} is missing required fields."
            )

    for record in records:
        category = record["category"]
        if not isinstance(category, str) or category not in BASELINE_CATEGORY_COUNTS:
            raise ValueError(
                f"QA case {record.get('id', '<unknown>')} has unsupported category."
            )

    counts = Counter(record["category"] for record in records)
    missing_baseline = {
        category: minimum
        for category, minimum in BASELINE_CATEGORY_COUNTS.items()
        if counts[category] < minimum
    }
    if missing_baseline:
        raise ValueError(
            "Notes QA fixture must retain baseline coverage of four cases per "
            "category; missing baseline coverage for "
            + ", ".join(sorted(missing_baseline))
            + "."
        )

    seen_ids = set()
    for record in records:
        case_id = record["id"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Every QA case must have a non-empty string id.")
        if case_id in seen_ids:
            raise ValueError(
                f"QA case ids must be unique; found duplicate id {case_id!r}."
            )
        seen_ids.add(case_id)

        if not isinstance(record["question"], str) or not record["question"].strip():
            raise ValueError(f"QA case {case_id} must have a non-empty question.")
        if not isinstance(record["answerable"], bool):
            raise ValueError(f"QA case {case_id} must define answerable as a boolean.")
        if (
            not isinstance(record["reference_answer"], str)
            or not record["reference_answer"].strip()
        ):
            raise ValueError(f"QA case {case_id} must have a reference answer.")
        sources = record["expected_sources"]
        evidence = record.setdefault("expected_evidence", [])
        if not isinstance(sources, list) or any(
            not isinstance(source, str) or not source.strip() for source in sources
        ):
            raise ValueError(f"QA case {case_id} has invalid expected_sources.")
        if not isinstance(evidence, list) or any(
            not isinstance(snippet, str) or not snippet.strip()
            for snippet in evidence
        ):
            raise ValueError(f"QA case {case_id} has invalid expected_evidence.")
        if record["answerable"] and not sources:
            raise ValueError(
                f"Answerable QA case {case_id} requires expected source annotations."
            )

    missing_seed_cases = SEED_CASE_IDS - seen_ids
    if missing_seed_cases:
        raise ValueError(
            "Notes QA fixture must retain the original seed cases; missing ids: "
            + ", ".join(sorted(missing_seed_cases))
            + "."
        )
    return records


def _fixture_digest() -> str:
    digest = hashlib.sha256()
    fixture_files = sorted([*NOTES_FIXTURE.rglob("*"), QA_FIXTURE])
    for path in fixture_files:
        if not path.is_file():
            continue
        if path == QA_FIXTURE:
            relative = Path("notes_qa.json")
        else:
            relative = path.relative_to(NOTES_FIXTURE.parent)
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@contextmanager
def _isolated_fixture_index() -> Iterator[tuple[Any, Any, Any]]:
    """Build the real notes index under a temporary fixture-only runtime."""
    environment_keys = ("NOTES_DIR", "STORAGE_DIR")
    previous = {key: os.environ.get(key) for key in environment_keys}
    previous_embedding_model = RagIndex._model
    RagIndex._model = None
    try:
        with tempfile.TemporaryDirectory(prefix="notes-rag-eval-") as temp_dir:
            root = Path(temp_dir)
            notes_dir = root / "notes"
            storage_dir = root / "storage"
            shutil.copytree(NOTES_FIXTURE, notes_dir)
            storage_dir.mkdir()
            os.environ["NOTES_DIR"] = str(notes_dir)
            os.environ["STORAGE_DIR"] = str(storage_dir)
            get_settings.cache_clear()

            settings = get_settings()
            rag = load_or_build_index()
            meta_store = MetaStore(storage_dir=storage_dir)
            try:
                yield settings, rag, meta_store
            finally:
                meta_store.close()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()
        RagIndex._model = previous_embedding_model


def _new_run_directory(output_root: Path) -> tuple[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{timestamp}-{suffix}"
        suffix += 1
    run_dir.mkdir()
    return timestamp, run_dir


def _log(run_log: Path, message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with run_log.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{timestamp} {message}\n")
    get_logger("cli").info("[EVAL] %s", message)


def _safe_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _mean(values: list[Any]) -> float | None:
    clean_values = [score for value in values if (score := _safe_score(value)) is not None]
    if not clean_values:
        return None
    return sum(clean_values) / len(clean_values)


def _aggregate_usage_fields(
    calls: list[dict[str, Any]],
    section: str,
    fields: tuple[str, ...],
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    values = {
        field: [call.get(section, {}).get(field) for call in calls]
        for field in fields
    }
    totals = {
        field: (
            sum(value for value in field_values if value is not None)
            if any(value is not None for value in field_values)
            else None
        )
        for field, field_values in values.items()
    }
    return values, totals


def _aggregate_usage(calls: list[dict[str, Any]]) -> dict[str, Any]:
    token_fields = ("input_tokens", "output_tokens", "total_tokens")
    token_values, token_totals = _aggregate_usage_fields(
        calls,
        "token_usage",
        token_fields,
    )
    token_status = (
        "unavailable"
        if not any(value is not None for values in token_values.values() for value in values)
        else "available"
        if calls and all(
            value is not None
            for values in token_values.values()
            for value in values
        )
        else "partial"
    )
    character_fields = ("input", "output")
    _, character_totals = _aggregate_usage_fields(
        calls,
        "characters",
        character_fields,
    )
    return {
        "call_count": len(calls),
        "token_usage": {"status": token_status, **token_totals},
        "character_usage": character_totals,
        "calls": calls,
    }


def _answerability_fields(case: dict[str, Any], answer: Any) -> dict[str, Any]:
    expected_answerable = bool(case["answerable"])
    if answer is None:
        return {
            "expected_answerable": expected_answerable,
            "predicted_answerable": None,
            "abstained": None,
            "correct_abstention": None,
            "answerability_correct": None,
        }
    abstained = (
        isinstance(answer, str)
        and answer.strip().casefold() == ABSTENTION_MESSAGE.casefold()
    )
    predicted_answerable = not abstained
    return {
        "expected_answerable": expected_answerable,
        "predicted_answerable": predicted_answerable,
        "abstained": abstained,
        "correct_abstention": not expected_answerable and abstained,
        "answerability_correct": predicted_answerable == expected_answerable,
    }


def _retrieval_coverage(
    case: dict[str, Any], evidence: list[dict[str, Any]]
) -> dict[str, Any]:
    sources = {item["source"] for item in evidence}
    combined_text = "\n".join(item["text"] for item in evidence).casefold()
    expected_sources = case["expected_sources"]
    expected_evidence = case["expected_evidence"]
    return {
        "expected_source_coverage": (
            sum(source in sources for source in expected_sources) / len(expected_sources)
            if expected_sources
            else None
        ),
        "expected_evidence_coverage": (
            sum(snippet.casefold() in combined_text for snippet in expected_evidence)
            / len(expected_evidence)
            if expected_evidence
            else None
        ),
    }


def _record_case(
    case: dict[str, Any],
    *,
    workflow: Workflow,
    rag: Any,
    meta_store: Any,
) -> dict[str, Any]:
    from rag_notes_helper.cli import run_query

    generation_calls: list[dict[str, Any]] = []
    workflow_error = None
    try:
        result = run_query(
            rag,
            meta_store,
            query=case["question"],
            agentic=workflow == "agentic",
            stream=False,
            on_generation=generation_calls.append,
        )
        answer = result.get("answer")
        citations = result.get("citations", [])
        evidence = result.get("retrieved_evidence", [])
    except Exception as exc:
        answer = None
        citations = []
        evidence = []
        workflow_error = type(exc).__name__

    evidence_records = [
        {
            "source": hit["source"],
            "chunk_id": hit["chunk_id"],
            "score": _safe_score(hit.get("score")),
            "text": hit["text"],
        }
        for hit in evidence
    ]
    citation_records = [
        {
            "source": citation["source"],
            "chunk_id": citation["chunk_id"],
            "score": _safe_score(citation.get("score")),
        }
        for citation in citations
    ]
    citation_references = {
        (citation["source"], citation["chunk_id"])
        for citation in citation_records
    }
    answer_contexts = [
        item["text"]
        for item in evidence_records
        if (item["source"], item["chunk_id"]) in citation_references
    ]
    record = {
        "case_id": case["id"],
        "category": case["category"],
        "workflow": workflow,
        "question": case["question"],
        "answer": answer,
        "citations": citation_records,
        "retrieved_evidence": evidence_records,
        "contexts": [item["text"] for item in evidence_records],
        "answer_contexts": answer_contexts,
        "reference_answer": case["reference_answer"],
        "expected_sources": case["expected_sources"],
        "expected_evidence": case["expected_evidence"],
        **_answerability_fields(case, answer),
        **_retrieval_coverage(case, evidence_records),
        "metrics": {name: None for name in METRIC_NAMES},
        "generation_usage": _aggregate_usage(generation_calls),
    }
    if workflow_error is not None:
        record["workflow_error"] = workflow_error
    return record


def _attach_ragas_scores(
    workflow_cases: list[dict[str, Any]],
    *,
    judge_calls: list[dict[str, Any]],
) -> None:
    scores = _run_ragas_isolated(
        workflow_cases,
        usage_recorder=judge_calls.append,
    )
    if len(scores) != len(workflow_cases):
        raise ValueError("Ragas returned a different number of scores than cases.")
    for case, case_scores in zip(workflow_cases, scores):
        case["metrics"] = {
            metric: _safe_score(case_scores.get(metric))
            for metric in METRIC_NAMES
        }


def _workflow_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_category[case["category"]].append(case)

    def summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "case_count": len(group),
        "ragas_case_count": sum(
            case["metrics"].get("faithfulness") is not None
            for case in group
        ),
            "metrics": {
                metric: _mean([case["metrics"].get(metric) for case in group])
                for metric in METRIC_NAMES
            },
            "expected_source_coverage": _mean(
                [case["expected_source_coverage"] for case in group]
            ),
            "expected_evidence_coverage": _mean(
                [case["expected_evidence_coverage"] for case in group]
            ),
            "answerability_accuracy": _mean(
                [case["answerability_correct"] for case in group]
            ),
            "abstention_rate": _mean([case["abstained"] for case in group]),
            "correct_abstention_rate": _mean(
                [case["correct_abstention"] for case in group]
            ),
        }

    return {
        **summarize_group(cases),
        "by_category": {
            category: summarize_group(group)
            for category, group in sorted(by_category.items())
        },
        "generation_usage": _aggregate_usage(
            [
                call
                for case in cases
                for call in case["generation_usage"]["calls"]
            ]
        ),
    }


def _judge_usage_call(record: dict[str, Any], settings: Any) -> dict[str, Any]:
    return {
        "call_type": "ragas_judge",
        "provider": settings.llm.effective_eval_provider,
        "model": settings.llm.effective_eval_model,
        "token_usage": record,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_evaluation(*, output_root: Path | None = None) -> Path:
    """Compare Classic and Agentic against the repository's isolated fixture set."""
    project_root = Path(__file__).resolve().parents[3]
    output_root = output_root or project_root / "eval"
    timestamp, run_dir = _new_run_directory(Path(output_root))
    run_log = run_dir / "run.log"
    run_log.touch()
    results_path = run_dir / "results.json"
    summary_path = run_dir / "summary.json"
    _log(run_log, "Starting paired synthetic-notes RAG evaluation.")

    results: dict[str, Any] = {
        "schema_version": 1,
        "cases": [],
    }
    try:
        cases = load_qa_fixture()
        _log(run_log, f"Loaded {len(cases)} reviewed QA cases.")
        _log(run_log, "Building isolated synthetic-notes index.")
        with _isolated_fixture_index() as (settings, rag, meta_store):
            _log(run_log, "Synthetic-notes index is ready.")
            results["run_metadata"] = {
                "timestamp_utc": timestamp,
                "fixture_sha256": _fixture_digest(),
                "answer_provider": settings.llm.provider,
                "answer_model": settings.llm.model,
                "judge_provider": settings.llm.effective_eval_provider,
                "judge_model": settings.llm.effective_eval_model,
                "embedding_model": settings.embed_model_name,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "top_k": settings.top_k,
                "min_retrieval_score": settings.min_retrieval_score,
                "max_answer_chunks": settings.llm.max_chunks,
                "agentic_max_searches": MAX_SEARCHES,
            }
            workflow_results: dict[Workflow, list[dict[str, Any]]] = {}
            judge_usage_by_workflow: dict[Workflow, list[dict[str, Any]]] = {}
            workflows: tuple[Workflow, ...] = ("classic", "agentic")
            progress_total = len(cases) * len(workflows) + len(workflows)

            with tqdm(
                total=progress_total,
                desc="RAG evaluation",
                unit="step",
                file=sys.stderr,
                disable=not sys.stderr.isatty(),
                dynamic_ncols=True,
            ) as progress:
                for workflow in workflows:
                    workflow_cases = []
                    progress.set_description(f"RAG evaluation ({workflow})")
                    for case_number, case in enumerate(cases, start=1):
                        progress.set_postfix_str(
                            f"{case_number}/{len(cases)} {case['id']}"
                        )
                        _log(
                            run_log,
                            f"[{workflow}] Starting case "
                            f"{case_number}/{len(cases)} ({case['id']}).",
                        )
                        record = _record_case(
                            case,
                            workflow=workflow,
                            rag=rag,
                            meta_store=meta_store,
                        )
                        workflow_cases.append(record)
                        results["cases"].append(record)
                        completion = (
                            f"[{workflow}] Completed case {case['id']} "
                            f"answerable={record['predicted_answerable']} "
                            f"abstained={record['abstained']} "
                            f"calls={record['generation_usage']['call_count']}"
                        )
                        if "workflow_error" in record:
                            completion += f" error={record['workflow_error']}"
                        _log(run_log, completion)
                        progress.update(1)
                    workflow_results[workflow] = workflow_cases

                for workflow, workflow_cases in workflow_results.items():
                    judge_calls: list[dict[str, Any]] = []
                    answerable_count = sum(
                        case["expected_answerable"] for case in workflow_cases
                    )
                    progress.set_description(f"RAG evaluation ({workflow} metrics)")
                    progress.set_postfix_str("Ragas metrics")
                    _log(
                        run_log,
                        f"[{workflow}] Starting Ragas metrics for "
                        f"{answerable_count} answerable cases.",
                    )
                    try:
                        _attach_ragas_scores(workflow_cases, judge_calls=judge_calls)
                        _log(run_log, f"[{workflow}] Ragas metrics completed.")
                    except Exception as exc:
                        error_name = type(exc).__name__
                        for record in workflow_cases:
                            record["ragas_error"] = error_name
                        _log(run_log, f"[{workflow}] Ragas failed ({error_name}).")
                    finally:
                        progress.update(1)

                    judge_usage_by_workflow[workflow] = [
                        _judge_usage_call(call, settings) for call in judge_calls
                    ]

            results["category_counts"] = dict(Counter(
                case["category"] for case in cases
            ))
            judge_usage = {
                "by_workflow": {
                    workflow: _aggregate_usage(calls)
                    for workflow, calls in judge_usage_by_workflow.items()
                },
                "total": _aggregate_usage([
                    call
                    for calls in judge_usage_by_workflow.values()
                    for call in calls
                ]),
            }
            results["judge_usage"] = judge_usage
            summary = {
                "schema_version": 1,
                "run_metadata": results["run_metadata"],
                "case_count": len(cases),
                "category_counts": results["category_counts"],
                "interpretation": (
                    "This fixture is a regression seed, not a statistically "
                    "representative benchmark. Its original 16 cases cover direct, "
                    "multi-hop, unanswerable, and ambiguous questions. Ragas metrics "
                    "apply only to expected-answerable cases; inspect abstention "
                    "cases through answerability and evidence coverage."
                ),
                "workflows": {
                    workflow: _workflow_summary(records)
                    for workflow, records in workflow_results.items()
                },
                "judge_usage": judge_usage,
            }
            _write_json(results_path, results)
            _write_json(summary_path, summary)
            _log(run_log, "Evaluation artifacts written successfully.")
    except KeyboardInterrupt:
        _log(run_log, "Evaluation interrupted by user.")
        raise
    except Exception as exc:
        _log(run_log, f"Evaluation failed ({type(exc).__name__}).")
        if results.get("run_metadata") is not None:
            _write_json(results_path, results)
        _write_json(
            summary_path,
            {
                "schema_version": 1,
                "timestamp_utc": timestamp,
                "error": type(exc).__name__,
            },
        )
        raise

    print(f"Evaluation artifacts saved to {run_dir}")
    return run_dir


def run_base_smoke_evaluation() -> None:
    """Run the legacy application-behavior smoke set separately from notes QA."""
    rag = load_or_build_index()
    meta_store = MetaStore()
    try:
        dataset = build_dataset(
            rag,
            meta_store,
            str(Path(__file__).resolve().parent / "testset" / "base.json"),
        )
        print("\nLegacy system smoke dataset built. Running Ragas evaluation...\n")
        scores = _run_ragas_rows_isolated(
            dataset.to_list(),
            usage_recorder=lambda _call: None,
        )
        df = pd.DataFrame(scores)
        print("\n=== LEGACY SYSTEM RAGAS RESULT ===\n")
        for column in df.columns:
            print(f"{column}: {df[column].mean()}")
    finally:
        meta_store.close()
