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
from pathlib import PurePosixPath, PureWindowsPath
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any, Iterator, Literal

import pandas as pd
from tqdm import tqdm

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.eval.dataset_builder import build_dataset
from rag_notes_helper.eval.ragas_runner import (
    GENERATION_METRIC_NAMES,
    LEGACY_METRIC_NAMES,
    METRIC_NAMES,
    RETRIEVAL_METRIC_NAMES,
)
from rag_notes_helper.eval.retrieval_metrics import calculate_retrieval_metrics
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
        [{"rows": rows, "metric_names": list(LEGACY_METRIC_NAMES)}],
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
    correctness_rows = [
        {
            "question": case["question"],
            "answer": case["answer"] or "",
            "reference_answer": case["reference_answer"],
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
            {"rows": correctness_rows, "metric_names": ["answer_correctness"]},
        ],
        usage_recorder=usage_recorder,
    )
    if any(len(scores) != len(scored_cases) for scores in score_sets):
        raise ValueError("Ragas returned a different number of scores than cases.")
    scored_results = []
    for batch_rows in zip(*score_sets):
        scores: dict[str, Any] = {}
        reasons: dict[str, Any] = {}
        latencies_ms: dict[str, Any] = {}
        for row_scores in batch_rows:
            scores.update({
                name: score
                for name, score in row_scores.items()
                if name not in {"metric_reasons", "metric_latencies_ms"}
            })
            reasons.update(row_scores.get("metric_reasons", {}))
            latencies_ms.update(row_scores.get("metric_latencies_ms", {}))
        scored_results.append({
            **scores,
            "metric_reasons": reasons,
            "metric_latencies_ms": latencies_ms,
        })
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
        if not isinstance(evidence, list):
            raise ValueError(f"QA case {case_id} has invalid expected_evidence.")
        fixture_root = NOTES_FIXTURE.resolve()
        for source in sources:
            _fixture_source_path(case_id, source, fixture_root)
        if record["answerable"] and not sources:
            raise ValueError(
                f"Answerable QA case {case_id} requires expected source annotations."
            )
        for annotation in evidence:
            if not isinstance(annotation, dict) or not {
                "source",
                "start_line",
                "end_line",
                "text",
            }.issubset(annotation):
                raise ValueError(
                    f"QA case {case_id} has invalid expected_evidence; each span "
                    "requires source, start_line, end_line, and text."
                )
            source = annotation["source"]
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"QA case {case_id} has an invalid evidence source path.")
            source_path = _fixture_source_path(case_id, source, fixture_root)
            if source not in sources:
                raise ValueError(
                    f"QA case {case_id} evidence source {source!r} is not listed "
                    "in expected_sources."
                )
            start_line = annotation["start_line"]
            end_line = annotation["end_line"]
            if (
                type(start_line) is not int
                or type(end_line) is not int
                or start_line < 1
                or end_line < start_line
            ):
                raise ValueError(
                    f"QA case {case_id} has an invalid evidence line range for "
                    f"{source!r}."
                )
            source_lines = source_path.read_text(encoding="utf-8").splitlines()
            if end_line > len(source_lines):
                raise ValueError(
                    f"QA case {case_id} evidence line range {start_line}-{end_line} "
                    f"is outside source {source!r} (1-{len(source_lines)})."
                )
            text = annotation["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"QA case {case_id} has empty evidence text.")
            actual_text = "\n".join(source_lines[start_line - 1 : end_line])
            if text != actual_text:
                raise ValueError(
                    f"QA case {case_id} evidence text does not match source "
                    f"{source!r} lines {start_line}-{end_line}."
                )

    missing_seed_cases = SEED_CASE_IDS - seen_ids
    if missing_seed_cases:
        raise ValueError(
            "Notes QA fixture must retain the original seed cases; missing ids: "
            + ", ".join(sorted(missing_seed_cases))
            + "."
        )
    return records


def _fixture_source_path(case_id: str, source: str, fixture_root: Path) -> Path:
    relative = PurePosixPath(source)
    windows_path = PureWindowsPath(source)
    if (
        relative.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in relative.parts
        or "\\" in source
    ):
        raise ValueError(f"QA case {case_id} has an unsafe source path {source!r}.")
    source_path = (fixture_root / Path(*relative.parts)).resolve()
    if not source_path.is_relative_to(fixture_root) or not source_path.is_file():
        raise ValueError(
            f"QA case {case_id} source path {source!r} is not a fixture file."
        )
    return source_path


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
    expected_evidence = [
        annotation["text"] for annotation in case["expected_evidence"]
    ]
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
    retrieval_calls: list[dict[str, Any]] = []
    workflow_error = None
    query_started = perf_counter()
    try:
        result = run_query(
            rag,
            meta_store,
            query=case["question"],
            agentic=workflow == "agentic",
            stream=False,
            on_generation=generation_calls.append,
            on_retrieval=retrieval_calls.append,
        )
        answer = result.get("answer")
        citations = result.get("citations", [])
        evidence = result.get("retrieved_evidence", [])
    except Exception as exc:
        answer = None
        citations = []
        evidence = []
        workflow_error = type(exc).__name__
    rag_query_latency_ms = round((perf_counter() - query_started) * 1000, 3)

    def evidence_record(hit: dict[str, Any]) -> dict[str, Any]:
        return {
            "source": hit["source"],
            "chunk_id": hit["chunk_id"],
            "start_line": hit.get("start_line"),
            "end_line": hit.get("end_line"),
            "score": _safe_score(hit.get("score")),
            "text": hit["text"],
        }

    evidence_records = [evidence_record(hit) for hit in evidence]
    raw_search_results = result.get("search_results") if workflow_error is None else None
    if not isinstance(raw_search_results, list):
        raw_search_results = [evidence]
    search_result_records = [
        [evidence_record(hit) for hit in search_hits]
        for search_hits in raw_search_results
    ]
    citation_records = [
        {
            "source": citation["source"],
            "chunk_id": citation["chunk_id"],
            "score": _safe_score(citation.get("score")),
            "start_line": citation.get("start_line"),
            "end_line": citation.get("end_line"),
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
    controller_latency_ms = round(sum(
        call.get("latency_ms") or 0
        for call in generation_calls
        if call.get("call_type") == "controller"
    ), 3)
    answer_latency_ms = round(sum(
        call.get("latency_ms") or 0
        for call in generation_calls
        if call.get("call_type") in {"answer", "final_answer"}
    ), 3)
    retrieval_latency_ms = round(sum(
        call.get("latency_ms") or 0 for call in retrieval_calls
    ), 3)
    retrieval_metrics = calculate_retrieval_metrics(
        expected_evidence=case["expected_evidence"],
        retrieval_lists=raw_search_results,
        top_k=get_settings().top_k,
    )
    record = {
        "case_id": case["id"],
        "category": case["category"],
        "workflow": workflow,
        "question": case["question"],
        "answer": answer,
        "citations": citation_records,
        "retrieved_evidence": evidence_records,
        "search_results": search_result_records,
        "contexts": [item["text"] for item in evidence_records],
        "answer_contexts": answer_contexts,
        "reference_answer": case["reference_answer"],
        "expected_sources": case["expected_sources"],
        "expected_evidence": case["expected_evidence"],
        **_answerability_fields(case, answer),
        **_retrieval_coverage(case, evidence_records),
        "retrieval_metrics": {
            **retrieval_metrics,
            **{name: None for name in RETRIEVAL_METRIC_NAMES},
        },
        "generation_metrics": {
            name: None for name in GENERATION_METRIC_NAMES
        },
        "metric_reasons": {},
        "generation_usage": _aggregate_usage(generation_calls),
        "judge_usage": _aggregate_usage([]),
        "judge_usage_by_metric": {},
        "judge_metrics": {},
        "latency": {
            "retrieval_calls": retrieval_calls,
            "retrieval_ms": retrieval_latency_ms,
            "controller_ms": controller_latency_ms,
            "answer_generation_ms": answer_latency_ms,
            "generation_ms": round(
                controller_latency_ms + answer_latency_ms,
                3,
            ),
            "judge_metrics_ms": {},
            "judge_ms": 0.0,
            "rag_query_ms": rag_query_latency_ms,
            "end_to_end_ms": rag_query_latency_ms,
        },
    }
    if workflow_error is not None:
        record["workflow_error"] = workflow_error
    return record


def _attach_ragas_scores(
    workflow_cases: list[dict[str, Any]],
    *,
    judge_calls: list[dict[str, Any]],
    settings: Any,
) -> None:
    scores = _run_ragas_isolated(
        workflow_cases,
        usage_recorder=judge_calls.append,
    )
    if len(scores) != len(workflow_cases):
        raise ValueError("Ragas returned a different number of scores than cases.")
    scored_case_indices = [
        index
        for index, case in enumerate(workflow_cases)
        if case["expected_answerable"]
    ]
    calls_by_row_metric: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for call in judge_calls:
        row_index = call.get("row_index")
        metric = call.get("metric")
        if isinstance(row_index, int) and isinstance(metric, str):
            calls_by_row_metric[(row_index, metric)].append(call)

    for scored_row_index, case_index in enumerate(scored_case_indices):
        case = workflow_cases[case_index]
        case_scores = scores[case_index]
        case["generation_metrics"] = {
            metric: _safe_score(case_scores.get(metric))
            for metric in GENERATION_METRIC_NAMES
        }
        case["retrieval_metrics"].update({
            metric: _safe_score(case_scores.get(metric))
            for metric in RETRIEVAL_METRIC_NAMES
        })
        case["metric_reasons"] = case_scores.get("metric_reasons", {})
        metric_latencies = case_scores.get("metric_latencies_ms", {})
        case_judge_calls = [
            _judge_usage_call(call, settings, case["case_id"])
            for (row_index, _metric), calls in calls_by_row_metric.items()
            if row_index == scored_row_index
            for call in calls
        ]
        by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for call in case_judge_calls:
            by_metric[call["metric"]].append(call)
        case["judge_usage"] = _aggregate_usage(case_judge_calls)
        case["judge_usage_by_metric"] = {
            metric: _aggregate_usage(calls)
            for metric, calls in sorted(by_metric.items())
        }
        case["judge_metrics"] = {
            metric: {
                "score": (
                    case["generation_metrics"].get(metric)
                    if metric in GENERATION_METRIC_NAMES
                    else case["retrieval_metrics"].get(metric)
                ),
                "reason": case["metric_reasons"].get(metric),
                "latency_ms": _safe_score(metric_latencies.get(metric)),
                "usage": case["judge_usage_by_metric"].get(
                    metric,
                    _aggregate_usage([]),
                ),
            }
            for metric in METRIC_NAMES
        }
        judge_metric_latencies = {
            metric: _safe_score(metric_latencies.get(metric))
            for metric in METRIC_NAMES
        }
        judge_latency_ms = round(sum(
            latency or 0 for latency in judge_metric_latencies.values()
        ), 3)
        case["latency"]["judge_metrics_ms"] = judge_metric_latencies
        case["latency"]["judge_ms"] = judge_latency_ms
        case["latency"]["end_to_end_ms"] = round(
            case["latency"]["rag_query_ms"] + judge_latency_ms,
            3,
        )


def _workflow_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_category[case["category"]].append(case)

    def summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "case_count": len(group),
            "ragas_case_count": sum(
                case["generation_metrics"].get("faithfulness") is not None
                for case in group
            ),
            "generation_metrics": {
                metric: _mean([
                    case["generation_metrics"].get(metric)
                    for case in group
                ])
                for metric in GENERATION_METRIC_NAMES
            },
            "retrieval_metrics": {
                metric: _mean([case["retrieval_metrics"].get(metric) for case in group])
                for metric in ("recall", "precision", "ndcg", *RETRIEVAL_METRIC_NAMES)
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
            "latency_ms": {
                name: _mean([
                    case["latency"].get(name)
                    for case in group
                ])
                for name in (
                    "retrieval_ms",
                    "controller_ms",
                    "answer_generation_ms",
                    "generation_ms",
                    "judge_ms",
                    "rag_query_ms",
                    "end_to_end_ms",
                )
            },
            "judge_usage_by_metric": {
                metric: _aggregate_usage([
                    call
                    for case in group
                    for call in case["judge_usage"].get("calls", [])
                    if call.get("metric") == metric
                ])
                for metric in METRIC_NAMES
            },
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


def _judge_usage_call(
    record: dict[str, Any],
    settings: Any,
    case_id: str | None = None,
) -> dict[str, Any]:
    return {
        "call_type": "judge",
        "metric": record.get("metric"),
        "row_index": record.get("row_index"),
        "case_id": case_id,
        "provider": settings.llm.effective_eval_provider,
        "model": settings.llm.effective_eval_model,
        "token_usage": record.get("token_usage", record),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_evaluation(
    *,
    output_root: Path | None = None,
    agentic: bool = False,
) -> Path:
    """Evaluate one selected workflow against the isolated fixture set."""
    workflow: Workflow = "agentic" if agentic else "classic"
    project_root = Path(__file__).resolve().parents[3]
    output_root = output_root or project_root / "eval"
    timestamp, run_dir = _new_run_directory(Path(output_root))
    run_log = run_dir / "run.log"
    run_log.touch()
    results_path = run_dir / "results.json"
    summary_path = run_dir / "summary.json"
    _log(run_log, f"Starting {workflow} synthetic-notes RAG evaluation.")

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
                "workflow": workflow,
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
            workflows: tuple[Workflow, ...] = (workflow,)
            progress_total = len(cases) + 1

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
                        _attach_ragas_scores(
                            workflow_cases,
                            judge_calls=judge_calls,
                            settings=settings,
                        )
                        _log(run_log, f"[{workflow}] Ragas metrics completed.")
                    except Exception as exc:
                        error_name = type(exc).__name__
                        for record in workflow_cases:
                            record["ragas_error"] = error_name
                        _log(run_log, f"[{workflow}] Ragas failed ({error_name}).")
                    finally:
                        progress.update(1)

                    answerable_cases = [
                        case
                        for case in workflow_cases
                        if case["expected_answerable"]
                    ]
                    judge_usage_by_workflow[workflow] = [
                        _judge_usage_call(
                            call,
                            settings,
                            answerable_cases[call["row_index"]]["case_id"]
                            if isinstance(call.get("row_index"), int)
                            and 0 <= call["row_index"] < len(answerable_cases)
                            else None,
                        )
                        for call in judge_calls
                    ]

            results["category_counts"] = dict(Counter(
                case["category"] for case in cases
            ))
            judge_usage = {
                "by_workflow": {
                    workflow: _aggregate_usage(calls)
                    for workflow, calls in judge_usage_by_workflow.items()
                },
                "by_metric": {
                    metric: _aggregate_usage([
                        call
                        for calls in judge_usage_by_workflow.values()
                        for call in calls
                        if call.get("metric") == metric
                    ])
                    for metric in METRIC_NAMES
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
