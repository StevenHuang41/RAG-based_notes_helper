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
import random
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any, Iterator, Literal
from zoneinfo import ZoneInfo

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
from rag_notes_helper.eval.report import write_readme
from rag_notes_helper.rag.answer import ABSTENTION_MESSAGE
from rag_notes_helper.rag.agentic import MAX_SEARCHES
from rag_notes_helper.rag.index import (
    RagIndex,
    build_and_save_rag,
    load_or_build_index,
)
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
    scored_cases = [
        (case_index, case)
        for case_index, case in enumerate(cases)
        if case["expected_answerable"]
    ]
    if not scored_cases:
        return [{} for _case in cases]

    scored_results = [
        {"metric_reasons": {}, "metric_latencies_ms": {}}
        for _case_index, _case in scored_cases
    ]
    scored_position_by_case_index = {
        case_index: scored_position
        for scored_position, (case_index, _case) in enumerate(scored_cases)
    }
    metric_specs = (
        (
            "answer_relevancy",
            [item for item in scored_cases if item[1]["answer"]],
            lambda case: {
                "question": case["question"],
                "answer": case["answer"] or "",
            },
            None,
        ),
        (
            "context_utilization",
            [item for item in scored_cases if item[1]["answer"] and item[1]["contexts"]],
            lambda case: {
                "question": case["question"],
                "answer": case["answer"] or "",
                "contexts": case["contexts"],
            },
            "no retrieved context was available",
        ),
        (
            "faithfulness",
            [item for item in scored_cases if item[1]["answer"] and item[1]["answer_contexts"]],
            lambda case: {
                "question": case["question"],
                "answer": case["answer"] or "",
                "contexts": case["answer_contexts"],
            },
            "no answer context was available",
        ),
        (
            "answer_correctness",
            [item for item in scored_cases if item[1]["answer"]],
            lambda case: {
                "question": case["question"],
                "answer": case["answer"] or "",
                "reference_answer": case["reference_answer"],
            },
            None,
        ),
    )

    for metric_name, eligible_cases, row_factory, skip_reason in metric_specs:
        eligible_positions = [
            scored_position_by_case_index[case_index]
            for case_index, _case in eligible_cases
        ]
        for scored_position, (_case_index, case) in enumerate(scored_cases):
            if not case["answer"]:
                scored_results[scored_position]["metric_reasons"][metric_name] = (
                    "Skipped because no generated answer was available."
                )
            elif skip_reason and not case[
                "contexts" if metric_name == "context_utilization" else "answer_contexts"
            ]:
                scored_results[scored_position]["metric_reasons"][metric_name] = (
                    f"Skipped because {skip_reason}."
                )
        if not eligible_cases:
            continue

        get_logger("cli").info(
            f"[EVAL] Starting Ragas {metric_name} for {len(eligible_cases)} cases."
        )

        def record_metric_usage(call: dict[str, Any]) -> None:
            row_index = call.get("row_index")
            if isinstance(row_index, int) and 0 <= row_index < len(eligible_positions):
                usage_recorder({
                    **call,
                    "row_index": eligible_positions[row_index],
                })

        score_rows = _run_ragas_batches_isolated(
            [{
                "rows": [row_factory(case) for _case_index, case in eligible_cases],
                "metric_names": [metric_name],
            }],
            usage_recorder=record_metric_usage,
        )[0]
        if len(score_rows) != len(eligible_cases):
            raise ValueError(
                f"Ragas returned a different number of {metric_name} scores than cases."
            )
        get_logger("cli").info(f"[EVAL] Ragas {metric_name} completed.")

        for scored_position, row_scores in zip(eligible_positions, score_rows):
            scored_results[scored_position].update({
                name: score
                for name, score in row_scores.items()
                if name not in {"metric_reasons", "metric_latencies_ms"}
            })
            scored_results[scored_position]["metric_reasons"].update(
                row_scores.get("metric_reasons", {})
            )
            scored_results[scored_position]["metric_latencies_ms"].update(
                row_scores.get("metric_latencies_ms", {})
            )

    scored_by_case_index = {
        case_index: scored_results[scored_position]
        for scored_position, (case_index, _case) in enumerate(scored_cases)
    }
    return [
        scored_by_case_index.get(case_index, {})
        for case_index, case in enumerate(cases)
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
        challenge_tags = record.setdefault("challenge_tags", [])
        if (
            not isinstance(challenge_tags, list)
            or any(not isinstance(tag, str) or not tag.strip() for tag in challenge_tags)
            or len(set(challenge_tags)) != len(challenge_tags)
        ):
            raise ValueError(
                f"QA case {case_id} must define challenge tags as unique, "
                "non-empty strings."
            )
        if record["category"] == "ambiguous":
            reason = record.get("ambiguity_reason")
            if record["answerable"] or not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    f"Ambiguous QA case {case_id} must be unanswerable and include "
                    "a non-empty ambiguity_reason."
                )
        if record["category"] == "unanswerable":
            missing_information = record.get("missing_information")
            if (
                record["answerable"]
                or not isinstance(missing_information, str)
                or not missing_information.strip()
            ):
                raise ValueError(
                    f"Unanswerable QA case {case_id} must include a non-empty "
                    "missing_information description."
                )
        if (
            not isinstance(record["reference_answer"], str)
            or not record["reference_answer"].strip()
        ):
            raise ValueError(f"QA case {case_id} must have a reference answer.")
        sources = record["expected_sources"]
        evidence = record.get("expected_evidence")
        if evidence is None:
            if record["answerable"]:
                raise ValueError(
                    f"QA case {record['id']} requires expected_evidence spans."
                )
            evidence = []
            record["expected_evidence"] = evidence
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
        if record["answerable"] and not evidence:
            raise ValueError(
                f"QA case {case_id} requires expected_evidence spans."
            )
        if record["category"] == "ambiguous" and (not sources or not evidence):
            raise ValueError(
                f"Ambiguous QA case {case_id} requires evidence for its ambiguity."
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
            _terminal_progress("Building isolated synthetic fixture index")
            rag = build_and_save_rag(
                success_message="Fixture index built and saved."
            )
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
    timezone_name = "Asia/Taipei"
    timestamp = datetime.now(ZoneInfo(timezone_name)).strftime("%Y-%m-%d-%H%M%S")
    run_dir = output_root / timestamp
    run_dir.mkdir()
    return timestamp, run_dir


def _log(run_log: Path, message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with run_log.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{timestamp} {message}\n")
    get_logger("cli").info("[EVAL] %s", message)


def _terminal_progress(message: str) -> None:
    if sys.stderr.isatty():
        print(f"[Evaluation] {message}", file=sys.stderr, flush=True)


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
    if not isinstance(answer, str) or not answer.strip():
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


def _answerability_group_summary(
    cases: list[dict[str, Any]],
    *,
    expected_answerable: bool,
) -> dict[str, Any]:
    group = [
        case
        for case in cases
        if case.get("expected_answerable") is expected_answerable
    ]
    if not group:
        return {
            "case_count": 0,
            "answered_count": None,
            "abstained_count": None,
            "unavailable_count": None,
            "error_count": None,
            "unavailable_or_error_count": None,
            "correct_behavior_count": None,
        }

    error_count = sum(bool(case.get("workflow_error")) for case in group)
    unavailable_count = sum(
        not case.get("workflow_error")
        and not case.get("answer")
        and case.get("abstained") is not True
        for case in group
    )
    answered_count = sum(
        not case.get("workflow_error")
        and bool(case.get("answer"))
        and case.get("abstained") is not True
        for case in group
    )
    abstained_count = sum(
        not case.get("workflow_error") and case.get("abstained") is True
        for case in group
    )
    correct_behavior_count = sum(
        not case.get("workflow_error")
        and bool(case.get("answer"))
        and case.get("answerability_correct") is True
        for case in group
    )
    return {
        "case_count": len(group),
        "answered_count": answered_count,
        "abstained_count": abstained_count,
        "unavailable_count": unavailable_count,
        "error_count": error_count,
        "unavailable_or_error_count": unavailable_count + error_count,
        "correct_behavior_count": correct_behavior_count,
    }


def _answerability_outcomes(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "expected_answerable": _answerability_group_summary(
            cases,
            expected_answerable=True,
        ),
        "expected_unanswerable": _answerability_group_summary(
            cases,
            expected_answerable=False,
        ),
    }


def _ragas_metric_is_eligible(case: dict[str, Any], metric: str) -> bool:
    if case.get("expected_answerable") is not True or not case.get("answer"):
        return False
    required_context = {
        "faithfulness": "answer_contexts",
        "context_utilization": "contexts",
    }.get(metric)
    return required_context is None or bool(case.get(required_context))


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
        record = {
            "source": hit["source"],
            "chunk_id": hit["chunk_id"],
            "start_line": hit.get("start_line"),
            "end_line": hit.get("end_line"),
            "score": _safe_score(hit.get("score")),
            "text": hit["text"],
        }
        for key in (
            "stage",
            "retrieval_stage",
            "method",
            "retrieval_method",
        ):
            if key in hit:
                record[key] = hit[key]
        return record

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
    retrieved_by_reference = {
        (hit["source"], hit["chunk_id"]): hit
        for search_hits in search_result_records
        for hit in search_hits
    }
    raw_generation_context = (
        result.get("generation_context", []) if workflow_error is None else []
    )
    if not isinstance(raw_generation_context, list):
        raw_generation_context = []
    generation_context_records = []
    for context_item in raw_generation_context:
        if not isinstance(context_item, dict):
            continue
        source = context_item.get("source")
        chunk_id = context_item.get("chunk_id")
        retrieved_item = retrieved_by_reference.get((source, chunk_id), {})
        context_text = context_item.get("text", retrieved_item.get("text"))
        content_changed = context_text != retrieved_item.get("text")
        generation_context_records.append({
            "source": source,
            "chunk_id": chunk_id,
            "start_line": context_item.get(
                "start_line", retrieved_item.get("start_line")
            ),
            "end_line": context_item.get(
                "end_line", retrieved_item.get("end_line")
            ),
            "content_changed": content_changed,
            "text": context_text if content_changed else None,
            "prompt_block": (
                context_item.get("prompt_block") if content_changed else None
            ),
        })
    if not generation_context_records:
        for citation in citation_records:
            retrieved_item = retrieved_by_reference.get(
                (citation["source"], citation["chunk_id"]),
                {},
            )
            if retrieved_item:
                text = retrieved_item.get("text")
                generation_context_records.append({
                    "source": citation["source"],
                    "chunk_id": citation["chunk_id"],
                    "start_line": retrieved_item.get("start_line"),
                    "end_line": retrieved_item.get("end_line"),
                    "text": text,
                    "prompt_block": (
                        f"file:{citation['source']}\n{text}"
                        if isinstance(text, str)
                        else None
                    ),
                })
    answer_contexts = []
    for item in generation_context_records:
        context_text = item.get("text")
        if context_text is None:
            context_text = retrieved_by_reference.get(
                (item.get("source"), item.get("chunk_id")), {}
            ).get("text")
        if isinstance(context_text, str):
            answer_contexts.append(context_text)
    search_queries = (
        result.get("search_queries", []) if workflow_error is None else []
    )
    if not isinstance(search_queries, list):
        search_queries = []
    if not search_queries and search_result_records:
        search_queries = [case["question"]] * len(search_result_records)
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
        "challenge_tags": case["challenge_tags"],
        "workflow": workflow,
        "question": case["question"],
        "answer": answer,
        "citations": citation_records,
        "retrieved_evidence": evidence_records,
        "search_results": search_result_records,
        "search_queries": search_queries,
        "generation_context": generation_context_records,
        "contexts": [item["text"] for item in evidence_records],
        "answer_contexts": answer_contexts,
        "reference_answer": case["reference_answer"],
        "ambiguity_reason": case.get("ambiguity_reason"),
        "missing_information": case.get("missing_information"),
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
    scored_position_by_case_index = {
        case_index: scored_position
        for scored_position, case_index in enumerate(scored_case_indices)
    }
    calls_by_row_metric: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for call in judge_calls:
        row_index = call.get("row_index")
        metric = call.get("metric")
        if isinstance(row_index, int) and isinstance(metric, str):
            calls_by_row_metric[(row_index, metric)].append(call)

    for case_index, case in enumerate(workflow_cases):
        scored_row_index = scored_position_by_case_index.get(case_index)
        case_scores = scores[case_index]
        case["generation_metrics"] = {
            metric: _safe_score(case_scores.get(metric))
            for metric in GENERATION_METRIC_NAMES
        }
        case["retrieval_metrics"].update({
            metric: _safe_score(case_scores.get(metric))
            for metric in RETRIEVAL_METRIC_NAMES
        })
        metric_reasons = case_scores.get("metric_reasons", {}).copy()
        if not case["expected_answerable"]:
            metric_reasons = {
                metric: "Skipped because expected answerability is unanswerable."
                for metric in METRIC_NAMES
            }
        case["metric_reasons"] = metric_reasons
        metric_latencies = case_scores.get("metric_latencies_ms", {})
        case_judge_calls = [
            _judge_usage_call(call, settings, case["case_id"])
            for (row_index, _metric), calls in calls_by_row_metric.items()
            if scored_row_index is not None and row_index == scored_row_index
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
                "reason": case["metric_reasons"].get(metric)
                or (
                    "Scored."
                    if (
                        case["generation_metrics"].get(metric)
                        if metric in GENERATION_METRIC_NAMES
                        else case["retrieval_metrics"].get(metric)
                    ) is not None
                    else "Not scored."
                ),
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
        generation_calls = [
            call
            for case in group
            for call in case["generation_usage"]["calls"]
        ]
        judge_calls = [
            call
            for case in group
            for call in case["judge_usage"].get("calls", [])
        ]

        return {
            "case_count": len(group),
            "ragas_case_count": sum(
                case["generation_metrics"].get("faithfulness") is not None
                for case in group
            ),
            "ragas_metric_eligible_case_counts": {
                metric: sum(
                    _ragas_metric_is_eligible(case, metric)
                    for case in group
                )
                for metric in METRIC_NAMES
            },
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
            "answerability_outcomes": _answerability_outcomes(group),
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
            "generation_usage": _aggregate_usage(generation_calls),
            "judge_usage": _aggregate_usage(judge_calls),
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
    sample: int | None = None,
    seed: int | None = None,
) -> Path:
    """Evaluate one selected workflow against the isolated fixture set."""
    workflow: Workflow = "agentic" if agentic else "classic"
    cases = load_qa_fixture()
    if sample is not None:
        if sample <= 0:
            raise ValueError("--sample must be a positive integer")
        if sample > len(cases):
            raise ValueError(
                f"--sample cannot exceed the fixture size ({len(cases)} cases)"
            )
        sampled_cases = random.Random(seed).sample(cases, sample)
        sampled_ids = {case["id"] for case in sampled_cases}
        cases = [case for case in cases if case["id"] in sampled_ids]

    project_root = Path(__file__).resolve().parents[3]
    output_root = output_root or project_root / "eval"
    timestamp, run_dir = _new_run_directory(Path(output_root))
    run_log = run_dir / "run.log"
    run_log.touch()
    results_path = run_dir / "results.json"
    readme_path = run_dir / "README.md"
    _log(run_log, f"Starting {workflow} synthetic-notes RAG evaluation.")

    results: dict[str, Any] = {
        "schema_version": 4,
        "run_metadata": {
            "timestamp_local": timestamp,
            "timezone": "Asia/Taipei",
            "workflow": workflow,
            "sample_count": sample,
            "seed": seed,
            "selected_case_ids": [case["id"] for case in cases],
        },
        "cases": [],
    }
    _terminal_progress(f"Starting {workflow} fixture evaluation")
    try:
        _log(run_log, f"Loaded {len(cases)} reviewed QA cases.")
        _log(run_log, "Building isolated synthetic-notes index.")
        with _isolated_fixture_index() as (settings, rag, meta_store):
            _log(run_log, "Synthetic-notes index is ready.")
            results["run_metadata"] = {
                "timestamp_local": timestamp,
                "timezone": "Asia/Taipei",
                "workflow": workflow,
                "sample_count": sample,
                "seed": seed,
                "selected_case_ids": [case["id"] for case in cases],
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
            results["summary"] = {
                "schema_version": 1,
                "run_metadata": results["run_metadata"],
                "case_count": len(cases),
                "category_counts": results["category_counts"],
                "interpretation": (
                    "This fixture is a regression set, not a statistically "
                    "representative benchmark. A sampled run may omit categories; "
                    "selected case IDs and category counts identify the coverage. "
                    "Ragas metrics apply only to eligible expected-answerable cases "
                    "that produced an answer; per-metric eligible counts are recorded. "
                    "Inspect abstention cases through grouped answerability outcomes "
                    "and reviewed evidence."
                ),
                "workflows": {
                    workflow: _workflow_summary(records)
                    for workflow, records in workflow_results.items()
                },
                "judge_usage": judge_usage,
            }
            _write_json(results_path, results)
            write_readme(readme_path, results)
            _log(run_log, "Evaluation artifacts written successfully.")
    except KeyboardInterrupt:
        _log(run_log, "Evaluation interrupted by user.")
        raise
    except Exception as exc:
        _log(run_log, f"Evaluation failed ({type(exc).__name__}).")
        results["error"] = type(exc).__name__
        _write_json(results_path, results)
        write_readme(readme_path, results)
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
