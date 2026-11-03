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

from tqdm import tqdm

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.eval.answerability import has_nonblank_answer
from rag_notes_helper.eval.ragas_runner import (
    GENERATION_METRIC_NAMES,
    METRIC_NAMES,
    RETRIEVAL_METRIC_NAMES,
)
from rag_notes_helper.eval.retrieval_metrics import (
    calculate_context_prefix_metrics,
    calculate_retrieval_metrics,
)
from rag_notes_helper.eval.report import write_readme
from rag_notes_helper.rag.answer import ABSTENTION_MESSAGE
from rag_notes_helper.rag.agentic import MAX_ANSWER_ATTEMPTS, MAX_SEARCHES
from rag_notes_helper.rag.index import (
    RagIndex,
    build_and_save_rag_from_isolated_corpus,
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
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_ROOT = PROJECT_ROOT / "evals"
NOTES_FIXTURE = EVALUATION_ROOT / "notes"
QA_FIXTURE = EVALUATION_ROOT / "qa" / "notes_qa.json"


def _require_evaluation_assets() -> None:
    """Fail clearly when the checked-out evaluation benchmark is incomplete."""
    if not NOTES_FIXTURE.is_dir():
        raise FileNotFoundError(
            f"Evaluation corpus directory is missing: {NOTES_FIXTURE}"
        )
    if not QA_FIXTURE.is_file():
        raise FileNotFoundError(
            f"Evaluation ground truth QA set is missing: {QA_FIXTURE}"
        )


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
    scored_cases = list(enumerate(cases))
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
            lambda case: {
                "question": case["question"],
                "answer": case["answer"],
            },
            None,
        ),
        (
            "context_utilization",
            lambda case: {
                "question": case["question"],
                "answer": case["answer"],
                "contexts": case["answer_contexts"],
            },
            "no answer context was available",
        ),
        (
            "faithfulness",
            lambda case: {
                "question": case["question"],
                "answer": case["answer"],
                "contexts": case["answer_contexts"],
            },
            "no answer context was available",
        ),
        (
            "answer_correctness",
            lambda case: {
                "question": case["question"],
                "answer": case["answer"],
                "reference_answer": case["reference_answer"],
            },
            None,
        ),
    )

    for metric_name, row_factory, skip_reason in metric_specs:
        eligible_cases = [
            item
            for item in scored_cases
            if _ragas_metric_is_eligible(item[1], metric_name)
        ]
        eligible_positions = [
            scored_position_by_case_index[case_index]
            for case_index, _case in eligible_cases
        ]
        for scored_position, (_case_index, case) in enumerate(scored_cases):
            if not has_nonblank_answer(case.get("answer")):
                scored_results[scored_position]["metric_reasons"][metric_name] = (
                    "Skipped because no generated answer was available."
                )
            elif skip_reason and not case.get("answer_contexts"):
                scored_results[scored_position]["metric_reasons"][metric_name] = (
                    "Unavailable because no final generation context was available."
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


def load_qa_fixture(path: Path | None = None) -> list[dict[str, Any]]:
    """Load reviewed QA cases while preserving the original seed coverage."""
    path = QA_FIXTURE if path is None else Path(path)
    if path == QA_FIXTURE:
        _require_evaluation_assets()
    records = json.loads(path.read_text(encoding="utf-8"))
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
            rag = build_and_save_rag_from_isolated_corpus(
                notes_dir,
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


def _measured_latency_total(values: list[Any]) -> float | None:
    measured = [_safe_score(value) for value in values]
    if not measured or any(value is None for value in measured):
        return None
    return round(sum(value for value in measured if value is not None), 3)


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


def _answerability_fields(
    case: dict[str, Any],
    answer: Any,
    *,
    forced_abstention: bool = False,
) -> dict[str, Any]:
    expected_answerable = bool(case["answerable"])
    if not has_nonblank_answer(answer):
        return {
            "expected_answerable": expected_answerable,
            "predicted_answerable": None,
            "abstained": None,
            "correct_abstention": None,
            "answerability_correct": None,
        }
    abstained = forced_abstention or (
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
        and not has_nonblank_answer(case.get("answer"))
        and case.get("abstained") is not True
        for case in group
    )
    answered_count = sum(
        not case.get("workflow_error")
        and has_nonblank_answer(case.get("answer"))
        and case.get("abstained") is not True
        for case in group
    )
    abstained_count = sum(
        not case.get("workflow_error") and case.get("abstained") is True
        for case in group
    )
    correct_behavior_count = sum(
        not case.get("workflow_error")
        and has_nonblank_answer(case.get("answer"))
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
    if not has_nonblank_answer(case.get("answer")):
        return False
    required_context = {
        "faithfulness": "answer_contexts",
        "context_utilization": "answer_contexts",
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


_RETRIEVAL_STAGE_NAMES = ("vector", "bm25", "rrf", "reranked")
_STAGE_RANK_FIELDS = {
    "vector": "vector_rank",
    "bm25": "bm25_rank",
    "rrf": "rank",
    "reranked": "rerank_rank",
}
_STAGE_SCORE_FIELDS = {
    "vector": "vector_score",
    "bm25": "bm25_score",
    "rrf": "fusion_score",
    "reranked": "reranker_score",
}


def _stage_candidate_id(hit: dict[str, Any]) -> tuple[Any, Any] | None:
    source = hit.get("source")
    chunk_id = hit.get("chunk_id")
    if source is None or chunk_id is None:
        return None
    try:
        hash((source, chunk_id))
    except TypeError:
        return None
    return source, chunk_id


def _stage_result_record(
    hit: dict[str, Any],
    *,
    stage: str,
    rank: int,
) -> dict[str, Any]:
    """Keep stage rank and score explicit while retaining candidate provenance."""
    score_field = _STAGE_SCORE_FIELDS[stage]
    score_value = hit.get(score_field)
    if score_field not in hit and stage != "reranked":
        score_value = hit.get("score")
    score = _safe_score(score_value)
    record = {
        "source": hit.get("source"),
        "chunk_id": hit.get("chunk_id"),
        "start_line": hit.get("start_line"),
        "end_line": hit.get("end_line"),
        "rank": rank,
        "score": score,
        "stage_score": score,
        "text": hit.get("text"),
    }
    if stage == "reranked":
        record["fusion_rank"] = hit.get("rank")
    for key in (
        "vector_rank",
        "bm25_rank",
        "rerank_rank",
    ):
        value = hit.get(key)
        if type(value) is int:
            record[key] = value
    for key in (
        "vector_score",
        "bm25_score",
        "fusion_score",
        "reranker_score",
    ):
        if key in hit:
            record[key] = _safe_score(hit.get(key))
    if isinstance(hit.get("search_provenance"), list):
        record["search_provenance"] = hit["search_provenance"]
    return record


def _stage_metrics(
    *,
    expected_evidence: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    top_k: int | None,
) -> dict[str, float | None]:
    if top_k is None or top_k < 0:
        return {"recall": None, "precision": None, "ndcg": None}
    return calculate_retrieval_metrics(
        expected_evidence=expected_evidence,
        retrieval_lists=[candidates],
        top_k=top_k,
    )


def _mean_stage_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, float | None]:
    return {
        metric: _mean([row.get(metric) for row in rows])
        for metric in ("recall", "precision", "ndcg")
    }


def _record_retrieval_stages(
    raw_records: Any,
    *,
    expected_evidence: list[dict[str, Any]],
    settings: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Normalize measured retrieval stages and score each at its cutoff."""
    if not isinstance(raw_records, list):
        raw_records = []
    configured_cutoffs = {
        "vector": getattr(settings, "vector_top_k", None),
        "bm25": getattr(settings, "bm25_top_k", None),
        "rrf": getattr(settings, "fusion_top_k", getattr(settings, "top_k", None)),
        "reranked": getattr(settings, "fusion_top_k", getattr(settings, "top_k", None)),
    }
    normalized_records: list[dict[str, Any]] = []
    metrics_by_stage: dict[str, list[dict[str, Any]]] = {
        name: [] for name in _RETRIEVAL_STAGE_NAMES
    }
    comparisons: list[dict[str, Any]] = []

    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            continue
        generation_top_k = raw_record.get("generation_top_k")
        if type(generation_top_k) is not int or generation_top_k < 0:
            generation_top_k = getattr(
                getattr(settings, "llm", None), "max_chunks", None
            )
        raw_stages = raw_record.get("stages")
        if not isinstance(raw_stages, dict):
            raw_stages = {}
        record_stages: dict[str, Any] = {}
        summaries: list[dict[str, Any]] = []
        for stage_name in _RETRIEVAL_STAGE_NAMES:
            raw_stage = raw_stages.get(stage_name)
            if not isinstance(raw_stage, dict):
                continue
            raw_candidates = raw_stage.get("results", [])
            if not isinstance(raw_candidates, list):
                raw_candidates = []
            top_k_value = raw_stage.get("top_k")
            if type(top_k_value) is not int or top_k_value < 0:
                top_k_value = configured_cutoffs[stage_name]
            top_k = (
                top_k_value
                if type(top_k_value) is int and top_k_value >= 0
                else None
            )
            metric_top_k = top_k
            normalized_candidates = []
            for index, hit in enumerate(raw_candidates, start=1):
                if not isinstance(hit, dict):
                    continue
                rank = hit.get(_STAGE_RANK_FIELDS[stage_name])
                if type(rank) is not int:
                    rank = index if stage_name == "reranked" else hit.get("rank")
                if type(rank) is not int:
                    rank = index
                normalized_candidates.append(_stage_result_record(
                    hit,
                    stage=stage_name,
                    rank=rank,
                ))
            metrics = _stage_metrics(
                expected_evidence=expected_evidence,
                candidates=normalized_candidates,
                top_k=metric_top_k,
            )
            raw_returned_count = raw_stage.get("returned_count")
            returned_count = (
                raw_returned_count
                if type(raw_returned_count) is int and raw_returned_count >= 0
                else len(normalized_candidates)
            )
            stage_record = {
                "top_k": top_k,
                "metric_top_k": metric_top_k,
                "returned_count": returned_count,
                "latency_ms": _safe_score(raw_stage.get("latency_ms")),
                "results": normalized_candidates,
                "metrics": metrics,
            }
            if stage_name == "reranked" and isinstance(
                raw_stage.get("provenance"),
                dict,
            ):
                stage_record["provenance"] = raw_stage["provenance"]
            record_stages[stage_name] = stage_record
            summary = {
                "stage": stage_name,
                "top_k": top_k,
                "metric_top_k": metric_top_k,
                "returned_count": returned_count,
                **metrics,
            }
            summaries.append(summary)
            metrics_by_stage[stage_name].append(metrics)

        normalized_record = {
            "search_id": raw_record.get("search_id"),
            "search_query": raw_record.get("search_query"),
            "evidence_mode": raw_record.get("evidence_mode"),
            "generation_top_k": generation_top_k,
            "effective_thresholds": (
                raw_record.get("effective_thresholds")
                if isinstance(raw_record.get("effective_thresholds"), dict)
                else {}
            ),
            "configured_top_k": (
                raw_record.get("configured_top_k")
                if isinstance(raw_record.get("configured_top_k"), dict)
                else {}
            ),
            "filter_diagnostics": (
                raw_record.get("filter_diagnostics")
                if isinstance(raw_record.get("filter_diagnostics"), dict)
                else {}
            ),
            "stage_summaries": summaries,
            "stages": record_stages,
        }
        normalized_records.append(normalized_record)

        fused = record_stages.get("rrf")
        reranked = record_stages.get("reranked")
        comparison_top_k = None
        if (
            fused is not None
            and reranked is not None
            and type(generation_top_k) is int
            and generation_top_k >= 0
            and type(fused["top_k"]) is int
            and type(reranked["top_k"]) is int
        ):
            comparison_top_k = min(
                generation_top_k,
                fused["top_k"],
                reranked["top_k"],
            )
        fused_ids = [
            _stage_candidate_id(hit) for hit in fused["results"]
        ] if fused is not None else []
        reranked_ids = [
            _stage_candidate_id(hit) for hit in reranked["results"]
        ] if reranked is not None else []
        candidate_pool_matches = (
            bool(fused_ids)
            and all(candidate_id is not None for candidate_id in fused_ids)
            and all(candidate_id is not None for candidate_id in reranked_ids)
            and Counter(fused_ids) == Counter(reranked_ids)
        )
        comparison = {
            "search_id": raw_record.get("search_id"),
            "candidate_pool_matches": candidate_pool_matches,
            "generation_top_k": generation_top_k,
            "common_top_k": comparison_top_k,
            "fused_metrics": (
                _stage_metrics(
                    expected_evidence=expected_evidence,
                    candidates=fused["results"],
                    top_k=comparison_top_k,
                )
                if candidate_pool_matches and comparison_top_k is not None
                else {"recall": None, "precision": None, "ndcg": None}
            ),
            "reranked_metrics": (
                _stage_metrics(
                    expected_evidence=expected_evidence,
                    candidates=reranked["results"],
                    top_k=comparison_top_k,
                )
                if candidate_pool_matches and comparison_top_k is not None
                else {"recall": None, "precision": None, "ndcg": None}
            ),
        }
        if fused is not None or reranked is not None:
            comparisons.append(comparison)

    stage_metrics = {
        stage_name: _mean_stage_metrics(metrics_by_stage[stage_name])
        for stage_name in _RETRIEVAL_STAGE_NAMES
        if metrics_by_stage[stage_name]
    }
    if not comparisons:
        rerank_comparison: dict[str, Any] = {}
    else:
        common_cutoffs = {item["common_top_k"] for item in comparisons}
        rerank_comparison = {
            "candidate_pool_matches": all(
                item["candidate_pool_matches"] for item in comparisons
            ),
            "generation_top_k": (
                comparisons[0]["generation_top_k"]
                if len({item["generation_top_k"] for item in comparisons}) == 1
                else None
            ),
            "common_top_k": (
                next(iter(common_cutoffs)) if len(common_cutoffs) == 1 else None
            ),
            "fused_metrics": _mean_stage_metrics([
                item["fused_metrics"] for item in comparisons
                if item["candidate_pool_matches"]
            ]),
            "reranked_metrics": _mean_stage_metrics([
                item["reranked_metrics"] for item in comparisons
                if item["candidate_pool_matches"]
            ]),
            "searches": comparisons,
        }
    return normalized_records, stage_metrics, rerank_comparison


def _record_agentic_retrieval_stages(
    raw_records: Any,
    *,
    expected_evidence: list[dict[str, Any]],
    settings: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Record each search and score stage unions plus per-search NDCG."""
    normalized_records, _, rerank_comparison = _record_retrieval_stages(
        raw_records,
        expected_evidence=expected_evidence,
        settings=settings,
    )
    stage_metrics: dict[str, Any] = {}
    for stage_name in _RETRIEVAL_STAGE_NAMES:
        candidates_by_search: list[list[dict[str, Any]]] = []
        actual_cutoffs: list[dict[str, Any]] = []
        for search in normalized_records:
            stage = search["stages"].get(stage_name)
            if not isinstance(stage, dict):
                continue
            top_k = stage.get("metric_top_k", stage.get("top_k"))
            if type(top_k) is not int or top_k < 0:
                continue
            candidates_by_search.append(stage["results"][:top_k])
            actual_cutoffs.append({
                "search_id": search.get("search_id"),
                "top_k": top_k,
            })

        if candidates_by_search:
            # The result lists have already been sliced to their own configured
            # cutoff. Using the largest cutoff lets the metrics function compute
            # the union over all searches and mean NDCG over each local ranking.
            metrics = calculate_retrieval_metrics(
                expected_evidence=expected_evidence,
                retrieval_lists=candidates_by_search,
                top_k=max(cutoff["top_k"] for cutoff in actual_cutoffs),
            )
        else:
            metrics = {"recall": None, "precision": None, "ndcg": None}
        stage_metrics[stage_name] = {
            **metrics,
            "search_count": len(candidates_by_search),
            "candidate_count": sum(map(len, candidates_by_search)),
            "actual_cutoffs": actual_cutoffs,
        }

    return normalized_records, stage_metrics, rerank_comparison


def _classic_stage_latency_totals(
    retrieval_stages: list[dict[str, Any]],
) -> dict[str, float | None]:
    if not retrieval_stages:
        return {stage: None for stage in _RETRIEVAL_STAGE_NAMES}
    totals: dict[str, float | None] = {}
    for stage_name in _RETRIEVAL_STAGE_NAMES:
        observed = [
            _safe_score(
                (record.get("stages", {}).get(stage_name, {}) or {}).get(
                    "latency_ms"
                )
            )
            for record in retrieval_stages
        ]
        totals[stage_name] = (
            round(sum(value for value in observed if value is not None), 3)
            if all(value is not None for value in observed)
            else None
        )
    return totals


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
    result: dict[str, Any] = {}
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
        structured_error = result.get("workflow_error")
        if isinstance(structured_error, str) and structured_error:
            workflow_error = structured_error
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
            "rank",
            "vector_rank",
            "bm25_rank",
            "rerank_rank",
        ):
            if key in hit:
                record[key] = hit[key]
        for key in (
            "vector_score",
            "bm25_score",
            "fusion_score",
            "reranker_score",
        ):
            if key in hit:
                record[key] = _safe_score(hit.get(key))
        return record

    evidence_records = [evidence_record(hit) for hit in evidence]
    raw_search_results = result.get("search_results")
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
    raw_generation_context = result.get("generation_context", [])
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
            **{
                key: context_item.get(key, retrieved_item.get(key))
                for key in (
                    "rank",
                    "vector_rank",
                    "bm25_rank",
                    "fusion_rank",
                    "rerank_rank",
                    "score",
                    "vector_score",
                    "bm25_score",
                    "fusion_score",
                    "reranker_score",
                )
                if context_item.get(key, retrieved_item.get(key)) is not None
            },
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
    search_queries = result.get("search_queries", [])
    if not isinstance(search_queries, list):
        search_queries = []
    if not search_queries and search_result_records:
        search_queries = [case["question"]] * len(search_result_records)
    retrieval_stages: list[dict[str, Any]] = []
    retrieval_stage_metrics: dict[str, Any] = {}
    rerank_comparison: dict[str, Any] = {}
    if workflow == "classic":
        (
            retrieval_stages,
            retrieval_stage_metrics,
            rerank_comparison,
        ) = _record_retrieval_stages(
            result.get("retrieval_stages", []),
            expected_evidence=case["expected_evidence"],
            settings=get_settings(),
        )
    elif workflow == "agentic":
        (
            retrieval_stages,
            retrieval_stage_metrics,
            rerank_comparison,
        ) = _record_agentic_retrieval_stages(
            result.get("retrieval_stages", []),
            expected_evidence=case["expected_evidence"],
            settings=get_settings(),
        )
    controller_latencies = [
        call.get("latency_ms")
        for call in generation_calls
        if call.get("call_type") == "controller"
    ]
    controller_latency_ms = _measured_latency_total(controller_latencies)
    answer_calls = [
        call
        for call in generation_calls
        if call.get("call_type") in {"answer", "final_answer"}
    ]
    answer_call_latencies = [
        _safe_score(call.get("latency_ms")) for call in answer_calls
    ]
    answer_latency_ms = (
        round(sum(value for value in answer_call_latencies if value is not None), 3)
        if answer_call_latencies and all(
            value is not None for value in answer_call_latencies
        )
        else None
    )
    retrieval_latency_ms = _measured_latency_total([
        call.get("latency_ms") for call in retrieval_calls
    ])
    retrieval_metrics = calculate_retrieval_metrics(
        expected_evidence=case["expected_evidence"],
        retrieval_lists=raw_search_results,
        top_k=get_settings().top_k,
    )
    context_prefix_metrics = calculate_context_prefix_metrics(
        expected_evidence=case["expected_evidence"],
        context_chunks=generation_context_records,
    )
    final_context_metrics = (
        context_prefix_metrics[-1]
        if context_prefix_metrics
        else {
            "context_prefix_k": None,
            "evidence_lines_covered": None,
            "ground_truth_line_count": sum(
                span["end_line"] - span["start_line"] + 1
                for span in case["expected_evidence"]
                if isinstance(span.get("start_line"), int)
                and isinstance(span.get("end_line"), int)
            ),
            "recall": None,
            "precision": None,
            "ndcg": None,
            "line_provenance_available": False,
        }
    )
    record = {
        "case_id": case["id"],
        "category": case["category"],
        "challenge_tags": case["challenge_tags"],
        "workflow": workflow,
        "question": case["question"],
        "answer": answer,
        "reason_code": result.get("reason_code"),
        "response_origin": result.get("response_origin"),
        "search_attempts": result.get("search_attempts"),
        "answer_attempts": result.get("answer_attempts"),
        "answer_retry_exhausted": result.get("answer_retry_exhausted", False),
        "citations": citation_records,
        "retrieved_evidence": evidence_records,
        "search_results": search_result_records,
        "search_queries": search_queries,
        **({
            "retrieval_stages": retrieval_stages,
            "retrieval_stage_metrics": retrieval_stage_metrics,
            "rerank_comparison": rerank_comparison,
        } if workflow in {"classic", "agentic"} else {}),
        "generation_context": generation_context_records,
        "context_prefix_metrics": context_prefix_metrics,
        "final_context_metrics": final_context_metrics,
        "contexts": [item["text"] for item in evidence_records],
        "answer_contexts": answer_contexts,
        "reference_answer": case["reference_answer"],
        "ambiguity_reason": case.get("ambiguity_reason"),
        "missing_information": case.get("missing_information"),
        "expected_sources": case["expected_sources"],
        "expected_evidence": case["expected_evidence"],
        **_answerability_fields(
            case,
            answer,
            forced_abstention=result.get("answer_retry_exhausted") is True,
        ),
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
            "retrieval_stages_ms": _classic_stage_latency_totals(
                retrieval_stages
            ),
            "controller_ms": controller_latency_ms,
            "answer_generation_ms": answer_latency_ms,
            "generation_ms": (
                answer_latency_ms
                if workflow == "classic"
                else round(controller_latency_ms + answer_latency_ms, 3)
                if controller_latency_ms is not None
                and answer_latency_ms is not None
                else None
            ),
            "judge_metrics_ms": {},
            "judge_ms": 0.0,
            "rag_query_ms": rag_query_latency_ms,
            "end_to_end_ms": rag_query_latency_ms,
        },
    }
    if workflow_error is not None:
        record["workflow_error"] = workflow_error
        if result.get("workflow_error_type") is not None:
            record["workflow_error_type"] = result["workflow_error_type"]
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
    calls_by_row_metric: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for call in judge_calls:
        row_index = call.get("row_index")
        metric = call.get("metric")
        if isinstance(row_index, int) and isinstance(metric, str):
            calls_by_row_metric[(row_index, metric)].append(call)

    for case_index, case in enumerate(workflow_cases):
        scored_row_index = case_index
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
        eligible_judge_metrics = [
            metric
            for metric in METRIC_NAMES
            if _ragas_metric_is_eligible(case, metric)
        ]
        measured_judge_latencies = [
            judge_metric_latencies[metric]
            for metric in eligible_judge_metrics
        ]
        judge_latency_ms = (
            round(sum(measured_judge_latencies), 3)
            if measured_judge_latencies
            and all(latency is not None for latency in measured_judge_latencies)
            else None
        )
        case["latency"]["judge_metrics_ms"] = judge_metric_latencies
        case["latency"]["judge_ms"] = judge_latency_ms
        if not eligible_judge_metrics:
            case["latency"]["end_to_end_ms"] = case["latency"]["rag_query_ms"]
        elif judge_latency_ms is not None:
            case["latency"]["end_to_end_ms"] = round(
                case["latency"]["rag_query_ms"] + judge_latency_ms,
                3,
            )
        else:
            case["latency"]["end_to_end_ms"] = None


def _workflow_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_category[case["category"]].append(case)

    def summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        semantic_group = [case for case in group if not case.get("workflow_error")]
        workflow_error_group = [case for case in group if case.get("workflow_error")]
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
                    for case in semantic_group
                ])
                for metric in GENERATION_METRIC_NAMES
            },
            "retrieval_metrics": {
                metric: _mean([
                    case["retrieval_metrics"].get(metric)
                    for case in (
                        semantic_group
                        if metric in RETRIEVAL_METRIC_NAMES
                        else group
                    )
                ])
                for metric in ("recall", "precision", "ndcg", *RETRIEVAL_METRIC_NAMES)
            },
            "workflow_error_ragas_metrics": {
                metric: _mean([
                    (case.get(source, {}) or {}).get(metric)
                    for case in workflow_error_group
                ])
                for metric, source in (
                    ("faithfulness", "generation_metrics"),
                    ("answer_relevancy", "generation_metrics"),
                    ("answer_correctness", "generation_metrics"),
                    ("context_utilization", "retrieval_metrics"),
                )
            },
            "semantic_response_case_count": len(semantic_group),
            "workflow_error_diagnostic_case_count": len(workflow_error_group),
            "final_context_metrics": {
                metric: _mean([
                    case.get("final_context_metrics", {}).get(metric)
                    for case in group
                ])
                for metric in ("recall", "precision", "ndcg")
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


def _workflow_reliability(cases: list[dict[str, Any]]) -> dict[str, Any]:
    selected_cases = len(cases)
    successful_cases = sum(not case.get("workflow_error") for case in cases)
    return {
        "successful_cases": successful_cases,
        "selected_cases": selected_cases,
        "ratio": (
            successful_cases / selected_cases if selected_cases else None
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

    output_root = output_root or EVALUATION_ROOT / "results"
    timestamp, run_dir = _new_run_directory(Path(output_root))
    run_log = run_dir / "run.log"
    run_log.touch()
    results_path = run_dir / "results.json"
    readme_path = run_dir / "README.md"
    _log(run_log, f"Starting {workflow} synthetic-notes RAG evaluation.")

    results: dict[str, Any] = {
        "schema_version": 8,
        "run_status": "running",
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
                "cross_encoder_model": getattr(settings, "cross_encoder_model", None),
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "top_k": settings.top_k,
                "min_retrieval_score": settings.min_retrieval_score,
                "max_answer_chunks": settings.llm.max_chunks,
                "agentic_max_searches": MAX_SEARCHES,
                "agentic_max_answer_attempts": MAX_ANSWER_ATTEMPTS,
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
                    judgeable_count = sum(
                        has_nonblank_answer(case.get("answer"))
                        for case in workflow_cases
                    )
                    progress.set_description(f"RAG evaluation ({workflow} metrics)")
                    progress.set_postfix_str("Ragas metrics")
                    _log(
                        run_log,
                        f"[{workflow}] Starting Ragas metrics for "
                        f"{judgeable_count} non-empty outcomes.",
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

                    judge_usage_by_workflow[workflow] = [
                        _judge_usage_call(
                            call,
                            settings,
                            workflow_cases[call["row_index"]]["case_id"]
                            if isinstance(call.get("row_index"), int)
                            and 0 <= call["row_index"] < len(workflow_cases)
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
            selected_workflow_cases = workflow_results[workflow]
            workflow_error_count = sum(
                bool(case.get("workflow_error"))
                for case in selected_workflow_cases
            )
            workflow_reliability = _workflow_reliability(
                selected_workflow_cases
            )
            results.update({
                "run_status": "completed",
                "workflow_outcome": (
                    "completed_with_errors"
                    if workflow_error_count
                    else "completed"
                ),
                "workflow_error_count": workflow_error_count,
                "workflow_reliability": workflow_reliability,
            })
            results["summary"] = {
                "schema_version": 2,
                "run_status": results["run_status"],
                "workflow_outcome": results["workflow_outcome"],
                "workflow_error_count": workflow_error_count,
                "workflow_reliability": workflow_reliability,
                "run_metadata": results["run_metadata"],
                "case_count": len(cases),
                "category_counts": results["category_counts"],
                "interpretation": (
                    "This fixture is a regression set, not a statistically "
                    "representative benchmark. A sampled run may omit categories; "
                    "selected case IDs and category counts identify the coverage. "
                    "Ragas applicability follows the produced non-empty outcome and "
                    "available Final generation context; per-metric eligible counts "
                    "and unavailable reasons are recorded. Workflow-error diagnostics "
                    "remain per-case and are excluded from semantic-response means. "
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
        results["run_status"] = "failed"
        results["error"] = type(exc).__name__
        _write_json(results_path, results)
        write_readme(readme_path, results)
        raise

    print(f"Evaluation artifacts saved to {run_dir}")
    return run_dir
