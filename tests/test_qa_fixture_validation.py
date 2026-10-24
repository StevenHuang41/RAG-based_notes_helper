import json
from pathlib import Path

import pytest

from rag_notes_helper.eval import eval_runner


def _write_fixture(tmp_path: Path, records) -> Path:
    path = tmp_path / "notes_qa.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def _fixture_records():
    return json.loads(eval_runner.QA_FIXTURE.read_text(encoding="utf-8"))


def test_fixture_loader_accepts_reviewed_cases_beyond_the_seed_set(tmp_path):
    records = _fixture_records()
    records.append({
        "id": "direct-05",
        "category": "direct",
        "question": "Which review is due by May 12?",
        "answerable": True,
        "reference_answer": "Legal review is due by May 12, 2026.",
        "expected_sources": ["delivery_timeline.md"],
        "expected_evidence": ["Legal review is due by May 12, 2026."],
    })

    loaded = eval_runner.load_qa_fixture(_write_fixture(tmp_path, records))

    assert len(loaded) == 17
    assert sum(record["category"] == "direct" for record in loaded) == 5


def test_fixture_loader_rejects_duplicate_ids(tmp_path):
    records = _fixture_records()
    records.append(records[0].copy())

    with pytest.raises(ValueError, match="unique"):
        eval_runner.load_qa_fixture(_write_fixture(tmp_path, records))


def test_fixture_loader_keeps_four_seed_cases_per_category(tmp_path):
    records = [
        record for record in _fixture_records()
        if record["id"] != "multi-hop-04"
    ]

    with pytest.raises(ValueError, match="baseline coverage"):
        eval_runner.load_qa_fixture(_write_fixture(tmp_path, records))


def test_fixture_loader_requires_all_original_seed_case_ids(tmp_path):
    records = _fixture_records()
    records = [record for record in records if record["id"] != "direct-01"]
    records.append({
        "id": "direct-05",
        "category": "direct",
        "question": "Which review is due by May 12?",
        "answerable": True,
        "reference_answer": "Legal review is due by May 12, 2026.",
        "expected_sources": ["delivery_timeline.md"],
    })

    with pytest.raises(ValueError, match="seed cases"):
        eval_runner.load_qa_fixture(_write_fixture(tmp_path, records))


def test_fixture_loader_defaults_missing_evidence_for_unanswerable_cases(tmp_path):
    records = _fixture_records()
    case = next(record for record in records if record["id"] == "unanswerable-01")
    del case["expected_evidence"]

    loaded = eval_runner.load_qa_fixture(_write_fixture(tmp_path, records))

    assert next(record for record in loaded if record["id"] == "unanswerable-01")[
        "expected_evidence"
    ] == []


def test_fixture_loader_allows_answerable_case_without_optional_evidence(tmp_path):
    records = _fixture_records()
    case = next(record for record in records if record["id"] == "direct-01")
    del case["expected_evidence"]

    loaded = eval_runner.load_qa_fixture(_write_fixture(tmp_path, records))

    assert next(record for record in loaded if record["id"] == "direct-01")[
        "expected_evidence"
    ] == []


def test_fixture_loader_rejects_unreviewed_candidate_drafts(tmp_path):
    records = _fixture_records()
    records.append({
        "candidate_id": "candidate-001",
        "review_status": "draft",
        "question": "What is the approved budget?",
        "reference_answer": "The approved budget is $24,000.",
        "reference_contexts": ["Approved total: $24,000."],
        "synthesizer_name": "SingleHopSpecificQuerySynthesizer",
    })

    with pytest.raises(ValueError, match="missing required fields"):
        eval_runner.load_qa_fixture(_write_fixture(tmp_path, records))


def test_fixture_loader_requires_source_for_answerable_cases(tmp_path):
    records = _fixture_records()
    records.append({
        "id": "direct-05",
        "category": "direct",
        "question": "Which review is due by May 12?",
        "answerable": True,
        "reference_answer": "Legal review is due by May 12, 2026.",
        "expected_sources": [],
    })

    with pytest.raises(ValueError, match="expected source"):
        eval_runner.load_qa_fixture(_write_fixture(tmp_path, records))

