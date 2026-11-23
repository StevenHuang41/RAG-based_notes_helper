import json
from collections import Counter
from pathlib import Path

import pytest

from rag_notes_helper.eval import eval_runner


def _write_fixture(tmp_path: Path, records) -> Path:
    path = tmp_path / "notes_qa.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def _fixture_records():
    return json.loads(eval_runner.QA_FIXTURE.read_text(encoding="utf-8"))


def test_reviewed_fixture_has_four_files_and_twenty_grounded_cases():
    cases = eval_runner.load_qa_fixture()
    assert len(list(eval_runner.NOTES_FIXTURE.glob("*.md"))) == 4
    assert Counter(case["category"] for case in cases) == {
        "direct": 5, "multi_hop": 5, "reasoning": 5, "unanswerable": 5,
    }
    for case in cases:
        assert case["answerable"] is (case["category"] != "unanswerable")
        assert case["reference_answer"] and case["expected_evidence"]
        if case["category"] in {"multi_hop", "reasoning"}:
            assert len(case["expected_sources"]) >= 2


@pytest.mark.parametrize("category", ["direct", "multi_hop", "reasoning"])
def test_fixture_loader_rejects_unanswerable_cases_in_answerable_categories(tmp_path, category):
    records = _fixture_records()
    case = next(record for record in records if record["category"] == category)
    case["answerable"] = False
    with pytest.raises(ValueError, match="must be answerable"):
        eval_runner.load_qa_fixture(_write_fixture(tmp_path, records))


def test_default_evaluation_assets_require_the_checked_out_corpus(monkeypatch, tmp_path):
    monkeypatch.setattr(eval_runner, "NOTES_FIXTURE", tmp_path / "missing-notes")

    with pytest.raises(FileNotFoundError, match="Evaluation corpus directory is missing"):
        eval_runner.load_qa_fixture()


def test_default_evaluation_assets_require_the_reviewed_qa_set(monkeypatch, tmp_path):
    monkeypatch.setattr(eval_runner, "QA_FIXTURE", tmp_path / "missing-qa.json")

    with pytest.raises(FileNotFoundError, match="Evaluation ground truth QA set is missing"):
        eval_runner.load_qa_fixture()


def test_fixture_loader_accepts_reviewed_cases_beyond_the_seed_set(tmp_path):
    records = _fixture_records()
    existing_count = len(records)
    existing_direct_count = sum(record["category"] == "direct" for record in records)
    records.append({
        "id": "direct-test-extra",
        "category": "direct",
        "question": "When was the accessibility review completed?",
        "answerable": True,
        "reference_answer": "The accessibility review was completed on 2027-04-16.",
        "expected_sources": ["delivery.md"],
        "expected_evidence": [{
            "source": "delivery.md",
            "start_line": 6,
            "end_line": 6,
            "text": "The accessibility review was completed on 2027-04-16.",
        }],
    })

    loaded = eval_runner.load_qa_fixture(_write_fixture(tmp_path, records))

    assert len(loaded) == existing_count + 1
    loaded_direct_count = sum(
        record["category"] == "direct" for record in loaded
    )
    assert loaded_direct_count == existing_direct_count + 1


def test_fixture_loader_rejects_duplicate_ids(tmp_path):
    records = _fixture_records()
    records.append(records[0].copy())

    with pytest.raises(ValueError, match="unique"):
        eval_runner.load_qa_fixture(_write_fixture(tmp_path, records))


def test_fixture_loader_keeps_four_seed_cases_per_category(tmp_path):
    records = _fixture_records()
    multi_hop_ids = [
        record["id"] for record in records if record["category"] == "multi_hop"
    ]
    remove_ids = set(multi_hop_ids[3:])
    records = [record for record in records if record["id"] not in remove_ids]

    with pytest.raises(ValueError, match="baseline coverage"):
        eval_runner.load_qa_fixture(_write_fixture(tmp_path, records))


def test_fixture_loader_requires_all_original_seed_case_ids(tmp_path):
    records = _fixture_records()
    records = [record for record in records if record["id"] != "direct-01"]
    records.append({
        "id": "direct-test-extra",
        "category": "direct",
        "question": "When was the accessibility review completed?",
        "answerable": True,
        "reference_answer": "The accessibility review was completed on 2027-04-16.",
        "expected_sources": ["delivery.md"],
        "expected_evidence": [{
            "source": "delivery.md",
            "start_line": 6,
            "end_line": 6,
            "text": "The accessibility review was completed on 2027-04-16.",
        }],
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


@pytest.mark.parametrize("evidence", [None, []])
def test_fixture_loader_requires_evidence_for_answerable_case(tmp_path, evidence):
    records = _fixture_records()
    case = next(record for record in records if record["id"] == "direct-01")
    if evidence is None:
        del case["expected_evidence"]
    else:
        case["expected_evidence"] = evidence

    with pytest.raises(ValueError, match="direct-01.*evidence"):
        eval_runner.load_qa_fixture(_write_fixture(tmp_path, records))


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
        "id": "direct-test-extra",
        "category": "direct",
        "question": "When was the accessibility review completed?",
        "answerable": True,
        "reference_answer": "The accessibility review was completed on 2027-04-16.",
        "expected_sources": [],
        "expected_evidence": [{
            "source": "delivery.md",
            "start_line": 6,
            "end_line": 6,
            "text": "The accessibility review was completed on 2027-04-16.",
        }],
    })

    with pytest.raises(ValueError, match="expected source"):
        eval_runner.load_qa_fixture(_write_fixture(tmp_path, records))


def test_fixture_loader_accepts_multiline_evidence_spans(tmp_path):
    records = _fixture_records()
    case = next(record for record in records if record["id"] == "direct-01")
    case["expected_evidence"] = [{
        "source": "project.md",
        "start_line": 3,
        "end_line": 4,
        "text": "Mira Chen is the product owner.\nThe pilot starts on 2027-04-05.",
    }]

    loaded = eval_runner.load_qa_fixture(_write_fixture(tmp_path, records))

    assert loaded[0]["expected_evidence"] == case["expected_evidence"]


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"start_line": 0}, "line range"),
        ({"end_line": 999}, "line range"),
        ({"source": "../../private.md"}, "source path"),
        ({"text": "not the source text"}, "does not match"),
    ],
)
def test_fixture_loader_rejects_invalid_evidence_annotations(tmp_path, update, message):
    records = _fixture_records()
    evidence = records[0]["expected_evidence"][0]
    evidence.update(update)

    with pytest.raises(ValueError, match=message):
        eval_runner.load_qa_fixture(_write_fixture(tmp_path, records))
