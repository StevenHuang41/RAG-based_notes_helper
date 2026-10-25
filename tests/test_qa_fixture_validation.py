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
    existing_count = len(records)
    existing_direct_count = sum(record["category"] == "direct" for record in records)
    records.append({
        "id": "direct-test-extra",
        "category": "direct",
        "question": "Which review is due by May 12?",
        "answerable": True,
        "reference_answer": "Legal review was due by 2026-05-12 and completed on 2026-05-11.",
        "expected_sources": ["delivery_timeline.md"],
        "expected_evidence": [{
            "source": "delivery_timeline.md",
            "start_line": 5,
            "end_line": 5,
            "text": "Legal review was due by 2026-05-12 and completed on 2026-05-11. The completion date is earlier than the due date; the old planning draft still shows the due date only.",
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
        "question": "Which review is due by May 12?",
        "answerable": True,
        "reference_answer": "Legal review was due by 2026-05-12 and completed on 2026-05-11.",
        "expected_sources": ["delivery_timeline.md"],
        "expected_evidence": [{
            "source": "delivery_timeline.md",
            "start_line": 5,
            "end_line": 5,
            "text": "Legal review was due by 2026-05-12 and completed on 2026-05-11. The completion date is earlier than the due date; the old planning draft still shows the due date only.",
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
        "question": "Which review is due by May 12?",
        "answerable": True,
        "reference_answer": "Legal review was due by 2026-05-12 and completed on 2026-05-11.",
        "expected_sources": [],
        "expected_evidence": [{
            "source": "delivery_timeline.md",
            "start_line": 5,
            "end_line": 5,
            "text": "Legal review was due by 2026-05-12 and completed on 2026-05-11. The completion date is earlier than the due date; the old planning draft still shows the due date only.",
        }],
    })

    with pytest.raises(ValueError, match="expected source"):
        eval_runner.load_qa_fixture(_write_fixture(tmp_path, records))


def test_fixture_loader_accepts_multiline_evidence_spans(tmp_path):
    records = _fixture_records()
    case = next(record for record in records if record["id"] == "direct-01")
    case["expected_evidence"] = [{
        "source": "product_overview.md",
        "start_line": 5,
        "end_line": 6,
        "text": "Product owner: Lena Ortiz. Launch coordinator: Marco Bell. QA lead: Priya Shah. These are separate responsibilities, even when meeting notes shorten them to “owner.”\nThe customer pilot began on 2026-04-06 with twelve invited coordinators. That date identifies the pilot start and must not be reported as the public launch date.",
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
