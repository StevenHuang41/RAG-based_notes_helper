import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from rag_notes_helper.eval import candidate_generator
from rag_notes_helper.eval.candidate_generator import map_candidate_records


def test_ragas_samples_become_unreviewed_candidates_without_source_claims():
    records = map_candidate_records([
        {
            "user_input": "What is the approved budget?",
            "reference": "The approved budget is $24,000.",
            "reference_contexts": ["Approved total: $24,000."],
            "synthesizer_name": "SingleHopSpecificQuerySynthesizer",
        }
    ])

    assert records == [{
        "candidate_id": "candidate-001",
        "review_status": "draft",
        "user_input": "What is the approved budget?",
        "reference": "The approved budget is $24,000.",
        "reference_contexts": ["Approved total: $24,000."],
        "synthesizer_name": "SingleHopSpecificQuerySynthesizer",
    }]
    assert not {"expected_sources", "expected_evidence"} & records[0].keys()


def test_candidate_mapping_preserves_ragas_values_exactly():
    sample = {
        "user_input": " What is the approved budget? ",
        "reference": " The approved budget is $24,000. ",
        "reference_contexts": [" Approved total: $24,000. "],
        "synthesizer_name": "SingleHopSpecificQuerySynthesizer",
    }

    record = map_candidate_records([sample])[0]

    assert record["user_input"] == sample["user_input"]
    assert record["reference"] == sample["reference"]
    assert record["reference_contexts"] == sample["reference_contexts"]
    assert record["synthesizer_name"] == sample["synthesizer_name"]


def test_candidate_generation_uses_fixture_documents_and_writes_a_draft(
    monkeypatch,
    tmp_path,
):
    notes_dir = tmp_path / "synthetic-notes"
    notes_dir.mkdir()
    (notes_dir / "budget.md").write_text(
        "Approved total: $24,000.", encoding="utf-8"
    )
    (notes_dir / "plan.md").write_text(
        "The review finishes on June 1.", encoding="utf-8"
    )

    answer_model = object()
    expected_embedding_model = object()
    monkeypatch.setattr(candidate_generator, "NOTES_FIXTURE", notes_dir)
    monkeypatch.setattr(candidate_generator, "get_eval_llm", lambda: answer_model)
    monkeypatch.setattr(
        candidate_generator,
        "get_embeddings",
        lambda: expected_embedding_model,
    )
    monkeypatch.setattr(
        candidate_generator,
        "get_settings",
        lambda: SimpleNamespace(
            llm=SimpleNamespace(
                effective_eval_provider="openrouter",
                effective_eval_model="openai/gpt-4o-mini",
            ),
            embed_model_name="sentence-transformers/test-model",
        ),
    )

    generation_calls = []

    class FakeTestsetGenerator:
        def __init__(self, *, llm, embedding_model):
            assert llm is answer_model
            assert embedding_model is expected_embedding_model

        def generate_with_langchain_docs(self, documents, *, testset_size):
            generation_calls.append((documents, testset_size))
            return SimpleNamespace(to_list=lambda: [{
                "user_input": "What is the approved budget?",
                "reference": "The approved budget is $24,000.",
                "reference_contexts": ["Approved total: $24,000."],
                "synthesizer_name": "SingleHopSpecificQuerySynthesizer",
            }])

    ragas_module = ModuleType("ragas")
    ragas_module.__path__ = []
    testset_module = ModuleType("ragas.testset")
    testset_module.TestsetGenerator = FakeTestsetGenerator
    ragas_module.testset = testset_module
    monkeypatch.setitem(sys.modules, "ragas", ragas_module)
    monkeypatch.setitem(sys.modules, "ragas.testset", testset_module)
    output_path = tmp_path / "candidate-draft.json"

    result = candidate_generator.generate_candidate_draft(
        testset_size=3,
        output_path=output_path,
    )

    assert result == output_path
    assert generation_calls[0][1] == 3
    documents = generation_calls[0][0]
    assert {document.metadata["source"] for document in documents} == {
        "budget.md",
        "plan.md",
    }
    assert [document.page_content for document in documents] == [
        "Approved total: $24,000.",
        "The review finishes on June 1.",
    ]
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["source_corpus"] == "tests/fixtures/notes"
    assert payload["ragas_config"] == {
        "judge_provider": "openrouter",
        "judge_model": "openai/gpt-4o-mini",
        "embedding_model": "sentence-transformers/test-model",
    }
    assert payload["candidates"][0]["review_status"] == "draft"
    assert payload["candidates"][0]["user_input"] == "What is the approved budget?"
    assert payload["candidates"][0]["reference"] == "The approved budget is $24,000."
    assert not {
        "expected_sources",
        "expected_evidence",
    } & payload["candidates"][0].keys()


def test_existing_draft_path_is_rejected_before_model_setup(monkeypatch, tmp_path):
    output_path = tmp_path / "candidate-draft.json"
    output_path.write_text("preserve this draft", encoding="utf-8")
    monkeypatch.setattr(
        candidate_generator,
        "get_eval_llm",
        lambda: pytest.fail("model setup should not happen for an existing path"),
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        candidate_generator.generate_candidate_draft(
            testset_size=3,
            output_path=output_path,
        )

    assert output_path.read_text(encoding="utf-8") == "preserve this draft"
