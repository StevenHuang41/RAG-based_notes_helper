import json

from rag_notes_helper.eval import eval_runner
from rag_notes_helper.rag.index import RagIndex


class AbstainingAnswerModel:
    provider_name = "fixture"
    model = "fixture-abstainer"
    last_usage = {
        "status": "available",
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
    }
    last_output = "I cannot find the answers in the notes."

    def generate(self, prompt, line_width):
        self.last_output = "I cannot find the answers in the notes."
        return self.last_output


def test_ambiguous_fixture_cases_are_auditable_and_abstain_through_evaluation(
    monkeypatch,
    tmp_path,
    fixture_evaluation_doubles,
):
    monkeypatch.setattr(
        RagIndex,
        "embed_model",
        property(lambda self: fixture_evaluation_doubles.embedding_model),
    )
    monkeypatch.setattr(
        "rag_notes_helper.rag.answer.get_llm",
        lambda: AbstainingAnswerModel(),
    )

    monkeypatch.setattr(
        eval_runner.subprocess,
        "run",
        fixture_evaluation_doubles.ragas_worker,
    )
    run_dir = eval_runner.run_evaluation(output_root=tmp_path / "runs")
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))

    ambiguous = [case for case in results["cases"] if case["category"] == "ambiguous"]
    assert len(ambiguous) == 12
    assert all(not case["expected_answerable"] for case in ambiguous)
    assert all(case["ambiguity_reason"].strip() for case in ambiguous)
    assert all(case["expected_sources"] for case in ambiguous)
    assert all(case["expected_evidence"] for case in ambiguous)
    assert all(case["answer"] == "I cannot find the answers in the notes." for case in ambiguous)
    assert all(case["correct_abstention"] is None for case in ambiguous)
    assert all(case["predicted_answerable"] is None for case in ambiguous)
