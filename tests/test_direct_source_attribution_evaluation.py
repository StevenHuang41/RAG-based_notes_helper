import json

from rag_notes_helper.eval import eval_runner
from rag_notes_helper.rag.index import RagIndex


class _AnswerModel:
    provider_name = "fixture"
    model = "fixture-answer"
    last_usage = {"status": "unavailable"}
    last_output = "A reviewed fixture answer."

    def generate(self, prompt, line_width):
        return self.last_output

    def stream(self, prompt, line_width):
        yield self.last_output

def test_direct_source_attribution_cases_complete_in_evaluation(
    monkeypatch, tmp_path, fixture_evaluation_doubles
):
    monkeypatch.setattr(
        RagIndex,
        "embed_model",
        property(lambda self: fixture_evaluation_doubles.embedding_model),
    )
    monkeypatch.setattr(
        "rag_notes_helper.rag.answer.get_llm",
        lambda: _AnswerModel(),
    )
    monkeypatch.setattr(
        eval_runner.subprocess,
        "run",
        fixture_evaluation_doubles.ragas_worker,
    )

    run_dir = eval_runner.run_evaluation(output_root=tmp_path)
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    direct_cases = [
        case for case in results["cases"] if case["category"] == "direct"
    ]
    reviewed_cases = eval_runner.load_qa_fixture()

    assert len(results["cases"]) == len(reviewed_cases)
    assert len(direct_cases) == 10
    assert {case["case_id"] for case in direct_cases} == {
        f"direct-{number:02}" for number in range(1, 11)
    }
    new_case_ids = {f"direct-{number:02}" for number in range(5, 11)}
    assert all(
        "source_attribution" in case["challenge_tags"]
        for case in direct_cases
        if case["case_id"] in new_case_ids
    )
    assert all(case["expected_sources"] for case in direct_cases)
    assert all(case["expected_evidence"] for case in direct_cases)
