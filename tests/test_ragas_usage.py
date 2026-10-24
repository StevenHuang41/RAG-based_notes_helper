from types import SimpleNamespace

from rag_notes_helper.eval.ragas_runner import _read_usage


def test_ragas_normalizes_provider_usage_and_marks_missing_usage_unavailable():
    response_with_usage = SimpleNamespace(
        generations=[
            [
                SimpleNamespace(
                    message=SimpleNamespace(
                        usage_metadata={
                            "input_tokens": 13,
                            "output_tokens": 4,
                            "total_tokens": 17,
                        },
                        response_metadata={},
                    )
                )
            ]
        ],
        llm_output=None,
    )
    response_without_usage = SimpleNamespace(generations=[], llm_output=None)

    assert _read_usage(response_with_usage) == {
        "status": "available",
        "input_tokens": 13,
        "output_tokens": 4,
        "total_tokens": 17,
    }
    assert _read_usage(response_without_usage) == {
        "status": "unavailable",
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }
