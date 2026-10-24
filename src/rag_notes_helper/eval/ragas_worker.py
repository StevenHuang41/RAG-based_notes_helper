"""Isolated Ragas judge process, kept separate from FAISS/OpenMP runtime."""

from contextlib import redirect_stdout
import io
import json
import math
import sys


def _json_score(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> None:
    request = json.load(sys.stdin)
    judge_calls = []
    try:
        from datasets import Dataset
        from rag_notes_helper.eval.ragas_runner import run_ragas

        with redirect_stdout(io.StringIO()):
            result = run_ragas(
                Dataset.from_list(request["rows"]),
                usage_recorder=judge_calls.append,
            )
        response = {
            "scores": [
                {key: _json_score(value) for key, value in row.items()}
                for row in result.scores
            ],
            "judge_calls": judge_calls,
        }
    except Exception as exc:
        response = {
            "error": type(exc).__name__,
            "judge_calls": judge_calls,
        }
    sys.stdout.write(json.dumps(response, allow_nan=False))


if __name__ == "__main__":
    main()
