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
            score_sets = []
            for batch in request["batches"]:
                result = run_ragas(
                    Dataset.from_list(batch["rows"]),
                    metric_names=batch["metric_names"],
                    usage_recorder=judge_calls.append,
                )
                score_sets.append([
                    {key: _json_score(value) for key, value in row.items()}
                    for row in result.scores
                ])
            response = {"scores": score_sets, "judge_calls": judge_calls}
    except Exception as exc:
        response = {
            "error": type(exc).__name__,
            "judge_calls": judge_calls,
        }
    sys.stdout.write(json.dumps(response, allow_nan=False))


if __name__ == "__main__":
    main()
