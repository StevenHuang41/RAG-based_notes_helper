import json
from datasets import Dataset

from rag_notes_helper.eval.rag_runner import run_single_query


def load_testset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset(rag, meta_store, testset_path: str) -> Dataset:
    records = load_testset(testset_path)

    rows = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for item in records:
        if item.get("type") == "system":
            continue
        
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        result = run_single_query(rag, meta_store, question)

        rows["question"].append(question)
        rows["answer"].append(result.answer)
        rows["contexts"].append(result.contexts)
        rows["ground_truth"].append(ground_truth)

    return Dataset.from_dict(rows)
