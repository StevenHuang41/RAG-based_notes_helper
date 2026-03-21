import pandas as pd
from datetime import datetime

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag.index import load_or_build_index
from rag_notes_helper.rag.meta_store import MetaStore
from rag_notes_helper.eval.dataset_builder import build_dataset
from rag_notes_helper.eval.ragas_runner import run_ragas


def run_evaluation():
    rag = load_or_build_index()
    meta_store = MetaStore()

    try :
        dataset = build_dataset(
            rag,
            meta_store,
            "src/rag_notes_helper/eval/testset/base.json",
        )

        print("\nDataset built. Running evaluation...\n")

        result = run_ragas(dataset)

        print("\n=== RAGAS RESULT ===\n")
        print(result)

        df = pd.DataFrame(result.scores)
        print("\n=== DETAIL ===\n")
        print(df)

        settings = get_settings()
        reports_dir = settings.reports_dir

        time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_fpath = reports_dir / f"run_{time_stamp}"
        summary_fpath = reports_dir / "summary.csv"

        # run file
        df.to_csv(run_fpath, index=False)
        print(f"\nSaved run to {run_fpath}")

        # summary file
        summary_row = {
            "time": time_stamp,
            "top_k": settings.top_k,
            "temperature": settings.llm.temperature,
            "min_retrieval_score": settings.min_retrieval_score,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "faithfulness": result["faithfulness"],
            "answer_relevancy": result["answer_relevancy"],
            "context_precision": result["context_precision"],
            "context_recall": result["context_recall"],
        }

        summary_df = pd.DataFrame([summary_row])

        # updated summary
        if summary_fpath.exists():
            existing = pd.read_csv(summary_fpath)
            summary_df = pd.concat([existing, summary_df], ignore_index=True)

        # save summary
        summary_df.to_csv(summary_fpath, index=False)
        print(f"Updated summary to {summary_fpath}")

    finally :
        meta_store.close()
