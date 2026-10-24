"""Generate review-only QA candidates from the repository's synthetic notes."""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.eval.embedding import get_embeddings
from rag_notes_helper.eval.llm import get_eval_llm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
NOTES_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "notes"
DEFAULT_DRAFT_ROOT = PROJECT_ROOT / "eval" / "candidate-drafts"
REVIEW_INSTRUCTIONS = [
    "Verify each question and reference answer against the synthetic source notes.",
    "Assign a reviewed fixture id, category, and answerability value.",
    (
        "Annotate expected_sources and expected_evidence from the source "
        "notes before promotion."
    ),
    (
        "Add only reviewed cases to tests/fixtures/notes_qa.json; keep "
        "unreviewed drafts out."
    ),
]


def map_candidate_records(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Map Ragas testset samples into a draft schema that is not eval-ready."""
    records = []
    required_fields = {
        "user_input",
        "reference",
        "reference_contexts",
        "synthesizer_name",
    }

    for index, sample in enumerate(samples, start=1):
        missing = required_fields - sample.keys()
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"Ragas candidate {index} is missing fields: {fields}.")

        question = sample["user_input"]
        reference_answer = sample["reference"]
        contexts = sample["reference_contexts"]
        synthesizer_name = sample["synthesizer_name"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Ragas candidate {index} has no question.")
        if not isinstance(reference_answer, str) or not reference_answer.strip():
            raise ValueError(f"Ragas candidate {index} has no reference answer.")
        if (
            not isinstance(contexts, list)
            or not contexts
            or any(
                not isinstance(context, str) or not context.strip()
                for context in contexts
            )
        ):
            raise ValueError(f"Ragas candidate {index} has invalid reference contexts.")
        if not isinstance(synthesizer_name, str) or not synthesizer_name.strip():
            raise ValueError(f"Ragas candidate {index} has no synthesizer name.")

        records.append({
            "candidate_id": f"candidate-{index:03d}",
            "review_status": "draft",
            "user_input": question,
            "reference": reference_answer,
            "reference_contexts": list(contexts),
            "synthesizer_name": synthesizer_name,
        })

    return records


def _fixture_documents():
    from langchain_core.documents import Document

    paths = sorted(path for path in NOTES_FIXTURE.rglob("*.md") if path.is_file())
    if not paths:
        raise ValueError(f"No synthetic Markdown notes found under {NOTES_FIXTURE}.")

    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": path.relative_to(NOTES_FIXTURE).as_posix()},
        )
        for path in paths
    ]


def _source_digest(documents) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document.metadata["source"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(document.page_content.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _new_default_output_path() -> Path:
    DEFAULT_DRAFT_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = DEFAULT_DRAFT_ROOT / f"ragas-candidates-{timestamp}.json"
    suffix = 1
    while output_path.exists():
        output_path = DEFAULT_DRAFT_ROOT / f"ragas-candidates-{timestamp}-{suffix}.json"
        suffix += 1
    return output_path


def generate_candidate_draft(
    *,
    testset_size: int,
    output_path: Path | None = None,
) -> Path:
    """Generate a review-only Ragas draft from synthetic fixture documents."""
    if testset_size <= 0:
        raise ValueError("Candidate test-set size must be greater than zero.")

    destination = (
        Path(output_path)
        if output_path is not None
        else _new_default_output_path()
    )
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing candidate draft: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    from ragas.testset import TestsetGenerator

    documents = _fixture_documents()
    settings = get_settings()
    generator = TestsetGenerator(
        llm=get_eval_llm(),
        embedding_model=get_embeddings(),
    )
    testset = generator.generate_with_langchain_docs(
        documents,
        testset_size=testset_size,
    )
    candidates = map_candidate_records(testset.to_list())
    if not candidates:
        raise ValueError("Ragas did not return any QA candidates.")

    payload = {
        "schema_version": 1,
        "review_status": "draft",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_corpus": "tests/fixtures/notes",
        "source_corpus_sha256": _source_digest(documents),
        "requested_candidate_count": testset_size,
        "candidate_count": len(candidates),
        "ragas_config": {
            "judge_provider": settings.llm.effective_eval_provider,
            "judge_model": settings.llm.effective_eval_model,
            "embedding_model": settings.embed_model_name,
        },
        "review_instructions": REVIEW_INSTRUCTIONS,
        "candidates": candidates,
    }
    with destination.open("x", encoding="utf-8") as draft_file:
        json.dump(payload, draft_file, ensure_ascii=False, indent=2)
        draft_file.write("\n")
    return destination


def _positive_size(value: str) -> int:
    try:
        size = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("size must be a positive integer") from exc
    if size <= 0:
        raise argparse.ArgumentTypeError("size must be a positive integer")
    return size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag_notes_helper.eval.candidate_generator",
        description="Generate review-only QA candidates from synthetic notes.",
    )
    parser.add_argument(
        "--size",
        type=_positive_size,
        required=True,
        help="Number of Ragas QA candidates to request (model calls may cost tokens).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Draft output path (defaults to a new ignored directory under eval/).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_path = generate_candidate_draft(
        testset_size=args.size,
        output_path=args.output,
    )
    print(f"Review-only Ragas candidates saved to {output_path}")
    print(
        "Verify each candidate and annotate ground truth before adding it "
        "to the QA fixture."
    )


if __name__ == "__main__":
    main()
