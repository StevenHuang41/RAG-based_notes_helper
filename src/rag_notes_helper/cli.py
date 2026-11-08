import argparse
import importlib
import sys
from time import perf_counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal
from pydantic import ValidationError
from collections.abc import Iterator

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag.index import (
    RagIndex,
    load_or_build_index,
    rebuild_index,
)
from rag_notes_helper.rag.meta_store import MetaStore
from rag_notes_helper.rag.prompt_context import (
    ConversationTurn,
    MAX_CONVERSATION_TURNS,
)
from rag_notes_helper.rag.retrieval import (
    CrossEncoderModel,
    HybridSearchResults,
    retrieve,
)
from rag_notes_helper.rag.answer import ABSTENTION_MESSAGE, rag_answer
from rag_notes_helper.rag.agentic import (
    MAX_SEARCHES,
    AgenticWorkflowError,
    ProgressStage,
    RETRIEVAL_FAILURE_MESSAGE,
    agentic_answer,
)
from rag_notes_helper.utils.logger import get_logger
from rag_notes_helper.utils.timer import time_block, deco_time_block


logger = get_logger("cli")


INVALID_QUERY_MESSAGE = "Please enter a non-empty question."
NO_EVIDENCE_MESSAGE = (
    "I could not find sufficient supporting evidence in the notes."
)


class EvaluationUnavailableError(RuntimeError):
    """Raised when a source-checkout-only evaluation command is unavailable."""


def _render_agentic_progress(stage: ProgressStage, search_count: int) -> None:
    if not sys.stderr.isatty():
        return

    if stage == "searching":
        message = f"Searching notes ({search_count}/{MAX_SEARCHES})"
    elif stage == "reviewing":
        message = "Reviewing evidence"
    elif stage == "generating":
        message = "Generating answer"
    else:
        message = "Insufficient evidence; abstaining"

    print(f"[Agentic] {message}", file=sys.stderr, flush=True)


def _load_evaluation_function(
    function_name: Literal["run_evaluation"],
) -> Callable[..., Any]:
    try:
        module = importlib.import_module("rag_notes_helper.eval.eval_runner")
    except ModuleNotFoundError as exc:
        raise EvaluationUnavailableError(
            "Evaluation commands require a source checkout with development "
            "dependencies installed."
        ) from exc
    return getattr(module, function_name)


def run_evaluation(
    *,
    output_root: Path | None = None,
    agentic: bool = False,
    sample: int | None = None,
    seed: int | None = None,
) -> Path:
    """Load the development-only evaluator only when the benchmark is requested."""
    evaluate = _load_evaluation_function("run_evaluation")

    if output_root is None:
        return evaluate(agentic=agentic, sample=sample, seed=seed)
    return evaluate(
        output_root=output_root,
        agentic=agentic,
        sample=sample,
        seed=seed,
    )


@deco_time_block
def show_config():
    print("\nChecking Configuration ...")
    try :
        settings = get_settings()
    except ValidationError as e:
        print("Configuration is INVALID\n")
        print(e)
        sys.exit(1)

    print("Configuration is VALID")

    print("\nLLM:")
    print(f"    Provider   : {settings.llm.provider}")
    print(f"    Model      : {settings.llm.model}")
    print(f"    API Key    : {settings.llm.api_key}")

    print("\nEmbedding:")
    print(f"    Model      : {settings.embed_model_name}")

    print("\nChunking:")
    print(f"    Size       : {settings.chunk_size}")
    print(f"    Overlap    : {settings.chunk_overlap}")

    print("\nRetrieval:")
    print(f"    VECTOR_TOP_K : {settings.vector_top_k}")
    print(f"    BM25_TOP_K   : {settings.bm25_top_k}")
    print(f"    FUSION_TOP_K : {settings.fusion_top_k}")
    print(f"    VECTOR_MIN_SCORE      : {settings.vector_min_score}")
    print(f"    BM25_MIN_SCORE_RATIO  : {settings.bm25_min_score_ratio}")

    print("\nFormat:")
    print(f"    stream     : {settings.stream}")
    print(f"    line_width : {settings.line_width}")

    print("\nPaths:")
    print(f"    Notes dir  : {settings.notes_dir}")
    print(f"    Storage dir: {settings.storage_dir}")

    print("\nConfig check completed")


@deco_time_block
def show_sources(meta_store: MetaStore):
    print("\nSOURCES:\n")
    for s in meta_store.list_indexed_sources():
        print(f"- {s}")


@deco_time_block
def show_citations(result, show_full: bool = False):
    if show_full:
        print("\nCITATIONS:\n")
        for c in result["citations"]:
            print(
                f"- {c['source']} "
                    f"(chunk={c['chunk_id']}, score={c['score']:.3f})"
            )

    else :
        print("\nCITATIONS:")
        print("- ", end="")
        source_set = {c["source"] for c in result["citations"]}
        print(", ".join(source_set))


@deco_time_block
def display_ansewr(answer):
    print("\nANSWER:\n")

    if isinstance(answer, Iterator) and not isinstance(answer, str):
        try :
            for chunk in answer:
                print(chunk, end="", flush=True)

            print()
        except Exception as e:
            logger.warning(f"[Stream Error]: {e}")
            print(f"\n[Stream Error]: {e}")

    else :
        print(answer)


def run_onetime(
    rag: RagIndex,
    meta_store: MetaStore,
    *,
    query: str,
    citations: bool = False,
    agentic: bool = False,
) -> None:
    logger.info((f"query: {query[:20]}{' ...' if len(query) > 20 else ''}"))
    result = run_query(
        rag,
        meta_store,
        query=query,
        agentic=agentic,
        on_progress=_render_agentic_progress if agentic else None,
    )

    display_ansewr(result["answer"])

    if citations and result["citations"]:
        show_citations(result)


def run_query(
    rag: RagIndex,
    meta_store: MetaStore,
    *,
    query: str,
    agentic: bool = False,
    stream: bool = False,
    on_generation: Callable[[dict[str, Any]], None] | None = None,
    on_progress: Callable[[ProgressStage, int], None] | None = None,
    on_retrieval: Callable[[dict[str, Any]], None] | None = None,
    cross_encoder: CrossEncoderModel | None = None,
    conversation_history: Sequence[ConversationTurn] = (),
) -> dict[str, Any]:
    if not query.strip():
        return {
            "answer": INVALID_QUERY_MESSAGE,
            "citations": [],
            "generation_context": [],
            "retrieved_evidence": [],
            "search_queries": [],
            "retrieval_stages": [],
            "reason_code": "invalid_query",
            "response_origin": "deterministic",
        }

    if agentic:
        kwargs = {
            "rag": rag,
            "meta_store": meta_store,
            "stream": stream,
        }
        if on_generation is not None:
            kwargs["on_generation"] = on_generation
        if on_progress is not None:
            kwargs["on_progress"] = on_progress
        if on_retrieval is not None:
            kwargs["on_retrieval"] = on_retrieval
        if cross_encoder is not None:
            kwargs["cross_encoder"] = cross_encoder
        if conversation_history:
            kwargs["conversation_history"] = tuple(
                conversation_history[-MAX_CONVERSATION_TURNS:]
            )
        return agentic_answer(query, **kwargs)

    retrieval_started = perf_counter()
    retrieval_kwargs = {}
    if cross_encoder is not None:
        retrieval_kwargs["cross_encoder"] = cross_encoder
    try:
        hits = retrieve(rag, meta_store, query=query, **retrieval_kwargs)
    except Exception as exc:
        if on_retrieval is not None:
            on_retrieval({
                "search_index": 1,
                "latency_ms": round((perf_counter() - retrieval_started) * 1000, 3),
                "result_count": None,
            })
        return {
            "answer": RETRIEVAL_FAILURE_MESSAGE,
            "citations": [],
            "generation_context": [],
            "retrieved_evidence": [],
            "search_results": [],
            "search_queries": [query],
            "retrieval_stages": [],
            "search_attempts": 1,
            "reason_code": "retrieval_failure",
            "response_origin": "deterministic",
            "workflow_error": "retrieval_failure",
            "workflow_error_type": type(exc).__name__,
        }
    if on_retrieval is not None:
        on_retrieval({
            "search_index": 1,
            "latency_ms": round((perf_counter() - retrieval_started) * 1000, 3),
            "result_count": len(hits),
        })
    retrieval_stages = [
        hits.to_stage_record(search_id=1, search_query=query)
    ] if isinstance(hits, HybridSearchResults) else []
    if not hits:
        return {
            "answer": NO_EVIDENCE_MESSAGE,
            "citations": [],
            "generation_context": [],
            "retrieved_evidence": hits,
            "search_queries": [query],
            "retrieval_stages": retrieval_stages,
            "reason_code": "no_evidence_after_searches",
            "response_origin": "deterministic",
        }
    kwargs = {"hits": hits, "stream": stream}
    if on_generation is not None:
        kwargs["on_generation"] = on_generation
    result = rag_answer(query, **kwargs)
    if isinstance(result["answer"], str) and not result["answer"].strip():
        result["answer"] = NO_EVIDENCE_MESSAGE
        result["reason_code"] = "insufficient_evidence"
        result["response_origin"] = "deterministic"
    else:
        result["reason_code"] = (
            "insufficient_evidence"
            if result["answer"] == ABSTENTION_MESSAGE
            else "answered"
        )
        result["response_origin"] = "answer_model"
    result["retrieved_evidence"] = hits
    result["search_queries"] = [query]
    result["retrieval_stages"] = retrieval_stages
    return result


def repl(
    rag: RagIndex | None = None,
    meta_store: MetaStore | None = None,
    *,
    citations: bool = False,
    agentic: bool = False,
):
    rag = rag or load_or_build_index()
    meta_store = meta_store or MetaStore()
    stream_response: bool = get_settings().stream
    conversation_history: list[ConversationTurn] = []

    print(
        "\nRAG-based Notes Helper\n"
        "(enter ':h' for help)\n"
    )


    try :
        while True:
            try :
                query = input("> ").strip()
                logger.info(f"[input]: {query}")
            except KeyboardInterrupt:
                print("\nBye~")
                break

            if query.startswith(";"):
                print("\nCommand Typo:\n  commands should start with a colon\n")
                continue

            if query.startswith(":"):
                if query in {":quit", ":q"}:
                    print("\nBye~")
                    break

                if query in {":update", ":ud", ":reindex", ":ri"}:
                    meta_store.close()

                    do_force = query in {":reindex", ":ri"}
                    with time_block(
                        f"rebuild_index({'force' if do_force else 'smart'})"
                    ):
                        rag = rebuild_index(force=do_force)

                    meta_store = MetaStore()

                    print()
                    continue

                if query in {":sources", ":so"}:
                    show_sources(meta_store)
                    print()
                    continue

                if query in {":config", ":co"}:
                    show_config()
                    print()
                    continue

                if query in {":citations", ":ci"}:
                    if citations:
                        print("\n/Hide Citations\n")
                    else :
                        print("\n/Show Citations\n")

                    citations = not citations
                    continue

                if query in {":stream", ":s"}:
                    if stream_response:
                        print("\n/Batch response\n")
                    else :
                        print("\n/Stream response\n")

                    stream_response = not stream_response
                    continue

                if query in {":evaluate", ":ev"}:
                    try:
                        run_evaluation()
                    except EvaluationUnavailableError as exc:
                        print(f"rag-app: {exc}", file=sys.stderr)
                    continue

                if query in {":help", ":h"}:
                    print(
                        "\nCommands:\n"
                        "  :quit      or  :q    -> exit app\n"
                        "  :help      or  :h    -> show instructions\n"
                        "  :reindex   or  :ri   -> rebuild rag (hard update)\n"
                        "  :update    or  :ud   -> update data (soft update)\n"
                        "  :citations or  :ci   -> show citation files\n"
                        "  :sources   or  :so   -> show all source files\n"
                        "  :config    or  :co   -> check configuration\n"
                        "  :stream    or  :s    -> toggle stream mode\n"
                        "  :evaluate  or  :ev   -> evaluate rag\n"
                    )
                    continue

                print("\nUnknown Command:\n  use :help to check valid commands\n")
                continue

            logger.info((f"query: {query[:20]}{' ...' if len(query) > 20 else ''}"))

            try:
                query_kwargs: dict[str, Any] = {
                    "query": query,
                    "agentic": agentic,
                    "stream": stream_response,
                    "on_progress": _render_agentic_progress if agentic else None,
                }
                if agentic and conversation_history:
                    query_kwargs["conversation_history"] = tuple(
                        conversation_history
                    )
                result = run_query(rag, meta_store, **query_kwargs)
            except AgenticWorkflowError as exc:
                logger.error(f"Agentic workflow error: {exc}")
                print(f"\nAgentic workflow error: {exc}", file=sys.stderr)
                print()
                continue

            display_ansewr(result["answer"])

            if agentic:
                conversation_history.append((query, str(result["answer"])))
                del conversation_history[:-MAX_CONVERSATION_TURNS]

            if citations and result["citations"]:
                show_citations(result, show_full=True)

            print()
    finally :
        conversation_history.clear()
        meta_store.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-app",
        description="RAG-based Notes Helper",
    )

    parser.add_argument(
        "query",
        nargs="*",
        help="Question to ask the RAG system (default mode).",
    )

    parser.add_argument(
        "-rp",
        "--repl",
        action="store_true",
        help="Start interactive REPL mode",
    )

    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Use the Agentic RAG workflow",
    )

    parser.add_argument(
        "-ev",
        "--eval",
        action="store_true",
        help="Evaluate Classic RAG, or Agentic RAG when used with --agentic",
    )

    parser.add_argument(
        "--sample",
        type=int,
        help="Evaluate this many randomly selected fixture cases (requires --eval)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Seed fixture case sampling when used with --sample (requires --eval)",
    )

    parser.add_argument(
        "-ud",
        "--update",
        action="store_true",
        help="Update the existing index without a full rebuild.",
    )

    parser.add_argument(
        "-ri",
        "--reindex",
        action="store_true",
        help="Rebuild the index from scratch.",
    )

    parser.add_argument(
        "-co",
        "--config",
        action="store_true",
        help="Display the current configuration.",
    )

    parser.add_argument(
        "-ci",
        "--citations",
        action="store_true",
        help="Show citation information with query results.",
    )

    parser.add_argument(
        "-so",
        "--sources",
        action="store_true",
        help="List all source files that have been indexed.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if (args.sample is not None or args.seed is not None) and not args.eval:
        parser.error("--sample and --seed can only be used with --eval")

    if args.sample is not None and args.sample <= 0:
        parser.error("--sample must be a positive integer")

    if args.eval:
        try:
            run_evaluation(
                agentic=args.agentic,
                sample=args.sample,
                seed=args.seed,
            )
        except EvaluationUnavailableError as exc:
            parser.error(str(exc))
        except ValueError as exc:
            parser.error(str(exc))
        return
    query_submitted = bool(args.query)
    query = (" ".join(args.query)).strip()
    logger.info(
        f"[input]: rag-app{f' {query}' if query else ''}"
        f"{' --repl' if args.repl else ''}"
        f"{' --agentic' if args.agentic else ''}"
        f"{' --reindex' if args.reindex else ''}"
        f"{' --update' if args.update else ''}"
        f"{' --citations' if args.citations else ''}"
        f"{' --sources' if args.sources else ''}"
        f"{' --config' if args.config else ''}"
        f"{' --eval' if args.eval else ''}"
    )
    logger.info(f"config: {get_settings().model_dump_json()}")

    with time_block("start up preparation"):
        rag = (
            rebuild_index(force=args.reindex)
            if args.update or args.reindex
            else load_or_build_index()
        )

        meta_store = MetaStore()

    if args.config:
        show_config()

    if args.sources:
        show_sources(meta_store)

    if args.repl:
        logger.info("==== REPL start ====")
        if query_submitted:
            try:
                run_onetime(
                    rag,
                    meta_store,
                    query=query,
                    citations=args.citations,
                    agentic=args.agentic,
                )
            except AgenticWorkflowError as exc:
                logger.error(f"Agentic workflow error: {exc}")
                print(f"{parser.prog}: Agentic workflow error: {exc}", file=sys.stderr)

        repl(
            rag,
            meta_store,
            citations=args.citations,
            agentic=args.agentic,
        )
        logger.info("==== REPL end ====")

    elif query_submitted and not args.repl:
        logger.info("==== run_onetime start ====")
        try:
            run_onetime(
                query=query,
                citations=args.citations,
                rag=rag,
                meta_store=meta_store,
                agentic=args.agentic,
            )
        except AgenticWorkflowError as exc:
            parser.exit(
                2,
                f"{parser.prog}: Agentic workflow error: {exc}\n",
            )
        logger.info("==== run_onetime end ====")

    elif not (
        args.config
        or args.sources
        or args.reindex
        or args.update
    ):
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
