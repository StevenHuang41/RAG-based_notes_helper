import argparse
import importlib
import sys
from collections.abc import Callable
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
from rag_notes_helper.rag.retrieval import retrieve
from rag_notes_helper.rag.answer import rag_answer
from rag_notes_helper.rag.agentic import (
    MAX_SEARCHES,
    AgenticWorkflowError,
    ProgressStage,
    agentic_answer,
)
from rag_notes_helper.utils.logger import get_logger
from rag_notes_helper.utils.timer import time_block, deco_time_block


logger = get_logger("cli")


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
    function_name: Literal["run_evaluation", "run_base_smoke_evaluation"],
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
) -> Path:
    """Load the development-only evaluator only when the benchmark is requested."""
    evaluate = _load_evaluation_function("run_evaluation")

    if output_root is None:
        return evaluate(agentic=agentic)
    return evaluate(output_root=output_root, agentic=agentic)


def run_base_smoke_evaluation() -> None:
    evaluate = _load_evaluation_function("run_base_smoke_evaluation")
    evaluate()

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
    print(f"    TOP_K      : {settings.top_k}")
    print(f"    Min Score  : {settings.min_retrieval_score}")

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
) -> dict[str, Any]:
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
        return agentic_answer(query, **kwargs)

    hits = retrieve(rag, meta_store, query=query)
    kwargs = {"hits": hits, "stream": stream}
    if on_generation is not None:
        kwargs["on_generation"] = on_generation
    result = rag_answer(query, **kwargs)
    result["retrieved_evidence"] = hits
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

            if query == "":
                continue

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
                result = run_query(
                    rag,
                    meta_store,
                    query=query,
                    agentic=agentic,
                    stream=stream_response,
                    on_progress=_render_agentic_progress if agentic else None,
                )
            except AgenticWorkflowError as exc:
                logger.error(f"Agentic workflow error: {exc}")
                print(f"\nAgentic workflow error: {exc}", file=sys.stderr)
                print()
                continue

            display_ansewr(result["answer"])

            if citations and result["citations"]:
                show_citations(result, show_full=True)

            print()
    finally :
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
        "--eval-base",
        action="store_true",
        help="Run the legacy base.json system smoke evaluation on the user index",
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

    if args.eval:
        try:
            run_evaluation(agentic=args.agentic)
        except EvaluationUnavailableError as exc:
            parser.error(str(exc))
        return
    if args.eval_base:
        try:
            run_base_smoke_evaluation()
        except EvaluationUnavailableError as exc:
            parser.error(str(exc))
        return

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
        f"{' --eval-base' if args.eval_base else ''}"
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
        if query:
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

    elif query and not args.repl:
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
        or args.eval_base
    ):
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
