import argparse
from pydantic import ValidationError
import sys

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag.index import (
    RagIndex,
    load_or_build_index,
    rebuild_index,
)
from rag_notes_helper.rag.meta_store import MetaStore
from rag_notes_helper.rag.retrieval import retrieve
from rag_notes_helper.rag.answer import rag_answer
from rag_notes_helper.utils.logger import get_logger
from rag_notes_helper.utils.timer import LapTimer


logger = get_logger("cli")

def show_config():
    logger.info("show_config")
    timer = LapTimer()
    print("\nChecking Configuration ...")
    try :
        settings = get_settings()
    except ValidationError as e:
        print("Configuration is INVALID\n")
        print(e)
        logger.info(f"show_config latency={timer.lap():.2f} ms")
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

    print("\nPaths:")
    print(f"    Notes dir  : {settings.notes_dir}")
    print(f"    Storage dir: {settings.storage_dir}")

    print("\nConfig check completed")
    logger.info(f"show_config latency={timer.lap():.2f} ms")


def show_sources(meta_store: MetaStore):
    print("\nSOURCES:\n")

    timer = LapTimer()
    for s in meta_store.list_indexed_sources():
        print(f"- {s}")

    logger.info(f"list_indexed_sources latency={timer.lap():.2f} ms")

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
        source_set = {c['source'] for c in result["citations"]}
        print(", ".join(source_set))


def run_onetime(
    rag: RagIndex,
    meta_store: MetaStore,
    *,
    query: str,
    citations: bool = False,
) -> None:
    timer = LapTimer()


    hits = retrieve(rag, meta_store, query=query)
    logger.info(f"retrieve latency={timer.lap():.2f} ms")

    logger.info((f"query: {query[:20]}{' ...' if len(query) > 20 else ''}"))
    timer.start()
    result = rag_answer(query, hits=hits)
    logger.info(f"rag_answer latency={timer.lap():.2f} ms")

    print("\nANSWER:\n")
    print(result["answer"])

    if citations and result['citations']:
        show_citations(result)

def repl(
    rag: RagIndex | None = None,
    meta_store: MetaStore | None = None,
    *,
    citations: bool = False,
):
    timer = LapTimer()
    rag = load_or_build_index()
    logger.info(f"load rag latency={timer.lap():.2f} ms")

    meta_store = MetaStore()
    logger.info(f"load MetaStore latency={timer.lap():.2f} ms")

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

            if query in {":quit", ":q"}:
                print("\nBye~")
                break

            if query in {":update", ":u", ":reindex", ":ri"}:
                timer.start()
                meta_store.close()
                logger.info(f"close MetaStore latency={timer.lap():.2f} ms")

                do_force = query in {":reindex", ":ri"}
                rag = rebuild_index(force=do_force)
                logger.info(
                    f"rebuild_index{'(force)' if do_force else ''} "
                    f"latency={timer.lap():.2f} ms"
                )

                meta_store = MetaStore()
                logger.info(f"reload meta latency={timer.lap():.2f} ms")
                print()
                continue

            if query in {":sources", ":so"}:
                show_sources(meta_store)
                print()
                continue

            if query in {":config", ":co"}:
                timer.start()
                show_config()
                logger.info(f"show_config latency={timer.lap()} ms")
                print()
                continue

            if query in {":citations", ":ci"}:
                if citations:
                    print("\n/Hide Citations\n")
                else :
                    print("\n/Show Citations\n")

                citations = not citations
                continue

            if query in {":help", ":h"}:
                print(
                    "\nCommands:\n"
                        "   :quit      or  :q    -> exit app\n"
                        "   :help      or  :h    -> show instructions\n"
                        "   :update    or  :u    -> update index\n"
                        "   :reindex   or  :ri   -> reindex rag\n"
                        "   :citations or  :ci   -> show citation files\n"
                        "   :sources   or  :so   -> show all source files\n"
                        "   :config    or  :co   -> check configuration\n"
                )
                continue

            logger.info((f"query: {query[:20]}{' ...' if len(query) > 20 else ''}"))

            timer.start()
            hits = retrieve(rag, meta_store, query=query)
            logger.info((f"retrieve latency={timer.lap()} ms"))

            result = rag_answer(query, hits=hits)
            logger.info((f"rag_answer latency={timer.lap()} ms"))

            print("\nANSWER:\n")
            print(result["answer"])

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
        help="Question to ask RAG (default mode)",
    )

    parser.add_argument(
        "--repl",
        action="store_true",
        help="Start interactive REPL",
    )

    parser.add_argument(
        "-u", "--update",
        action="store_true",
        help="Update index",
    )

    parser.add_argument(
        "-r", "--reindex",
        action="store_true",
        help="Rebuild index",
    )

    parser.add_argument(
        "-co", "--config",
        action="store_true",
        help="Check configuration",
    )

    parser.add_argument(
        "-ci", "--citations",
        action="store_true",
        help="Show citations",
    )

    parser.add_argument(
        "-so", "--sources",
        action="store_true",
        help="Show source files",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    query = (" ".join(args.query)).strip()
    logger.info(
        f"[input]: rag-app{f' {query}' if query else ''}"
        f"{' --repl' if args.repl else ''}"
        f"{' --config' if args.config else ''}"
        f"{' --update' if args.update else ''}"
        f"{' --reindex' if args.reindex else ''}"
        f"{' --citations' if args.citations else ''}"
        f"{' --sources' if args.sources else ''}"
    )
    logger.info(f"config: {get_settings().model_dump_json()}")

    timer = LapTimer()
    rag = rebuild_index(force=args.reindex) \
          if args.update or args.reindex else load_or_build_index()
    logger.info(f"main: load rag latency={timer.lap():.2f} ms")

    meta_store = MetaStore()
    logger.info(f"main: load meta_store latency={timer.lap():.2f} ms")

    if args.config:
        show_config()
        logger.info(f"main: show_config latency={timer.lap():.2f} ms")

    if args.sources:
        show_sources(meta_store)
        logger.info(f"main: show_sources latency={timer.lap():.2f} ms")

    if args.repl:
        logger.info("==== REPL start ====")
        if query:
            run_onetime(
                rag,
                meta_store,
                query=query,
                citations=args.citations,
            )

        repl(
            rag,
            meta_store,
            citations=args.citations,
        )
        logger.info("==== REPL end ====")

    elif query and not args.repl:
        logger.info("==== run_onetime start ====")
        timer.start()
        run_onetime(
            query=query,
            citations=args.citations,
            rag=rag,
            meta_store=meta_store,
        )
        logger.info("==== run_onetime end ====")
        logger.info(f"main: run_onetime latency={timer.lap():.2f} ms")

    elif not (args.config or args.sources):
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
