import argparse
from inspect import getlineno
from pydantic import ValidationError
import sys
import time

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag.ingest import load_notes
from rag_notes_helper.rag.index import (
    build_index,
    save_index,
    load_index,
)
from rag_notes_helper.rag.meta_store import MetaStore
from rag_notes_helper.rag.retrieval import retrieve
from rag_notes_helper.rag.answer import rag_answer
from rag_notes_helper.utils.logger import get_logger
from rag_notes_helper.utils.timer import LapTimer


logger = get_logger("cli")

def build_or_load_index():
    timer = LapTimer()

    try:
        logger.info(f"load existing index{timer.start()}")
        rag = load_index()
        logger.info(f"load existing index latency={timer.lap():.2f} ms")
        return rag

    except FileNotFoundError:
        logger.info("no existing index")

        print("\nIndex not found. Building index from notes ...")
        chunks = load_notes()
        logger.info(f"load notes latency={timer.lap():.2f} ms")

        rag = build_index(chunks)
        logger.info(f"build index latency={timer.lap():.2f} ms")

        save_index(rag)
        logger.info(f"save index latency={timer.lap():.2f} ms")

        print("Index built and saved.\n")
        return rag

def rebuild_index():
    timer = LapTimer()

    print("\nRebuilding index from notes ...")

    logger.info(f"load notes{timer.start()}")
    chunks = load_notes()
    logger.info(f"load notes latency={timer.lap():.2f} ms")

    rag = build_index(chunks)
    logger.info(f"build index latency={timer.lap():.2f} ms")

    save_index(rag)
    logger.info(f"save index latency={timer.lap():.2f} ms")

    print("Index built and saved.\n")
    return rag

def show_config():
    print("\nChecking Configuration ...\n")
    try :
        settings = get_settings()
    except ValidationError as e:
        print("Configuration is INVALID\n")
        print(e)
        sys.exit(1)

    print("Configuration is VALID")

    print("\nLLM:")
    print(f"    Provider   : {settings.LLM_PROVIDER}")
    print(f"    Model      : {settings.LLM_MODEL}")
    print(
          f"    API Key    : "
          f"{'SET' if settings.LLM_API_KEY else 'NOT SET'}"
    )

    print("\nEmbedding:")
    print(f"    Model      : {settings.EMBEDDING_MODEL}")

    print("\nChunking:")
    print(f"    Size       : {settings.CHUNK_SIZE}")
    print(f"    Overlap    : {settings.CHUNK_OVERLAP}")

    print("\nRetrieval:")
    print(f"    TOP_K      : {settings.TOP_K}")
    print(f"    Min Score  : {settings.MIN_RETRIEVAL_SCORE}")

    print("\nPaths:")
    print(f"    Notes dir  : {settings.NOTES_DIR}")
    print(f"    Storage dir: {settings.STORAGE_DIR}")

    print("\nConfig check completed.\n")

def run_onetime(
    query: str,
    *,
    reindex: bool,
    show_citations: bool
) -> None:
    logger.info(f"mode: onetime query_start reindex: {reindex} qeury: {query[:20]}")

    timer = LapTimer()
    rag = rebuild_index() if reindex else build_or_load_index()
    logger.info(f"mode: onetime load rag latency={timer.lap():.2f} ms")

    with MetaStore() as meta_store:


        logger.info((f"mode: onetime retrieve{timer.start()}"))
        hits = retrieve(query=query, rag=rag, meta_store=meta_store)
        logger.info(f"mode: onetime retrieve latency={timer.lap():.2f} ms")

        result = rag_answer(query=query, hits=hits)
        logger.info(f"mode: onetime generate answer latency={timer.lap():.2f} ms")

    print("\nANSWER:\n")
    print(result["answer"])

    if show_citations and result['citations']:
        print("\nCITATIONS:")
        print("- ", end="")
        source_set = {c['source'] for c in result["citations"]}
        print(", ".join(source_set))

def repl():
    logger.info("mode: repl")

    timer = LapTimer()
    rag = build_or_load_index()
    logger.info(f"mode: repl load rag latency={timer.lap():.2f} ms")

    meta_store = MetaStore()
    logger.info(f"mode: repl load meta latency={timer.lap():.2f} ms")

    print(
        "\nRAG-based Notes Helper\n"
        "Type your question.\n"
    )

    show_citations = False

    try :
        while True:
            try :
                query = input("> ").strip()
            except KeyboardInterrupt:
                print("\nBye~")
                break

            if not query:
                continue

            if query in {":quit", ":q"}:
                print("\nBye~")
                break

            if query in {":reindex", ":ri"}:
                logger.info((f"mode: repl close meta{timer.start()}"))
                meta_store.close()
                logger.info((f"mode: repl close meta latency={timer.lap()} ms"))

                rag = rebuild_index()
                logger.info((f"mode: repl rebuild index latency={timer.lap()} ms"))

                meta_store = MetaStore()
                logger.info((f"mode: repl reload meta latency={timer.lap()} ms"))
                continue

            if query in {":sources", ":so"}:
                print("\nSOURCES:\n")

                logger.info((f"mode: repl list_indexed_sources{timer.start()}"))
                for s in meta_store.list_indexed_sources():
                    print(f"- {s}")

                logger.info((f"mode: repl list_indexed_sources latency={timer.lap()} ms"))

                print()
                continue

            if query in {":config", ":co"}:
                show_config()
                continue

            if query in {":citations", ":ci"}:
                if show_citations:
                    print("\n/Hide Citations\n")
                else :
                    print("\n/Show Citations\n")

                show_citations = not show_citations
                continue

            if query in {":help", ":h"}:
                print(
                    "\nCommands:\n"
                        "   :quit      or  :q    -> exit app\n"
                        "   :help      or  :h    -> show instructions\n"
                        "   :reindex   or  :ri   -> reindex rag\n"
                        "   :citations or  :ci   -> show citation files\n"
                        "   :sources   or  :so   -> show all source files\n"
                        "   :config    or  :co   -> check configuration\n"
                )
                continue

            logger.info((f"mode: repl query_start  qeury: {query[:20]}{timer.start()}"))
            hits = retrieve(query=query, rag=rag, meta_store=meta_store)
            logger.info((f"mode: repl retrieve latency={timer.lap()} ms"))

            result = rag_answer(query=query, hits=hits)
            logger.info((f"mode: repl generate answer latency={timer.lap()} ms"))

            print("\nANSWER:\n")
            print(result["answer"])

            if show_citations and result["citations"]:
                print("\nCITATIONS:")
                for c in result["citations"]:
                    print(
                        f"- {c['source']} "
                            f"(chunk={c['chunk_id']}, score={c['score']:.3f})"
                    )

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
        "-r", "--reindex",
        action="store_true",
        help="Rebuild index",
    )

    parser.add_argument(
        "--config",
        action="store_true",
        help="Check configuration",
    )

    parser.add_argument(
        "-c", "--citations",
        action="store_true",
        help="Show citations",
    )

    return parser


def main():
    main_timer = LapTimer()
    parser = build_parser()
    logger.info(f"main build_parser latency={main_timer.lap()} ms")

    args = parser.parse_args()
    logger.info(f"main parse_args latency={main_timer.lap()} ms")

    if args.config:
        show_config()
        return

    if args.reindex and not args.query:
        rebuild_index()
        return

    if args.repl or (not args.query and not args.reindex):
        repl()
        return

    query = " ".join(args.query)

    logger.info(f"main run_onetime{main_timer.start()}")
    run_onetime(
        query=query,
        reindex=args.reindex,
        show_citations=args.citations,
    )
    logger.info(f"main run_onetime latency={main_timer.lap()} ms")


if __name__ == "__main__":
    main()
