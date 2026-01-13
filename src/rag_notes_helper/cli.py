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
        logger.info("index exists")

        logger.info(f"load_index{timer.start()}")
        rag = load_index()
        logger.info(f"latency={timer.lap():.2f} ms")
        return rag

    except FileNotFoundError:
        logger.info("index not exists")

        print("\nIndex not found. Building index from notes ...")

        logger.info(f"load_index{timer.start()}")
        chunks = load_notes()
        logger.info(f"latency={timer.lap():.2f} ms")

        rag = build_index(chunks)
        logger.info(f"build_index latency={timer.lap():.2f} ms")

        save_index(rag)
        logger.info(f"save_index latency={timer.lap():.2f} ms")

        print("Index built and saved.\n")
        return rag

def rebuild_index():
    timer = LapTimer()
    logger.info(f"----rebuild_index----{timer.start()}")

    print("\nRebuilding index from notes ...")

    logger.info(f"load_notes{timer.start()}")
    chunks = load_notes()
    logger.info(f"latency={timer.lap():.2f} ms")

    rag = build_index(chunks)
    logger.info(f"build_index latency={timer.lap():.2f} ms")

    save_index(rag)
    logger.info(f"save_index latency={timer.lap():.2f} ms")

    print("Index built and saved.\n")
    logger.info(f"----rebuild_index latency={timer.lap()} ms----")
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
    timer = LapTimer()
    logger.info(f"----run_onetime----{timer.start()}")
    logger.info(f"reindex: {reindex}")


    logger.info(f"building rag{timer.start()}")
    rag = rebuild_index() if reindex else build_or_load_index()
    logger.info(f"latency={timer.lap():.2f} ms")

    with MetaStore() as meta_store:
        logger.info(f"load meta latency={timer.lap():.2f} ms")

        logger.info((f"query: {query[:10]}{' ...' if len(query) > 10 else ''}"))
        hits = retrieve(query=query, rag=rag, meta_store=meta_store)
        logger.info(f"retrieve latency={timer.lap():.2f} ms")

        result = rag_answer(query=query, hits=hits)
        logger.info(f"generate answer latency={timer.lap():.2f} ms")

    print("\nANSWER:\n")
    print(result["answer"])

    if show_citations and result['citations']:
        print("\nCITATIONS:")
        print("- ", end="")
        source_set = {c['source'] for c in result["citations"]}
        print(", ".join(source_set))

    logger.info(f"----run_onetime latency={timer.lap()} ms----")

def repl(*, show_citations: bool):
    timer = LapTimer()
    logger.info(f"----repl----{timer.start()}")

    rag = build_or_load_index()
    logger.info(f"load rag latency={timer.lap():.2f} ms")

    meta_store = MetaStore()
    logger.info(f"load meta latency={timer.lap():.2f} ms")

    print(
        "\nRAG-based Notes Helper\n"
        "Type your question.\n"
    )

    try :
        while True:
            try :
                query = input("> ").strip()
            except KeyboardInterrupt:
                print("\nBye~")
                break

            if query == "":
                continue

            if query in {":quit", ":q"}:
                print("\nBye~")
                break

            if query in {":reindex", ":ri"}:
                logger.info((f"close meta{timer.start()}"))
                meta_store.close()
                logger.info((f"latency={timer.lap()} ms"))

                rag = rebuild_index()
                logger.info((f"rebuild_index latency={timer.lap()} ms"))

                meta_store = MetaStore()
                logger.info((f"reload meta latency={timer.lap()} ms"))
                continue

            if query in {":sources", ":so"}:
                print("\nSOURCES:\n")

                logger.info((f"list_indexed_sources{timer.start()}"))
                for s in meta_store.list_indexed_sources():
                    print(f"- {s}")

                logger.info((f"latency={timer.lap()} ms"))

                print()
                continue

            if query in {":config", ":co"}:
                logger.info(f"show_config{timer.start()}")
                show_config()
                logger.info(f"latency={timer.lap()} ms")
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

            logger.info((f"query: {query[:10]}{' ...' if len(query) > 10 else ''}"))
            logger.info((f"retrieve{timer.start()}"))
            hits = retrieve(query=query, rag=rag, meta_store=meta_store)
            logger.info((f"latency={timer.lap()} ms"))

            result = rag_answer(query=query, hits=hits)
            logger.info((f"generate answer latency={timer.lap()} ms"))

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

    logger.info(f"----repl latency={timer.lap()} ms----")



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
    parser = build_parser()
    args = parser.parse_args()
    query = (" ".join(args.query)).strip()

    if query:
        # > 'rag-app --config [query]'
        if args.config:
            show_config()

        # > 'rag-app [query]'
        run_onetime(
            query=query,
            reindex=args.reindex,
            show_citations=args.citations,
        )
    else :
        # > 'rag-app --config'
        if args.config:
            show_config()
            return

        # > 'rag-app --reindex'
        if args.reindex:
            rebuild_index()
            return

        # > 'rag-app' or > 'rag-app --repl'
        repl(show_citations=args.citations)


if __name__ == "__main__":
    main()
