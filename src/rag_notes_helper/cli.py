import argparse
from logging import currentframe
import time
from pydantic import ValidationError
import sys

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag.ingest import get_changed_doc_ids, get_stable_doc_id, load_notes
from rag_notes_helper.rag.index import (
    RagIndex,
    build_index,
    save_index,
    load_index,
    smart_rebuild,
)
from rag_notes_helper.rag.loaders import is_text_file
from rag_notes_helper.rag.meta_store import MetaStore
from rag_notes_helper.rag.retrieval import retrieve
from rag_notes_helper.rag.answer import rag_answer
from rag_notes_helper.utils.logger import get_logger
from rag_notes_helper.utils.timer import LapTimer


logger = get_logger("cli")

def build_or_load_index():
    logger.info("----build_or_load_index start----")
    timer = LapTimer()

    try:
        logger.info("index exists:")

        timer.start()
        rag = load_index()
        logger.info(f"load_index latency={timer.lap():.2f} ms")

        logger.info("----build_or_load_index end----")
        return rag

    except FileNotFoundError:
        logger.info("index not exists:")
        print("\nIndex not found. Building index from notes ...")

        timer.start()
        chunks = load_notes()
        logger.info(f"load_notes latency={timer.lap():.2f} ms")

        rag = build_index(chunks)
        logger.info(f"build_index latency={timer.lap():.2f} ms")

        save_index(rag)
        logger.info(f"save_index latency={timer.lap():.2f} ms")

        print("Index built and saved.\n")
        logger.info("----build_or_load_index end----")
        return rag

def rebuild_index():
    logger.info("----rebuild_index start----")
    print("\nRebuilding index from notes ...")

    timer = LapTimer()

    # 1. get current files' hash value
    try :
        with MetaStore() as meta_store:
            timer.start()
            old_doc_ids = meta_store.get_all_doc_id()
            logger.info(f"get_all_doc_id latency={timer.lap():.2f} ms")
    except Exception:
        logger.info("No existing meta, rebuilding full index")
        old_doc_ids = set()

    timer.start()
    # 2. get the changed and unchanged file
    changed_ids, unchanged_ids = get_changed_doc_ids(old_doc_ids)
    logger.info(f"get_changed_doc_ids latency={timer.lap():.2f} ms")

    if changed_ids:
        timer.start()
        rag = smart_rebuild(changed_ids, unchanged_ids)
        logger.info(f"smart_rebuild latency={timer.lap():.2f} ms")

        save_index(rag)
        logger.info(f"save_index latency={timer.lap():.2f} ms")

        print("Index built and saved.\n")
    else :
        rag = load_index()
        print("Index is already up to date\n")

    logger.info("----rebuild_index end----")
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
    logger.info("----run_onetime start----")
    logger.info(f"reindex: {reindex}")


    timer = LapTimer()
    rag = rebuild_index() if reindex else build_or_load_index()
    logger.info(f"building rag latency={timer.lap():.2f} ms")

    with MetaStore() as meta_store:
        logger.info(f"load meta latency={timer.lap():.2f} ms")

        logger.info((f"query: {query[:20]}{' ...' if len(query) > 20 else ''}"))
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

    logger.info("----run_onetime end----")

def repl(*, show_citations: bool):
    logger.info("----repl start----")

    timer = LapTimer()
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
                logger.info(f"[input]: {query}")
            except KeyboardInterrupt:
                print("\nBye~")
                logger.info("----repl end----")
                break

            if query == "":
                continue

            if query in {":quit", ":q"}:
                print("\nBye~")
                logger.info("----repl end----")
                break

            if query in {":reindex", ":ri"}:
                timer.start()
                meta_store.close()
                logger.info((f"close meta latency={timer.lap()} ms"))

                rag = rebuild_index()
                logger.info((f"rebuild_index latency={timer.lap()} ms"))

                meta_store = MetaStore()
                logger.info((f"reload meta latency={timer.lap()} ms"))
                continue

            if query in {":sources", ":so"}:
                print("\nSOURCES:\n")

                timer.start()
                for s in meta_store.list_indexed_sources():
                    print(f"- {s}")

                logger.info((f"list_indexed_sources latency={timer.lap()} ms"))

                print()
                continue

            if query in {":config", ":co"}:
                timer.start()
                show_config()
                logger.info(f"show_config latency={timer.lap()} ms")
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

            logger.info((f"query: {query[:20]}{' ...' if len(query) > 20 else ''}"))

            timer.start()
            hits = retrieve(query=query, rag=rag, meta_store=meta_store)
            logger.info((f"retrieve latency={timer.lap()} ms"))

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
    logger.info(
        f"[input]: rag-app{f' {query}' if query else ''}"
        f"{' --repl' if args.repl else ''}"
        f"{' --config' if args.config else ''}"
        f"{' --reindex' if args.reindex else ''}"
        f"{' --citations' if args.citations else ''}"
    )

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
