import argparse
from pydantic import ValidationError
import sys

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


def build_or_load_index():
    try:
        return load_index()
    except FileNotFoundError:
        print("\nIndex not found. Building index from notes ...")
        chunks = load_notes()
        rag = build_index(chunks)
        save_index(rag)
        print("Index built and saved.\n")
        return rag

def rebuild_index():
    print("\nRebuilding index from notes ...")
    chunks = load_notes()
    rag = build_index(chunks)
    save_index(rag)
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
    rag = rebuild_index() if reindex else build_or_load_index()

    with MetaStore() as meta_store:
        hits = retrieve(query=query, rag=rag, meta_store=meta_store)
        result = rag_answer(query=query, hits=hits)

    print("\nANSWER:\n")
    print(result["answer"])

    if show_citations and result['citations']:
        print("\nCITATIONS:")
        print("- ", end="")
        source_set = {c['source'] for c in result["citations"]}
        print(", ".join(source_set))

def repl():
    rag = build_or_load_index()
    meta_store = MetaStore()

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
                meta_store.close()
                rag = rebuild_index()
                meta_store = MetaStore()
                continue

            if query in {":sources", ":so"}:
                print("\nSOURCES:\n")
                for s in meta_store.list_indexed_sources():
                    print(f"- {s}")

                print()
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
                )
                continue

            hits = retrieve(query=query, rag=rag, meta_store=meta_store)
            result = rag_answer(query=query, hits=hits)

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

    run_onetime(
        query=query,
        reindex=args.reindex,
        show_citations=args.citations,
    )


if __name__ == "__main__":
    main()
