import argparse

from rag_notes_helper.rag.ingest import load_notes
from rag_notes_helper.rag.index import (
    build_index,
    save_index,
    load_index,
    list_indexed_sources,
)
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
    print("Index rebuilt.\n")
    return rag

def run_onetime(
    query: str,
    *,
    reindex: bool,
    show_citations: bool
) -> None:
    rag = rebuild_index() if reindex else build_or_load_index()

    hits = retrieve(query=query, rag=rag)
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

    print(
        "\nRAG-based Notes Helper\n"
        "Type your question.\n"
    )

    show_citations = False

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
            rag = rebuild_index()
            continue

        if query in {":sources", ":so"}:
            print("\nSOURCES:\n")
            for s in list_indexed_sources(rag):
                print(f"- {s}")

            print()
            continue

        if query in {":citations", ":ci"}:
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


        hits = retrieve(query=query, rag=rag)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-app",
        description="RAG-based Notes Helper",
    )

    parser.add_argument(
        "query",
        nargs="*",
        help="Question to ask RAG, or command (repl, reindex)",
    )

    parser.add_argument(
        "-r", "--reindex",
        action="store_true",
        help="Rebuild index before generating answers",
    )

    parser.add_argument(
        "-c", "--citations",
        action="store_true",
        help="Show citation files"
    )

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    if len(args.query) == 1:
        cmd = args.query[0]

        if cmd == "repl":
            repl()
            return

        if cmd == "reindex":
            rebuild_index()
            return

    query = " ".join(args.query)

    run_onetime(
        query=query,
        reindex=args.reindex,
        show_citations=args.citations,
    )

