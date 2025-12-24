from faiss import bucket_sort
from numpy import add
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


def main():
    rag = build_or_load_index()

    print(
        "\nRAG-based Notes Helper\n"
        "Type your question. Type 'exit' or 'quit' to leave.\n"
    )

    while True:
        try :
            query = input("> ").strip()

            if not query:
                continue

            if query.lower() in {":quit", ":q"}:
                print("Bye~")
                break

            if query in {":reindex", ":ri"}:
                rag = rebuild_index()
                continue

            if query in {":source", ":so"}:
                print("\nSOURCES:\n")
                for s in list_indexed_sources(rag):
                    print(f"- {s}")

                print()
                continue


            hits = retrieve(query=query, rag=rag)
            result = rag_answer(query=query, hits=hits)

            print("\nANSWER:\n")
            print(result["answer"])

            ## version 1:
            # if result["citations"]:
            #     print("\nCITATIONS:")
            #     for c in result["citations"]:
            #         print(
            #             f"- {c['source']} "
            #             f"(chunk {c['chunk_id']}, score={c['score']:.3f})"
            #         )

            ## version 2:
            if result["citations"]:
                print("\nCITATIONS:")
                print("- ", end="")

                source_set = {c['source'] for c in result["citations"]}
                print(", ".join(source_set))

            print()

        except KeyboardInterrupt:
            print("\nBye~")
            break


if __name__ == "__main__":
    main()

