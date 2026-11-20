# Notes Helper

## What Is Notes Helper?

Notes Helper is a local-first Retrieval-Augmented Generation (RAG) tool for
asking questions about notes in `data/`. It retrieves relevant passages and uses
a configurable LLM to generate an answer grounded in those passages.

You can use Notes Helper from the CLI or an interactive REPL. A local read-only
MCP search tool also lets a connected assistant search the notes and inspect the
returned evidence.

## What Can Notes Helper Do?

- Answer questions and help summarize information found in notes.
- Read `.md`, `.txt`, `.pdf`, and `.py` files in the notes directory.
- Show source citations with CLI answers when citations are enabled.
- Return ranked note chunks and source metadata through the MCP search tool.
- Indicate when retrieved evidence is insufficient, though results depend on
  retrieval and the configured model.

## Classic and Agentic RAG

**Classic RAG** runs one retrieval search and generates an answer from the
selected context. If retrieval returns no chunks, it returns a no-evidence
response without calling the answer model.

**Agentic RAG** can review the retrieved evidence, perform additional searches,
and review answer candidates. Search attempts and answer attempts have separate
configured limits, each including the first attempt. Answer review uses the
selected generation context; it does not reopen retrieval.

## How Retrieval Works

1. Notes are split into overlapping chunks. The chunks and their embeddings are
   stored in an index with source metadata.
2. FAISS Vector search and BM25 independently find candidate chunks for a query.
3. Reciprocal Rank Fusion (RRF) combines the candidate rankings, and a
   CrossEncoder reranks the fused results.
4. Selected chunks are provided as context to the LLM for answer generation.

Updating an index makes changed notes available to retrieval without fine-tuning
the LLM. The selected context may contain only part of a note or miss relevant
information.

## Using and Updating Notes

Ask a question or start a REPL from the project directory:

```bash
uv run rag-app "What can Notes Helper do?" --citations
uv run rag-app "What can Notes Helper do?" --agentic --citations
uv run rag-app --repl
```

Add or edit notes in `data/`, then update the existing index before asking about
the changes:

```bash
uv run rag-app --update
```

Use `uv run rag-app --reindex` to rebuild the index from scratch. In the REPL,
use `:update` (or `:ud`) to update changed notes and `:reindex` (or `:ri`) to
rebuild without restarting the session.

Other REPL commands include `:citations` to toggle source display, `:sources`
to list indexed files, `:config` to inspect settings, `:stream` to toggle
streaming, `:help` for commands, and `:quit` to exit.

The development evaluation commands `uv run rag-app --eval` and
`uv run rag-app --agentic --eval` use a separate reviewed evaluation corpus and
temporary index; they do not evaluate the user's `data/` notes or persistent
`storage/` index. See the project README for setup and details.

## Limitations and Privacy

Retrieval can miss relevant passages, and an LLM can produce incomplete or
incorrect answers even when instructed to use the notes as evidence. Check the
cited sources and the original notes when accuracy matters.

A remote LLM provider receives selected note passages as part of the prompt.
The local MCP search tool returns note content to its connected client; that
client may send the content to its model provider. Consider this before adding
sensitive information to your notes.

## Test Your Notes

1. Add a short fact under **My test note** below and save this file. For example,
   write a project name and its start date.
2. Run `uv run rag-app --update`, or enter `:update` in the REPL.
3. Ask when your project starts, using the name you wrote. Use `--citations` on
   the CLI or enable `:citations` in the REPL to inspect the source.
4. Change the fact, save the file, update the index again, and repeat the
   question. Check the cited note if the answer is missing or incorrect.

### My test note

Add your own fact here.
