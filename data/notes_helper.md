# Notes Helper

## What Can Notes Helper Do

- answer questions based on notes in `data/`
- summarize conepts using the most relevant chunks selected by faiss
- help reviewing and recalling topics from notes
- show what files were used to generate the answer

**must not**:

- invent information
- use external knowledge
- halluciante answers

---

## How To Use Notes Helper

### Asking Questions

* type your question after '> '

    - Example:

    ```
    Who are you?
    How can you help me?
    What is xxx?
    ```

* Notes helper will:

    1. search relevant content in `data/`
    2. retrieve relevant chunks
    3. generate a readable answer
    4. show citations to the source notes

### Commands

#### one time mode

Answer a single question, optionally reindex or show citations

* `[query]`
    RAG generates answer as usual

* `[query] -r`
    RAG reindex before generating answer

* `[query] -c`
    Show citations file with answer

#### REPL mode

Ask multiple questions interactively, reindex nots without restarting,
inspect sources, toggle citations

* `:quit` or `:q`
    Exit Notes Helper

* `:reindex` or `:ri`
    Rebuild the index to include new or updated notes in `data/`

* `:sources` or `:so`
    Show the indexed files in RAG

* `:citations` or `:ci`
    Toggle citation files in RAG

You do **NOT** need to restart the program to reindex

See CLI usage in README.md

---

## How It Works (RAG Overview)

Notes Helper uses a standard RAG pipeline:

1. Notes in `data/` are read using **memory-safe streaming loaders**

2. Text is split into **overlapping chunks**

3. Each chunk is embedded using a **Sentence Transformer**

4. Embeddings are indexed with **Faiss**

5. User queries are embedded using the same model

6. Only the **top-k relevant chunks** are retrieved

7. LLM generates an answer **strictly from those chunks**

- The LLM never sees:

    - The full notes
    - Unrelated chunks
    - Any external knowledge

---

## Design Principles

- **Source of truth stays external**
  Notes remain the only knowledge base

- **No fine-tuning**
  Knowledge updates without retraining

- **Memory safety**
  Large note files are processed line-by-line

- **Transparency**
  Answers can be traced back to source files

---

## Limitations

- Notes Helper cannot answer questions outside the stored notes
- Answers depend on the quality and completeness of the notes
- The system may return short or conservative answer if the notes are short

---

## Reminder

Notes Helper is a tool for querying your notes, not an general-purpose ai assistant.
If query cannot be found in notes, it will reuturn nothing


## Tips For Better Answer

- write clear, rich, descriptive notes
- keep one concept per question
- run `:reindex` after adding or updating notes
- change the configuration in `.env` according to your needs
