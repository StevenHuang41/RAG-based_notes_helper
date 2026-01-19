# Notes Helper

## System Identity

- Who are you: I am Notes Helper, your personal assistant for managing and searching your notes
- What is Notes Helper: A RAG-based tool designed to help you reason across your personal data directory
- Who am I: You are the Owner of these notes
- Your data: All information I have comes from the files in your local data directory

## What Can Notes Helper Do

- answer questions based on notes in `data/`
- summarize concepts using the most relevant chunks
- help reviewing and recalling topics from notes
- show what files were used to generate the answer

**must not**:

- invent information not present in the notes
- use external or prior knowledge
- hallucinate answers beyond retrieved content

---

## How To Use Notes Helper

### Asking Questions

* type your question after '> '

    - Example:

    ```
    What is xxx
    Who are you
    How can you help me
    ```

* Notes helper will:

    1. search relevant content in `data/`
    2. retrieve relevant chunks
    3. generate a readable answer
    4. show citations to the source notes

### Commands:

* `--help` or `-h`
    - Show help message

* `--repl`
    - Run in REPL mode

* `--reindex` or `-r`
    - Process all files in data/ to rebuild rag index

* `--update` or `-u`
    - Only process files that changed its content, faster than `--reindex` if only few changes

* `--citations` or `-ci`
    - Toggle citations display

* `--sources` or `-so`
    - Show indexed source files

* `--config` or `-co`
    - Show configuration


#### REPL mode
Ask multiple questions interactively, reindex notes without restarting,
inspect sources, toggle citations
The index and embedding model are loaded once and reused during the session
for faster interaction

- `:quit`       /   `:q`      (exit)
- `:help`       /   `:h`      (show instructions)
- `:reindex`    /   `:ri`     (reindex all files without exiting)
- `:update`     /   `:u`      (update only changed files)
- `:citations`  /   `:ci`     (toggle citations display)
- `:sources`    /   `:so`     (show indexed files)
- `:config`     /   `:co`     (check configuration)

You do **NOT** need to restart the system to update or reindex

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
  Knowledge updates via reindexing, without retraining models

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

Notes Helper is a tool for querying your notes, not a general-purpose AI assistant
If relevant information cannot be found in the notes, it will respond accordingly without guessing

## Tips For Better Answer

- write clear, rich, descriptive notes
- keep one concept per question
- based on experience, avoid using question mark '?'
- run `:reindex` after adding or updating notes
- change the configuration in `.env` according to your needs
