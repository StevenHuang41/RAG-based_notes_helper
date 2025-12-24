# RAG-based Notes Helper

A Retrieval-Augmented AI assistant for querying and reviewing your own notes using local embeddings and LLMs.

The system indexes plain text files, retrieves relevant content via vector similarity, and generates answers using LLM.

---

## Features

- Retrieval-Augmented Generation (RAG) over notes
- Streaming text chunking (memory-safe for large files)
- Faiss-based dense vector retrieval
- Pluggable LLM backend (Hugging Face)
- Source-aware answers with citation
- Interactive CLI with live re-indexing

---

## Architecture

Notes (`data/`)

Text loaders (memory-safe)

Streaming chunking

Sentence-Transformer embedding

faiss vector index

Query embedding

Top-k retrieval

LLM

---

## Project Structure

.
├── ask.py                          # entry point
├── src/
│   └── rag_notes_helper/
│       ├── core/
│       │   └── config.py
│       └── rag/
│           ├── answer.py
│           ├── chunking.py
│           ├── index.py
│           ├── ingest.py
│           ├── retrieval.py
│           ├── loaders.py
│           └── llm/
│               ├── hf.py
│               ├── openai_api.py
│               └── router.py
├── tests/                          # unit tests
│   ├── test_answer.py
│   ├── test_chunking.py
│   ├── test_index.py
│   ├── test_ingest.py
│   ├── test_loaders.py
│   └── test_retrieval.py
├── data/                           # user notes
│   ├── EDA.md
│   ├── ml_flows.md
│   ├── notes_helper.md
│   └── workflow
├── storage/                        # faiss index + metadata
│   ├── faiss.index
│   └── meta.jsonl
├── LICENSE
├── pytest.ini
├── pyproject.toml
└── README.md


## How It Works (RAG Pipeline)

1. **Ingestion**
    - Files in `data/` are scanned
    - Binary files are skipped
    - Text files are streamed line-by-line

2. **Chunking**
    - Text is split into overlapping chunks
    - Streaming design avoids loading large files into memory

3. **Indexing**
    - Each chunk is embedded using SentenceTransformer
    - Embeddings are normalized and indexed with faiss

4. **Retrieval**
    - User query is embedded with the same sentence transformer model
    - faiss retrieves top-k relevant chunks
    - Low-score matches are filtered out

5. **Generation**
    - Retrieved chunks are written into an LLM prompt
    - LLM is instructed to answer strictly from context
    - Citations are shown with the answer

---

## Usage

## Configuration

## Testing

## Design Decisions

## Limitations

## Future Work

## License

