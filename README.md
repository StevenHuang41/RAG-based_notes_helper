# RAG-based Notes Helper

A **Retrieval-Augmented Generation (RAG)** assistant for querying and reviewing your own notes using **local embeddings** (Hugging Face), **Vector search** (Faiss), and **LLM backends** (Hugging Face / OpenAI).

This project emphasizes **correct RAG design**, **memory-safe ingestion**, **testable**, and **real-world workflows** (Docker + CI).

---

## Why

This project shows how to build an **LLM system without fine-tuning**

### Why not fine-tuning

Fine-tuning is often **costly, unnecessary** for note-based knowledge system

- Require retraining when knowledge updates
- Higher infrastructure and maintenance cost
- Introduces model drift and reproducibility issue
- Hard to debug hallucinations

### Why RAG

- **Knowledge stay external and inspectable**
    - Your notes remain the only source of truth

- **Instant updates**
    - New documents are indexed without retraining

- **Lower cost**
    - Only relevant chunks are sent to the LLM rather than entire file

- **Reducing hallucinations**
    - LLM only generates content strictly based on high-quality retrieval

---

## Features

- Retrieval-Augmented Generation (**RAG**) over notes
- **Memory-safe & streaming ingestion** for large note files
- Overlapping chunking with configurable window size
- Sentence-Transformer embeddings
- Dense vector retrieval using **Faiss**
- LLM backends:
    - Hugging Face (free token)
    - OpenAI (paid API)
- Source-aware answers with citation
- Interactive CLI with live re-indexing
- Unit-tested components
- Containerization using docker
- CI/CD-enabled (github actions)
    - Automatic test execution on push and pull requests
    - Docker build validation
- Dockerized for reproducible execution

---

## Architecture

```text
Notes

Text loaders

Streaming chunking

Sentence-Transformer embedding

faiss vector index

Query embedding

Top-k retrieval

LLM
```

---

## Project Structure

```text
.
├── ask.py                          # entry point
├── src/
│   └── rag_notes_helper/
│       ├── core/
│       │   └── config.py
│       └── rag/
│           ├── ingest.py
│           ├── loaders.py
│           ├── index.py
│           ├── chunking.py
│           ├── retrieval.py
│           ├── answer.py
│           └── llm/
│               ├── hf.py
│               ├── openai_api.py
│               └── router.py
├── tests/                          # unit tests
│   ├── test_ingest.py
│   ├── test_loaders.py
│   ├── test_index.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_answer.py
├── data/                           # place your notes here!
│   └── notes_helper.md             # base knowledge for this app
├── storage/                        # faiss index + metadata
│   ├── faiss.index
│   └── meta.jsonl
├── .env.example                    # template for creating .env
├── .gitignore
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── pytest.ini
├── pyproject.toml
└── README.md
```

## How It Works

1. **Ingestion**
    - Files in `data/` are scanned
    - Binary files are skipped via null-byte detection
    - Text is processed **line-by-line** to avoid memory exhaustion

2. **Chunking**
    - Streaming chunk generator splits text into overlapping windows
    - Prevents memory exhaustion
    - Configurable chunk and overlap size

3. **Indexing**
    - Each chunk is embedded using SentenceTransformer
    - Embeddings are normalized and indexed with faiss

4. **Retrieval**
    - User query is embedded with the same embedding model
    - faiss retrieves top-k relevant chunks
    - Low-scoring matches are filtered out

5. **Generation**
    - Retrieved chunks are written into an LLM prompt
    - LLM is instructed to answer **only from retrieved context**
    - Citations are shown with the answer

---

## Setup

This app can be run from source or via Docker
Regardless of the methods, the following directories and file setup is required

### 1. `.env`

Create `.env` based on `.env.example`

```text
LLM_PROVIDER=hf
HUGGINGFACE_API_KEY=hf_xxxxxx
...
```

ensure it exists at the project root file

At minimum, `HUGGINGFACE_API_KEY` is required in `.env` to run the app

See https://huggingface.co/docs/hub/security-tokens for creating free tokens

### 2. `data/`

Create a `data/` and place your notes inside it

```text
.
└── data/                           # place your notes here!
    ├── [your notes]
    └── notes_helper.md
```

### 3. `storage/`

Create an empty `storage/`

```text
.
└── storage/                        # faiss index + metadata
```

`storage/` is for storing faiss index and chunk metadata

Contents wil be built automatically at runtime

---

## Usage

After completing the setup steps above, choose one of the following running methods

### Running via source:

```bash
git clone https://github.com/StevenHuang41/RAG-based_notes_helper.git
cd RAG-based_notes_helper
# create .env
# ensure data/ and storage/ exist
python ask.py
```


or run via docker compose
```bash
# python ask.py
docker compose up --build
```


### Running via Docker

If prefer not to clone the repo, pull the pre-built image and mount the required directoies

Setup:
```bash
mkdir rag_application
cd rag_application
touch .env
# setup up for api keys

mkdir data storage
# place your notes in data/
```

Runs:
```bash
docker pull ghcr.io/stevenhuang41/rag-based-notes-helper/rag-app:latest

docker run -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/storage:/app/storage \
  --env-file .env \
  ghcr.io/stevenhuang41/rag-based-notes-helper/rag-app:latest
```


### Commands:

- Type a question after `> `
- `:reindex`/`:ri` (rebuild index without restarting)
- `:quit`/`:q` (exit)

 ---

## Testing

Run unit tests at project root:

```bash
pytest
```

---

## Limitations

- No conversation memory (single-turn only)
- PDF ingestion not implemented
- Designed for personal note collections

---

## Future Work

- Conversational memory
- PDF loaders

## License

MIT License

