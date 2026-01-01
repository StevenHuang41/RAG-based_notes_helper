# RAG-based Notes Helper

A **Retrieval-Augmented Generation (RAG)** assistant for querying and reviewing your own notes using **local embeddings** (Hugging Face), **Vector search** (Faiss), and **LLM backends** (Hugging Face / OpenAI).

This project emphasizes **correct RAG design**, **memory-safe ingestion**, **testable**, and **real-world workflows** (Docker + CI).

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

[**Quick Start**](#running-via-docker)


---

## Overview

 - [Why](#why)
 - [Features](#features)
 - [Architecture](#architecture)
 - [Project Structure](#project-structure)
 - [How It Works](#how-it-works)
 - [Instinstallation-setup](#installation-setup)
 - [Usage](#usage)
 - [Testing](#testing)
 - [Limitations](#limitations)
 - [Future Work](#future-work)
 - [License](#license)

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
Notes → Chunking → Embedding → Index
                                  ↓
Query   →   Embedding   →   Retrieval   →   LLM   →   Answer
```

---

## Project Structure

```text
RAG-based_notes_helper/
├── README.md
├── pyproject.toml
├── .env.example                    # template for creating .env
├── Dockerfile
├── docker-compose.yml              # runtime-only, pull-only
├── LICENSE
├── pytest.ini
├── .gitignore
├── .dockerignore
│
├── src/
│   └── rag_notes_helper/
│       ├── cli.py                  # entry point (rag-app)
│       │
│       ├── core/
│       │   └── config.py
│       │
│       └── rag/
│           ├── ingest.py
│           ├── loaders.py
│           ├── index.py
│           ├── chunking.py
│           ├── retrieval.py
│           ├── answer.py
│           │
│           └── llm/
│               ├── hf.py
│               ├── openai_api.py
│               └── router.py
│
├── tests/                          # unit tests
│   ├── test_ingest.py
│   ├── test_loaders.py
│   ├── test_index.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_answer.py
│
├── data/                           # place your notes here!!!
│   └── notes_helper.md             # base knowledge
│
├── storage/                        # rag matadata and faiss
│   ├── faiss.index
│   └── meta.jsonl
│
├── hf_cache/                       # huggingface cache
│
└── .github/                        # github action
    └── workflows/
        ├── ci.yml
        └── cd.yml
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

## Installation & Setup

Make sure the following requirements have been setup before running the app

### Prerequisites

- **python** >= 3.10
- `pip` or **uv** (recommended)
- **Docker** >= 20.10
- **git** (optional)

### 1. Configuration (`.env`)

Create `.env` at project root based on `.env.example`:

```bash
curl -L -o .env.example \
https://raw.githubusercontent.com/StevenHuang41/RAG-based_notes_helper/main/.env.example
```

```bash
cp .env.example .env
```

At minimum, `HUGGINGFACE_API_KEY` must be assigned in `.env`:
```text
HUGGINGFACE_API_KEY=hf_xxxxxx   # replace hf_xxxxxx with your token
```
You need to create a [Hugging Face Token](https://huggingface.co/docs/hub/security-tokens)

If using OpenAI api instead, configure:
```text
LLM_PROVIDER=openai
OPENAI_API_KEY=sk_xxxxxx
```

You can also customize other configuration values in `.env`

### 2. Notes directory (`data/`)

Create a `data/` and place your notes inside it:
```bash
mkdir data
```

Example structure:
```text
RAG-based_notes_helper/
└── data/
    ├── notes_helper.md
    └── [your notes]
```

To download the `notes_helper.md`:
```bash
cd data
curl -L -o notes_helper.md \
https://raw.githubusercontent.com/StevenHuang41/RAG-based_notes_helper/main/data/notes_helper.md
```

You mush have **at least one text file** in `data/` to avoid no chunk to index error.

---

## Usage

### Running from Source:

```bash
git clone https://github.com/StevenHuang41/RAG-based_notes_helper.git
cd RAG-based_notes_helper
# get in venv
uv pip install -e .
```

[Setup `.env`](#1-configuration-env)

[Setup `data/`](#2-notes-directory-data)

#### One time query
```bash
rag-app what is xxx
rag-app "what is ...?"
rag-app "what is xxx" -r                    # reindex before answering
rag-app "what is xxx" -c                    # including citations in answer
rag-app "I wanna know ..." > answer.txt     # save generated answer
```

#### Interactive REPL
Runs rag-app repeatedly in repl mode:
```bash
rag-app repl
```

#### Reindex
Reindex data if you update in `data/`
```bash
rag-app reindex
```

### Running via Docker

No need to clone git repo

Setup:
```bash
mkdir rag_application
cd rag_application

# get docker-compose.yml
curl -L -o docker-compose.yml \
https://raw.githubusercontent.com/StevenHuang41/Rag-based_notes_helper/main/docker-compose.yml

# get docker image
docker pull ghcr.io/stevenhuang41/rag-based-notes-helper:latest
```

[Setup `.env`](#1-configuration-env)

[Setup `data/`](#2-notes-directory-data)

Runs at project root:
```bash
docker compose run --rm rag-app "query"
```

Do **NOT** use `docker compose up` for interactive CLI

(optional)
```bash
docker run --rm -it \
  --env-file .env \
  -v ./data:/app/data \
  -v ./storage:/app/storage \
  -v ./hf_cache:/root/.cache/huggingface \
  ghcr.io/stevenhuang41/rag-based-notes-helper:latest \
  "query"
```

### Commands:

#### One time mode

* `[query]`
    RAG generates answer as usual

    - query = repl

        Start REPL mode

    - query = reindex

        Reindex notes in `data/`

* `[query] -r`
    RAG reindex before generating answer

* `[query] -c`
    Show citations file with answer


#### REPL mode

- `:quit`       /   `:q`      (exit)
- `:help`       /   `:h`      (show instructions)
- `:reindex`    /   `:ri`     (reindex without exiting)
- `:citations`  /   `:ci`     (toggle citation files with answer)
- `:sources`    /   `:so`     (show indexed files)

 ---

## Testing

Runs unit tests at project root:

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

