# RAG-based Notes Helper

![CI](https://github.com/StevenHuang41/RAG-based_notes_helper/actions/workflows/ci.yml/badge.svg)
![CD](https://github.com/StevenHuang41/RAG-based_notes_helper/actions/workflows/cd.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-blue)
![Docker](https://img.shields.io/badge/Docker-GHCR-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

A local-first Retrieval-Augmented Generation (RAG) application for querying personal notes and documents. The system indexes notes with SentenceTransformer embeddings and FAISS, combines Vector and BM25 search results with Reciprocal Rank Fusion, and sends grounded context to a configurable LLM backend.

The application can run from source or as a containerized CLI. It supports Classic RAG by default and an opt-in Agentic RAG workflow, one-time queries, an interactive REPL, source citations, smart re-indexing, multiple LLM providers, PDF ingestion, and a fixture-backed RAGAS benchmark.

<p align="center">
    <a href="docs/demo.mp4">
        <img src="docs/demo.gif" width="800" alt="RAG-based Notes Helper demo" />
    </a>
</p>

---

## Contents

- [Why RAG for Notes](#why-rag-for-notes)
- [Features](#features)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Evaluation](#evaluation)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Local Development](#local-development)
- [Usage](#usage)
- [Docker Usage](#docker-usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [License](#license)

---

## Why RAG for Notes

A note assistant should answer from the current knowledge base without requiring model fine-tuning every time the notes change.

Fine-tuning is usually a poor fit for personal notes because:

- knowledge updates require additional training runs
- model behavior becomes harder to debug and reproduce
- private notes may be mixed into model weights
- hallucination sources are harder to inspect
- infrastructure cost increases for a problem that can be solved with retrieval

This project keeps knowledge external and inspectable. Notes are chunked, embedded, indexed, retrieved at query time, and passed to the LLM as explicit context.

---

## Features

### Retrieval pipeline

- Ingests `.txt`, `.md`, `.pdf`, and `.py` files from `data/`.
- Splits documents into overlapping chunks with configurable chunk size and overlap.
- Generates local embeddings with SentenceTransformer.
- Stores vectors in FAISS for similarity search.
- Combines Vector and BM25 candidate rankings with Reciprocal Rank Fusion (RRF).
- Stores chunk metadata separately in JSONL with an offset index for efficient lookup.
- Supports smart indexing so unchanged files do not need to be reprocessed.

### Generation

- Supports multiple LLM backends:
  - Hugging Face Inference API
  - Google Gemini
  - OpenAI
  - OpenRouter
  - Ollama for local inference
- Grounds answers in retrieved context.
- Offers an opt-in Agentic RAG workflow with independently configurable Search and evidence-reviewed Answer attempt limits (defaults: three each; allowed range: one through five).
- Supports optional citation output for source visibility.
- Supports streaming responses in one-shot and REPL modes when `STREAM=true`.

### CLI workflow

- One-time query mode for shell usage and scripting.
- Interactive REPL mode for repeated queries after the index is loaded.
- Commands for re-indexing, smart updates, configuration inspection, source listing, citation toggling, stream toggling, and evaluation.
- Workflow selection is independent of interaction style: Classic is the default; `--agentic` selects Agentic RAG, and `--repl` selects interactive use.

### Engineering workflow

- Runtime configuration validation with Pydantic Settings.
- Docker image published through GitHub Container Registry.
- GitHub Actions workflows for tests and container builds.
- Unit tests for ingestion, loading, chunking, indexing, retrieval, metadata storage, and answer generation.
- Latency and runtime logs under `logs/`.

---

## Architecture

```mermaid
flowchart TD
    notes["Notes in data/"] --> load["Loaders and chunking"]
    load --> embed["SentenceTransformer embeddings"]
    embed --> vector["FAISS vector index"]
    load --> metadata["Chunk metadata and BM25 index"]
    vector --> hybrid["Hybrid retrieval: Vector + BM25, RRF, reranking"]
    metadata --> hybrid
    user["User query"] --> choose{"Workflow"}
    choose --> classic["Classic RAG"]
    choose --> agentic["Agentic RAG: bounded search and evidence review"]
    classic --> hybrid
    agentic --> hybrid
    hybrid --> context["Selected evidence"]
    context --> answer["Answer generation and citations"]
    assistant["Local coding assistant"] -->|"search_notes over stdio"| mcp["Local MCP server"]
    mcp --> hybrid
    hybrid -->|"Ranked chunks and provenance"| mcp
    mcp -->|"Evidence"| assistant
    assistant -->|"Answer based on evidence"| user
```

The local MCP path returns retrieval evidence to the coding assistant; the RAG application's answer-generation workflow is not invoked by the MCP server.

---

## How It Works

1. **Configuration validation**
   - The CLI reads `.env` through Pydantic Settings.
   - Required provider/model/key fields are validated before runtime.

2. **Document loading**
   - Files in `data/` are scanned for supported extensions.
   - Loaders normalize source content into text streams.

3. **Chunking**
   - Documents are split into overlapping chunks.
   - Chunk size and overlap are controlled by `CHUNK_SIZE` and `CHUNK_OVERLAP`.

4. **Indexing**
   - Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` by default.
   - Embeddings are normalized and stored in a FAISS index.
   - Chunk text and source metadata are stored outside the vector index.

5. **Retrieval**
   - The user query is embedded with the same embedding model for Vector search.
   - BM25 searches the indexed chunk text independently of the embedding model.
   - Vector matches below `VECTOR_MIN_SCORE` are filtered. BM25 candidates below `BM25_MIN_SCORE_RATIO` times that query's highest positive BM25 score are filtered; the ratio is query-relative rather than a cross-query score comparison.
   - Reciprocal Rank Fusion combines both ranked lists into the configured fusion candidate list.

6. **Workflow selection and answer generation**
   - Classic RAG sends one retrieval result to answer generation.
   - Agentic RAG uses a LangGraph StateGraph with two phases: bounded evidence recovery, then generation and same-context Answer review. Each Search appends original chunks to a pool deduplicated by source/chunk identity, preserving first-seen order and all Search provenance. The controller judges the full pool against the original question as zero, partial, or sufficient evidence; no separate round coverage state is maintained.
   - The first Search uses the original question and configured thresholds. For zero/partial evidence, later searches may rewrite the query and adjust Vector/BM25 thresholds within `[0, 1]`, while configured top-k, models, and generation limits remain unchanged. Normal Retrieval stops at sufficient evidence or the Search cap. Coverage and stop reason are separate: exhausted searches do not prove corpus absence.
   - Search and Answer total-attempt limits are independently configurable, defaulting to three each with an allowed range of one through five. Each includes the first attempt. Controller validation retries consume neither budget. Answer review cannot reopen Retrieval or expand its selected context.
   - Once Retrieval finishes, pools larger than `LLM_MAX_CHUNKS` are reranked against the original question and reduced to top-k. Smaller pools retain first-seen order without final reranking. A sufficient pool can lose necessary evidence during selection, so Answer review uses the exact context actually supplied to generation.
   - A non-empty selected pool reaches generation even when evidence is irrelevant. Accepted answers can be full, partial with explicit gaps, or abstentions. Unsupported claims or undisclosed gaps trigger revisions with the same context. Candidates remain buffered until accepted; revision exhaustion, empty generation, or failed review returns a fixed warning without exposing a rejected candidate. With `STREAM=true`, the CLI displays the accepted Agentic answer in chunks after review; Classic streams during generation.
   - When the accumulated pool is empty, an intelligibility check can request rephrasing of an unintelligible original question. An intelligible question can continue recovery; a final empty pool returns deterministic search-limitation text without Answer generation. A later empty Search does not erase earlier chunks or trigger that check when the pool is non-empty.
   - Invalid Retrieval decisions receive one validation retry. Persistent invalidity, provider failure, or insufficient reviewer context budget records degradation and uses the available pool for final selection; unknown coverage is not treated as zero evidence. User-facing warnings avoid internal failure codes, while evaluation retains sanitized causes.
   - Retrieved chunks are inserted into the LLM prompt as context.
   - The answer can include source citations.
   - REPL mode keeps the loaded index available for faster repeated queries.

---

## Evaluation

From a source checkout with development dependencies installed, `--eval` evaluates Classic RAG and `--agentic --eval` evaluates Agentic RAG against the fixed evaluation corpus under `evals/notes/` and the reviewed QA set at `evals/qa/notes_qa.json`. Each run builds a temporary index and does not read or modify the user's `data/` or `storage/` directories:

```bash
rag-app --eval
rag-app --eval --sample 5 --seed 42
rag-app --agentic --eval --sample 5 --seed 42
```

Each run writes `README.md`, `results.json`, and `run.log` under a local-time, Taipei timestamped directory in `evals/results/`. These generated reports are ignored by Git. When stderr is a terminal, evaluation shows a progress bar for case execution and RAGAS metric stages. Run metadata records the selected workflow, sampling settings, and selected case IDs. Sample selection is uniform without replacement; cases run in evaluation-corpus order. The same corpus, QA set, sample count, and seed select the same cases. Without a seed, a sampled run may select a different subset each time. A seed without `--sample` still runs the full QA set. `--sample` must be a positive integer no larger than the QA set size.

The evaluation corpus and QA set are a fixed regression set, not a statistically representative benchmark. A small sample can omit categories; inspect the selected case IDs and category counts in the report. Evaluation logs use case IDs and do not include query or note text. Results keep deterministic answerability, reason-code, response-origin, workflow-error, Retrieval evidence, and line-grounded Recall/Precision/NDCG checks separate from model-based RAGAS diagnostics.

Accepted generated full answers, partial answers, and abstentions enter per-metric Ragas applicability checks, including accepted answers after degraded Retrieval review. Deterministic empty-pool/rephrasing/warning outcomes and rejected candidates do not enter accepted-final-answer means. Answer Correctness requires the reviewed Reference answer; Faithfulness and Context Utilization additionally require the exact Final generation context. Each metric records `scored`, `unavailable`, or `failed` with applicable/scored counts and reasons. RAGAS scores are provider-dependent observations, not pass/fail guarantees.

The report distinguishes artifact process status from workflow outcome. A successfully written run containing query-level failures or degradation is `completed` with workflow outcome `completed_with_errors`; it reports failure/degradation counts and reliability independently of answer quality. Agentic query details compare accumulated and selected pools in two metadata-only tables, even when final reranking is skipped, then show actual generation context separately. Per-query Search registries retain effective thresholds and every executed Search. Final context evidence metrics use only the exact context sent to generation. Real Retrieval review, final pool rerank, Answer review, and judge costs remain distinguishable. Token counts are provider-reported; unavailable values remain unavailable, while character counts are recorded separately.

The QA set covers direct lookup, multi-hop integration, unanswerable questions, and ambiguous questions. Evaluation always uses this reviewed evaluation corpus and isolated synthetic index; it never evaluates a user's `data/` directory or persistent `storage/` snapshot. If the corpus changes, review and update any affected line-anchored QA evidence in the same change; validation rejects stale evidence rather than rewriting it automatically.

`--eval` is a repository development command. The evaluation code, evaluation assets, and RAGAS development dependencies are not included in the runtime wheel. In a source checkout, use an editable development install to run this command.

---

## Tech Stack

| Layer | Tools |
|---|---|
| CLI / runtime | Python, argparse, Pydantic Settings |
| Retrieval | FAISS, BM25, Reciprocal Rank Fusion, SentenceTransformer, NumPy |
| LLM backends | Hugging Face, Gemini, OpenAI, OpenRouter, Ollama |
| Evaluation | Ragas 0.4 collections metrics and provider adapters |
| Document parsing | PyMuPDF for PDF support, text/markdown/python loaders |
| Packaging | uv, hatchling, pyproject.toml |
| Testing | pytest |
| Containerization | Docker, Docker Compose, GitHub Container Registry |
| CI/CD | GitHub Actions |

---

## Project Structure

```text
RAG-based_notes_helper/
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── .github/workflows/
│   ├── ci.yml
│   └── cd.yml
├── data/
│   └── notes_helper.md             # example/base note file tracked by git
├── docs/
│   ├── demo.gif
│   ├── demo.mp4
│   └── test.png
├── evals/
│   ├── notes/                       # version-controlled evaluation corpus
│   ├── qa/
│   │   └── notes_qa.json            # reviewed QA ground truth
│   └── results/                     # generated evaluation reports; ignored by Git
├── src/rag_notes_helper/
│   ├── cli.py                      # rag-app entry point
│   ├── mcp_server.py               # local read-only MCP search server
│   ├── core/config.py              # runtime settings and validation
│   ├── eval/                       # RAGAS evaluation
│   ├── rag/
│   │   ├── answer.py
│   │   ├── chunking.py
│   │   ├── index.py
│   │   ├── ingest.py
│   │   ├── loaders.py
│   │   ├── meta_store.py
│   │   ├── retrieval.py
│   │   └── llm/                    # provider implementations
│   └── utils/
└── tests/                           # pytest tests
```

Runtime directories such as `storage/`, `logs/`, and `hf_cache/` are created locally and ignored by Git. Evaluation run artifacts are stored under `evals/results/` and ignored by Git. Historical root-level `eval/` artifacts remain local and ignored; evaluation no longer writes there.

---

## Local Development

### Prerequisites

- Python 3.11+
- `uv` recommended
- Docker optional for containerized usage
- API key for the selected remote LLM provider, unless using Ollama locally

### Install from source

```bash
git clone https://github.com/StevenHuang41/RAG-based_notes_helper.git
cd RAG-based_notes_helper
uv sync
```

Create a local environment file:

```bash
cp .env.example .env
```

Edit `.env` and choose one LLM provider.

The repository includes `data/notes_helper.md` as an example document. Add your own notes under `data/`.

---

## Usage

### One-time query

```bash
uv run rag-app "What does my note say about RAG?"
```

Useful options:

```bash
uv run rag-app "What is RAG?" --update      # smart re-index before answering
uv run rag-app "What is RAG?" --reindex     # rebuild the full index before answering
uv run rag-app "What is RAG?" --citations   # include source citations
uv run rag-app --sources                     # list indexed source files
uv run rag-app --config                      # show validated configuration
uv run rag-app "What is RAG?" --agentic      # use Agentic RAG for one query
uv run rag-app --eval                        # evaluate Classic RAG on synthetic notes
uv run rag-app --eval --sample 5 --seed 42   # reproducible small sample
uv run rag-app --agentic --eval --sample 5 --seed 42 # sample Agentic RAG
```

### Interactive REPL

```bash
uv run rag-app --repl
```

Select Agentic RAG for the whole REPL session with `--agentic`. A question supplied at startup is answered first, then the REPL accepts fresh queries:

```bash
uv run rag-app --repl --agentic
uv run rag-app --repl --agentic "What changed before launch?"
```

Each query starts with fresh Agentic state and performs a new Retrieval. The Agentic REPL keeps up to five recent user/assistant turns in memory for resolving intent; prompts label them as conversational context, never evidence. This history is cleared when the REPL exits. One-shot queries and evaluation cases start with empty history. Agentic candidates stay buffered until evidence review accepts them; with `STREAM=true`, the CLI then displays the accepted answer in chunks. Set `STREAM=false` to display it all at once.

REPL commands:

```text
:quit      or :q     exit
:help      or :h     show commands
:reindex   or :ri    rebuild the index from scratch
:update    or :ud    smart update changed files
:citations or :ci    toggle citation display
:sources   or :so    show indexed files
:config    or :co    show configuration
:stream    or :s     toggle stream mode
:evaluate  or :ev    run evaluation
```

### Local MCP search

The `rag-mcp-server` command exposes one read-only `search_notes` tool over local stdio. It refreshes the runtime index from `data/` at startup, runs the shared Hybrid retrieval pipeline, and returns ranked chunks with relative source paths, line ranges, and ranking provenance. Restart the server to guarantee later note changes are indexed. It does not generate an answer.

The optional `top_k` defaults to 5 and is capped at 20, independently of `LLM_MAX_CHUNKS`. It is a maximum: upstream Vector, BM25, and fusion limits and available evidence can yield fewer chunks. Source notes and the index remain local, but returned evidence enters the connected assistant's context and may be sent to its model provider.

#### Set up a local client

Install from source and configure the project as described in [Local Development](#local-development). Add your notes under `data/`. The application currently validates `LLM_PROVIDER` and `LLM_MODEL` when the server starts, even though MCP search does not invoke an answer-generation model; choose a valid provider configuration in `.env`. Embedding and reranking models must be available locally or downloadable on first use. The client starts the server itself; there is no need to launch a separate terminal for `rag-mcp-server`.

From the repository directory, note the absolute paths shown by these commands:

```bash
pwd
command -v uv
```

Use those paths below. The client launches the local process with your user permissions; it has no network listener or OAuth layer. When configuring a client, allow enough startup time for indexing and first-time model downloads.

**Codex app / CLI**

From this repository directory, register the server with the Codex CLI. `--directory` selects this checkout even when you use Codex from another directory:

```bash
codex mcp add rag_notes -- "$(command -v uv)" --directory "$PWD" run rag-mcp-server
codex mcp list
```

Set `startup_timeout_sec = 120` in the resulting `rag_notes` entry in `~/.codex/config.toml`, then start a new local Codex app session and check that `rag_notes` is available. Alternatively, add the following entry directly, replacing both paths with the absolute paths from above:

```toml
[mcp_servers.rag_notes]
command = "/absolute/path/to/uv"
args = ["--directory", "/absolute/path/to/RAG-based_notes_helper", "run", "rag-mcp-server"]
startup_timeout_sec = 120
```

Use one registration method, not both. See [Codex MCP configuration](https://developers.openai.com/codex/mcp) for current options.

**Claude Code**

From this repository directory, register a local stdio server:

```bash
claude mcp add --transport stdio rag_notes -- "$(command -v uv)" --directory "$PWD" run rag-mcp-server
claude mcp get rag_notes
```

Start Claude Code in the same project and use `/mcp` to inspect its server status. This configuration follows [Claude Code MCP documentation](https://code.claude.com/docs/en/mcp); it has not been verified with an actual Claude Code tool call.

#### Search and verify

In your coding assistant, try:

```text
Use the rag_notes search_notes tool to find what my notes say about RAG.
Return at most 3 chunks, cite their source paths and line ranges, and answer only from those chunks.
```

Inspect the actual tool call and response to confirm that `top_k` is 3 and that each result includes text, relative source path, chunk ID, line range, and ranking provenance. Then compare the answer and citations with the returned evidence. The assistant decides whether to call the tool and how to use its results; instructions alone do not guarantee a tool call or an evidence-grounded answer. An empty result means no chunks were returned for that search, while a tool error indicates a failure.

If the server does not connect, check that the absolute `uv` and repository paths exist, dependencies are installed, and the project's `.env` is valid. Inspect the client's MCP logs or server stderr for configuration, model download, and index errors; stdout is reserved for the MCP protocol. The first startup can take longer while building the index or downloading models. Restart the server after modifying notes.

---

## Docker Usage

Create a working directory with `.env`, `data/`, and `docker-compose.yml`:

```bash
mkdir rag-application
cd rag-application

curl -L -o docker-compose.yml \
  https://raw.githubusercontent.com/StevenHuang41/RAG-based_notes_helper/main/docker-compose.yml

curl -L -o .env.example \
  https://raw.githubusercontent.com/StevenHuang41/RAG-based_notes_helper/main/.env.example

cp .env.example .env
mkdir -p data storage logs hf_cache
```

Add notes under `data/`, then run:

```bash
docker compose run --rm rag-app "What do my notes say about RAG?"
```

For Ollama from Docker, set:

```text
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

Do not use `docker compose up` for the interactive CLI workflow; use `docker compose run --rm rag-app ...` instead.

Equivalent `docker run` example:

```bash
docker run --rm -it \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/storage:/app/storage" \
  -v "$(pwd)/hf_cache:/root/.cache/huggingface" \
  -v "$(pwd)/logs:/app/logs" \
  -e TZ=Asia/Taipei \
  ghcr.io/stevenhuang41/rag-based-notes-helper:latest \
  "What do my notes say about RAG?"
```

---

## Configuration

The application reads configuration from `.env`.

Minimal remote-provider configuration:

```text
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash
LLM_API_KEY=your-api-key
```

Minimal Ollama configuration:

```text
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
LLM_API_KEY=not-used
OLLAMA_BASE_URL=http://localhost:11434
```

Minimal OpenRouter configuration:

```text
LLM_PROVIDER=openrouter
LLM_MODEL=provider/model
LLM_API_KEY=your-openrouter-api-key
```

Use an OpenRouter model identifier in `provider/model` form. See the [OpenRouter quickstart](https://openrouter.ai/docs/quickstart) for account and model setup.

Ragas evaluation uses the generation provider and model as its judge by default. To select a different judge, set both `LLM_EVAL_PROVIDER` and `LLM_EVAL_MODEL`. `LLM_EVAL_PROVIDER` accepts `hf`, `openai`, `ollama`, `gemini`, or `openrouter`. A same-provider judge reuses `LLM_API_KEY` unless `LLM_EVAL_API_KEY` is set; a different remote provider requires `LLM_EVAL_API_KEY`. Ollama judges do not require an evaluation API key.

For example, use OpenRouter for answers and a different provider for evaluation:

```text
LLM_PROVIDER=openrouter
LLM_MODEL=provider/answer-model
LLM_API_KEY=your-openrouter-api-key

LLM_EVAL_PROVIDER=gemini
LLM_EVAL_MODEL=gemini-2.5-flash
LLM_EVAL_API_KEY=your-gemini-api-key
```

If only one of `LLM_EVAL_PROVIDER` or `LLM_EVAL_MODEL` is set, configuration fails instead of pairing it with an unintended default.

Common settings:

| Variable | Purpose | Default / example |
|---|---|---|
| `LLM_PROVIDER` | LLM backend | `hf`, `gemini`, `openai`, `openrouter`, `ollama` |
| `LLM_MODEL` | Model name for selected provider | `gemini-2.0-flash` |
| `LLM_API_KEY` | API key for remote providers | required except Ollama |
| `LLM_EVAL_PROVIDER` | Optional Ragas judge provider; defaults to `LLM_PROVIDER` | `hf`, `openai`, `ollama`, `gemini`, `openrouter` |
| `LLM_EVAL_MODEL` | Optional Ragas judge model; defaults to `LLM_MODEL`; set with `LLM_EVAL_PROVIDER` | provider-specific model name |
| `LLM_EVAL_API_KEY` | Optional judge key; required for a different remote judge provider | `your-evaluation-provider-api-key` |
| `LLM_MAX_CHUNKS` | Maximum selected chunks sent to Answer generation | `5` |
| `LLM_MAX_TOKENS` | Maximum output tokens for each generation | `1024` |
| `LLM_TEMPERATURE` | Generation temperature | `0.1` |
| `CHUNK_SIZE` | Chunk size for ingestion | `800` |
| `CHUNK_OVERLAP` | Chunk overlap | `200` |
| `VECTOR_TOP_K` | Human-configured maximum candidates returned by Vector search; Agentic actions cannot change it | `30` |
| `BM25_TOP_K` | Human-configured maximum candidates returned by BM25 search; Agentic actions cannot change it | `20` |
| `FUSION_TOP_K` | Human-configured maximum candidates returned after RRF fusion; Agentic actions cannot change it | `15` |
| `VECTOR_MIN_SCORE` | Starting Vector similarity threshold in `[0, 1]`; later Agentic searches may adapt it | `0.3` |
| `BM25_MIN_SCORE_RATIO` | Starting query-relative BM25 threshold ratio in `[0, 1]`; later Agentic searches may adapt it | `0.0` |
| `CROSS_ENCODER_MODEL` | Sentence Transformers model used to rerank fused candidates | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| `AGENTIC_MAX_SEARCH_ATTEMPTS` | Maximum total Retrieval searches per Agentic query, including the original search | `3` (range `1`–`5`) |
| `AGENTIC_MAX_ANSWER_ATTEMPTS` | Maximum total Answer generations per Agentic query, including the first generation | `3` (range `1`–`5`) |
| `STREAM` | Stream model output where supported | `true` |

`TOP_K` remains accepted as a legacy alias for `FUSION_TOP_K`, and `MIN_RETRIEVAL_SCORE` remains accepted as a legacy alias for `VECTOR_MIN_SCORE`. Existing local configuration therefore continues to work while new configuration uses the stage-specific names.

The application retains the most recent five complete REPL turns for intent context without trimming them by a local token estimate. `LLM_MAX_TOKENS` controls the separate output limit. The retired `LLM_CONTEXT_WINDOW_TOKENS` setting is ignored if it remains in an existing environment file.

Agentic sends the complete accumulated chunks pool to Retrieval review against the original question after each Search. Once Retrieval stops, an oversized pool is reranked and at most `LLM_MAX_CHUNKS` chunks are selected for Answer generation; those chunks remain intact through Answer review and revisions. Model and deployment capacity still applies. If a provider rejects review input, existing degraded Retrieval fallback retains the pool for final selection; Answer generation or Answer review failure returns the existing safe warning. The application does not provision provider context capacity or guarantee that a provider consumes every submitted token.

---

## Testing

Run unit tests from the project root:

```bash
uv run pytest
```

Test evidence:

![Test result](./docs/test.png)

---

## CI/CD

| Workflow | Purpose |
|---|---|
| `.github/workflows/ci.yml` | Install dependencies and run pytest |
| `.github/workflows/cd.yml` | Build and push the Docker image to GitHub Container Registry |

The published image name is:

```text
ghcr.io/stevenhuang41/rag-based-notes-helper:latest
```

---

## Limitations

- Agentic retrieval and conversation history are scoped to one query and one REPL session. Search and Answer attempt limits are configurable, defaulting to three each.
- Retrieval quality depends on note quality, chunk size, and embedding model choice.
- Runtime artifacts such as FAISS indexes and logs are local by default; evaluation reports are local to the source checkout.
- API-based LLM providers require valid credentials in `.env`.

---

## Future Work

- Evaluate whether a graph framework is justified if the bounded Python agent loop grows more complex.
- Evaluate provider-specific prompt tokenizers for tighter context-budget estimates without weakening the conservative fallback.
- Add an optional web UI for browsing sources, retrieved chunks, answers, and citations.
- Improve deployment examples for local Ollama and remote LLM providers.

---

## License

This project is licensed under the [MIT License](./LICENSE).
