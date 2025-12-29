FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip uv

COPY pyproject.toml ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache-dir --requirements pyproject.toml

COPY src/ src/

RUN mkdir -p data storage hf_cache

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m", "rag_notes_helper.cli"]
