FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip uv

COPY pyproject.toml ./

COPY src/ src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache-dir .

RUN mkdir -p data storage hf_cache

ENV PYTHONPATH=/app/src

ENTRYPOINT ["rag-app"]
