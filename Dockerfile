FROM python:3.11-slim AS builder

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt \
    pip install --no-cache-dir --upgrade pip uv
    # apt-get update && apt-get install -y \
    # build-essential \
    # && rm -rf /var/lib/apt/lists/* \

COPY src/ src/

COPY pyproject.toml ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache-dir .



# Runtime
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

RUN mkdir -p data storage

ENTRYPOINT ["rag-app"]
CMD [ "repl" ]
