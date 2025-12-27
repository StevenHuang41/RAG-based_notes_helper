FROM python:3.11-slim

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

COPY pyproject.toml ./
COPY src/ src/

RUN --mount=type=cache,target=/root/.cache/uv \
    pip install --no-cache-dir --upgrade pip \
    && uv pip install --system --no-cache-dir .

RUN mkdir -p data storage

ENTRYPOINT ["rag-app"]
CMD [ "repl" ]
