FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

COPY pyproject.toml ./
COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip \
    && uv pip install --system --no-cache-dir .

COPY ask.py .

RUN mkdir -p data storage

CMD ["python", "ask.py"]

