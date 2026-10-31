import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rag_notes_helper.core.config import get_settings


class _AllOnesEmbeddingModel:
    def encode(self, texts, *, normalize_embeddings, convert_to_numpy):
        return np.ones((len(texts), 4), dtype=np.float32)


def _fake_ragas_worker(_command, *, input, capture_output, text, check):
    request = json.loads(input)
    score_sets = [
        [
            {
                **{metric: 1.0 for metric in batch["metric_names"]},
                "metric_reasons": {},
                "metric_latencies_ms": {},
            }
            for _ in batch["rows"]
        ]
        for batch in request["batches"]
    ]
    return SimpleNamespace(
        stdout=json.dumps({"scores": score_sets, "judge_calls": []}),
        returncode=0,
    )


@pytest.fixture
def fixture_evaluation_doubles():
    return SimpleNamespace(
        embedding_model=_AllOnesEmbeddingModel(),
        ragas_worker=_fake_ragas_worker,
    )


@pytest.fixture(autouse=True)
def test_settings(monkeypatch, tmp_path):
    """ The 'autouse=True' ensures this runs for every test file """

    notes_dir: Path = tmp_path / "data"
    storage_dir: Path = tmp_path / "storage"
     
    notes_dir.mkdir(exist_ok=True)
    storage_dir.mkdir(exist_ok=True)

    monkeypatch.setenv("NOTES_DIR", str(notes_dir))
    monkeypatch.setenv("STORAGE_DIR", str(storage_dir))
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama3.1")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    get_settings.cache_clear()

    yield
