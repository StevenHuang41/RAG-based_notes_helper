import pytest
from pathlib import Path

from rag_notes_helper.core.config import get_settings


@pytest.fixture(autouse=True)
def test_settings(monkeypatch, tmp_path):
    """ The 'autouse=True' ensures this runs for every test file """

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama3.1")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    notes_dir: Path = tmp_path / "data"
    storage_dir: Path = tmp_path / "storage"
    notes_dir.mkdir(exist_ok=True)
    storage_dir.mkdir(exist_ok=True)

    monkeypatch.setenv("NOTES_DIR", str(notes_dir))
    monkeypatch.setenv("STORAGE_DIR", str(storage_dir))

    get_settings.cache_clear()

    yield

    get_settings.cache_clear()
