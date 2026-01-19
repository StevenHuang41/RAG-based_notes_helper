from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import (
    Field,
    SecretStr,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMSettings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LLM_",
        env_ignore_empty=True,
        extra="ignore",
    )

    provider: Literal["hf", "openai", "ollama", "gemini"]
    model: str
    api_key: SecretStr | None = None

    max_chunks: int = Field(5, gt=0, le=50)
    max_tokens: int = Field(1024, gt=0)
    temperature: float = Field(0.3, gt=0, le=1)


    @property
    def api_key_str(self) -> str | None:
        return self.api_key.get_secret_value() if self.api_key else None

    @model_validator(mode="after")
    def validate_llm(self) -> LLMSettings:
        if self.provider != "ollama" and not self.api_key:
            raise ValueError(f"API Key is required for {self.provider}")

        if self.provider == "ollama" and "/" in self.model:
            raise ValueError("Ollama model name should be like 'llama3'")

        return self



class Settings(BaseSettings):
    # read .env file configuration variables
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore"
    )

    # llm settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    ollama_base_url: str = Field("http://localhost:11434", frozen=True)

    # embedding model
    embed_model_name: str = Field("sentence-transformers/all-MiniLM-L6-v2")

    # chunking strategy
    chunk_size: int = Field(1000, gt=0)
    chunk_overlap: int = Field(200, gt=0)

    # retrieval
    top_k: int = Field(5, gt=0, le=50)
    min_retrieval_score: float = Field(0.2, ge=0, le=1)

    # format
    stream: bool = True
    line_width: int = 80

    # path
    project_root: Path = Path(__file__).resolve().parents[3]
    notes_dir: Path = project_root / "data"
    storage_dir: Path = project_root / "storage"
    logs_dir: Path = project_root / "logs"

    @model_validator(mode="after")
    def validate_cross_logic(self) -> Settings:
        # chunking logic
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP should be < CHUNK_SIZE")

        # top k & llm max_chunks
        if self.top_k > self.llm.max_chunks:
            raise ValueError(
                f"TOP_K ({self.top_k}) should not "
                f"exceed LLM_MAX_CHUNKS ({self.llm.max_chunks})"
            )

        # path logic
        if not self.notes_dir.exists():
            raise ValueError("data/ is missing")

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

