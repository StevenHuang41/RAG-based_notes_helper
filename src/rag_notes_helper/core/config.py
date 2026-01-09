from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import (
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    LLM_PROVIDER: Literal["hf", "openai", "ollama"] | None = None
    LLM_MODEL: str | None = None
    LLM_API_KEY: str | None = None

    # only for ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # LLM runtime settings
    LLM_MAX_CHUNKS: int = 5
    LLM_MAX_TOKENS: int = 2048
    LLM_TEMPERATURE: float = 0.3

    # path
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
    NOTES_DIR: Path = PROJECT_ROOT / "data"
    STORAGE_DIR: Path = PROJECT_ROOT / "storage"

    # embedding
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # chunking strategy
    CHUNK_SIZE: int = 200        # characters
    CHUNK_OVERLAP: int = 50     # characters

    # retrieval
    TOP_K: int = 5
    MIN_RETRIEVAL_SCORE: float = 0.15

    # read .env file configuration variables
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    @field_validator("TOP_K")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("TOP_K should > 0")

        return v


    @field_validator("MIN_RETRIEVAL_SCORE")
    @classmethod
    def validate_retri_score(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("MIN_RETRIEVAL_SCORE should be 0 <= v <= 1")

        return v


    @field_validator("CHUNK_SIZE", "CHUNK_OVERLAP")
    @classmethod
    def validate_chunk_params(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Chunk parameters should be > 0")

        return v


    @field_validator("EMBEDDING_MODEL")
    @classmethod
    def validate_embedding_model(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("EMBEDDING_MODEL cannot be empty")

        return v


    @field_validator("NOTES_DIR")
    @classmethod
    def validate_notes_dir(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"data dir does not exist: {v}")

        if not v.is_dir():
            raise ValueError(f"data is not a directory: {v}")

        return v


    @field_validator("STORAGE_DIR")
    @classmethod
    def validate_storage_dir(cls, v: Path) -> Path:
        try :
            v.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create storage dir: {v}") from e

        return v


    @model_validator(mode="after")
    def validate_semantics(self):
        # chunking logic
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(
                "CHUNK_OVERLAP must < CHUNK_SIZE"
            )

        # LLM configuration
        if not self.LLM_PROVIDER:
            raise ValueError("LLM_PROVIDER must be set")

        if not self.LLM_MODEL:
            raise ValueError("LLM_MODEL must be set")

        if self.LLM_PROVIDER != "ollama" and not self.LLM_API_KEY:
            raise ValueError(
                f"LLM_API_KEY is required for '{self.LLM_PROVIDER}'"
            )

        # provider/model sanity check
        if self.LLM_PROVIDER == "ollama" and "/" in self.LLM_MODEL:
            raise ValueError(
                "OLLAMA models should be a local name, ex: llama3.1"
            )

        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

