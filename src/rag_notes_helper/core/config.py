from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    LLM_PROVIDER: str = "hf" # "hf" or "openai"

    # HuggingFace settings
    HF_MODEL: str = "openai/gpt-oss-120b"
    HUGGINGFACE_API_KEY: str | None = None

    # OpenAI settings
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_API_KEY: str | None = None

    # LLM settings
    LLM_MAX_CHUNKS: int = 10
    LLM_MAX_TOKENS: int = 2048
    LLM_TEMPERATURE: float = 0.3

    # path
    PROJECT_ROOT: Path = Path.cwd()
    NOTES_DIR: Path = PROJECT_ROOT / "data"
    STORAGE_DIR: Path = PROJECT_ROOT / "storage"

    # embedding
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # chunking strategy
    CHUNK_SIZE: int = 500        # characters
    CHUNK_OVERLAP: int = 100     # characters

    # retrieval
    TOP_K: int = 10
    MIN_RETRIEVAL_SCORE: float = 0.12


    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

settings = Settings()
