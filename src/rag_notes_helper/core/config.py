from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    LLM_PROVIDER: str | None = None
    LLM_MODEL: str | None = None
    LLM_API_KEY: str | None = None

    # only for ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # # HuggingFace settings
    # HF_MODEL: str = "openai/gpt-oss-120b"
    # HUGGINGFACE_API_KEY: str | None = None
    #
    # # OpenAI settings
    # OPENAI_MODEL: str = "gpt-4o-mini"
    # OPENAI_API_KEY: str | None = None

    # LLM settings
    LLM_MAX_CHUNKS: int = 5
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
    TOP_K: int = 5
    MIN_RETRIEVAL_SCORE: float = 0.15


    # read .env file configuration variables
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

settings = Settings()
