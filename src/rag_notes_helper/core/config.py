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
    LLM_MAX_CHUNKS: int = 5
    LLM_MAX_TOKENS: int = 256
    LLM_TEMPERATURE: float = 0.3

    # path
    PROJECT_ROOT: Path = Path.cwd()
    # PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
    NOTES_DIR: Path = PROJECT_ROOT / "data"
    STORAGE_DIR: Path = PROJECT_ROOT / "storage"

    # embedding
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # chunking strategy
    CHUNK_SIZE: int = 1000        # characters
    CHUNK_OVERLAP: int = 200     # characters

    # retrieval
    TOP_K: int = 5
    MIN_RETRIEVAL_SCORE: float = 0.2


    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )



settings = Settings()
