from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "LearnForge AI Support Assistant"
    data_dir: Path = Path("data")
    storage_dir: Path = Path("storage")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    min_similarity: float = 0.34
    min_confidence: float = 0.58
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
