import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger("uvicorn")


# Define the project root directory, which is the parent of the 'src' directory.
# This makes it easy to reference files in 'files/', 'docs/', etc.
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """
    Validates and stores all environment variables for the application.
    It reads from a .env file and throws an error if required variables are missing.
    """

    # --- Required ---
    # These must be set in the .env file or the environment.
    OPENAI_API_KEY: str
    GEMINI_API_KEY: str
    CHROMA_API_KEY: str
    CHROMA_TENANT: str
    CHROMA_DATABASE: str

    # --- Optional with Defaults ---
    # These have default values but can be overridden.
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K: int = 3
    PORT: int = 8000
    ENVIRONMENT: str = "development"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


try:
    settings = Settings()  # type: ignore
    log.info("Configuration loaded successfully.")
except Exception as e:
    log.error(f"Configuration error: {e}")
    raise
