"""Application configuration.

All settings are loaded from environment variables (or a local `.env` file).
We use `pydantic-settings` so every value is validated and type-checked.

To add a new setting:
  1. Add a field below (with a sensible default).
  2. Add the matching KEY to `.env.example`.
That's it — it becomes available everywhere via `get_settings()`.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- OpenAI ---------------------------------------------------------
    # The field names are lower-case; pydantic matches them to UPPER-CASE
    # environment variables automatically (OPENAI_API_KEY -> openai_api_key).
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # --- App ------------------------------------------------------------
    app_env: str = "local"
    app_name: str = "AI Agent Starter Kit"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unknown env vars instead of crashing
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    `lru_cache` means the `.env` file is read only once per process, and the
    same object is reused everywhere — cheap and consistent.
    """
    return Settings()
