"""Application settings, read from environment variables (and a local .env file)."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM provider. SecretStr keeps keys out of repr()/logs: they print as '**********'.
    llm_provider: Literal["openai", "azure"] = "openai"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4.1-mini"
    azure_openai_api_key: SecretStr | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_deployment: str | None = None
    openai_api_version: str = "2024-10-21"

    # ARES public REST API
    ares_base_url: str = "https://ares.gov.cz/ekonomicke-subjekty-v-be/rest"
    ares_timeout_seconds: float = 10.0

    # Agent: hard cap on agent (LLM) turns per question.
    agent_max_steps: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
