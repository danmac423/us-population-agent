from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: SecretStr
    anthropic_model_name: str

    log_level: str = "WARNING"


@lru_cache
def get_settings() -> Settings:
    return Settings()
