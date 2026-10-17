from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: SecretStr
    anthropic_model_name: str

    database_url: str

    snowflake_account: str
    snowflake_user: str
    snowflake_password: SecretStr
    snowflake_role: str
    snowflake_warehouse: str
    # App database: Cortex Search service, columns table and geography table.
    snowflake_database: str
    snowflake_schema: str
    snowflake_search_service: str
    snowflake_columns_table: str
    snowflake_geography_table: str
    # Census data shared from the Marketplace (read-only).
    snowflake_census_database: str
    snowflake_census_schema: str

    log_level: str = "WARNING"


@lru_cache
def get_settings() -> Settings:
    return Settings()
