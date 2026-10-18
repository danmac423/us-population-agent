from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: SecretStr
    # Light model for classification, rewriting and replies; the strong one is for writing SQL.
    anthropic_model_name: str
    anthropic_strong_model_name: str = "claude-sonnet-5-5"

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

    @property
    def census_schema(self) -> str:
        return f"{self.snowflake_census_database}.{self.snowflake_census_schema}"

    @property
    def geography_table(self) -> str:
        return f"{self.snowflake_database}.{self.snowflake_schema}.{self.snowflake_geography_table}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
