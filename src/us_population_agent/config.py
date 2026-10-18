from functools import lru_cache

from pydantic import Field, SecretStr
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
    default_thread_id: str = "demo"  # conversation used when --thread is not given

    # Agent limits.
    max_repairs: int = Field(default=2, ge=0)  # how many times a failing query is rewritten before giving up
    max_search_queries: int = Field(default=4, ge=1)  # catalog search queries per question
    max_candidates: int = Field(default=8, ge=1)  # candidate tables passed to the table selector
    search_limit: int = Field(default=5, ge=1)  # tables returned by one catalog search query
    max_context_tokens: int = Field(default=2000, ge=1)  # conversation history passed to the model (estimated tokens)

    # Model calls: transient errors (network, 429, 5xx) are retried with exponential backoff.
    llm_max_retries: int = Field(default=4, ge=0)
    llm_timeout_seconds: int = Field(default=30, ge=1)

    # Snowflake timeouts.
    query_timeout_seconds: int = Field(default=30, ge=1)
    login_timeout_seconds: int = Field(default=20, ge=1)

    @property
    def census_schema(self) -> str:
        return f"{self.snowflake_census_database}.{self.snowflake_census_schema}"

    @property
    def geography_table(self) -> str:
        return f"{self.snowflake_database}.{self.snowflake_schema}.{self.snowflake_geography_table}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
