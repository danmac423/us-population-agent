FROM python:3.13-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1


# The API: runtime dependencies only, no dev tools and no Streamlit.
FROM base AS api

# Dependencies first, so this layer is cached until pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD ["uvicorn", "--factory", "us_population_agent.api:create_app", "--host", "0.0.0.0", "--port", "8000"]


# The Streamlit UI: only the `frontend` group, it never imports the agent package.
FROM base AS frontend

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --only-group frontend --no-install-project

COPY frontend ./frontend

EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
