FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1

# Dependencies first, so this layer is cached until pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --group frontend --no-install-project

COPY src ./src
COPY frontend ./frontend
RUN uv sync --frozen --no-dev --group frontend

# The API by default; docker-compose overrides the command for the frontend.
EXPOSE 8000 8501
CMD ["uvicorn", "--factory", "us_population_agent.api:create_app", "--host", "0.0.0.0", "--port", "8000"]
