.PHONY: install fmt lint typecheck test test-unit test-integration eval eval-generate eval-judge check api frontend run clean

install:
	uv sync
	uv run pre-commit install

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run ty check

test:
	uv run pytest --cov

test-unit:
	uv run pytest tests/unit

test-integration:
	uv run pytest tests/integration

eval:
	uv run python tests/eval/run_eval.py run $(ARGS)

eval-generate:
	uv run python tests/eval/run_eval.py generate $(ARGS)

eval-judge:
	uv run python tests/eval/run_eval.py judge $(ARGS)

check: lint typecheck test

api:
	uv run uvicorn --factory us_population_agent.api:create_app --reload

frontend:
	uv run --group frontend streamlit run frontend/app.py

run:
	uv run us_population_agent

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache \) -not -path "./.venv/*" -prune -exec rm -rf {} +
	rm -rf .coverage coverage.xml htmlcov dist build *.egg-info
