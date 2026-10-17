# Import from elsewhere in the package: absolute (always starts with the package name).
from us_population_agent.config import APP_NAME, DEFAULT_NAME
from us_population_agent.services.greeter import greet


def main() -> None:
    print(f"Starting {APP_NAME}...")
    print(greet(DEFAULT_NAME))
