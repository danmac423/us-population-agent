from us_population_agent.config import DEFAULT_NAME
from us_population_agent.services.greeter import greet


def test_greet_default() -> None:
    assert greet(DEFAULT_NAME) == "Hello, world!"


def test_greet_loud() -> None:
    assert greet("Daniel", loud=True) == "HELLO, DANIEL!"
