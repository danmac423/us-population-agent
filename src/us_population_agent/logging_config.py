import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configures root logging with the application log format.

    Args:
        level: Logging level name, case-insensitive (e.g. "info", "WARNING").
    """
    logging.basicConfig(level=level.upper(), format=LOG_FORMAT)
