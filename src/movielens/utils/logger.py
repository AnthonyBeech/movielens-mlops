import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from movielens.config.config import PROJECT_ROOT


def setup_logging(
    logger_name: str | None = None,
    log_level: int = logging.INFO,
    log_file: str = PROJECT_ROOT / "logs/src.log",
    max_bytes: int = 10_000_000,  # 10 MB
    backup_count: int = 3,
) -> logging.Logger:
    """
    Configure and return a logger for the given name or the root logger.

    Args:
        logger_name (str, optional): The name of the logger. If None, the root logger is used.
        log_level (int, optional): Logging level (e.g., logging.DEBUG, logging.INFO, etc.). Defaults to logging.INFO.
        log_file (str, optional): File path for the log file. Defaults to 'src.log'.
        max_bytes (int, optional): Max size (in bytes) of the log file before a new one is created.
                                   Defaults to 10,000,000 (10 MB).
        backup_count (int, optional): Number of rotated log files to keep. Defaults to 3.

    Returns:
        logging.Logger: Configured logger object.

    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Clear existing handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Common log format
    formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 1. Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (Rotating)
    if log_file:
        log_file_path = Path(log_file)

        # IMPORTANT: Create the parent folder for the log file if it doesn't exist
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=str(log_file_path),  # convert Path to str for older Python versions
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
