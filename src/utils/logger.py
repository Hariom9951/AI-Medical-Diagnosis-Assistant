"""Logging Utility.

Provides centralized logger provisioning, standard formats, rotating file handlers,
and decorators for logging function-level exceptions with full trace details.
"""

import functools
import json
import logging
import logging.config
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

from src.constants import LOGGING_CONFIG_PATH

# Generic callable type for decorator annotations
F = TypeVar("F", bound=Callable[..., Any])


class AppLogger:
    """Configures and provides standard loggers for the application."""

    _configured: bool = False

    @classmethod
    def configure(
        self,
        config_path: Path = LOGGING_CONFIG_PATH,
        default_level: int = logging.INFO,
        default_file: Path = Path("logs/app.log"),
    ) -> None:
        """Initializes and configures the global logging settings.

        This method attempts to read from a standard JSON configuration file.
        If the file does not exist or is malformed, it sets up a robust fallback
        consisting of a console stream handler and a rotating file handler.

        Args:
            config_path (Path): Path to the JSON configuration file.
            default_level (int): Default logging level if fallback is triggered.
            default_file (Path): Log file target path for fallback rotation handler.
        """
        if self._configured:
            return

        # Ensure directory for logs exists
        default_file.parent.mkdir(parents=True, exist_ok=True)

        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                logging.config.dictConfig(config)
                self._configured = True
                return
            except Exception as e:
                # Direct fallback initialization
                logging.basicConfig(level=default_level)
                logging.warning("Failed to configure logging via JSON at %s. Error: %s", config_path, e)
        
        # Programmatic Fallback Configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(default_level)

        # Clear existing handlers to prevent duplicate outputs
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        # Define logging format
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console Handler (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(default_level)
        root_logger.addHandler(console_handler)

        # Rotating File Handler (10MB per file, 5 back-ups)
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=default_file,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(default_level)
            root_logger.addHandler(file_handler)
        except (OSError, PermissionError) as e:
            logging.warning("Unable to initialize rotating file handler at %s: %s", default_file, e)

        self._configured = True

    @classmethod
    def get_logger(self, name: str) -> logging.Logger:
        """Retrieves a logger instance configured for the specified namespace.

        Automatically ensures that logging configurations are booted at request time.

        Args:
            name (str): The namespace identifier (usually __name__).

        Returns:
            logging.Logger: The configured Logger instance.
        """
        if not self._configured:
            self.configure()
        return logging.getLogger(name)


def log_exception(logger: logging.Logger) -> Callable[[F], F]:
    """Decorator factory that intercepts and logs unhandled exceptions.

    Logs the exception type, message, and full traceback at the ERROR level
    before re-raising the exception for caller handling.

    Args:
        logger (logging.Logger): The logger instance to record the exception.

    Returns:
        Callable: The wrapped decorator function.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(
                    "Unhandled exception captured in function: '%s' inside module: '%s'. Details: %s",
                    func.__name__,
                    func.__module__,
                    e,
                )
                raise

        return cast(F, wrapper)
    return decorator
