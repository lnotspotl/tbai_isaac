#!/usr/bin/env python3

import logging
from rich.logging import RichHandler
from rich.console import Console


def get_logger(
    name: str, log_to_stdout: bool = False, log_to_file: bool = False, level: str = "INFO", log_file: str = "./logs.txt"
) -> logging.Logger:
    assert log_to_stdout or log_to_file, "At least one of log_to_stdout or log_to_file must be True"
    assert level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], f"Invalid log level: {level}"

    handlers = list()
    if log_to_file:
        console = Console(file=open(log_file, "w"))  # TODO: this file should be closed
        handlers.append(RichHandler(console=console))

    if log_to_stdout:
        handlers.append(RichHandler(rich_tracebacks=True, tracebacks_show_locals=True))

    FORMAT = "%(name)s: %(message)s"
    logging.basicConfig(
        level=logging.CRITICAL,
        format=FORMAT,
        datefmt="[%X]",
        handlers=handlers,
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
