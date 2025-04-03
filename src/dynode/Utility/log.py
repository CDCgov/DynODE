"""Module that contains relevant functions for logging in DynODE.

use_logging is the primary function that sets up and configures the global dynode logger.
"""

import datetime
import logging
import os
import sys
from typing import Literal

from .custom_log_formatter import CustomLogFormatter

logger = logging.getLogger("dynode")


def use_logging(
    level: Literal[
        "none", "debug", "info", "warn", "error", "critical"
    ] = "info",
    output: Literal["file", "console", "both"] = "file",
    log_path: str = "./logs",
) -> None:
    """Set or disable logging within the dynode package.

    Uses standard python logging library to set up and customize a logger for DynODE. Logger instance can be retrieved
    from anywhere using logging.getLogger("dynode").

    Parameters
    ----------
    level : str, optional
        Log level desired. Choices from "none", "debug", "info", "warn", "error" and "critical". Defaults to "info".
    output : str, optional
        Output for logs. Choices from "console", "file", and "both". Defaults to "file".
    log_path : str, optional
        folder path to store log files. Defaults to "./logs".

    Notes
    -----
    Log level of NONE is considered CRITICAL + 1 which you may see in various places such as in this function as logging.CRITICAL + 1
    """
    # remove any loggers set previously
    global logger
    # clear logger handlers to avoid duplication in outputs
    logger.handlers.clear()
    # get the log level
    match level.lower():
        case "none":
            log_level = logging.CRITICAL + 1
            level_name = "NONE"
        case "debug":
            log_level = logging.DEBUG
            level_name = "DEBUG"
        case "info":
            log_level = logging.INFO
            level_name = "INFO"
        case "warn" | "warning":
            log_level = logging.WARN
            level_name = "WARN"
        case "error":
            log_level = logging.ERROR
            level_name = "ERROR"
        case "critical":
            log_level = logging.CRITICAL
            level_name = "CRITICAL"
        case _:
            print(
                f"Did not recognize {level} as a valid log level. Using INFO."
            )
            log_level = logging.INFO

    # logger.setLevel(log_level)
    logger.setLevel(log_level)
    formatter = CustomLogFormatter(
        "[%(levelname)s] %(asctime)s - %(filename)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d_%H:%M:%S",
    )

    # make log_path folder
    os.makedirs(log_path, exist_ok=True)
    # get logfile path
    start_time = datetime.datetime.now()
    now_string = f"{start_time:%Y-%m-%d_%Hh-%Mm-%Ss}"
    logfile = os.path.join(log_path, f"{now_string}.log")

    if not os.path.exists(logfile):
        with open(logfile, "w") as file:
            file.write("")

    # setting up console and file handlers
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)

    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    # check output
    if output.lower().startswith("console"):
        logger.addHandler(stream_handler)
    elif output.lower().startswith("file"):
        logger.addHandler(file_handler)
    elif output.lower().startswith("both"):
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    else:
        # set to stdout
        logger.addHandler(stream_handler)
        print("Did not recognize {output}. Saving to stdout.")
    print(f"Setting log level {level_name}.")
