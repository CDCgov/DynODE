import sys, os, datetime
import logging
from dynode.utility.custom_log_formatter import CustomLogFormatter

logger = logging.getLogger("dynode")


def use_logging(level: str = "INFO", output: str = "stdout", log_path: str = "./logs") -> None:
    """
    Sets or disables logging with the dynode package.

    Args:
        level (str, optional): Log level desired. Choices from "None", "DEBUG", "INFO", "WARN", "ERROR" and "CRITICAL". Defaults to "INFO".
        output (str, optional): Output for logs. Choices from "stdout", "file", and "both". Defaults to "stdout".
        log_path (str, optional): folder path to store log files. Defaults to "./logs".
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
            print(f"Did not recognize {level} as a valid log level. Using INFO.")
            log_level = logging.INFO

    # logger.setLevel(log_level)
    logger.setLevel(log_level)
    formatter = CustomLogFormatter("[%(levelname)s] %(asctime)s - %(filename)s - %(funcName)s: %(message)s", datefmt="%Y-%m-%d_%H:%M:%S")

    # make log_path folder
    os.makedirs(log_path, exist_ok=True)
    # get logfile path
    run_time = datetime.datetime.now()
    now_string = f"{run_time:%Y-%m-%d_%Hh-%Mm-%Ss}"
    logfile = os.path.join(log_path, f"{now_string}.log")

    if not os.path.exists(logfile):
        with open(logfile, 'w') as file:
            file.write('')

    # setting up console and file handlers
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)

    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    # check output
    if output.lower().startswith("std") or output.lower().startswith("console"):
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

