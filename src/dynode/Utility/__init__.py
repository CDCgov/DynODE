"""Utility package to contain all utility modules"""

import logging

from . import log
from .custom_log_formatter import CustomLogFormatter
from .log_decorator import log_decorator

# Fetching the global logger called dynode
logger = logging.getLogger("dynode")

__all__ = ["log", "log_decorator", "CustomLogFormatter", "logger"]
