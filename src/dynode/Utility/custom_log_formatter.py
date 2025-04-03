import logging


class CustomLogFormatter(logging.Formatter):
    """Custom Log Formatter class that inherits Python's logging.Formatter class

    Uses standard python logging library to inherit logging.Formatter class.
    CustomLogFormatter overrides the expected format function in order to check if the log record
    contains either the func_name_override or file_name_override attribute. If the record contains one or either of those
    attributes it sets the records funcName and/or filename attribute.

    CustomLogFormatter was created and intended for when a function is decorated with log_decorator().
    It captures the decorated functions name and file name to include in the log.

    For inline logging calls the CustomLogFormatter will behave exactly the same as logging.Formatter.
    Unless the user passes func_name_override and/or file_name_override attributes into logging.[log level](exra=[extra arguments])

    Below parameters are passed to CustomLogFormatter when used and derive from logging.Formatter

    Parameters
    ----------
    fmt : str, optional
        A format string in the given style for the logged output as a whole.
        The possible mapping keys are drawn from the LogRecord object’s LogRecord attributes.
        If not specified, '%(message)s' is used, which is just the logged message. Defaults to None.
    datefmt : str, optional
        A format string in the given style for the date/time portion of the logged output.
        If not specified, the default described in formatTime() is used. Defaults to None.
    style : str, optional
        Can be one of '%', '{' or '$' and determines how the format string will be merged with its data: using one of printf-style String Formatting (%), str.format() ({) or string.Template ($).
        This only applies to fmt and datefmt (e.g. '%(message)s' versus '{message}'), not to the actual log messages passed to the logging methods.
        However, there are other ways to use {- and $-formatting for log messages. Defaults to '%'.
    validate : bool, optional
        If True (the default), incorrect or mismatched fmt and style will raise a ValueError; for example, logging.Formatter('%(asctime)s - %(message)s', style='{').
    defaults : dict[str, Any], optional
        A dictionary with default values to use in custom fields. For example, logging.Formatter('%(ip)s %(message)s', defaults={"ip": None}). Defaults to None.
    """

    def format(self, record):
        if hasattr(record, "func_name_override"):
            record.funcName = record.func_name_override
        if hasattr(record, "file_name_override"):
            record.filename = record.file_name_override
        return super(CustomLogFormatter, self).format(record)
