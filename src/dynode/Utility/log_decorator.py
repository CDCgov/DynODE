"""A module that defines a decorator for DynODEs global logger."""

import logging
import os
from datetime import datetime
from functools import wraps
from inspect import getframeinfo, stack


def log_decorator(_func=None):
    """Outermost log decorator function.

    log_decorator is the function used to wrap a function the user wishes to log.
    This outermost function allows the user to either use @log_decorator() convention
    or call log_decorator() and pass the function they wish to wrap.

    Parameters
    ----------
    _func : function, optional
        log_decorator can be called as a function and passed the function it wraps. Defaults to None.
    """

    def log_decorator_info(func):
        """Middle log decorator function.

        log_decorator_info exists mostly for the @wraps(func) call which allows us to pass the func metadata into the log_decorator_wrapper.
        Allowing for log_decorator_wrapper to grab the func's *args, **kwargs, and other info such as func.__name__.
        Also allowing for the decorator to do work before and after the func is called.

        Parameters
        ----------
        func : function
            The function that is wrapped by the decorator.
        """

        @wraps(func)
        def log_decorator_wrapper(self, *args, **kwargs):
            """Innermost log decorator function.

            Gets decorated functions name, file name, arguments passed, and starts an execution timer.
            Then creates a log using all of that information. Next we try executing the function, stop the timer, and store return value if any exists.
            A log is created for using all the after execution information. If any exceptions occur during execution we log and raise.
            Finally returning any value that was returned from the function.
            """
            logger = logging.getLogger("dynode")

            args_passed_in_function = [repr(a) for a in args]
            kwargs_passed_in_function = [
                f"{k}={v!r}" for k, v in kwargs.items()
            ]

            """ The lists of positional and keyword arguments is joined together to form final string """
            formatted_arguments = ", ".join(
                args_passed_in_function + kwargs_passed_in_function
            )

            """ Grabbing file and function name from stack then overriding those values in the CustomLogFormatter """
            py_file_caller = getframeinfo(stack()[1][0])
            extra_args = {
                "func_name_override": func.__name__,
                "file_name_override": os.path.basename(
                    py_file_caller.filename
                ),
            }

            start_time = datetime.now()
            logger.info(
                f"Arguments: {formatted_arguments} - Begin function",
                extra=extra_args,
            )
            try:
                """log return value from the function"""
                value = func(self, *args, **kwargs)

                end_time = datetime.now()
                execution_time = end_time - start_time

                log_value = "\n".join(map(str, value))
                logger.info(
                    f"Execution Time: {execution_time}", extra=extra_args
                )
                logger.info(
                    f"Returned: - End function \n{log_value}", extra=extra_args
                )
            except Exception as ex:
                """log exception if occurs in function"""
                logger.error(f"Exception: {ex}", extra=extra_args)
                raise ex

            return value

        return log_decorator_wrapper

    if _func is None:
        return log_decorator_info
    else:
        return log_decorator_info(_func)
