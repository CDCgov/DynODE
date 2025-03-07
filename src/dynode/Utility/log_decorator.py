import sys
import os
from datetime import datetime
from functools import wraps
from inspect import getframeinfo, stack
import logging


def log_decorator(_func=None):
    def log_decorator_info(func):
        @wraps(func)
        def log_decorator_wrapper(self, *args, **kwargs):
            logger = logging.getLogger("dynode")

            args_passed_in_function = [repr(a) for a in args]
            kwargs_passed_in_function = [f"{k}={v!r}" for k, v in kwargs.items()]

            ''' The lists of positional and keyword arguments is joined together to form final string '''
            formatted_arguments = ", ".join(args_passed_in_function + kwargs_passed_in_function)

            ''' Grabbing file and function name from stack then overriding those values in the CustomLogFormatter '''
            py_file_caller = getframeinfo(stack()[1][0])
            extra_args = {'func_name_override': func.__name__,
                          'file_name_override': os.path.basename(py_file_caller.filename)}

            start_time = datetime.now()
            logger.info(f"Arguments: {formatted_arguments} - Begin function", extra=extra_args)
            try:
                ''' log return value from the function '''
                value = func(self, *args, **kwargs)

                end_time = datetime.now()
                execution_time = end_time - start_time

                log_value = "\n".join(map(str,value))
                logger.info(f"Execution Time: {execution_time}", extra=extra_args)
                logger.info(f"Returned: - End function \n{log_value}", extra=extra_args)
            except Exception as ex:
                ''' log exception if occurs in function '''
                logger.error(f"Exception: {ex}", extra=extra_args)
                raise ex

            return value

        return log_decorator_wrapper

    if _func is None:
        return log_decorator_info
    else:
        return log_decorator_info(_func)