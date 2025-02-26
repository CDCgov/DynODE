import sys, os, functools
import logging

def log_decorator(_func=None):
    def log_decorator_info(func):
        @functools.wraps(func)
        def log_decorator_wrapper(self, *args, **kwargs):
            ''' Build logger object '''
            logger = logging.getLogger("dynode")

            ''' log function begining '''
            logger.info("Begin function")
            try:
                ''' log return value from the function '''
                value = func(self, *args, **kwargs)
                logger.info(f"Returned: - End function {value!r}")
            except:
                ''' log exception if occurs in function '''
                logger.error(f"Exception: {str(sys.exc_info()[1])}")
                raise

            return value

        return log_decorator_wrapper

    if _func is None:
        return log_decorator_info
    else:
        return log_decorator_info(_func)