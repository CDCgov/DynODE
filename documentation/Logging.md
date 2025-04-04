# Logging Functions & Decorator - Documentation

### Overview
This document is intended to provide user with information on logging in DynODE.
As well as, document the various features, functions, and decorators that may be available.
Additionally, you will find useful examples regarding inline logging, decorating your functions, turning on/off logging, and where to expect output.

## Logging Components
### Directory
```
./src/dynode/utility
./src/logs
```
### Files
```
log.py
log_decorator.py
custom_log_formatter.py
```
###  Use Logging and CLI Subparsers
```python
import argparse
from dynode.utility import log

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

log_parser = subparsers.add_parser('log', help="Subcommands for logging")
log_parser.add_argument(
    '-l',
    '--level',
    default='info',
    choices=['debug', 'info', 'warning', 'error', 'critical'],
    help="set the logging level the default if info"
)
log_parser.add_argument(
    '-o',
    '--output',
    default='file',
    choices=['file', 'console', 'both'],
    help="print logs to console, file, or both the default is file"
)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.subcommand == 'log':
        log.use_logging(level=args.level, output=args.output)
```
#### import log
```python
from dynode.utility import log
```
#### use_logging
Notice the function above use_logging located in log.py initializes a global logger called "dynode".
```python
if args.subcommand == 'log':
        log.use_logging(level=args.level, output=args.output)
```
#### log subparser cli command
--help command will display a list of commands and their options
```bash
dynode_main.py log --help

usage: example_end_to_end_run.py log [-h]
                                     [-l {debug,info,warning,error,critical}]
                                     [-o {file,console,both}]

options:
  -h, --help            show this help message and exit
  -l, --level {debug,info,warning,error,critical}
                        set the logging level the default if info
  -o, --output {file,console,both}
                        print logs to console, file, or both the default is file
```
The log command can be called by itself. The default value for level will be set to info and the default value for output will be file. \
Note: If you do not include the command log then logging will be turned off.
```bash
dynode_main.py log
```
log --level (same as log -l) sets the logger level to one of five options debug, info, warning, error, critical.
Each log level will include logs from every level above it. Debug being the lowest level will include logs from every level. \
```bash
dynode_main.py log --level debug
```
log --output (same as log -o) sets the logger output to one of three options file, console, or both. If the output is set to file or both
then log file will be written to the above directory located under Directory ./src/logs.
```bash
dynode_main.py log --output both
```

## Inline Logging and Logging Decorator
### Logging Decorator
First you will want to import the decorator in which ever file you want to use it.

```python
from utility.log_decorator import log_decorator
```
Next all you need to do is decorate any function you would like to log. \
Note: Your function can take arguments or not and it can return a value or not. Function arguments and return values will be logged if they are present.
```python
@log_decorator()
def some_function():
    '''function code'''
    return value
```
### Inline Logging
The global logger instance is created in the utility packages __init__.py file. \
The dynode logger can be fetched in the dynode package using the below import statement.
```python
from .utility import logger

# If needed dynode logger can be fetched using the below line
logging.getLogger("dynode")
```
Next you can add logging statements by specifying at which of the 5 levels you would like to log anywhere in your functions and classes. \
Note: This can be used in addition to a logging decorator on a function and is encouraged for more specific things you would like to log in a function.
```python
logger.debug("This is a debug statement")
logger.info("This is an info statement")
logger.warning("This is a warning statement")
logger.error("This is an error statement")
logger.critical("This is a critical statement")

# example with f-string
try:
    some_function()
except Exception as ex:
    logger.error(f"Exception: {ex}")
    raise ex
```

## Output
Example output below. Run at level debug so logs of all levels are present.
```python
[DEBUG] 2025-03-04_14:19:27 - abstract_initializer.py - load_initial_population_fractions: Creating populations_path based on DEMOGRAPHIC_DATA_PATH in config.
[DEBUG] 2025-03-04_14:19:27 - abstract_initializer.py - load_initial_population_fractions: Set populations path as examples/data/demographic-data/population_rescaled_age_distributions/.
[DEBUG] 2025-03-04_14:19:27 - abstract_initializer.py - load_initial_population_fractions: Returning values from utils.load_age_demographics()
[INFO] 2025-03-04_14:19:29 - example_end_to_end_run.py - get_initial_state: Arguments:  - Begin function
[INFO] 2025-03-04_14:19:29 - example_end_to_end_run.py - get_initial_state: Execution Time: 0:00:00.000240
[INFO] 2025-03-04_14:19:29 - example_end_to_end_run.py - get_initial_state: Returned: - End function
[[[[0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
    5.23313424e+04]
   [2.44241556e+02 2.03953596e+03 4.87913342e+02 2.29051125e+03
    3.84410877e+01]
   [9.44460863e+03 3.14222477e+03 3.95783878e+03 7.82299939e+03
    4.50363904e+01]]
...
```
The format for each log line is below.
```python
[LEVEL] yyyy-mm-dd_h:m:s - file name - function name: log message

# code in log.py that defines the format
formatter = CustomLogFormatter("[%(levelname)s] %(asctime)s - %(filename)s - %(funcName)s: %(message)s", datefmt="%Y-%m-%d_%H:%M:%S")
```
