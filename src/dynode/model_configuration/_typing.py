from typing import Annotated

from pydantic import BeforeValidator

# an str with no spaces and no leading numbers.


def _verify_name(name: str) -> str:
    """Validate to ensure names have no spaces and dont begin with a number."""
    if name[0].isnumeric():
        raise ValueError("Name can not start with a number.")
    elif " " in name:
        raise ValueError("Name can not have spaces.")
    return name


# a str with no spaces or leading numbers
DynodeName = Annotated[str, BeforeValidator(_verify_name)]
