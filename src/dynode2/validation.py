import dataclasses

import typing
from typing import Callable, TypeVar

class ValidationError(Exception):
    def __init__(self, fieldname, value, validator, *args):
        # TODO: automatically construct human-readable message
        super().__init__(*args, fieldname, value, validator)

@dataclasses.dataclass
class Validator:
    def validate(self, p, x) -> bool:
        pass

@dataclasses.dataclass
class Inclusive(Validator):
    x: any

    def validate(self, p, x):
        pass

@dataclasses.dataclass
class Exclusive(Validator):
    x: any

    def validate(self, p, x):
        pass


@dataclasses.dataclass
class InRange(Validator):
    min_value: any
    max_value: any

    def validate(self, p, x):
        pass

@dataclasses.dataclass
class MaxValue:
    max_value: any

    def validate(self, p, x):
        pass

@dataclasses.dataclass
class GreaterThan:
    value: any

    def validate(self, p, x):
        pass

@dataclasses.dataclass
class LessThan:
    value: any

    def validate(self, p, x):
        pass

@dataclasses.dataclass
class LessThanOrEqualTo:
    value: any

    def validate(self, p, x):
        pass

@dataclasses.dataclass
class GreaterThanOrEqualTo:
    value: any

    def validate(self, p, x):
        pass

Positive = GreaterThan(0)
Negative = LessThan(0)
Nonnegative = GreaterThanOrEqualTo(0)

@dataclasses.dataclass
class Assert:
    expr: Callable[[], bool] | str # A string makes it easier to print/save for error reporting (makes you miss R)

    def validate(self, p, fieldname, value):
        locals().update(p.__dict__) # Bind all attributes of p as local variables

        if isinstance(self.expr, str):
            func = eval("lambda: {}".format(self.exp_str))
        else:
            func = self.expr
        result = func()
        if not isinstance(result, bool):
            if isinstance(self.expr, str):
                msg = "Validation string did not evaluate to bool: {}".format(self.expr)
            else:
                msg = "Validation lambda did not return bool"
            raise ValidationError(fieldname, value, self, msg)
        return result

def validate(p):
    for fieldname, typ in typing.get_type_hints(p, include_extras = True):
        if isinstance(typ, typing.Annotated):
            for mditem in typ.__metadata__:
                if isinstance(mditem, Validator):
                    value = getattr(p, fieldname)
                    if not validate(p, fieldname, value):
                        raise ValidationError(fieldname, value, mditem)
