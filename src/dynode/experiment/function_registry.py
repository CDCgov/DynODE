from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .function_ref import FunctionRef


@dataclass
class FunctionRegistry:
    """Small runtime registry for serializable function refs."""

    functions: dict[str, Callable[..., Any]] = field(default_factory=dict)

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        if not callable(fn):
            raise TypeError(f"Registered object {name!r} is not callable.")
        self.functions[name] = fn

    def resolve(self, ref: str | FunctionRef) -> Callable[..., Any]:
        name = ref if isinstance(ref, str) else ref.name
        try:
            return self.functions[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown function {name!r}. Known functions are: {sorted(self.functions)}."
            ) from exc

    def call(
        self,
        ref: FunctionRef,
        *args: Any,
        context: Mapping[str, Any] | None = None,
        data: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        fn = self.resolve(ref)
        return fn(
            *args, **ref.evaluate_kwargs(context=context, data=data), **kwargs
        )
