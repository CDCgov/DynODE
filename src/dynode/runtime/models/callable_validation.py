from __future__ import annotations

from typing import Any


def require_callable(value: Any, field_name: str) -> None:
    if not callable(value):
        raise TypeError(f"{field_name} must be callable.")


def require_optional_callable(value: Any, field_name: str) -> None:
    if value is not None and not callable(value):
        raise TypeError(f"{field_name} must be callable if provided.")
