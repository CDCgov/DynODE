from __future__ import annotations


class CompileError(ValueError):
    """
    Raised when a validated spec cannot be compiled into a runtime object.
    """