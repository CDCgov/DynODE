from __future__ import annotations


class OdeSolverError(RuntimeError):
    """
    Raised when the ODE solve cannot be configured or completed.
    """


class ParameterSamplingError(RuntimeError):
    """
    Raised when parameters cannot be sampled or deterministic parameters cannot
    be resolved.
    """


class StateBuilderError(ValueError):
    """
    Raised when an initial state cannot be built from RuntimeModel and
    InitializerSpec.
    """
