"""Public runtime API for DynODE.

The implementation is organized by concern under subpackages:
- dynode.runtime.compile
- dynode.runtime.context
- dynode.runtime.execution
- dynode.runtime.layout
- dynode.runtime.models

Legacy module paths such as dynode.runtime.compile_model remain as compatibility
re-exports.
"""

from .compile import (
    CompileError,
    CompileOptions,
    ExperimentCompileError,
    compile_experiment,
    compile_model,
    compile_model_from_dict,
    compile_parameter_block,
)
from .context import (
    DataBundle,
    ExperimentRuntime,
    ModelInstanceRuntime,
    RuntimeContext,
)
from .execution import (
    OdeSolverError,
    OdeSolverOptions,
    ParameterSamplingError,
    ParameterSamplingOptions,
    StateBuilderError,
    StateBuilderOptions,
    build_initial_state,
    build_initial_state_dict,
    build_initial_state_flat,
    evaluate_parameter_context_without_numpyro,
    sample_parameters,
    solution_final_state_dict,
    solution_final_state_flat,
    solution_ys_as_state_dict,
    solve_ode,
    split_parameter_context,
)
from .layout import (
    RuntimeCompartment,
    RuntimeDimension,
    RuntimeModel,
    RuntimeParameterLayout,
    RuntimeTransmission,
    StateLayout,
)
from .models import DynodeExperiment, DynodeModel

__all__ = [name for name in globals() if not name.startswith("_")]
