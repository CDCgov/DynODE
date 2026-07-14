from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpyro

from dynode.runtime.layout.runtime_model import RuntimeParameterLayout

from .errors import ParameterSamplingError
from .parameter_options import ParameterSamplingOptions
from .parameter_utils import (
    name_of,
    prior_to_numpyro,
    sample_site_name,
    validate_dependencies_available,
)


def sample_prior_parameters(
    *,
    parameter_layout: RuntimeParameterLayout,
    context: MutableMapping[str, Any],
    data: Any | None = None,
    options: ParameterSamplingOptions | None = None,
) -> MutableMapping[str, Any]:
    """
    Sample all prior parameters in the compiled prior order.

    Prior order is determined by compile_model.py. This allows priors to depend
    on previously sampled priors when needed.
    """
    options = options or ParameterSamplingOptions()

    for prior_name in parameter_layout.prior_names:
        prior = parameter_layout.get_prior(prior_name)

        sample_prior_parameter(
            prior=prior,
            context=context,
            data=data,
            options=options,
        )

    return context


def sample_prior_parameter(
    *,
    prior: Any,
    context: MutableMapping[str, Any],
    data: Any | None = None,
    options: ParameterSamplingOptions | None = None,
) -> Any:
    """
    Sample a single PriorSpec and write the sampled value into context.
    """
    options = options or ParameterSamplingOptions()

    prior_name = name_of(prior)

    if prior_name in context and not options.allow_initial_context_overwrite:
        raise ParameterSamplingError(
            f"Parameter context already contains prior {prior_name!r}. "
            "Set allow_initial_context_overwrite=True only if this is intentional."
        )

    if options.validate_dependencies:
        validate_dependencies_available(
            obj=prior,
            context=context,
            obj_label=f"Prior {prior_name!r}",
        )

    distribution = prior_to_numpyro(
        prior=prior,
        context=context,
        data=data,
    )

    site_name = sample_site_name(prior, options.scope)

    value = numpyro.sample(
        site_name,
        distribution,
    )

    context[prior_name] = value

    return value
