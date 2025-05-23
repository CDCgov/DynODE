"""Utility package to contain all utility modules."""

import logging

from . import log
from .custom_log_formatter import CustomLogFormatter
from .datetime_utils import (
    date_to_epi_week,
    date_to_sim_day,
    sim_day_to_date,
    sim_day_to_epiweek,
)
from .log_decorator import log_decorator
from .splines import base_equation, conditional_knots, evaluate_cubic_spline
from .utils import (
    drop_keys_with_substring,
    flatten_list_parameters,
    identify_distribution_indexes,
    vectorize_objects,
)
from .vis_utils import (
    plot_checkpoint_inference_correlation_pairs,
    plot_mcmc_chains,
    plot_model_overview_subplot_matplotlib,
    plot_prior_distributions,
    plot_violin_plots,
)

# Fetching the global logger called dynode
logger = logging.getLogger("dynode")

__all__ = [
    "log",
    "log_decorator",
    "CustomLogFormatter",
    "logger",
    "sim_day_to_date",
    "sim_day_to_epiweek",
    "date_to_sim_day",
    "date_to_epi_week",
    "vectorize_objects",
    "flatten_list_parameters",
    "drop_keys_with_substring",
    "identify_distribution_indexes",
    "base_equation",
    "conditional_knots",
    "evaluate_cubic_spline",
    "plot_violin_plots",
    "plot_prior_distributions",
    "plot_mcmc_chains",
    "plot_checkpoint_inference_correlation_pairs",
    "plot_model_overview_subplot_matplotlib",
]
