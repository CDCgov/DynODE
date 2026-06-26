from __future__ import annotations

from datetime import date

import jax.numpy as jnp

STRAIN_NAMES: tuple[str, str, str] = ("H1", "H3", "B")

AGE_BIN_EDGES: tuple[tuple[int, int], ...] = (
    (0, 4),
    (5, 17),
    (18, 49),
    (50, 64),
    (65, 99),
)

AGE_LABELS: tuple[str, ...] = ("0-4", "5-17", "18-49", "50-64", "65+")

US_POPULATION_BY_AGE_5 = jnp.array([19.3e6, 55.0e6, 137.9e6, 63.8e6, 54.5e6])

CONTACT_MATRIX_5 = jnp.array(
    [
        [0.10045306, 0.06873629, 0.06493611, 0.03236584, 0.02867914],
        [0.18407544, 0.48844703, 0.16735415, 0.1111201, 0.0815753],
        [0.4098531, 0.40611078, 0.54371794, 0.40482455, 0.23568303],
        [0.09430659, 0.12378313, 0.18494981, 0.28700569, 0.14051782],
        [0.05519881, 0.05888843, 0.06989965, 0.09549161, 0.2165984],
    ]
)

DEFAULT_DURATION_DAYS = 364 + 49

DEFAULT_INIT_DATES: dict[int, date] = {
    2015: date(2015, 8, 15),
    2016: date(2016, 8, 13),
    2017: date(2017, 8, 12),
    2018: date(2018, 8, 11),
    2019: date(2019, 8, 10),
    2022: date(2022, 8, 13),
    2023: date(2023, 8, 12),
    2024: date(2024, 8, 10),
    2025: date(2025, 8, 9),
}

# Reported vaccine effectiveness against hospitalization from the old example.
HOSPITALIZATION_VE_BY_YEAR: dict[int, float] = {
    2015: 0.48,
    2016: 0.40,
    2017: 0.38,
    2018: 0.29,
    2019: 0.39,
    2022: 0.36,
    2023: 0.40,
    2024: 0.48,
    2025: 0.37,
}


# The old experiment translated VE against hospitalization to VE against infection.
def ve_hospitalization_to_ve_infection(ve_hospitalization: float):
    return 2 / 3 - 1 / 3 * jnp.sqrt(4 - 3 * ve_hospitalization)


HOSP_LIKELIHOOD_WEIGHT = 1.0
SUBTYPE_LIKELIHOOD_WEIGHT = 1.0
B_SHARE_WEIGHT = 5.0
HOSP_NEGBIN_INF_FACTOR = 1 / 100
SUBTYPE_DIRMUL_INF_FACTOR = 1 / 80

A_WANING_DAYS = 900.0
B_WANING_DAYS = 1800.0
RECOVERED_TO_SUSCEPTIBLE_DAYS = 21.0

DEFAULT_BETA_CHANGE_POINTS = jnp.array([0.0, 141.0, 155.0])
