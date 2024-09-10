"""
This script provides an easy way to pull down all publically available data used in the model.

This script uses available APIs to pull down data sources, if data is not publically available it will not be
pulled, even if you have permissions to view it (you must pull the data manually in that case).
"""

# import os

import pandas as pd

# import requests

states = pd.read_csv("fips_to_name.csv")
# pull demographic-data
for state in states["stname"]:
    # get population_rescaled_age_distributions

    state = state.replace(" ", "_")
    if state == "United_states":
        filename = "United_States_country_level_age_distribution_85.csv"
    else:
        filename = (
            "United_States_subnational_%s_age_distribution_85.csv" % state
        )


# pull covid hosp data

# pull serological-data (?)

# pull vaccination-data (?)

# pull variant-data
