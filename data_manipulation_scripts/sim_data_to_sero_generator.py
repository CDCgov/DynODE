import datetime
import sqlite3

import pandas as pd

SIM_START_DATE = datetime.date(2020, 2, 10)
SIM_INPUT_PATH = "data/abm-data/sim_data_scaled_us.sqlite"
MODEL_INIT_DATE = datetime.date(2022, 2, 11)
OUTPUT_DATA_PATH = "data/abm_population.csv"


# Create your connection.
cnx = sqlite3.connect(SIM_INPUT_PATH)
res = cnx.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("available tables: ")
tables = [name[0] for name in res.fetchall()]
print(tables)

retention = 1 if "retention" in tables else 0
if retention:
    print("subsampling of infection history detected")

days_diff = (MODEL_INIT_DATE - SIM_START_DATE).days
print("model init date: " + str(days_diff))

### LOAD IN TABLES ##########################################################

sim_people = pd.read_csv(
    "data/abm-data/sim_ages.txt", sep=" "
)  # all people in the simulation and their ages
# all infections that occured before init date.
# we exclude those who died from their infections, unless they died after the model init date
if retention:
    sql = """SELECT inf.inf_owner_id, inf.strain, inf.infected_time, inf.infectious_start,
    inf.infectious_end, r.retain
    FROM infection_history AS inf
    JOIN retention AS r ON inf.inf = r.inf
    WHERE ((death_time == 2147483647 AND infected_time <= {}) OR
    (death_time > {} AND infected_time <= {})) AND retain = 1"""
else:
    sql = """SELECT inf_owner_id, strain, infected_time, infectious_start, infectious_end
    FROM infection_history
    WHERE (death_time == 2147483647 AND infected_time <= {}) OR
    (death_time > {} AND infected_time <= {})"""

infection_history = pd.read_sql_query(
    sql=sql.format(days_diff, days_diff, days_diff),
    con=cnx,
)

vaccination_history = (
    pd.read_sql_query(
        "SELECT * FROM vaccination_history WHERE vax_sim_day <= "
        + str(days_diff),
        cnx,
    )
    .groupby(by="p_id", as_index=False)
    .agg({"dose": len, "vax_sim_day": max})
)

### JOIN TABLES TOGETHER ##########################################################

infection_history_by_infected = infection_history.groupby(
    "inf_owner_id", as_index=False
).agg(
    {
        "strain": list,
        "infected_time": max,
        "infectious_start": max,
        "infectious_end": max,
    }
)

infection_history_by_infected = pd.merge(
    sim_people,
    infection_history_by_infected,
    left_on="pid",
    right_on="inf_owner_id",
    how="left",
).drop("inf_owner_id", axis=1)

infection_history_by_infected = pd.merge(
    infection_history_by_infected,
    vaccination_history,
    left_on="pid",
    right_on="p_id",
    how="left",
).drop("p_id", axis=1)

infection_history_by_infected.columns = pd.Index(
    [
        "pid",
        "age",
        "strains",
        "last_infected_date",
        "last_infectious_start_date",
        "last_infectious_end_date",
        "num_doses",
        "last_vax_date",
    ]
)

### FILL NA VALUES WITH GOOD DEFAULTS AND SET TO INT FROM FLOAT.
infection_history_by_infected["strains"] = (
    infection_history_by_infected["strains"].fillna("").apply(list)
)
infection_history_by_infected["num_doses"] = (
    infection_history_by_infected["num_doses"].fillna(0).astype(int)
)
infection_history_by_infected["last_vax_date"] = (
    infection_history_by_infected["last_vax_date"].fillna(-1).astype(int)
)
infection_history_by_infected["last_infected_date"] = (
    infection_history_by_infected["last_infected_date"].fillna(-1).astype(int)
)
infection_history_by_infected["last_infectious_start_date"] = (
    infection_history_by_infected["last_infectious_start_date"]
    .fillna(-1)
    .astype(int)
)
infection_history_by_infected["last_infectious_end_date"] = (
    infection_history_by_infected["last_infectious_end_date"]
    .fillna(-1)
    .astype(int)
)

### CREATE TSLIE (time since last immunogenetic event) #######################################
# last_vax_date can be a max of MODEL_INIT_DATE
# last_infectious_end_date can be greater than MODEL_INIT_DATE,
#       producing a negative TSLIE value. Inidicating active infection / exposure
infection_history_by_infected["TSLIE"] = [
    int(days_diff - max(last_inf_date, last_vax_date))
    for (last_inf_date, last_vax_date) in zip(
        infection_history_by_infected["last_infectious_end_date"],
        infection_history_by_infected["last_vax_date"],
    )
]

### CREATE NUM INFECTIONS COL #######################################
infection_history_by_infected["num_infections"] = [
    len(x) for x in infection_history_by_infected["strains"]
]

infection_history_by_infected["infectious"] = [
    int(start_date <= days_diff)
    for start_date in infection_history_by_infected[
        "last_infectious_start_date"
    ]
]
infection_history_by_infected.loc[
    infection_history_by_infected["TSLIE"] >= 0, "infectious"
] = 0

# drop date columns since they arent needed anymore
infection_history_by_infected = infection_history_by_infected.drop(
    [
        "last_infected_date",
        "last_infectious_start_date",
        "last_infectious_end_date",
        "last_vax_date",
    ],
    axis=1,
)


#### REPLACE EMPTY SET WITH NONE  #################################
def replace_empty_set_with_none(input_set):
    if len(input_set) == 0:
        return ""
    else:
        return ",".join(input_set)


infection_history_by_infected["strains"] = infection_history_by_infected[
    "strains"
].apply(replace_empty_set_with_none)

infection_history_by_infected.to_csv(OUTPUT_DATA_PATH, index=False)

print("successfully created intermediate sim data")
