import pandas as pd
import json
import numpy as np
import os

# Local path to the medians csv file.
az_output_path = "/output/fifty_state_5strain_2202_2404/SMH_5strains_071624/"


def retrieve_checkpoint(states):
    total = {}
    total_df = pd.DataFrame(total)
    for state in states:
        state_config_path = os.path.join(az_output_path, state)
        print("Retrieving " + state + "\n")
        state_checkpoint_path = os.path.join(state_config_path, "checkpoint.json")
        df = pd.read_csv(state_checkpoint_path)
        for x in df.columns:
            df[x] = pd.Series(df[x].to_numpy().flatten())
        total_df = pd.concat([total_df, df], axis=0)
    return total_df


def show_statistics(states, bad):
    if bad == True:
        total_df = retrieve_checkpoint(states)
        statistics = {}
        statistics["mean_b"] = total_df.mean()
        statistics["min_b"] = total_df.min()
        statistics["max_b"] = total_df.max()
        statistics["std_b"] = total_df.std()
        statistics["median_b"] = total_df.median()
        statistics = pd.DataFrame(statistics)
    else:
        total_df = retrieve_checkpoint(states)
        statistics = {}
        statistics["mean_g"] = total_df.mean()
        statistics["min_g"] = total_df.min()
        statistics["max_g"] = total_df.max()
        statistics["std_g"] = total_df.std()
        statistics["median_g"] = total_df.median()
        statistics = pd.DataFrame(statistics)
    return statistics


if __name__ == "__main":

    # Select bad and good sets of states.
    # BUT ONE NEEDS TO CONSIDER THE STATES_OMIT! seguir a logica do postaz!!!
    bad_states = [
        "DE",
        "GA",
        "HI",
        "IA",
        "ID",
        "IL",
        "IN",
        "KS",
        "KY",
        "MD",
        "VA",
        "VT",
        "WI",
        "WY",
    ]
    good_states = [
        "AK",
        "AR",
        "CA",
        "FL",
        "LA",
        "MN",
        "TX",
        "NH",
        "NJ",
        "NM",
        "NY",
        "UT",
        "SD",
        "MT",
        "NC",
    ]

    # Fazer bem parecido, mas ao inves de acessar o medians csv file,
    # acessa o checkpoint de cada estado bad, concatena esses checkpoints vertically, e calcula as estatisticas std, median, mean, max, min de cada coluna
    # pode fazer tbm inter-quantile distance.

    statistics_bad = show_statistics(bad_states, bad=True)
    statistics_good = show_statistics(good_states, bad=0)

    df_total = pd.concat([statistics_bad, statistics_good], axis=1)
    output_file = "good_bad_stats.csv"
    output_dir = f"output/{output_file}"
    df_total.to_csv(output_dir, index=True)
