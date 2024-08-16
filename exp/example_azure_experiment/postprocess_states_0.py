"""A super basic postprocessing script which just combines all the azure_visualizer_timeline's together for each state into one """

import argparse
import os

import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument(
    "-j",
    "--jobid",
    type=str,
    help="job-id of the state being run on Azure",
    required=True,
)
args = parser.parse_args()
jobid = args.jobid
output_path = os.path.join("/output/example_azure_experiment", jobid)
states = os.listdir(output_path)
dfs = list()
for st in states:
    print(st)
    state_path = os.path.join(output_path, st)
    csv_path = os.path.join(state_path, "azure_visualizer_timeline.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["state"] = st
        dfs.append(df)

final_df = pd.concat(dfs)
final_df.to_csv(
    os.path.join(output_path, "all_states_timeseries.csv"), index=False
)
