# %%
import argparse
import os

import pandas as pd
from tqdm import tqdm

OUTPUT_PATH = "/output/projections_2407_2507/"


def save_collated_timeseries(output_path, job_id):
    job_path = os.path.join(output_path, job_id)
    states = os.listdir(job_path)
    state_dfs = []
    for st in tqdm(states, desc="processing states "):
        scenario_dfs = []
        state_path = os.path.join(job_path, st)
        scens = os.listdir(state_path)

        for sc in scens:
            csv_path = os.path.join(
                state_path, sc, "azure_visualizer_timeline.csv"
            )
            if os.path.exists(
                os.path.join(state_path, sc, "azure_visualizer_timeline.csv")
            ):
                df = pd.read_csv(
                    csv_path,
                    usecols=[
                        "chain_particle",
                        "date",
                        "pred_hosp_0_17",
                        "pred_hosp_18_49",
                        "pred_hosp_50_64",
                        "pred_hosp_65+",
                        "JN1_strain_proportion",
                        "KP_strain_proportion",
                        "X_strain_proportion",
                    ],
                )

                df["state"] = st
                df["scenario"] = sc
                scenario_dfs.append(df)

        state_df = pd.concat(scenario_dfs)
        state_df.to_csv(
            os.path.join(state_path, "all_scens_projections.csv"), index=False
        )
        state_dfs.append(state_df)
    all_states_df = pd.concat(state_dfs)
    all_states_df.to_csv(
        os.path.join(job_path, "all_states_projections.csv"), index=False
    )


parser = argparse.ArgumentParser(description="Experiment Azure Launcher")
parser.add_argument(
    "--job_id",
    "-j",
    type=str,
    help="job ID of the azure job, must be unique",
    required=True,
)

if __name__ == "__main__":
    args = parser.parse_args()
    job_id: str = args.job_id
    save_collated_timeseries(OUTPUT_PATH, job_id)
