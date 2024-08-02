# %%
import os

import pandas as pd

OUTPUT_PATH = "/output/projections_2407_2507/"
JOB_ID = "proj_2024_allscen_v6"

job_path = os.path.join(OUTPUT_PATH, JOB_ID)
states = os.listdir(job_path)
for st in states:
    print(st)
    dfs = list()
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
            dfs.append(df)

    final_df = pd.concat(dfs)
    final_df.to_csv(os.path.join(state_path, f"projections.csv"), index=False)
