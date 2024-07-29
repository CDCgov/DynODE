# %%
import os

import pandas as pd

OUTPUT_PATH = "/output/projections_2407_2507/"
JOB_ID = "proj_2024_240728_wseason"
suffix = "v9_6strain_wseason"

job_path = os.path.join(OUTPUT_PATH, JOB_ID)
states = os.listdir(job_path)
dfs = list()
for st in states:
    print(st)
    state_path = os.path.join(job_path, st)
    scens = os.listdir(state_path)

    for sc in scens:
        csv_path = os.path.join(
            state_path, sc, "azure_visualizer_timeline.csv"
        )
        if os.path.exists(
            os.path.join(state_path, sc, "azure_visualizer_timeline.csv")
        ):
            df = pd.read_csv(csv_path)
            df_select = df[
                [
                    "chain_particle",
                    "date",
                    "pred_hosp_0_17",
                    "pred_hosp_18_49",
                    "pred_hosp_50_64",
                    "pred_hosp_65+",
                ]
            ].copy()

            df_select["state"] = st
            df_select["scenario"] = sc
            dfs.append(df_select)

final_df = pd.concat(dfs)
final_df.to_csv(f"./output/projections_2407_2507_{suffix}.csv", index=False)
