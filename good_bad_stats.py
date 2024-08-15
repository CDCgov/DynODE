import pandas as pd
import json
import numpy as np
import os

# Local path to the medians csv file.
az_output_path = "/output/fifty_state_2204_2407_6strain/smh_6str_prelim_7/"


# This function retrieves the particles from the checkpoint json files and concatenate them all state by state
def retrieve_checkpoint(states, az_output_path):
    total_df = pd.DataFrame()
    for state in states:
        state_config_path = os.path.join(az_output_path, state)
        print(f"Retrieving {state}")
        state_checkpoint_path = os.path.join(state_config_path, "checkpoint.json")
        if not os.path.exists(state_checkpoint_path):
            print(f"File not found: {state_checkpoint_path}")
            continue  # Skip to the next state if file doesn't exist
        try:
            with open(state_checkpoint_path, "r") as json_file:
                data = json.load(json_file)
            # Convert the JSON data to a DataFrame
            df = pd.DataFrame(data)
            # Check if the DataFrame is empty
            if df.empty:
                print(f"DataFrame is empty for {state_checkpoint_path}")
                continue

            # Flatten lists in each column and concatenate
            flattened_df = pd.DataFrame()
            for column in df.columns:
                flattened_column = pd.Series(
                    [item for sublist in df[column] for item in sublist]
                )
                # Convert the flattened column to float
                flattened_df[column] = flattened_column.astype(float)
            # Concatenate the flattened DataFrame
            total_df = pd.concat([total_df, flattened_df], axis=0, ignore_index=True)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {state_checkpoint_path}")
        except Exception as e:
            print(f"Error processing {state_checkpoint_path}: {str(e)}")
    print(total_df)
    return total_df


# This function creates a data frame with the statistics given the output of the previous function
def show_statistics(states, bad, az_output_path):
    total_df = retrieve_checkpoint(states, az_output_path)
    if total_df.empty:
        print("No data retrieved for any state.")
        return pd.DataFrame()

    statistics = {}
    prefix = "b" if bad else "g"
    statistics[f"mean_{prefix}"] = total_df.mean()
    statistics[f"min_{prefix}"] = total_df.min()
    statistics[f"max_{prefix}"] = total_df.max()
    statistics[f"std_{prefix}"] = total_df.std()
    statistics[f"median_{prefix}"] = total_df.median()
    return pd.DataFrame(statistics)


# Based on your own judgement on their plots this function groups the states into two categories: the good and bad states, returning the last function's output-the statistics of the concatenated checkpoint json files.
def main():
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

    statistics_bad = show_statistics(
        bad_states, bad=True, az_output_path=az_output_path
    )
    statistics_good = show_statistics(
        good_states, bad=False, az_output_path=az_output_path
    )
    df_total = pd.concat([statistics_bad, statistics_good], axis=1)
    for x in df_total.columns:
        print(df_total[x].dtype)
    # Save the DataFrame to CSV
    output_path = "output/good_bad_stats.csv"
    df_total.to_csv(output_path, index=True)
    # print("DataFrame saved to:", output_path)
    # print("\nDataFrame contents:")


if __name__ == "__main__":
    main()
