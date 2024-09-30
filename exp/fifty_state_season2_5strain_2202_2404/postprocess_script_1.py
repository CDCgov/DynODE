import argparse
import csv
import json
import os
import warnings

from tqdm import tqdm

OUTPUT_PATH = "/output/fifty_state_season2_5strain_2202_2404/"


def collate_checkpoints_to_csv(output_path, jobid):
    """Grabs all states under output_path/jobid reads in their checkpoint.json, and saves
    the values as a csv

    Parameters
    ----------
    output_path : str
        output path as str where job is stored
    jobid : str
        jobid of the job run
    """
    job_path = os.path.join(output_path, jobid)
    output_csv = os.path.join(job_path, "checkpoints_collated.csv")
    scenarios = os.listdir(job_path)
    # all the scenarios have the same checkpoints, so lets just go into one of them
    scenario = scenarios[0]
    states = os.listdir(job_path)
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row to the CSV file
        writer.writerow(
            ["state", "parameter_name", "chain", "sample_num", "value"]
        )
        for st in tqdm(states, desc="processing states "):
            print(st)
            # only looking at one scenario per state
            state_path = os.path.join(job_path, st, scenario)
            checkpoint_path = os.path.join(state_path, "checkpoint.json")
            if os.path.exists(checkpoint_path):
                checkpoint = json.load(open(checkpoint_path, "r"))
                # Iterate over each parameter in the JSON data
                for parameter_name, values in checkpoint.items():
                    # Iterate over each chain and sample number in the 2D list of values
                    for chain, row in enumerate(values):
                        for sample_num, value in enumerate(row):
                            # Write a row to the CSV file with state, parameter_name, chain, sample_num, and value
                            writer.writerow(
                                [st, parameter_name, chain, sample_num, value]
                            )

            else:
                warnings.warn(
                    "%s state path lacks a checkpoint.json file, check your paths or job for this state did not complete"
                    % state_path
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
    collate_checkpoints_to_csv(OUTPUT_PATH, job_id)
