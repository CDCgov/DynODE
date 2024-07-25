import json
import os
import csv
import warnings

OUTPUT_PATH = "/output/fifty_state_5strain_2202_2404/"
JOB_ID = "SMH_5strains_071624"


def collate_checkpoints_to_csv(output_path, jobid, suffix="v0"):
    """Grabs all states under output_path/jobid reads in their checkpoint.json, and saves
    the values as a csv

    Parameters
    ----------
    output_path : str
        output path as str where job is stored
    jobid : str
        jobid of the job run
    suffix : str, optional
        suffix to place onto csv, by default "v0"
    """
    job_path = os.path.join(output_path, jobid)
    output_csv = os.path.join(job_path, "checkpoints_collated_%s.csv" % suffix)
    states = os.listdir(job_path)
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row to the CSV file
        writer.writerow(
            ["state", "parameter_name", "chain", "sample_num", "value"]
        )
        for st in states:
            print(st)
            state_path = os.path.join(job_path, st)
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


if __name__ == "__main__":
    collate_checkpoints_to_csv(OUTPUT_PATH, JOB_ID)
