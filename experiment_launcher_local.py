"""
a script which is passed two flags, one to dictate the experiment folder on which to run
and another to specify the location of the state-specific runner script, which will perform analysis on a single state.
"""

import argparse
import os
import subprocess


def run_experiment(directory, runner_file):
    """
    a file which looks into `directory` and runs the python file pointed to by
    `runner_file` for each subdirectory within `directory`.
    Passing the subdirectory path to the runner_file file via the -s flag.

    Parameters
    ----------
    `directory`: str
        relative or absolute path to the experiment directory, under which are state folders

    runner_file: str
        relative or absolute path to the runner file which will perform the experiment on a single state
        and output the requested results into that states /output folder.

    Returns
    ----------
    None
    """
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            command = ["python3", runner_file, "-s", subdir_path]
            subprocess.run(command, cwd=os.getcwd())


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Experiment Runner")
parser.add_argument(
    "--folder",
    type=str,
    help="Directory of experiment",
    required=True,
)
parser.add_argument(
    "--runner",
    type=str,
    help="Path to single state runner Python file",
    required=True,
)
args = parser.parse_args()
# Run experiments on each subdirectory
run_experiment(args.folder, args.runner)
