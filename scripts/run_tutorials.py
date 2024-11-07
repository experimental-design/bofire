from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


# This script is based on the corresponding one from botorch: https://github.com/pytorch/botorch/blob/main/scripts/run_tutorials.py


def run_script(
    tutorial: Path,
    timeout_minutes: int = 20,
    env: Optional[Dict[str, str]] = None,
    inplace: bool = False,
):
    tutorial_path = str(tutorial.absolute())
    output_path = tutorial_path if inplace else "temp.ipynb"

    utils_path = {"PYTHONPATH": str(tutorial.parent)}
    if env is not None:
        env = {**os.environ, **env, **utils_path}
    else:
        env = {**os.environ, **utils_path}

    try:
        run_out = subprocess.run(
            ["papermill", tutorial_path, output_path],
            capture_output=True,
            text=True,
            env=env,
            encoding="utf-8",
            timeout=timeout_minutes * 60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"{tutorial} exceeded max. runtime ({timeout_minutes*60} s)... ")
        if not inplace:
            os.remove(output_path)
        return None

    if not inplace:
        os.remove(output_path)

    return run_out


def run_tutorials(
    name: Optional[str] = None,
    smoke_test: bool = False,
    inplace: bool = False,
) -> None:
    """Run each tutorial, print statements on how it ran, and write a data set
    as a csv to a directory.
    """
    timeout_minutes = 30 if smoke_test is False else 2

    print(f"Running Tutorials, smoke_test_flag = {smoke_test}")

    tutorial_dir = Path(os.getcwd()).joinpath("tutorials")
    num_runs = 0
    num_errors = 0

    tutorials = sorted(t for t in tutorial_dir.rglob("*.ipynb") if t.is_file)
    env = {"SMOKE_TEST": "True"} if smoke_test else None
    if name is not None:
        tutorials = [t for t in tutorials if t.name == name]
        if len(tutorials) == 0:
            raise RuntimeError(f"Specified tutorial {name} not found in directory.")

    df = pd.DataFrame(
        {
            "name": [t.name for t in tutorials],
            "ran_successfully": False,
            "message": "",
            "runtime": float("nan"),
        },
    ).set_index("name")

    # TODO: take care
    # here are notebooks which are not tested due to random issues
    blacklist = []

    for tutorial in tutorials:
        print(42 * "#", tutorial)
        # # for now we skip all tutorials but the one for which we have implemented SMOKE_TEST. This will change soon!
        if str(tutorial).split("/")[-1] in blacklist:
            print("Skipping", str(tutorial))
            continue
        num_runs += 1
        t1 = time.time()
        run_out = run_script(
            tutorial,
            env=env,
            timeout_minutes=timeout_minutes,
            inplace=inplace,
        )
        elapsed_time = time.time() - t1
        print(f"time elapsed:{elapsed_time:.2f}")
        if run_out is None:  # in this case it bumped against max wall time
            df.loc[tutorial.name, "ran_successfully"] = False
            df.loc[tutorial.name, "message"] = "walltime exceeded"
            continue
        print(f"statuscode: {run_out.returncode}")

        if run_out.returncode != 0:
            num_errors += 1
            df.loc[tutorial.name, "message"] = run_out.stderr
            print(run_out.stderr)

        else:
            print(
                f"Running tutorial {tutorial.name} took {elapsed_time:.2f} seconds.",
            )
            df.loc[tutorial.name, "ran_successfully"] = True

    df.to_csv("notebook_test_stats.csv")

    # delete temporary test notebook file
    if os.path.exists("temp.ipynb"):
        os.remove("temp.ipynb")

    if num_errors > 0:
        raise RuntimeError(
            f"Running {num_runs} tutorials resulted in {num_errors} errors.",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs tutorials.")

    parser.add_argument(
        "-n",
        "--name",
        help="Run a specific tutorial by name.",
    )

    parser.add_argument(
        "-l",
        "--long",
        action="store_true",
        help="Run the full version of the notebook. Will take a long time.",
    )

    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Run the tests in place, without creating a temporary notebook. Used to update all outputs.",
    )

    args = parser.parse_args()

    run_tutorials(
        name=args.name,
        smoke_test=not args.long,
        inplace=args.in_place,
    )
