from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def run_script(
    tutorial: Path, timeout_minutes: int = 5, env: Optional[Dict[str, str]] = None
):
    utils_path = {"PYTHONPATH": str(tutorial.parent)}
    if env is not None:
        env = {**os.environ, **env, **utils_path}
    else:
        env = {**os.environ, **utils_path}

    run_out = subprocess.run(
        ["papermill", str(tutorial.absolute()) , "temp.ipynb"],  # , "|"
        capture_output=True,
        text=True,
        env=env,
        encoding="utf-8",
        timeout=timeout_minutes * 60,
    )
    return run_out


def run_tutorials(
    name: Optional[str] = None,
) -> None:
    """
    Run each tutorial, print statements on how it ran, and write a data set as a csv
    to a directory.
    """

    print("This may take a long time...")

    tutorial_dir = Path(os.getcwd()).joinpath("tutorials")
    num_runs = 0
    num_errors = 0

    tutorials = sorted(t for t in tutorial_dir.rglob("*.ipynb") if t.is_file)
    if name is not None:
        tutorials = [t for t in tutorials if t.name == name]
        if len(tutorials) == 0:
            raise RuntimeError(f"Specified tutorial {name} not found in directory.")

    df = pd.DataFrame(
        {
            "name": [t.name for t in tutorials],
            "ran_successfully": False,
            "runtime": float("nan"),
        }
    ).set_index("name")

    for tutorial in tutorials:
        num_runs += 1
        t1 = time.time()
        run_out = run_script(tutorial)
        elapsed_time = time.time() - t1
        print(50*'#', tutorial)
        print(f'time elapsed:{elapsed_time:.2f}')
        print(f'statuscode: {run_out.returncode}')
        # print(run_out.stdout)

        if run_out.returncode != 0:
            num_errors += 1
            print(run_out.stderr)

        else:
            print(
                f"Running tutorial {tutorial.name} took " f"{elapsed_time:.2f} seconds."
            )
            df.loc[tutorial.name, "ran_successfully"] = True
        
        
    # delete temporary test notebook file
    os.remove("temp.ipynb")

    if num_errors > 0:
        raise RuntimeError(
            f"Running {num_runs} tutorials resulted in {num_errors} errors."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs tutorials.")

    parser.add_argument(
        "-n",
        "--name",
        help="Run a specific tutorial by name.",
    )
    parser.add_argument(
        '-l',
        '--long',

    )

    parser.add_argument(
        "-p", "--path", metavar="path", required=False, help="bofire repo directory."
    )
    args = parser.parse_args()
    run_tutorials(
        name=args.name,
    )
