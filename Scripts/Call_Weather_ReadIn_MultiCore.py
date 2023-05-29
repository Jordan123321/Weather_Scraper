#!/usr/bin/env python3

"""
Multi-processes the script Call_Weather_ReadIn_MultiCore.py to process data for each month
between January 2011 and December 2021. Uses the concurrent.futures module to create a pool
of worker processes. Each worker process calls Call_Weather_ReadIn_MultiCore.py with a
specific year and month as command-line arguments.
"""

import os
import concurrent.futures as cft
from itertools import product
import subprocess
import time
from typing import Tuple

def worker(year_month: Tuple[int, int]) -> Tuple[int, int, float]:
    """
    Worker function to process data for a specific year and month.

    Args:
    year_month: A tuple containing the year and month.

    Returns:
    A tuple with year, month, and elapsed time for the process.
    """
    year, month = year_month
    assert isinstance(year, int), "Year must be an integer."
    assert isinstance(month, int), "Month must be an integer."
    assert 1 <= month <= 12, "Month must be between 1 and 12."

    print(f"Processing data for {year}-{month:02d}...")
    start = time.time()

    # Get the current working directory
    cwd = os.getcwd()

    # Create file paths for stdout and stderr
    stdout_file = os.path.join(cwd, f'log/stdout_{year}_{month}.log')
    stderr_file = os.path.join(cwd, f'log/stderr_{year}_{month}.log')

    with open(stdout_file, 'w') as out, open(stderr_file, 'w') as err:
        # Call the original script with subprocess and pass year and month as arguments
        script_path = os.path.join(cwd, 'Call_Weather_ReadIn_MultiCore.py')
        subprocess.call(['python', script_path, str(year), str(month).zfill(2)], stdout=out, stderr=err)

    end = time.time()
    elapsed_time = end - start
    print(f"Finished processing data for {year}-{month:02d} in {elapsed_time:.2f} seconds.")

    return (year, month, elapsed_time)

def print_result(future: cft.Future) -> None:
    """
    Callback function to print the result of a completed task.

    Args:
    future: A Future object representing the completed task.

    Returns:
    None
    """
    year, month, elapsed_time = future.result()
    print(f"Task completed: {year}-{month:02d}, elapsed time: {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    """
    Main function to start worker processes.

    Args:
    None

    Returns:
    None
    """
    # Prepare a list of (year, month) tuples for each month between 2011 and 2021
    year_month_pairs = list(product(range(2019, 2021), range(1, 4)))

    # Get number of cores - 1
    n_cores = os.cpu_count() - 1 if os.cpu_count() > 1 else 1

    # Create a pool of workers and a list to hold the Future objects
    with cft.ProcessPoolExecutor(max_workers=n_cores) as executor:
        # Use the worker function to process each (year, month) pair
        futures = {executor.submit(worker, pair) for pair in year_month_pairs}

        for future in cft.as_completed(futures):
            future.add_done_callback(print_result)
