"""
Summary:
This script processes and merges weather data files for the years 1900-2022 using the Weather_ReadIn.py script.
It runs the Weather_ReadIn.py script in parallel using multiprocessing, utilizing one less core than the maximum core count.
It prints the processing time for each year in real time.
"""

import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Union

def run_weather_readin(year: Union[int, str]) -> str:
    """
    Run the Weather_ReadIn.py script for the given year and return the processing time and output.

    Args:
        year (Union[int, str]): The year to process.

    Returns:
        str: A string containing the processing time and output for the given year.
    """
    if not isinstance(year, (int, str)):
        raise ValueError("Year must be an integer or a string.")

    year = str(year)
    start_time = time.perf_counter()
    script_path = os.path.join(os.getcwd(), "Weather_ReadIn.py")
    result = subprocess.run(["python", script_path, year], capture_output=True, text=True)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return f"Processing year {year}:\n{result.stdout}\nElapsed time: {elapsed_time:.2f} seconds\n"

def print_result(future):
    """
    Print the result of a completed future.

    Args:
        future (concurrent.futures.Future): The completed future.
    """
    print(future.result())

if __name__ == "__main__":
    script_path = os.path.join(os.getcwd(), "Station_ReadIn.py")
    subprocess.run(["python", script_path])
    num_cores = os.cpu_count() - 1
    years = range(1900, 2023)

    # Create a ProcessPoolExecutor with a specified number of worker processes.
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit the run_weather_readin function with each year as an argument and store the resulting futures.
        futures = {executor.submit(run_weather_readin, year): year for year in years}
        # Iterate through the completed futures as they become available.
        for future in as_completed(futures):
            # Add the print_result function as a callback to each completed future.
            future.add_done_callback(print_result)
