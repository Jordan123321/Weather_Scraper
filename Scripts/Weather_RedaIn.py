#!/usr/bin/env python3

import os
import sys
import pandas as pd
import io
from typing import Optional

# Get the current working directory and its parent directory
PATH1 = os.getcwd()
PATH0 = os.path.dirname(PATH1)
PROCESSED_DATA_DIR = os.path.join(PATH0, 'Processed_Data')


def read_metadata() -> pd.DataFrame:
    """Read the processed_stations_metadata.csv file."""
    metadata_file = os.path.join(PROCESSED_DATA_DIR, 'processed_stations_metadata.csv')
    return pd.read_csv(metadata_file, index_col='src_id')


def find_csv_in_file_location(file_location: str, year: int) -> Optional[str]:
    """Find the CSV file with the specified year in the file location."""
    for file in os.listdir(file_location):
        if file.endswith(f"{year}.csv"):
            return os.path.join(file_location, file)
    return None


def read_csv_data(file: str) -> pd.DataFrame:
    """Read the CSV file data between 'data' and 'end data' entries."""
    with open(file, 'r') as f:
        lines = f.readlines()

    start_data = False
    data_lines = []

    for line in lines:
        if 'end data' in line.strip():
            break
        elif start_data:
            data_lines.append(line)
        elif 'data' == line.strip():
            start_data = True

    return pd.read_csv(io.StringIO(''.join(data_lines)), header=None)


if __name__ == "__main__":
    # Commented out the command-line argument and set year to 2019 for testing
    # if len(sys.argv) < 2:
    #     print("Usage: python script_name.py [year]")
    #     sys.exit(1)

    # year = int(sys.argv[1])
    year = 2019

    # Read the processed stations metadata DataFrame
    df = read_metadata()

    # Process the first matching CSV file
    for _, row in df.iterrows():
        if row['first_year'] <= year <= row['last_year']:
            file_locations = {
                "temp": row["file_location_temp"],
                "weather": row["file_location_weather"],
                "precip": row["file_location_precip"],
            }

            data_dfs = {}

            for data_type, file_location in file_locations.items():
                if file_location:
                    csv_file = find_csv_in_file_location(file_location, year)
                    if csv_file:
                        data_df = read_csv_data(csv_file)
                        data_dfs[data_type] = data_df

            if data_dfs:
                # Print the first few rows of each DataFrame
                for data_type, data_df in data_dfs.items():
                    print(f"Data Type: {data_type}")
                    print(data_df.head())
                    print()
                break