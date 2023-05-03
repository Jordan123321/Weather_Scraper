#!/usr/bin/env python3
"""
This script reads metadata CSV files from three directories, merges them to find the intersection
of stations based on their src_id, and adds the file location columns for each type of weather data
(temperature, general weather, and precipitation). The merged DataFrame is saved as a new CSV file
in the 'Processed_Data' directory.

Functions:
find_metadata_csv(directory: str) -> Optional[str]: Find the metadata CSV file in the specified directory.
read_metadata(file: str) -> pd.DataFrame: Read the metadata CSV file and return it as a DataFrame.
check_location_exists(row: pd.Series, directory: str) -> Union[str, None]: Check if the folder exists and return the folder location if it exists, otherwise return None.
intersection_dataframes(df0: pd.DataFrame, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame: Find the intersection of the three DataFrames based on their index and add the file location columns.
save_merged_dataframe(df: pd.DataFrame): Save the merged DataFrame.
"""

#!/usr/bin/env python3

import os
import sys
import io
import pandas as pd
from typing import Optional, Union
from pathlib import Path

# Get the current working directory and its parent directory
PATH1 = os.getcwd()
PATH0 = os.path.dirname(PATH1)

# Define the directories containing the metadata CSV files
DIRECTORY0 = os.path.join(PATH0, 'Raw_Data', 'Temperature_Daily', 'dataset-version-202207')
DIRECTORY1 = os.path.join(PATH0, 'Raw_Data', 'Weather_Daily', 'dataset-version-202207')
DIRECTORY2 = os.path.join(PATH0, 'Raw_Data', 'Precipitation_Daily', 'dataset-version-202207')


def find_metadata_csv(directory: str) -> Optional[str]:
    """Find the metadata CSV file in the specified directory."""
    for file in os.listdir(directory):
        if file.endswith('metadata.csv'):
            return os.path.join(directory, file)
    return None


def read_metadata(file: str) -> pd.DataFrame:
    """Read the metadata CSV file and return it as a DataFrame."""
    try:
        with open(file, 'r') as f:
            lines = f.readlines()

        start_data = False
        data_lines = []

        for line in lines:
            if 'end data' == line.strip():
                break
            elif start_data:
                data_lines.append(line)
            elif 'data' == line.strip():
                start_data = True

        df = pd.read_csv(io.StringIO(''.join(data_lines)), index_col='src_id')

    except pd.errors.ParserError as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
    return df


def check_location_exists(row: pd.Series, directory: str) -> Union[str, None]:
    """Check if the folder exists and return the folder location if it exists, otherwise return None."""
    # Zero-pad the src_id (row.name)
    src_id_padded = str(row.name).zfill(5)

    # Construct the partial folder location string
    partial_location = f"{row['historic_county']}/{src_id_padded}_{row['station_file_name']}/qc-version-1"

    # Replace backslashes with forward slashes
    partial_location = partial_location.replace("\\", "/")

    # Create a Path object from the given directory and resolve it to its absolute path
    directory_path = Path(directory).resolve()

    # Search for the matching folder location
    matching_location = None
    for folder in directory_path.rglob(partial_location):
        if folder.is_dir():
            matching_location = str(folder)
            break

    if matching_location is not None:
        return matching_location
    else:
        print(f"Folder not found: {directory}/{partial_location}")
        return None
    
def intersection_dataframes(df0: pd.DataFrame, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Find the intersection of the three DataFrames based on their index and add the file location columns."""
    df1 = df1.drop(columns=['station_name', 'station_file_name', 'historic_county',
                             'station_latitude', 'station_longitude', 'station_elevation'])
    df2 = df2.drop(columns=['station_name', 'station_file_name', 'historic_county',
                             'station_latitude', 'station_longitude', 'station_elevation'])

    intersected_df = df0.merge(df1, how='inner', left_index=True, right_index=True, suffixes=('', '_y1')).merge(
        df2, how='inner', left_index=True, right_index=True, suffixes=('', '_y2'))

    intersected_df['first_year'] = intersected_df[['first_year', 'first_year_y1', 'first_year_y2']].max(axis=1)
    intersected_df['last_year'] = intersected_df[['last_year', 'last_year_y1', 'last_year_y2']].min(axis=1)

    intersected_df.drop(columns=['first_year_y1', 'first_year_y2', 'last_year_y1', 'last_year_y2'], inplace=True)

    intersected_df['file_location_temp'] = intersected_df.apply(check_location_exists, args=(DIRECTORY0,), axis=1)
    intersected_df['file_location_weather'] = intersected_df.apply(check_location_exists, args=(DIRECTORY1,), axis=1)
    intersected_df['file_location_precip'] = intersected_df.apply(check_location_exists, args=(DIRECTORY2,), axis=1)

    return intersected_df


def save_merged_dataframe(df: pd.DataFrame):
    """Save the merged DataFrame."""
    output_directory = os.path.join(PATH0, 'Processed_Data')
    os.makedirs(output_directory, exist_ok=True)

    output_file = os.path.join(output_directory, 'processed_stations_metadata.csv')
    df.to_csv(output_file)
    
    
    
if __name__ == "__main__":
    # Read metadata CSVs
    metadata_file0 = find_metadata_csv(DIRECTORY0)
    metadata_file1 = find_metadata_csv(DIRECTORY1)
    metadata_file2 = find_metadata_csv(DIRECTORY2)

    df0 = read_metadata(metadata_file0)
    df1 = read_metadata(metadata_file1)
    df2 = read_metadata(metadata_file2)

    # Find the intersection of the DataFrames and add file location columns
    intersected_df = intersection_dataframes(df0, df1, df2)

    # Save the intersected DataFrame
    save_merged_dataframe(intersected_df)