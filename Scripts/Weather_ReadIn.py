
"""
Weather Data Processing Script

This script processes weather data from various sources and combines them into a single Pandas DataFrame.
The output DataFrame contains weather classification, temperature categories, and station information for the specified year.

The script takes the following steps:

Read metadata for weather stations
For each station, load temperature, weather, and precipitation data for the specified year
Process the loaded data by cleaning, aggregating, and calculating relevant features
Merge the processed DataFrames and apply weather classification
Keep only the required columns in the merged DataFrame
Concatenate the results for all stations into a single DataFrame
Save the final DataFrame to a CSV file
Usage:
python weather_data_processing.py <year>

Arguments:
<year> The year for which to process the data (e.g., 2019)

Dependencies:
pandas, numpy, os, io, argparse, typing
"""

#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import io
import sys
import argparse
from typing import Optional, Tuple, Dict

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

    df = pd.read_csv(io.StringIO(''.join(data_lines)))
    if 'ob_end_time' in df.columns:
        df['ob_end_time'] = pd.to_datetime(df['ob_end_time'])
        df['date'] = df['ob_end_time'].dt.date
    return df

def classify_weather(row: pd.Series) -> Tuple[str, str]:
    """
    Classify the weather and temperature category for a given row of data.

    Args:
        row (pd.Series): A Pandas Series containing weather data.

    Returns:
        Tuple[str, str]: A tuple containing the weather classification and the temperature category.
    """
    weather = "Unknown"
    temp_category = "Unknown"

    if "prcp_amt" in row:
        if row["prcp_amt"] < 1.0:
            weather = "Dry"
        elif row["prcp_amt"] < 2.5:
            weather = "Drizzle"
        elif row["prcp_amt"] < 7.5:
            weather = "Mild Rain"
        else:
            weather = "Heavy Rain"

    if "snow_day_id" in row and "min_air_temp" in row and "min_grss_temp" in row and "snow_depth" in row and "prcp_amt" in row:
        if (
            (row["snow_day_id"] >= 1 and row["snow_day_id"] <= 6)
            or (
                (row["min_air_temp"] <= 0 and row["prcp_amt"] > 0)
                or (row["min_grss_temp"] <= 0 and row["prcp_amt"] > 0)
            )
            and row["snow_depth"] > 0
        ):
            weather = "Snowy"

    if "hail_day_id" in row:
        if row["hail_day_id"] >= 1 and row["hail_day_id"] <= 6:
            weather = "Hail"

    if "thunder_day_flag" in row:
        if row["thunder_day_flag"] >= 1 and row["thunder_day_flag"] <= 6:
            weather = "Thunderstorm"

    if "gale_day_flag" in row:
        if row["gale_day_flag"] >= 1 and row["gale_day_flag"] <= 6:
            weather = "Gale"

    if "min_air_temp" in row and "max_air_temp" in row:
        if row["min_air_temp"] <= -5:
            temp_category = "Freezing"
        elif row["max_air_temp"] <= 5:
            temp_category = "Cold"
        elif row["max_air_temp"] <= 15:
            temp_category = "Cool"
        elif row["max_air_temp"] <= 25:
            temp_category = "Mild"
        else:
            temp_category = "Hot"

    return weather, temp_category



def load_data_dfs(file_locations: Dict[str, str], year: int) -> Dict[str, pd.DataFrame]:
    """
    Load data DataFrames from the provided file locations and year.

    :param file_locations: A dictionary containing the file locations for temperature, weather, and precipitation data.
    :type file_locations: Dict[str, str]
    :param year: The year for which the data needs to be loaded.
    :type year: int
    :return: A dictionary containing the loaded DataFrames.
    :rtype: Dict[str, pd.DataFrame]
    """
    if not isinstance(file_locations, dict):
        raise ValueError("file_locations must be a dictionary")

    if not isinstance(year, int):
        raise ValueError("year must be an integer")

    data_dfs = {}
    for data_type, file_location in file_locations.items():
        if file_location:
            csv_file = find_csv_in_file_location(file_location, year)
            if csv_file:
                data_df = read_csv_data(csv_file)
                data_dfs[data_type] = data_df
    return data_dfs


def process_weather_df(data_dfs: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Process the weather DataFrame by selecting only the numeric columns, excluding the 'date' column,
    and then grouping by the 'date' column and calculating the mean.

    :param data_dfs: A dictionary containing the loaded DataFrames with keys "temp", "weather", and "precip".
    :type data_dfs: Dict[str, pd.DataFrame]

    :return: The processed weather DataFrame, or None if the 'weather' key is not present in data_dfs.
    :rtype: Optional[pd.DataFrame]
    """
    if "weather" not in data_dfs:
        return None

    weather_df = data_dfs["weather"]
    weather_df["date"] = pd.to_datetime(weather_df["date"])

    quality_flag_columns = [col for col in weather_df.columns if col.endswith('_q')]

    columns_to_drop = ["id", "ob_hour_count", "version_num", "src_id", "rec_st_ind", "ob_end_time", "midas_stmp_etime"] + quality_flag_columns
    weather_df = weather_df.drop(columns=columns_to_drop)

    # Keep the 'date' column in a separate DataFrame
    date_df = weather_df[['date']]

    # Select only the numeric columns, excluding the 'date' column
    numeric_columns = weather_df.select_dtypes(include=[np.number]).columns
    weather_df = weather_df[numeric_columns]

    # Add the 'date' column back to the DataFrame
    weather_df['date'] = date_df['date']

    # Group by date and calculate the mean
    grouped_weather_df = weather_df.groupby("date").mean()

    # Set date as index
    grouped_weather_df = grouped_weather_df.reset_index().set_index("date")

    return grouped_weather_df



def process_precip_df(data_dfs: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Process the precipitation DataFrame if present in the data_dfs dictionary.

    :param data_dfs: A dictionary containing the loaded DataFrames.
    :type data_dfs: Dict[str, pd.DataFrame]
    :return: The processed precipitation DataFrame or None if not present in data_dfs.
    :rtype: Optional[pd.DataFrame]
    """
    if not isinstance(data_dfs, dict):
        raise ValueError("data_dfs must be a dictionary")

    if "precip" not in data_dfs:
        return None

    precip_df = data_dfs["precip"]
    precip_df["ob_date"] = pd.to_datetime(precip_df["ob_date"]).dt.date
    precip_df = precip_df.set_index("ob_date")

    columns_to_drop = ["id", "id_type", "version_num", "met_domain_name", "ob_end_ctime", "ob_day_cnt", "src_id", "ob_day_cnt_q", "meto_stmp_time", "midas_stmp_etime", "prcp_amt_j", "prcp_amt_q", "rec_st_ind"]
    precip_df = precip_df.drop(columns=columns_to_drop)

    return precip_df


def process_temp_df(data_dfs: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Process the temperature DataFrame if present in the data_dfs dictionary.

    :param data_dfs: A dictionary containing the loaded DataFrames.
    :type data_dfs: Dict[str, pd.DataFrame]
    :return: The processed temperature DataFrame or None if not present in data_dfs.
    :rtype: Optional[pd.DataFrame]
    """
    if not isinstance(data_dfs, dict):
        raise ValueError("data_dfs must be a dictionary")

    if "temp" not in data_dfs:
        return None

    temp_df = data_dfs["temp"]
    temp_df["date"] = pd.to_datetime(temp_df["date"])
    temp_df = temp_df.set_index("date")

    columns_to_drop = ["ob_end_time", "id_type", "id", "ob_hour_count", "version_num", "met_domain_name", "src_id", "rec_st_ind",
                       "meto_stmp_time", "midas_stmp_etime", "max_air_temp_q", "min_air_temp_q", "min_grss_temp_q",
                       "min_conc_temp_q", "max_air_temp_j", "min_air_temp_j", "min_grss_temp_j", "min_conc_temp_j"]
    temp_df = temp_df.drop(columns=columns_to_drop)

    return temp_df


def apply_weather_classification(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the weather classification to the merged DataFrame.

    :param merged_df: The merged DataFrame containing the weather data.
    :return: The merged DataFrame with the weather classification applied.
    """
    classification_results = merged_df.apply(classify_weather, axis=1)
    if len(classification_results) > 0:
        merged_df["weather_classifier"], merged_df["temp_category"] = zip(*classification_results)
    else:
        merged_df["weather_classifier"] = "Unknown"
        merged_df["temp_category"] = "Unknown"

    return merged_df


def keep_required_columns(merged_df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """
    Keep only the required columns in the merged DataFrame.

    :param merged_df: The merged DataFrame with weather classification applied.
    :type merged_df: pd.DataFrame
    :param row: A row from the metadata DataFrame containing station information.
    :type row: pd.Series
    :return: The merged DataFrame with only the required columns.
    :rtype: pd.DataFrame
    """
    if not isinstance(merged_df, pd.DataFrame):
        raise ValueError("merged_df must be a DataFrame")

    if not isinstance(row, pd.Series):
        raise ValueError("row must be a Series")

    station_latitude = row["station_latitude"]
    station_longitude = row["station_longitude"]
    station_name = row["station_name"]

    merged_df["station_latitude"] = station_latitude
    merged_df["station_longitude"] = station_longitude
    merged_df["station_name"] = station_name

    columns_to_keep = [
        "date",
        "station_latitude",
        "station_longitude",
        "weather_classifier",
        "temp_category",
        "station_name",
    ]
    merged_df_ = merged_df[columns_to_keep]

    return merged_df_


def merge_dataframes(temp_df: Optional[pd.DataFrame], grouped_weather_df: Optional[pd.DataFrame], precip_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Merge the temperature, grouped_weather, and precipitation DataFrames.

    :param temp_df: The temperature DataFrame.
    :param grouped_weather_df: The grouped_weather DataFrame.
    :param precip_df: The precipitation DataFrame.
    :return: The merged DataFrame containing all three input DataFrames.
    """
    dataframes = [df for df in [temp_df, grouped_weather_df, precip_df] if df is not None]

    if len(dataframes) < 2:
        return None

    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = merged_df.merge(df, left_index=True, right_index=True, how='outer')

    # Reset the index and rename it to 'date'
    merged_df = merged_df.reset_index().rename(columns={'index': 'date'})

    return merged_df

def process_and_merge_dataframes(year: int, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and merge the data for the given year and metadata DataFrame.

    :param year: The year for which the data needs to be processed and merged.
    :type year: int
    :param metadata_df: The metadata DataFrame containing station information.
    :type metadata_df: pd.DataFrame
    :return: The merged DataFrame containing the weather and temperature data for the given year and metadata.
    :rtype: pd.DataFrame
    """
    if not isinstance(year, int):
        raise ValueError("year must be an integer")

    if not isinstance(metadata_df, pd.DataFrame):
        raise ValueError("metadata_df must be a pandas DataFrame")

    all_merged_df = pd.DataFrame(columns=["station_latitude", "station_longitude", "weather_classifier", "temp_category", "station_name"])
    total_stations = len([row for _, row in metadata_df.iterrows() if row['first_year'] <= year <= row['last_year']])
    processed_stations = 0

    for _, row in metadata_df.iterrows():
        if row['first_year'] <= year <= row['last_year']:
            file_locations = {
                "temp": row["file_location_temp"],
                "weather": row["file_location_weather"],
                "precip": row["file_location_precip"],
            }
            data_dfs = load_data_dfs(file_locations, year)

            if data_dfs:
                grouped_weather_df = process_weather_df(data_dfs)
                precip_df = process_precip_df(data_dfs)
                temp_df = process_temp_df(data_dfs)

                merged_df = merge_dataframes(temp_df, grouped_weather_df, precip_df)
                if merged_df is not None:
                    merged_df = apply_weather_classification(merged_df)
                    merged_df_ = keep_required_columns(merged_df, row)
                    all_merged_df = pd.concat([all_merged_df, merged_df_], ignore_index=True)




            processed_stations += 1
            print(f"Processed {processed_stations} of {total_stations} stations")

    return all_merged_df



if __name__ == "__main__":
    print(f"Processing year {sys.argv[1]} in Weather_ReadIn.py...")
    parser = argparse.ArgumentParser(description="Process and merge weather data.")
    parser.add_argument("year", type=int, help="The year for which to process the data.")
    args = parser.parse_args()

    #year = 2019

    year = args.year
    metadata_df = read_metadata()
    all_merged_df = process_and_merge_dataframes(year, metadata_df)

    output_file_path = os.path.join(PROCESSED_DATA_DIR, f"Weather_Classified_{year}.csv")
    all_merged_df.to_csv(output_file_path, index=False)
    print(f"Saved all_merged_df to {output_file_path}")
