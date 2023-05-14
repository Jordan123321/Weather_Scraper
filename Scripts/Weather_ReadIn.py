#!/usr/bin/env python3
"""
Weather Data Processing Script

This script processes weather data from various sources and combines them into a single Pandas DataFrame.
The output DataFrame contains weather classification, temperature categories, sunshine type, and station information for the specified year.

The script takes the following steps:

1. Read metadata for weather stations.
2. For each station, load temperature, weather, and precipitation data for the specified year.
3. Process the loaded data by cleaning, aggregating, and calculating relevant features.
4. Merge the processed DataFrames and apply weather classification.
5. Keep only the required columns in the merged DataFrame.
6. Concatenate the results for all stations into a single DataFrame.
7. Save the final DataFrame to a CSV file.

Usage:
python weather_data_processing.py <year>

Arguments:
<year> The year for which to process the data (e.g., 2019)

Dependencies:
pandas, numpy, os, io, sys, argparse, typing
"""


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


def classify_weather(row: pd.Series) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Classify the weather, temperature category, and sunshine type for a given row of data.

    Args:
        row (pd.Series): A Pandas Series containing weather data.

    Returns:
        Tuple[Optional[str], Optional[str], Optional[str]]: A tuple containing the weather classification,
        temperature category, and sunshine type.
    """
    weather = classify_weather_condition(row)
    temp_category = classify_temp_category(row)
    sunshine_type = classify_sunshine_type(row)

    return weather, temp_category, sunshine_type


def apply_weather_classification(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the weather classification to the merged DataFrame.

    :param merged_df: The merged DataFrame containing the weather data.
    :return: The merged DataFrame with the weather classification applied.
    """
    classification_results = merged_df.apply(classify_weather, axis=1)
    merged_df["weather_classifier"], merged_df["temp_category"], merged_df["sunshine_type"] = zip(*classification_results)

    #print(merged_df["sunshine_type"].head())
    return merged_df


def classify_weather_condition(row: pd.Series) -> str:
    """
    Classify the weather condition for a given row of data.

    This function analyzes the weather data in the provided Pandas Series, `row`, and classifies the weather condition
    based on various attributes. The function retrieves specific weather attributes using the `row.get` method and assigns
    default values of `np.nan` when the attributes are missing.

    The classification follows the following order of conditions:
    1. Snowy condition: If any of the snow-related attributes (snow_day_id, min_air_temp, min_grss_temp, prcp_amt, snow_depth)
       are present and meet the specified conditions, the condition is classified as "Snowy".
    2. Hail condition: If the hail_day_id attribute is present and falls within the range of 1 to 6, the condition is classified as "Hail".
    3. Thunderstorm condition: If the thunder_day_flag attribute is present and falls within the range of 1 to 6, the condition is classified as "Thunderstorm".
    4. Gale condition: If the gale_day_flag attribute is present and falls within the range of 1 to 6, the condition is classified as "Gale".
    5. Precipitation conditions: If the prcp_amt attribute is present, it is checked against different thresholds to classify the condition
       as "Dry", "Drizzle", "Mild Rain", or "Heavy Rain".

    If none of the conditions are met, the function returns `np.nan`.

    Args:
        row (pd.Series): A Pandas Series containing weather data.

    Returns:
        str: The weather condition classification.
    """


    snow_day_id = row.get("snow_day_id", default=np.nan)
    min_air_temp = row.get("min_air_temp", default=np.nan)
    min_grss_temp = row.get("min_grss_temp", default=np.nan)
    snow_depth = row.get("snow_depth", default=np.nan)
    prcp_amt = row.get("prcp_amt", default=np.nan)


    if (
        (
            (not pd.isna(snow_day_id) or (not pd.isna(min_air_temp) or not pd.isna(min_grss_temp)))
            and not pd.isna(prcp_amt)
        )
        and not pd.isna(snow_depth)
    ):
        if (
            (snow_day_id >= 1 and snow_day_id <= 6) or
            (
                (min_air_temp <= 0 and prcp_amt > 0) or
                (min_grss_temp <= 0 and prcp_amt > 0)
            )
        ) and snow_depth > 0:
            return "Snowy"

    hail_day_id = row.get("hail_day_id", default=np.nan)

    if not pd.isna(hail_day_id):
        if hail_day_id >= 1 and hail_day_id <= 6:
            return "Hail"

    thunder_day_flag = row.get("thunder_day_flag", default=np.nan)

    if not pd.isna(thunder_day_flag):
        if thunder_day_flag >= 1 and thunder_day_flag <= 6:
            return "Thunderstorm"

    gale_day_flag = row.get("gale_day_flag", default=np.nan)

    if not pd.isna(gale_day_flag):
        if gale_day_flag >= 1 and gale_day_flag <= 6:
            return "Gale"



    if not pd.isna(prcp_amt):
        if prcp_amt < 1.0:
            return "Dry"
        elif prcp_amt < 2.5:
            return "Drizzle"
        elif prcp_amt < 7.5:
            return "Mild Rain"
        else:
            return "Heavy Rain"
    return np.nan


def classify_temp_category(row: pd.Series) -> str:
    """
    Classify the temperature category for a given row of data.

    Args:
        row (pd.Series): A Pandas Series containing weather data.

    Returns:
        str: The temperature category classification.
    """
    min_air_temp = row.get("min_air_temp", default=np.nan)
    max_air_temp = row.get("max_air_temp", default=np.nan)

    if not pd.isna(min_air_temp) or not pd.isna(max_air_temp):
        if min_air_temp <= -5:
            return "Freezing"
        elif max_air_temp <= 5:
            return "Cold"
        elif max_air_temp <= 15:
            return "Cool"
        elif max_air_temp <= 25:
            return "Mild"
        else:
            return "Hot"

    return np.nan


def classify_sunshine_type(row: pd.Series) -> Optional[str]:
    """
    Classify the sunshine type for a given row of data.

    Args:
        row (pd.Series): A Pandas Series containing weather data.

    Returns:
        Optional[str]: The sunshine type classification. Possible values are:
            - "Sunny": If either "cs_24hr_sun_dur" or "wmo_24hr_sun_dur" is greater than 3.
            - "Partially Overcast": If either "cs_24hr_sun_dur" or "wmo_24hr_sun_dur" is between 0 and 3 (inclusive).
            - "Overcast": If both "cs_24hr_sun_dur" and "wmo_24hr_sun_dur" are NaN or less than 0.
            - None: If both "cs_24hr_sun_dur" and "wmo_24hr_sun_dur" are NaN.
    """
    cs_sun_dur = row.get("cs_24hr_sun_dur", default=np.nan)
    wmo_sun_dur = row.get("wmo_24hr_sun_dur", default=np.nan)

    if pd.isna(cs_sun_dur) and pd.isna(wmo_sun_dur):
        return None

    if cs_sun_dur > 3 or wmo_sun_dur > 3:
        return "Sunny"
    elif 0 <= cs_sun_dur <= 3 or 0 <= wmo_sun_dur <= 3:
        return "Partially Overcast"
    else:
        return "Overcast"


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
        "sunshine_type",
        "station_name",
    ]
    merged_df_ = merged_df[columns_to_keep]

    return merged_df, merged_df_


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

def process_and_merge_dataframes(year: int, metadata_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process and merge the data for the given year and metadata DataFrame.

    :param year: The year for which the data needs to be processed and merged.
    :type year: int
    :param metadata_df: The metadata DataFrame containing station information.
    :type metadata_df: pd.DataFrame
    :return: The merged DataFrame containing the weather and temperature data for the given year and metadata,
             and the DataFrame with all columns merged.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
    """
    if not isinstance(year, int):
        raise ValueError("year must be an integer")

    if not isinstance(metadata_df, pd.DataFrame):
        raise ValueError("metadata_df must be a pandas DataFrame")

    weather_data_df = pd.DataFrame(columns=[
        "date", "station_name", "station_latitude", "station_longitude", "weather_classifier", "temp_category",
        "sunshine_type"
    ])

    full_weather_data_df = pd.DataFrame(columns=[
        "date", "station_name", "station_latitude", "station_longitude", "weather_classifier", "temp_category",
        "sunshine_type", "max_air_temp", "min_air_temp", "min_grss_temp", "min_conc_temp", "cs_24hr_sun_dur",
        "wmo_24hr_sun_dur", "conc_state_id", "lying_snow_flag", "snow_depth", "frsh_snw_amt", "snow_day_id",
        "hail_day_id", "thunder_day_flag", "gale_day_flag", "frsh_mnt_snwfall_flag", "drv_24hr_sun_dur",
        "lying_snow_ht", "prcp_amt"
    ])

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
                    merged_df_, merged_df = keep_required_columns(merged_df, row)
                    weather_data_df = pd.concat([weather_data_df, merged_df], ignore_index=True)
                    full_weather_data_df = pd.concat([full_weather_data_df, merged_df_], ignore_index=True)

            processed_stations += 1
            #print(f"Processed {processed_stations} of {total_stations} stations")

    return weather_data_df, full_weather_data_df

if __name__ == "__main__":
    print(f"Processing year {sys.argv[1]} in Weather_ReadIn.py...")
    parser = argparse.ArgumentParser(description="Process and merge weather data.")
    parser.add_argument("year", type=int, help="The year for which to process the data.")
    args = parser.parse_args()

    year = args.year

    #year = 1972

    metadata_df = read_metadata()
    weather_data_df, full_weather_data_df = process_and_merge_dataframes(year, metadata_df)
    short_weather_data_file_path = os.path.join(PROCESSED_DATA_DIR, f"Weather_Data_Short_{year}.csv")
    full_weather_data_file_path = os.path.join(PROCESSED_DATA_DIR, f"Weather_Data_Full_{year}.csv")
    full_weather_data_df.to_csv(full_weather_data_file_path, index=False)
    weather_data_df.to_csv(short_weather_data_file_path, index=False)

    print(f"Saved short weather data to: {short_weather_data_file_path}")
    print(f"Saved full weather data to: {full_weather_data_file_path}")
