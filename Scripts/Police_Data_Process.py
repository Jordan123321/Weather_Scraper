#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import argparse
import logging
import time
from typing import List, Tuple
from sklearn.neighbors import BallTree
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)



# Get the current working directory and its parent directory
PATH1 = os.getcwd()
PATH0 = os.path.dirname(PATH1)
PROCESSED_DATA_DIR = os.path.join(PATH0, 'Processed_Data')
RAW_POLICE_DATA_DIR = os.path.join(PATH0, 'Raw_Police_Data')
PROCESSED_POLICE_DATA_DIR = os.path.join(PATH0, 'Processed_Police_Data')
PROCESSED_POLICE_DATA_AGG_DIR = os.path.join(PATH0, 'Processed_Police_Data_Agg')

# Create directories if they do not exist
for directory in [PROCESSED_DATA_DIR, RAW_POLICE_DATA_DIR, PROCESSED_POLICE_DATA_DIR, PROCESSED_POLICE_DATA_AGG_DIR]:
    os.makedirs(directory, exist_ok=True)


earth_radius=6371

def read_crime_data(year: str, month: str) -> pd.DataFrame:
    """
    Reads crime data for a specific year and month.
    """
    # Create a list to hold dataframes
    dfs = []

    # Define the subfolder path
    subfolder_path = os.path.join(RAW_POLICE_DATA_DIR, f'{year}-{month}')

    # Specify the columns to read in
    cols_to_read = ['Month', 'Falls within', 'Longitude', 'Latitude', 'Crime type', 'Last outcome category']

    # Loop through all files in the subfolder
    for filename in os.listdir(subfolder_path):
        # Check if the file is a CSV and ends with "street.csv"
        if filename.endswith("street.csv"):
            # Create the full file path
            file_path = os.path.join(subfolder_path, filename)
            # Read the CSV file and append it to the list
            dfs.append(pd.read_csv(file_path, usecols=cols_to_read))

    # Concatenate all dataframes in the list
    df = pd.concat(dfs, ignore_index=True)

    return df

def read_weather_data(year: str, month: str) -> pd.DataFrame:
    """
    Reads weather data for a specific year and month.
    """
    # Define the weather data file path
    weather_data_file_path = os.path.join(PROCESSED_DATA_DIR, f'Monthly_Weather_Data_Short_{year}.csv')

    # Read the weather data CSV file
    weather_df = pd.read_csv(weather_data_file_path)

    # Filter the weather data DataFrame to only include rows where the 'month' column matches the specified month
    # Note: The day is assumed to be the last day of the month
    last_day_of_month = pd.to_datetime(f'{year}-{month}', format='%Y-%m') + pd.offsets.MonthEnd(1)
    weather_df = weather_df[weather_df['month'] == last_day_of_month.strftime('%Y-%m-%d')]

    return weather_df



def process_weather_category(df: pd.DataFrame, weather_df: pd.DataFrame, category: str) -> Tuple[BallTree, pd.DataFrame]:
    """
    Process a specific weather category.
    """
    # Create a DataFrame for the category
    category_df = weather_df.dropna(subset=[category]).copy()
    category_df['station_latitude_rad'] = np.radians(category_df['station_latitude'])
    category_df['station_longitude_rad'] = np.radians(category_df['station_longitude'])

    # Create a BallTree
    tree = BallTree(category_df[['station_latitude_rad', 'station_longitude_rad']].values, leaf_size=15, metric='haversine')

    # Query the BallTree for nearest weather station
    dist, idx = tree.query(df[['Latitude_rad', 'Longitude_rad']].values, return_distance=True)

    return tree, dist, idx

def process_data(year_month: Tuple[str, str]) -> None:
    """
    Main function to process the data.
    """
    start_time = time.time()
    year, month = year_month
    logging.info(f'Starting process for {year}-{month}')

    df = read_crime_data(year, month)
    weather_df = read_weather_data(year, month)

    # Create a DataFrame of rows with missing latitude or longitude
    error_df = df[df['Latitude'].isna() | df['Longitude'].isna()]

    # Calculate the percentage of data dropped
    percent_drop = len(error_df) / len(df) * 100

    # Create a DataFrame for the log
    log_df = pd.DataFrame({'Year': [year], 'Month': [month], 'Percent Drop': [percent_drop]})

    # Remove rows with missing latitude or longitude from the original DataFrame
    df = df.dropna(subset=['Latitude', 'Longitude'])

    # Convert coordinates to radians for BallTree
    df['Latitude_rad'] = np.radians(df['Latitude'])
    df['Longitude_rad'] = np.radians(df['Longitude'])

    # Process each weather category
    for category in ['weather_classifier', 'temp_category', 'sunshine_type']:
        tree, dist, idx = process_weather_category(df, weather_df, category)
        df[category] = weather_df.iloc[idx.ravel()][category].values
        df[f'distance_to_{category}_km'] = dist.ravel() * earth_radius

    # Define the file name
    file_name = f'processed_police_and_weather_data_{year}_{month}.csv'

    # Define the full file path
    file_path = os.path.join(PROCESSED_POLICE_DATA_DIR, file_name)

    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)

    # Create a shorter version of the DataFrame
    short_df = df[['Crime type', 'Latitude', 'Longitude', 'weather_classifier', 'temp_category', 'sunshine_type']]

    # Define the file name for the short version
    short_file_name = f'processed_police_and_weather_data_short_version_{year}_{month}.csv'

    # Define the full file path for the short version
    short_file_path = os.path.join(PROCESSED_POLICE_DATA_DIR, short_file_name)

    # Save the short version of the DataFrame to a CSV file
    short_df.to_csv(short_file_path, index=False)


    # Aggregate the short version of DataFrame
    aggregated_df = short_df.groupby(['Crime type', 'weather_classifier', 'temp_category', 'sunshine_type']).size().reset_index(name='count')
    aggregated_df['date_month'] = f'{year}-{month}'

    elapsed_time = time.time() - start_time
    logging.info(f'Finished process for {year}-{month}. Elapsed time: {elapsed_time:.2f} seconds')

    return aggregated_df, log_df




if __name__ == "__main__":
    # Get the number of available cores and decrement by 1 to leave one free for other tasks
    n_cores = os.cpu_count() - 1

    # Generate a list of year/month pairs for the desired range
    # We want data from years 2011 to 2023, inclusive, for all months
    year_month_pairs = [('%04d' % year, '%02d' % month) for year in range(2011, 2022) for month in range(1, 13)]

    aggregated_dfs = []
    log_dfs = []

    # Use a ProcessPoolExecutor to parallelize the function execution
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        # Create a future for each year-month pair
        futures = {executor.submit(process_data, pair): pair for pair in year_month_pairs}

        # Collect the results as they become available
        for future in as_completed(futures):
            year_month = futures[future]
            try:
                agg_df, log_df = future.result()
                aggregated_dfs.append(agg_df)
                log_dfs.append(log_df)
            except Exception as exc:
                logging.error(f'{year_month} generated an exception: {exc}')
            else:
                logging.info(f'{year_month} processed successfully')

    # Concatenate all returned dataframes into a single one
    final_df = pd.concat(aggregated_dfs)

    # Concatenate all logs into a single DataFrame
    log_df = pd.concat(log_dfs)

    # Format the 'Percent Drop' column
    log_df['Percent Drop'] = log_df['Percent Drop'].astype(float).apply(lambda x: '{:.2f}%'.format(x))

    # Write the DataFrame to the log file
    log_df.to_csv('log/data_drop_log.csv', index=False)

    # Define the file path where to save the final aggregated data
    agg_file_path = os.path.join(PROCESSED_POLICE_DATA_AGG_DIR, 'final_aggregated_data.csv')

    # Save the final dataframe to a csv file
    final_df.to_csv(agg_file_path, index=False)
