import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta

# Function to scrape the weather data for a given ICAO code, date, and coordinates
def scrape_weather_data(icao, lat, lon, date):
    url = f"https://www.wunderground.com/history/daily/gb/{icao}/date/{date}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    weather_data = {
        "date": date,
        "icao": icao,
        "lat": lat,
        "lon": lon,
        "temperature": None,
        "weather_type": None
    }

    try:
        weather_summary = soup.find("td", class_="mat-cell cdk-cell cdk-column-Conditions mat-column-Conditions ng-star-inserted").text.strip()
        weather_data["weather_type"] = weather_summary

        temp_avg = soup.find("td", class_="mat-cell cdk-cell cdk-column-TemperatureAvg mat-column-TemperatureAvg ng-star-inserted").text.strip()
        weather_data["temperature"] = float(temp_avg)

    except AttributeError:
        pass

    return weather_data

def generate_dates():
    today = datetime.now()
    last_year = today - timedelta(days=365)
    dates = [last_year + timedelta(days=x) for x in range(366)]
    return [date.strftime("%Y-%m-%d") for date in dates]

# Fetch UK ICAO codes with coordinates
uk_weather_stations = pd.read_csv("uk_icao_codes_with_coordinates.csv")

# Generate dates for the past year
dates = generate_dates()

# Initialize an empty list to store weather data for all weather stations
weather_data = []

# Loop through each weather station and date to fetch weather data
for index, station in uk_weather_stations.iterrows():
    icao = station["icao"]
    lat = station["lat"]
    lon = station["lon"]

    for date in dates:
        station_data = scrape_weather_data(icao, lat, lon, date)
        weather_data.append(station_data)
        
        print(f"Done for {date} in airport {icao}")
    print(f"Done for airport {icao}\n")
    break
    

# Store the fetched weather data in a pandas DataFrame
weather_df = pd.DataFrame(weather_data)

# Save the DataFrame to a CSV file
weather_df.to_csv("uk_weather_data_with_coordinates.csv", index=False)