import requests
from datetime import datetime, timedelta
import pandas as pd

def fetch_weather_data(api_key, lat, lon, date):
    # Unix timestamp for the requested date
    timestamp = int(datetime.strptime(date, "%Y-%m-%d").timestamp())

    # Fetch weather data using the OpenWeatherMap API
    url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={timestamp}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()

    # Extract weather information from the response
    weather_data = {
        "date": date,
        "lat": lat,
        "lon": lon,
        "temperature": data["current"]["temp"],
        "weather_type": data["current"]["weather"][0]["description"]
    }

 
    return weather_data


def generate_dates():
    today = datetime.now()
    last_year = today - timedelta(days=365)
    dates = [last_year + timedelta(days=x) for x in range(366)]
    return [date.strftime("%Y-%m-%d") for date in dates]

# Replace 'YOUR_API_KEY' with your actual API key from OpenWeatherMap
api_key = "0b4dd54658240cc9a2136ea933fcb137"

# Example usage
uk_weather_stations = [
    {"icao": "EGLL", "lat": 51.4775, "lon": -0.461389},  # London Heathrow
    # Add more weather stations here
]



dates = generate_dates()

weather_data = []

for station in uk_weather_stations:
    lat = station["lat"]
    lon = station["lon"]

    for date in dates:
        station_data = fetch_weather_data(api_key, lat, lon, date)
        weather_data.append(station_data)

weather_df = pd.DataFrame(weather_data)
weather_df.to_csv("uk_weather_data_with_coordinates.csv", index=False)