import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

def get_weather_data(date_str):
    url = f"https://www.wow.metoffice.gov.uk/archive?siteID=0&siteName=0&date={date_str}&submit=1"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('table', {'class': 'data-table'})
    if not table:
        print(f"No data found for {date_str}.")
        return

    headers = [th.text.strip() for th in table.find_all('th')]
    data = []

    for row in table.find_all('tr')[1:]:
        data.append({headers[i]: value.text.strip() for i, value in enumerate(row.find_all('td'))})

    return data

def save_weather_data_to_file(data, file_name):
    with open(file_name, 'w') as f:
        for row in data:
            f.write(','.join(row.values()) + '\n')

if __name__ == '__main__':
    start_date = datetime.strptime('2023-04-01', '%Y-%m-%d')
    end_date = datetime.strptime('2023-04-30', '%Y-%m-%d')

    all_data = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        weather_data = get_weather_data(date_str)
        if weather_data:
            all_data.extend(weather_data)
        current_date += timedelta(days=1)

    if all_data:
        file_name = 'historical_uk_weather_data.csv'
        save_weather_data_to_file(all_data, file_name)
        print(f"Weather data saved to {file_name}.")
    else:
        print("No weather data found.")