import requests
import csv
from io import StringIO

def get_icao_codes_with_coordinates(country):
    # Fetch airport data for the specified country using the OurAirports API
    url = f"https://ourairports.com/countries/{country}/airports.csv"
    response = requests.get(url)
    response_text = response.text

    # Read the CSV data
    csv_data = StringIO(response_text)
    reader = csv.DictReader(csv_data)

    # Extract ICAO codes, latitude, and longitude
    icao_codes_with_coordinates = []
    for row in reader:
        icao = row["ident"]
        lat = float(row["latitude_deg"])
        lon = float(row["longitude_deg"])

        if icao.startswith(country):  # Filter ICAO codes based on the country code
            icao_codes_with_coordinates.append({"icao": icao, "lat": lat, "lon": lon})

    return icao_codes_with_coordinates

def write_to_csv(icao_codes_with_coordinates, output_filename):
    with open(output_filename, "w", newline="") as csvfile:
        fieldnames = ["icao", "lat", "lon"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in icao_codes_with_coordinates:
            writer.writerow(row)

# Example usage
uk_icao_codes_with_coordinates = get_icao_codes_with_coordinates("EG")
write_to_csv(uk_icao_codes_with_coordinates, "uk_icao_codes_with_coordinates.csv")
