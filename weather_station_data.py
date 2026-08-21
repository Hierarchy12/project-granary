import openmeteo_requests
import pandas as pd
import requests
import requests_cache
from retry_requests import retry
import datetime

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
params = {
	"latitude": 35.764,
	"longitude": -79.064,
	"hourly": "precipitation_probability",
	"timezone": "GMT",
    "past_days": 7,
    "forecast_days": 2,
	"temperature_unit": "fahrenheit",
}
responses = openmeteo.weather_api(url, params = params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_precipitation_probability = hourly.Variables(0).ValuesAsNumpy()

hourly_data = {
	"date": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)
}

hourly_data["precipitation_probability"] = hourly_precipitation_probability

precip_df = pd.DataFrame(data = hourly_data)

# Import the CSV file
df = pd.read_csv('project_granary_data.csv')

# Main station
resp_main = requests.get("https://api.weather.com/v2/pws/observations/hourly/7day?stationId=KNCCHAPE335&format=json&units=e&apiKey=000f1365a4884f368f1365a4885f3629&numericPrecision=decimal")
# Station nearby (needed to obtain pressure data)
resp_near = requests.get("https://api.weather.com/v2/pws/observations/hourly/7day?stationId=KNCPITTS117&format=json&units=e&apiKey=000f1365a4884f368f1365a4885f3629&numericPrecision=decimal")

if resp_main.status_code == 200 and resp_near.status_code == 200:
    data_main = resp_main.json()
    data_near = resp_near.json()
    len_main = len(data_main["observations"])
    len_near = len(data_near["observations"])

    for i in range(len_main):
        # Add new row to the dataframe with most data from the main station and pressure data from the nearby station
        df.loc[len(df), "date"] = data_main["observations"][i]["obsTimeUtc"]
        df["date"] = pd.to_datetime(df["date"], format='ISO8601', errors='coerce')
        df.loc[len(df) - 1, "temperature_2m"] = data_main["observations"][i]["imperial"]["tempAvg"] 
        df.loc[len(df) - 1, "relative_humidity_2m"] = data_main["observations"][i]["humidityAvg"] 
        df.loc[len(df) - 1, "surface_pressure"] = ((data_near["observations"][len_near - len_main + i]["imperial"]["pressureMax"] + data_near["observations"][len_near - len_main + i]["imperial"]["pressureMin"]) / 2) * 33.8639
        df.loc[len(df) - 1, "wind_speed_10m"] = data_main["observations"][i]["imperial"]["windspeedAvg"]
        df.loc[len(df) - 1, "wind_direction_10m"] = data_main["observations"][i]["winddirAvg"]
        df.loc[len(df) - 1, "wind_gusts_10m"] = data_main["observations"][i]["imperial"]["windgustHigh"]
        df.loc[len(df) - 1, "precipitation_probability"] = 0.0

    # Round the date column to the nearest hour and drop duplicates
    df['date'] = df['date'].dt.round('h')
    df.drop_duplicates(subset = ['date'], inplace = True)
    precip_df['date'] = precip_df['date'].dt.round('h')

    # Get current time and round it down to the nearest hour to avoid duplicates when merging with the precipitation data
    dt = datetime.datetime.now(datetime.timezone.utc)
    dt = dt.replace(minute=0, second=0, microsecond=0)

    # Merge the precipitation data with the main dataframe on the date column, keeping only rows with dates less than or equal to the current time
    df = pd.merge(df, precip_df, on='date', how='left')
    df = df[df['date'] <= dt]
    for i in range(len(df)):
        if pd.notnull(df.loc[i, 'precipitation_probability_y']):
            df.loc[i, 'precipitation_probability_x'] = df.loc[i, 'precipitation_probability_y']
    df = df.drop(columns=['precipitation_probability_y'], errors='ignore')
    df = df.rename(columns={'precipitation_probability_x': 'precipitation_probability'})

else:
   print(f"API returned code {resp_main.status_code}, API in Pittsboro returned code {resp_near.status_code}")

# Export the dataframe to the CSV file
df.to_csv("project_granary_data.csv", index = False)
