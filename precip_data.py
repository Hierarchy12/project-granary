import openmeteo_requests

import pandas as pd
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
	"start_date": "2024-04-25",
	"end_date": "2026-10-03",
	"hourly": "precipitation",
    "precipitation_unit": "inch",
}
responses = openmeteo.weather_api(url, params = params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_precipitation = hourly.Variables(0).ValuesAsNumpy()

hourly_data = {
	"date": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)
}

hourly_data["precipitation"] = hourly_precipitation

precip_df = pd.DataFrame(data = hourly_data)
    
precip_df['date'] = precip_df['date'].dt.round('h')
precip_df.drop_duplicates(subset = ['date'], inplace = True)
precip_df = precip_df[precip_df['date'] >= '2024-04-25 15:00:00+00:00']
dt = datetime.datetime.now(datetime.timezone.utc)
dt = dt.replace(minute=0, second=0, microsecond=0)
precip_df = precip_df[precip_df['date'] <= dt]
precip_df.to_csv('precipitation_data.csv', index = False)
df = pd.read_csv('project_granary_data.csv')
df['date'] = pd.to_datetime(df['date'], utc=True)




# Merge the precipitation data with the main dataframe on the date column, keeping only rows with dates less than or equal to the current time
print("Left columns:", df.columns.tolist())
print("Right columns:", precip_df.columns.tolist())
df = pd.merge(df, precip_df, on = 'date', how = 'left', suffixes = ('_main', '_precipitation'))
df.to_csv("project_granary_data.csv", index = False)