import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

df = pd.read_csv("project_granary_data.csv") # Dataset
fs = 24  # Forecast steps
past_index = pd.date_range(start=pd.to_datetime(df.iat[-1, 0]) - pd.Timedelta(25, unit="h"), periods=fs+1, freq="h")
forecast_index = pd.date_range(start=pd.to_datetime(df.iat[-1, 0]) - pd.Timedelta(1, unit="h"), periods=fs, freq="h")

def fit_arimax(data, order=(1, 1, 1), endog_cols=None, exog_cols=None, forecast_steps=24):
    # Prepare the endogenous and exogenous variables
    endog = data[endog_cols] if endog_cols is not None else None
    exog = data[exog_cols] if exog_cols is not None else None 

    # Fit the ARIMAX model
    model = SARIMAX(endog, exog=exog, order=order)
    model_fit = model.fit(disp=False)

    # Forecast future values
    forecast = model_fit.get_forecast(steps=forecast_steps, exog=exog.tail(forecast_steps))
    forecast_mean = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()

    forecast_df = pd.DataFrame({'Forecast': forecast_mean}, index=forecast_index)   
    forecast_df['Lower CI'] = forecast_conf_int.iloc[:, 0]
    forecast_df['Upper CI'] = forecast_conf_int.iloc[:, 1]

    return model_fit, forecast_mean, forecast_conf_int

# (3, 1, 3) order is chosen based on the AIC/BIC values and model diagnostics
model_fit, forecast_mean, forecast_conf_int = fit_arimax(df, order=(3, 1, 3), endog_cols=["precipitation_probability"], exog_cols=["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"], forecast_steps=fs)

print("ARIMAX Model Summary:")
print(model_fit.summary())
print("\nForecasted Values:")
print(forecast_mean)
print("\nForecast Confidence Intervals:")
print(forecast_conf_int)

plt.figure(figsize=(12, 6))
plt.plot(past_index, df.iloc[len(df)-fs-2:len(df)-1, 7], label="Observed")
plt.plot(forecast_index, forecast_mean, label="Forecast")
plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color="yellow", alpha=0.3)
plt.legend()
plt.title("ARIMAX Model Forecast")
plt.show()
