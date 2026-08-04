import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv('project_granary_data.csv')

p = range(0, 4)
d = range(0, 3)
q = range(0, 4)
pdq = list(itertools.product(p, d, q))
best_aic = np.inf
best_order = None
best_model = None

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

    return model_fit, forecast_mean, forecast_conf_int

model_fit, forecast_mean, forecast_conf_int = fit_arimax(df, order=(3, 1, 3), endog_cols=['precipitation_probability'], exog_cols=['temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m'], forecast_steps=24)

print("ARIMAX Model Summary:")
print(model_fit.summary())
print("\nForecasted Values:")
print(forecast_mean)
print("\nForecast Confidence Intervals:")
print(forecast_conf_int)
