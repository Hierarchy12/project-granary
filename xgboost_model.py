import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from random import random
import datetime
import matplotlib.pyplot as plt

# Load the dataset and forecast periods
fs = 24  # Forecast steps
df = pd.read_csv('project_granary_data.csv')
dt = datetime.datetime.now(datetime.timezone.utc)
dt = dt.replace(minute=0, second=0, microsecond=0)
past_index = pd.date_range(start=pd.to_datetime(df.iat[-1, 0]) - pd.Timedelta(23, unit="h"), periods=25, freq="h")
forecast_index = pd.date_range(start=pd.to_datetime(df.iat[-1, 0]) + pd.Timedelta(1, unit="h"), periods=fs + 1, freq="h")
forecast_df = pd.DataFrame({'forecast': [df.iloc[len(df) - 1]['precipitation_probability']] + [None] * (len(forecast_index) - 1)}, index=forecast_index)

# Set up dataset variables
def granary_split(df, test_size=0.2, threshold=5.0, forecast_hour=1):
    x_train = pd.DataFrame(columns = [
        'temp_1','temp_2','temp_3','temp_4','temp_5','temp_6',
        'humidity_1','humidity_2','humidity_3','humidity_4','humidity_5','humidity_6',
        'pressure_1','pressure_2','pressure_3','pressure_4','pressure_5','pressure_6',
        'ws_1','ws_2','ws_3','ws_4','ws_5','ws_6',
        'wd_1','wd_2','wd_3','wd_4','wd_5','wd_6',
        'gust_1','gust_2','gust_3','gust_4','gust_5','gust_6',
        'precip_1','precip_2','precip_3','precip_4','precip_5','precip_6'
    ])
    x_test = pd.DataFrame(columns = [
        'temp_1','temp_2','temp_3','temp_4','temp_5','temp_6',
        'humidity_1','humidity_2','humidity_3','humidity_4','humidity_5','humidity_6',
        'pressure_1','pressure_2','pressure_3','pressure_4','pressure_5','pressure_6',
        'ws_1','ws_2','ws_3','ws_4','ws_5','ws_6',
        'wd_1','wd_2','wd_3','wd_4','wd_5','wd_6',
        'gust_1','gust_2','gust_3','gust_4','gust_5','gust_6',
        'precip_1','precip_2','precip_3','precip_4','precip_5','precip_6'
    ])
    y_train = pd.Series(dtype='float64')
    y_test = pd.Series(dtype='float64')

    for i in range(len(df) - 5 - forecast_hour):
        new_row = []
        for j in range(6):
            new_row.append(df.iloc[i + j]['temperature_2m'])
            new_row.append(df.iloc[i + j]['relative_humidity_2m'])
            new_row.append(df.iloc[i + j]['surface_pressure'])
            new_row.append(df.iloc[i + j]['wind_speed_10m'])
            new_row.append(df.iloc[i + j]['wind_direction_10m'])
            new_row.append(df.iloc[i + j]['wind_gusts_10m'])
            new_row.append(df.iloc[i + j]['precipitation_probability'])
        if random() < test_size:
            x_test.loc[len(x_test)] = new_row
            if df.iloc[i + 5 + forecast_hour]['precipitation_probability'] >= threshold:
                y_test.loc[len(y_test)] = 1
            else:
                y_test.loc[len(y_test)] = 0
        else:
            x_train.loc[len(x_train)] = new_row
            if df.iloc[i + 5 + forecast_hour]['precipitation_probability'] >= threshold:
                y_train.loc[len(y_train)] = 1
            else:
                y_train.loc[len(y_train)] = 0

    return x_train, x_test, y_train, y_test

def current_conditions(df):
    x = pd.DataFrame(columns = [
        'temp_1','temp_2','temp_3','temp_4','temp_5','temp_6',
        'humidity_1','humidity_2','humidity_3','humidity_4','humidity_5','humidity_6',
        'pressure_1','pressure_2','pressure_3','pressure_4','pressure_5','pressure_6',
        'ws_1','ws_2','ws_3','ws_4','ws_5','ws_6',
        'wd_1','wd_2','wd_3','wd_4','wd_5','wd_6',
        'gust_1','gust_2','gust_3','gust_4','gust_5','gust_6',
        'precip_1','precip_2','precip_3','precip_4','precip_5','precip_6'
        ])
    
    new_row = []
    for j in range(6):
        new_row.append(df.iloc[len(df) - 6 + j]['temperature_2m'])
        new_row.append(df.iloc[len(df) - 6 + j]['relative_humidity_2m'])
        new_row.append(df.iloc[len(df) - 6 + j]['surface_pressure'])
        new_row.append(df.iloc[len(df) - 6 + j]['wind_speed_10m'])
        new_row.append(df.iloc[len(df) - 6 + j]['wind_direction_10m'])
        new_row.append(df.iloc[len(df) - 6 + j]['wind_gusts_10m'])
        new_row.append(df.iloc[len(df) - 6 + j]['precipitation_probability'])
    return new_row



current_obs = current_conditions(df.drop(columns=['date']))

for i in range(1, fs + 1):
    X_train, X_test, y_train, y_test = granary_split(df.drop(columns=['date']), test_size=0.2, threshold=5.0, forecast_hour=i)
    bst = XGBClassifier(n_estimators=2, eval_metric='mlogloss', max_depth=6, learning_rate=1, objective='binary:logistic')

    bst.fit(X_train, y_train)
    preds = bst.predict(X_test)
    future_preds = bst.predict_proba([current_obs])
    forecast_dt = dt + datetime.timedelta(hours=i)

    accuracy = accuracy_score(y_test, preds)
    print(f"For forecast hour {i} ({forecast_dt}):")
    print("Detailed Classification Report:")
    print(classification_report(y_test, preds, zero_division=1))
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Current period prediction: {future_preds[0][1] * 100:.2f}% chance of precipitation")
    forecast_df.loc[forecast_index[i], 'forecast'] = future_preds[0][1] * 100

plt.figure(figsize=(12, 6))
plt.plot(past_index, df.iloc[len(df) - 26 : len(df) - 1, 7], label="Observed")
plt.plot(forecast_index, forecast_df['forecast'], label="Forecast")
plt.legend()
plt.title("XGBoost Model Forecast")
plt.show()