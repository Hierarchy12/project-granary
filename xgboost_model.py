import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import gradient_descent as gd
from sklearn.metrics import accuracy_score, classification_report
from random import random

# Load the dataset
df = pd.read_csv('project_granary_data.csv')

# Set up dataset variables
def granary_split(df, test_size=0.2):
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

    for i in range(len(df) - 6):
        new_row = []
        for j in range(6):
            new_row.append(df.iloc[i + j]['temperature_2m'])
        for j in range(6):
                new_row.append(df.iloc[i + j]['relative_humidity_2m'])
        for j in range(6):
                new_row.append(df.iloc[i + j]['surface_pressure'])
        for j in range(6):
                new_row.append(df.iloc[i + j]['wind_speed_10m'])
        for j in range(6):
                new_row.append(df.iloc[i + j]['wind_direction_10m'])
        for j in range(6):
                new_row.append(df.iloc[i + j]['wind_gusts_10m'])
        for j in range(6):
                new_row.append(df.iloc[i + j]['precipitation_probability'])
        if random() < test_size:
            x_test.loc[len(x_test)] = new_row
            if df.iloc[i + 6]['precipitation_probability'] >= 33.0:
                y_test.loc[len(y_test)] = 1
            else:
                y_test.loc[len(y_test)] = 0
        else:
            x_train.loc[len(x_train)] = new_row
            if df.iloc[i + 6]['precipitation_probability'] >= 33.0:
                y_train.loc[len(y_train)] = 1
            else:
                y_train.loc[len(y_train)] = 0

    return x_train, x_test, y_train, y_test
            

X_train, X_test, y_train, y_test = granary_split(df.drop(columns=['date']), test_size=0.2)
bst = XGBClassifier(n_estimators=2, eval_metric='mlogloss', max_depth=2, learning_rate=1, objective='binary:logistic', num_class=2)

bst.fit(X_train, y_train)
preds = bst.predict(X_test)

accuracy = accuracy_score(y_test, preds)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("Detailed Classification Report:")
print(classification_report(y_test, preds, zero_division=1))


