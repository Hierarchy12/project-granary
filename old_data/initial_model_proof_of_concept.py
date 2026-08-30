import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
df = pd.read_csv('project_granary_data.csv')
X = df.drop(columns=['precipitation_probability', 'date'])
y = df['precipitation_probability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')

bst.fit(X_train, y_train)
preds = bst.predict(X_test)

accuracy = accuracy_score(y_test, preds)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("Detailed Classification Report:")
print(classification_report(y_test, preds, zero_division=1))

X_test_df = pd.DataFrame(X_test, columns=X.columns)
X_test_df['Predictions'] = preds

print(X_test_df.describe())
