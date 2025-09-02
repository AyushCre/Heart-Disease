import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Define the exact column names as per the dataset's structure
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'target'
]

# 1. Load the dataset using the defined column names
try:
    # We tell pandas to use our names and skip the original header row (if any)
    df = pd.read_csv('dataset1.csv', header=0, names=column_names)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: dataset1.csv not found. Make sure it's in the same folder.")
    exit()

# 2. Prepare the data
# Ensure all data is numeric, converting any non-numeric values to NaN and then filling with the median
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.fillna(df.median(), inplace=True)

X = df.drop('target', axis=1)
y = df['target']

# 3. Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)
print("Model training complete.")

# 4. Save the trained model to a file
joblib.dump(model, 'heart_disease_model.pkl')
print("Model saved successfully as heart_disease_model.pkl")