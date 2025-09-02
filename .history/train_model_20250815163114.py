import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib # Used to save the model

# 1. Load the dataset
try:
    df = pd.read_csv('dataset1.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: dataset1.csv not found. Make sure it's in the same folder.")
    exit()

# The PDF shows chest pain type values are 1-4, but datasets often start at 0.
# Let's adjust the column names to be more Python-friendly (no spaces).
df.columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'target'
]

# 2. Prepare the data
# We need to drop the 'target' column to create our features (X)
X = df.drop('target', axis=1)
y = df['target']

# 3. Train the model
# We will use the whole dataset for training to create the final model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)
print("Model training complete.")

# 4. Save the trained model to a file
joblib.dump(model, 'heart_disease_model.pkl')
print("Model saved successfully as heart_disease_model.pkl")