import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
try:
    df = pd.read_csv("heart.csv")
    print("dataset loaded")
except FileNotFoundError:
    print("file not found")
    exit()

# Column names from dataset
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Force numeric, fill missing with median
df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(df.median(numeric_only=True))

# Split features/labels
X = df.drop("target", axis=1)
y = df["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "heart_disease_model.pkl")
print("model saved")
