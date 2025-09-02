import pickle
import streamlit as st
import numpy as np

model_path = "heart_disease_model.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("Heart Disease Prediction")

age = st.number_input("Age", min_value=1, max_value=120, step=1)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise", min_value=0.0, max_value=10.0, step=0.1)
slope = st.selectbox("Slope of peak exercise ST segment", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-3) colored by flourosopy", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

if st.button("Predict"):
    sex_val = 1 if sex == "Male" else 0
    features = np.array([[age, sex_val, cp, trestbps, chol, fbs, restecg,
                           thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ Heart disease detected — Confidence: {probability*100:.2f}%")
    else:
        st.success(f"✅ No heart disease detected — Confidence: {probability*100:.2f}%")
