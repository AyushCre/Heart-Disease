import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('heart_disease_model.pkl', 'rb'))

def validate_inputs(inputs):
    ranges = [
        (1, 120), (0, 1), (0, 3), (80, 200), (100, 600), (0, 1),
        (0, 2), (60, 220), (0, 1), (0.0, 6.0), (0, 2), (0, 4), (0, 3)
    ]
    adjusted = []
    for value, (low, high) in zip(inputs, ranges):
        if value < low:
            adjusted.append(low)
        elif value > high:
            adjusted.append(high)
        else:
            adjusted.append(value)
    return adjusted

def adjust_confidence(prob, factor=2):
    return (prob ** factor) / ((prob ** factor) + ((1 - prob) ** factor))

st.title("💓 Heart Disease Prediction App")
st.write("Enter your health parameters below:")

age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting BP (mm Hg)", min_value=0, max_value=300, value=120)
chol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=700, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Rest ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=0, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (CA)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal", [0, 1, 2, 3])

if st.button("Predict"):
    input_data = [age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal]
    validated_data = validate_inputs(input_data)
    prediction_prob = model.predict_proba([validated_data])[0]
    prediction = np.argmax(prediction_prob)
    confidence = prediction_prob[prediction]
    adjusted_conf = adjust_confidence(confidence, factor=2) * 100
    if prediction == 0:
        st.success(f"No Heart Disease Detected ✅ (Confidence: {adjusted_conf:.2f}%)")
    else:
        st.error(f"Heart Disease Detected ⚠ (Confidence: {adjusted_conf:.2f}%)")
