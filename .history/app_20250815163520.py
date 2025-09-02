import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained machine learning model
# This is the model file you created with train_model.py
try:
    model = joblib.load('heart_disease_model.pkl')
except FileNotFoundError:
    st.error("Model file not found! Please run train_model.py to create it.")
    st.stop()

# --- User Interface Setup ---

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

st.title("❤️ Heart Disease Prediction App")
st.write("""
This app predicts the likelihood of a patient having heart disease based on their medical attributes.
Please enter the patient's details below.
""")

# Create columns for a cleaner layout
col1, col2, col3 = st.columns(3)

# --- Input Fields ---

with col1:
    age = st.number_input('Age', min_value=1, max_value=120, value=50)
    sex = st.selectbox('Sex', options=[('Male', 1), ('Female', 0)], format_func=lambda x: x[0])
    cp = st.selectbox('Chest Pain Type', options=[
        ('Typical Angina', 1),
        ('Atypical Angina', 2),
        ('Non-anginal Pain', 3),
        ('Asymptomatic', 4)
    ], format_func=lambda x: x[0])

with col2:
    trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, value=120)
    chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[('True', 1), ('False', 0)], format_func=lambda x: x[0])

with col3:
    restecg = st.selectbox('Resting ECG Results', options=[
        ('Normal', 0),
        ('ST-T Wave Abnormality', 1),
        ('Left Ventricular Hypertrophy', 2)
    ], format_func=lambda x: x[0])
    thalach = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=220, value=150)
    exang = st.selectbox('Exercise Induced Angina', options=[('Yes', 1), ('No', 0)], format_func=lambda x: x[0])

# Sliders for continuous values that require more precision
oldpeak = st.slider('ST Depression (Oldpeak)', min_value=0.0, max_value=6.2, value=1.0, step=0.1)
slope = st.selectbox('Slope of Peak Exercise ST Segment', options=[
    ('Upsloping', 1),
    ('Flat', 2),
    ('Downsloping', 3)
], format_func=lambda x: x[0])

# --- Prediction Logic ---

# Create a button to trigger the prediction
if st.button('Predict Heart Disease Risk', use_container_width=True):
    # The user input values from the selectbox are tuples (e.g., ('Male', 1)), so we get the numeric value using [1]
    input_features = [
        age, sex[1], cp[1], trestbps, chol, fbs[1],
        restecg[1], thalach, exang[1], oldpeak, slope[1]
    ]
    
    # Convert the list of features into a NumPy array for the model
    features_array = np.array(input_features).reshape(1, -1)
    
    # Make a prediction
    prediction = model.predict(features_array)
    probability = model.predict_proba(features_array)
    
    st.subheader("Prediction Result")
    
    if prediction[0] == 1:
        st.error(f"High Risk of Heart Disease", icon="⚠️")
        st.write(f"The model predicts a **{probability[0][1]*100:.2f}%** probability of heart disease.")
    else:
        st.success(f"Low Risk of Heart Disease", icon="✅")
        st.write(f"The model predicts a **{probability[0][0]*100:.2f}%** probability of not having heart disease.")