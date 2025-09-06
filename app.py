# Import necessary libraries
import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

# Set the page title and icon
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

# Page Title
st.title('Heart Disease Prediction App ❤️')

# --- Input Fields for User Data ---
st.header("Enter Patient's Details")

# Layout with columns
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Age', min_value=1, max_value=120, value=52)
    sex = st.selectbox('Sex', ('Male', 'Female'))
    cp = st.selectbox('Chest Pain Type', ('Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'))
    trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, value=120)

with col2:
    chol = st.number_input('Serum Cholestoral (mg/dl)', min_value=100, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ('True', 'False'))
    restecg = st.selectbox('Resting Electrocardiographic Results', ('Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'))
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)

with col3:
    exang = st.selectbox('Exercise Induced Angina', ('Yes', 'No'))
    oldpeak = st.number_input('ST depression induced by exercise', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox('Slope of the peak exercise ST segment', ('Upsloping', 'Flat', 'Downsloping'))

# --- Prediction Button ---
if st.button('Predict Heart Disease Risk'):
    
    # --- Data Conversion ---
    # Convert categorical text inputs to numbers that the model understands
    sex_num = 1 if sex == 'Male' else 0
    fbs_num = 1 if fbs == 'True' else 0
    exang_num = 1 if exang == 'Yes' else 0

    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    cp_num = cp_map[cp]

    restecg_map = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
    restecg_num = restecg_map[restecg]

    slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    slope_num = slope_map[slope]

    # Create a numpy array with the converted numeric data
    input_data = np.array([[age, sex_num, cp_num, trestbps, chol, fbs_num, restecg_num, thalach, exang_num, oldpeak, slope_num]])

    # --- Make Prediction ---
    prediction = model.predict(input_data)

    # --- Display Result ---
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error('High Risk of Heart Disease Detected 💔')
        st.warning("Please consult a doctor for further evaluation and advice.")
    else:
        st.success('Low Risk of Heart Disease Detected 💚')
        st.info("Maintain a healthy lifestyle to keep your heart healthy.")