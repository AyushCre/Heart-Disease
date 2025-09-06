# ❤️ Heart Disease Prediction App

This is a simple web application that predicts the risk of heart disease using a machine learning model. Users can input their health details such as age, sex, blood pressure, and cholesterol to get a real-time risk prediction.

## 🚀 Quick Start

For those familiar with Python projects, here are the essential commands to get the app running:

1.  **Install the required libraries:**
    ```bash
    pip install streamlit pandas scikit-learn numpy
    ```
2.  **Run the application:**
    ```bash
    streamlit run app.py
    ```
*(Note: Make sure your CSV file is named `heart.csv`)*

---

## ✨ Features

-   **User-Friendly Interface:** Built with Streamlit for a clean and interactive user experience.
-   **Real-time Predictions:** Instantly get a prediction after submitting the health data.
-   **Easy to Set Up:** The project can be set up and run on a local machine with minimal steps.
-   **Powered by Scikit-learn:** Uses a robust Random Forest Classifier for predictions.

---

## ⚙️ How It Works

The project follows a standard machine learning workflow:

1.  **Data:** The model is trained on the `heart.csv` dataset.
2.  **Training:** The `train_model.py` script trains a Random Forest Classifier model.
3.  **Model Saving:** The trained model is saved as `heart_disease_model.pkl`.
4.  **Web Interface:** `app.py` creates a web interface using Streamlit.
5.  **Prediction:** The app loads the saved model to make and display predictions based on user input.

---

## 💻 Technologies Used

-   **Python**
-   **Streamlit**
-   **Pandas**
-   **Scikit-learn**
-   **Joblib**

---

## 📋 Detailed Setup and Usage

Follow these steps to run the project on your local machine.

### 1. Get the Project Files
Make sure all the project files (`app.py`, `train_model.py`, `dataset1.csv`, etc.) are in the same directory.

### 2. Install Dependencies
Open your terminal or command prompt and run the following command to install the necessary libraries:
```bash
pip install streamlit pandas scikit-learn numpy