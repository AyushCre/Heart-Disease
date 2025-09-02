from flask import Flask, request, jsonify
import joblib # Or 'import pickle' if you used that
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load your trained machine learning model
# Make sure the 'heart_disease_model.pkl' file is in the same folder
try:
    model = joblib.load('heart_disease_model.pkl')
except FileNotFoundError:
    print("Model file not found! Please train and save the model first.")
    model = None

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    # Get the JSON data sent from the form
    data = request.get_json(force=True)

    # The order of features MUST match the order your model was trained on
    # Based on the PDF, there are 11 features.
    features = [
        data['age'], data['sex'], data['cp'], data['trestbps'],
        data['chol'], data['fbs'], data['restecg'], data['thalach'],
        data['exang'], data['oldpeak'], data['slope']
    ]

    # Convert features to a NumPy array for the model
    final_features = [np.array(features)]
    
    # Make a prediction
    prediction = model.predict(final_features)

    # Send back the result as JSON
    # prediction[0] will be 0 (Low Risk) or 1 (High Risk)
    return jsonify({'prediction': int(prediction[0])})

# Run the app
if __name__ == '__main__':
    app.run(port=5000, debug=True)