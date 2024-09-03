from flask import Flask, jsonify, request
import numpy as np
import pickle
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    print("I am in home")
    return "Welcome to the Simple Flask API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data['inputText']

    # Convert the input_text to a numpy array
    input_array = np.array(input_text)

    # Use the loaded model to make a prediction
    prediction = model.predict(input_array.reshape(1, -1))  # Reshape for a single sample

    # Determine the result based on the prediction
    result = 'Cancerous' if prediction[0] == 1 else 'Not Cancerous'

    # Prepare the response
    response = {
        'input_text': input_array.tolist(),  # Convert numpy array back to list for JSON
        'prediction': result,
        'message': 'Prediction made successfully!'
    }

    print(response)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
