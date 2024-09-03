from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the features from the form
        features = request.form['feature']
        features = features.split(',')
        np_features = np.asarray(features, dtype=np.float32)
        
        # Ensure the correct number of features are provided
        if len(np_features) != model.n_features_in_:
            message = "Invalid input: Please provide the correct number of features."
        else:
            # Make the prediction
            pred = model.predict(np_features.reshape(1, -1))
            message = 'Cancerous' if pred[0] == 1 else 'Not Cancerous'
    
    except ValueError:
        message = "Invalid input: Please ensure all features are numerical values."
    
    return render_template('index.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)
