import numpy as np
from flask import Flask, request, jsonify
import pickle

# Load the model
with open("best_logistic_pipeline.bin", "rb") as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict endpoint that receives a JSON payload, processes it,
    and returns a prediction.
    """
    try:
        # Parse JSON input
        input_data = request.json
        
        # Ensure input data is provided
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert input data to a 2D list
        input_features = [list(input_data.values())]

        # Get prediction from the model
        prediction = model.predict(input_features)
        prediction_proba = model.predict_proba(input_features).tolist()  # Convert to Python list
        
        # Prepare the response
        response = {
            "prediction": int(prediction[0]),  # Convert numpy int64 to Python int
            "probabilities": prediction_proba  # Already converted to Python list
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
