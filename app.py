from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained models and label encoder
model_target = joblib.load('model_target.pkl')
model_failure_type = joblib.load('model_failure_type.pkl')
le_failure_type = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[
        float(data['features']['Air temperature [K]']),
        float(data['features']['Process temperature [K]']),
        float(data['features']['Rotational speed [rpm]']),
        float(data['features']['Torque [Nm]']),
        float(data['features']['Tool wear [min]'])
    ]])
    
    # Perform predictions
    target_prediction = model_target.predict(features)[0]
    failure_type_prediction = le_failure_type.inverse_transform(model_failure_type.predict(features))[0]
    
    # Return predictions as JSON response
    return jsonify({
        'target_prediction': target_prediction,
        'failure_type_prediction': failure_type_prediction
    })

if __name__ == "__main__":
    app.run(debug=True)
