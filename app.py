from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and the label encoder
model = pickle.load(open('model.pkl', 'rb'))
role_encoder = pickle.load(open('role_encoder.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([data[feature] for feature in sorted(data.keys())]).reshape(1, -1)
    prediction = model.predict(features)
    role = role_encoder.inverse_transform(prediction)
    return jsonify({'prediction': role[0]})

if __name__ == '__main__':
    app.run(debug=True)
