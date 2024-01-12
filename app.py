from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the model and the label encoder
model = pickle.load(open('model.pkl', 'rb'))
role_encoder = pickle.load(open('role_encoder.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([data[feature] for feature in sorted(data.keys())]).reshape(1, -1)

    class_probs = model.predict_proba(features)

    top_three_indices = np.argsort(class_probs[0])[::-1][:3]

    top_three_roles = role_encoder.inverse_transform(top_three_indices)

    return jsonify({'predictions': top_three_roles.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
