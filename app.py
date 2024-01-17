from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from ai_util import get_reduced_data

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

@app.route('/suggest', methods=['POST'])
def get_recommendations():
    try:
        data = request.json

        if 'user_career' not in data:
            return jsonify({'error': 'Missing "user_career" in request data'}), 400

        user_career = data['user_career']

        certifications_prompt = f"I am a {user_career}. Can you suggest which certifications or qualifications are needed for this career?"
        skills_prompt = f"I am a {user_career}. Can you suggest which skills are essential for this career?"
        advice_prompt = f"I am a {user_career}. Do you have any general career advice for someone in this field?"

        try:
            certifications = get_reduced_data(certifications_prompt)
            skills = get_reduced_data(skills_prompt)
            career_advice = get_reduced_data(advice_prompt)
        except Exception as e:
            return jsonify({'error': f'Error processing data: {str(e)}'}), 500

        response_data = {
            'certifications': certifications,
            'skills': skills,
            'career_advice': career_advice,
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
