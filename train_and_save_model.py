import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load CSV data
data = pd.read_csv('dataset9000.csv')
data.dropna(inplace=True)

# Define the mapping and feature columns
skill_level_mapping = {'Poor': 1, 'Beginner': 2, 'Average': 3, 'Intermediate': 4,
                        'Professional': 5, 'Excellent': 6, 'Not Interested': 7}
feature_columns = ['Database Fundamentals', 'Computer Architecture', 'Distributed Computing Systems',
                   'Cyber Security', 'Networking', 'Software Development', 'Programming Skills',
                   'Project Management', 'Computer Forensics Fundamentals', 'Technical Communication',
                   'AI ML', 'Software Engineering', 'Business Analysis', 'Communication skills',
                   'Data Science', 'Troubleshooting skills', 'Graphics Designing']

# Map skill levels and encode roles
for col in feature_columns:
    data[col] = data[col].map(skill_level_mapping)
role_encoder = LabelEncoder()
data['Role'] = role_encoder.fit_transform(data['Role'])

# Prepare data for training
X = data[feature_columns]
y = data['Role']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Initialize and train the model
#Using RandomForestClassifier since it performed better than the other models
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and label encoder to disk
with open('model.pkl', 'wb') as model_file, open('role_encoder.pkl', 'wb') as encoder_file:
    pickle.dump(model, model_file)
    pickle.dump(role_encoder, encoder_file)
