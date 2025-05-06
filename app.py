from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

# Symptom vocabulary
symptom_keywords = {
    'fever': ['fever', 'high temperature', 'hot'],
    'cough': ['cough', 'dry throat', 'sore chest'],
    'headache': ['headache', 'migraine', 'pain in head'],
    'sore_throat': ['sore throat', 'pain swallowing', 'scratchy throat'],
    'fatigue': ['fatigue', 'tiredness', 'exhausted', 'lack of energy'],
    'body_ache': ['body ache', 'muscle pain', 'body pain', 'muscle ache']
}

# Function to extract symptoms from user input
def extract_symptoms(user_input):
    user_input_lower = user_input.lower()
    symptoms_detected = {symptom: any(keyword in user_input_lower for keyword in keywords)
                         for symptom, keywords in symptom_keywords.items()}
    return symptoms_detected

app = Flask(__name__)

# Severity mapping
severity_mapping = {
    'fever': 2,
    'cough': 1,
    'headache': 1,
    'sore_throat': 1,
    'fatigue': 1,
    'body_ache': 1
}

def calculate_severity(detected_symptoms):
    severity_score = sum(severity_mapping[symptom] for symptom, present in detected_symptoms.items() if present)
    if severity_score > 3:
        return "high"
    elif severity_score == 3:
        return "moderate"
    else:
        return "low"

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Mapping diseases to medicines or advice
disease_to_medicines = {
    'flu': ['Paracetamol', 'Antihistamines', 'Rest and Fluids'],
    'cold': ['Cetirizine', 'Decongestant', 'Warm fluids'],
    'migraine': ['Ibuprofen', 'Sumatriptan', 'Caffeine in moderation'],
    'allergy': ['Loratadine', 'Cetirizine', 'Avoid allergens'],
    'general_tiredness': ['Rest', 'Stay Hydrated', 'Nutritious Food'],
    'muscle_strain': ['Rest', 'Ice packs', 'Mild stretching exercises']
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_text = data['symptoms']

    # Extract symptoms
    detected_symptoms = extract_symptoms(user_text)

    # Predict disease
    input_df = pd.DataFrame([detected_symptoms])
    prediction = model.predict(input_df)[0]

    # Calculate severity
    severity = calculate_severity(detected_symptoms)

    # Medicines or advice
    medicines = disease_to_medicines.get(prediction, [])

    # Prepare medicine list
    if medicines:
        medicines_advice = "\n💊 Recommended: " + ", ".join(medicines)
    else:
        medicines_advice = ""

    # Response
    response = (
        f"🔍 Based on your symptoms, you may have {prediction.replace('_', ' ')} "
        f"with {severity} severity. Please consult a doctor for confirmation."
        f"{medicines_advice}"
    )

    return jsonify({'disease': response})

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)


