import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Dataset with one disease per symptom and the rest set to 0
data = {
    'fever': [1, 0, 0, 0, 0, 0],
    'cough': [0, 1, 0, 0, 0, 0],
    'headache': [0, 0, 1, 0, 0, 0],
    'sore_throat': [0, 0, 0, 1, 0, 0],
    'fatigue': [0, 0, 0, 0, 1, 0],
    'body_ache': [0, 0, 0, 0, 0, 1],
    'disease': [
        'flu',               # Fever only
        'cold',              # Cough only
        'migraine',          # Headache only
        'allergy',           # Sore throat only
        'general_tiredness', # Fatigue only
        'muscle_strain'      # Body ache only
    ]
}

# Convert the dataset to a DataFrame
df = pd.DataFrame(data)

# Features and labels
X = df[['fever', 'cough', 'headache', 'sore_throat', 'fatigue', 'body_ache']]
y = df['disease']

# Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained with mild conditions and saved as model.pkl")
