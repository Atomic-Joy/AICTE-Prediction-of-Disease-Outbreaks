import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained models
def load_model(model_path):
    with open(model_path, "rb") as f:
        model, scaler = pickle.load(f)
    return model, scaler

parkinsons_model, parkinsons_scaler = load_model("../Models/parkinsons_model.pkl")
heart_model, heart_scaler = load_model("../Models/heart_model.pkl")
diabetes_model, diabetes_scaler = load_model("../Models/diabetes_model.pkl")

# Title
st.title("Disease Prediction App")

# Model selection
model_option = st.selectbox("Choose a model for prediction:", ["Parkinson's Disease", "Heart Disease", "Diabetes"])

# User input form
st.subheader("Enter Patient Data:")
input_data = []

if model_option == "Parkinson's Disease":
    features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 
                'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 
                'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
elif model_option == "Heart Disease":
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
elif model_option == "Diabetes":
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

for feature in features:
    value = st.number_input(f"Enter {feature}", value=0.0)
    input_data.append(value)

# Predict
if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    if model_option == "Parkinson's Disease":
        scaled_input = parkinsons_scaler.transform(input_array)
        prediction = parkinsons_model.predict(scaled_input)
    elif model_option == "Heart Disease":
        scaled_input = heart_scaler.transform(input_array)
        prediction = heart_model.predict(scaled_input)
    else:
        scaled_input = diabetes_scaler.transform(input_array)
        prediction = diabetes_model.predict(scaled_input)

    st.subheader("Prediction Result:")
    st.write("Positive" if prediction[0] == 1 else "Negative")
