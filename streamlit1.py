#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 21:27:34 2024

@author: kelseymoorhead
"""

import streamlit as st
import pickle
import numpy as np

# Explicit paths for the model, scaler, and features
model_path = '/Users/kelseymoorhead/Desktop/streamlit/best_model.pkl'
scaler_path = '/Users/kelseymoorhead/Desktop/streamlit/scaler.pkl'
features_path = '/Users/kelseymoorhead/Desktop/streamlit/features.pkl'

# Load the model
try:
    model = pickle.load(open(model_path, 'rb'))
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Model file not found at {model_path}. Please ensure the file exists and retry.")
    st.stop()

# Load the scaler
try:
    scaler = pickle.load(open(scaler_path, 'rb'))
    st.success("Scaler loaded successfully!")
except FileNotFoundError:
    st.error(f"Scaler file not found at {scaler_path}. Please ensure the file exists and retry.")
    st.stop()

# Load the feature names
try:
    features = pickle.load(open(features_path, 'rb'))
    st.success("Feature list loaded successfully!")
except FileNotFoundError:
    st.error(f"Feature file not found at {features_path}. Please ensure the file exists and retry.")
    st.stop()

# Title and user input form
st.title("Chronic Kidney Disease Prediction")
st.write("Enter the patient details below to predict CKD:")

# Collect user inputs for all features
user_inputs = {}
for feature in features:
    if feature == 'Age':
        user_inputs[feature] = st.number_input(feature, min_value=0, max_value=120, step=1)
    elif feature == 'Blood Pressure':
        user_inputs[feature] = st.number_input(feature, min_value=50, max_value=200, step=1)
    elif feature == 'Specific Gravity':
        user_inputs[feature] = st.selectbox(feature, options=[1.005, 1.010, 1.015, 1.020, 1.025])
    elif feature == 'Albumin':
        user_inputs[feature] = st.slider(feature, min_value=0, max_value=5, step=1)
    elif feature == 'Sugar':
        user_inputs[feature] = st.slider(feature, min_value=0, max_value=5, step=1)
    else:
        # Adjust inputs for other features as needed
        user_inputs[feature] = st.number_input(feature, value=0.0)

# Prepare the data for prediction
if st.button("Predict"):
    try:
        # Combine user inputs into a single feature array in the correct order
        input_data = np.array([[user_inputs[feature] for feature in features]])

        # Scale the inputs
        input_data_scaled = scaler.transform(input_data)

        # Predict using the model
        prediction = model.predict(input_data_scaled)
        probability = model.predict_proba(input_data_scaled)[0][1]

        # Display the results
        st.subheader("Prediction Results")
        st.write("Chronic Kidney Disease Detected!" if prediction[0] == 1 else "No Chronic Kidney Disease Detected.")
        st.write(f"Confidence: {probability:.2%}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
