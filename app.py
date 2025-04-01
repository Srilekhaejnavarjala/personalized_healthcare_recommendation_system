import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("C:/Users/91951/PycharmProjects/Personalized_Healthcare_System/healthcare_model.pkl")
scaler = joblib.load("C:/Users/91951/PycharmProjects/Personalized_Healthcare_System/scaler (2).pkl")

# Define feature names
feature_names = ["Recency", "Frequency", "Monetary", "Time"]

# Streamlit App UI
st.title("Personalized Healthcare Recommendation System")
st.write("Enter the patient details below to get a recommendation.")

# Input fields for user
recency = st.number_input("Recency (months since last donation)", min_value=0, value=2)
frequency = st.number_input("Frequency (total donations)", min_value=0, value=20)
monetary = st.number_input("Monetary (total volume donated)", min_value=0, value=5000)
time = st.number_input("Time (months since first donation)", min_value=0, value=45)

# Predict button
if st.button("Get Recommendation"):
    # Prepare input data
    input_data = pd.DataFrame([[recency, frequency, monetary, time]], columns=feature_names)
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Display recommendation
    if prediction == 1:
        st.success("Recommended for donation!")
    else:
        st.error("Not recommended for donation.")
