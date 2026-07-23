import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("bank_model.pkl")
columns = joblib.load("columns.pkl")

st.title("Bank Deposit Prediction")

# User Inputs
age = st.number_input("Age", 18, 100, 30)
balance = st.number_input("Balance", value=1000)
day = st.number_input("Day", 1, 31, 1)
duration = st.number_input("Duration", value=100)
campaign = st.number_input("Campaign", value=1)
pdays = st.number_input("Pdays", value=-1)
previous = st.number_input("Previous", value=0)

if st.button("Predict"):

    data = pd.DataFrame(0, index=[0], columns=columns)

    data["age"] = age
    data["balance"] = balance
    data["day"] = day
    data["duration"] = duration
    data["campaign"] = campaign
    data["pdays"] = pdays
    data["previous"] = previous

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("Customer Will Subscribe: YES")
    else:
        st.error("Customer Will Subscribe: NO")