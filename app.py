import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Advertising Sales Prediction",
    page_icon="📊",
    layout="centered"
)

# Load trained model
model = joblib.load("model/sales_model.pkl")

# Title
st.title("📊 Advertising Sales Prediction")

st.write(
    "Enter the advertising budget for TV, Radio, and Newspaper "
    "to predict the expected Sales."
)

# Input fields
tv = st.number_input(
    "TV Advertising Budget",
    min_value=0.0,
    value=100.0,
    step=0.1
)

radio = st.number_input(
    "Radio Advertising Budget",
    min_value=0.0,
    value=25.0,
    step=0.1
)

newspaper = st.number_input(
    "Newspaper Advertising Budget",
    min_value=0.0,
    value=25.0,
    step=0.1
)

# Input Summary
st.subheader("📋 Input Summary")

st.write(f"**TV Advertising:** {tv}")
st.write(f"**Radio Advertising:** {radio}")
st.write(f"**Newspaper Advertising:** {newspaper}")

# Prediction
if st.button("Predict Sales"):

    input_data = pd.DataFrame({
        "TV": [tv],
        "Radio": [radio],
        "Newspaper": [newspaper]
    })

    prediction = model.predict(input_data)
    predicted_sales = prediction[0]

    st.success(f"Predicted Sales: {predicted_sales:.2f}")

# About the Prediction
st.subheader("About the Prediction")

st.write(
    "The prediction is generated using a Linear Regression model "
    "trained on the advertising dataset."
)