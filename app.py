import streamlit as st
import pandas as pd
import joblib

# Load the ML model
loaded_model = joblib.load("best_model.pkl")

# Create the Streamlit app
st.title("Car Selling Price Prediction")

# Sidebar
st.sidebar.header("User Input Features")
# Create input fields for user input
year = st.sidebar.slider("Year", 2000, 2023, 2015)
km_driven = st.sidebar.slider("Kilometers Driven", 0, 500000, 50000)
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.sidebar.selectbox("Seller Type", ["Individual", "Dealer"])
transmission_type = st.sidebar.selectbox("Transmission Type", ["Manual", "Automatic"])
owner_type = st.sidebar.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])

# Convert user input to DataFrame
user_input = pd.DataFrame({
    "year": [year],
    "km_driven": [km_driven],
    "fuel_" + fuel_type: [1],
    "seller_type_" + seller_type: [1],
    "transmission_" + transmission_type: [1],
    "owner_" + owner_type: [1]
})

# Make predictions
prediction = loaded_model.predict(user_input)

# Show the prediction
st.subheader("Car Selling Price Prediction")
st.write(f"The estimated selling price of the car is Rs. {prediction[0]:.2f} Lakhs.")
