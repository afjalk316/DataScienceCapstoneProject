import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load the original dataset from the CSV file
df = pd.read_csv("CAR DETAILS.csv")

# Drop the "name" column as it contains non-numeric values and is not required for modeling
df.drop("name", axis=1, inplace=True)

# One-Hot Encoding for Categorical Variables
df_encoded = pd.get_dummies(df, columns=["fuel", "seller_type", "transmission", "owner"], drop_first=True)

# Feature Scaling (MinMax Scaling)
scaler = MinMaxScaler()
df_encoded[["year", "km_driven"]] = scaler.fit_transform(df_encoded[["year", "km_driven"]])

# Separate the Features and Target
X = df_encoded.drop("selling_price", axis=1)  # Features (excluding "selling_price")
y = df_encoded["selling_price"]  # Target variable ("selling_price")

# Load the saved best model
loaded_model = joblib.load("best_model.pkl")

# Streamlit App
st.title("Car Price Prediction")
st.write("Enter car details to predict the selling price:")

year = st.slider("Year", min_value=int(X["year"].min()), max_value=int(X["year"].max()), value=int(X["year"].mean()))
km_driven = st.slider("Kilometers Driven", min_value=int(X["km_driven"].min()), max_value=int(X["km_driven"].max()), value=int(X["km_driven"].mean()))
fuel_petrol = st.checkbox("Petrol")
fuel_diesel = st.checkbox("Diesel")
seller_type_individual = st.checkbox("Individual Seller Type")
seller_type_dealer = st.checkbox("Dealer Seller Type")
transmission_manual = st.checkbox("Manual Transmission")
transmission_automatic = st.checkbox("Automatic Transmission")
owner_first = st.checkbox("First Owner")
owner_second = st.checkbox("Second Owner")
owner_third = st.checkbox("Third Owner")

if st.button("Predict"):
    # Ensure the correct feature names are used for prediction
    fuel_Petrol = 1 if fuel_petrol else 0
    fuel_Diesel = 1 if fuel_diesel else 0
    seller_type_Individual = 1 if seller_type_individual else 0
    transmission_Manual = 1 if transmission_manual else 0
    owner_Second = 1 if owner_second else 0
    
    input_data = pd.DataFrame({
        "year": [year],
        "km_driven": [km_driven],
        "fuel_Diesel": [fuel_Diesel],
        "fuel_Petrol": [fuel_Petrol],
        "seller_type_Individual": [seller_type_Individual],
        "transmission_Manual": [transmission_Manual],
        "owner_Second": [owner_Second]
    })
    
    # Scale the input data using the same MinMaxScaler
    input_data[["year", "km_driven"]] = scaler.transform(input_data[["year", "km_driven"]])
    
    # Make prediction using the loaded model
    prediction = loaded_model.predict(input_data)
    st.write(f"Predicted Selling Price: {prediction[0]:.2f} Lakh(s)")
