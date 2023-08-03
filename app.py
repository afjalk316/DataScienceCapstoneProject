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

# Checkboxes for fuel types
fuel_options = df_encoded.filter(regex=r'^fuel_', axis=1).columns
fuel_selected = st.multiselect("Select Fuel Types", fuel_options, default=fuel_options)

# Checkboxes for seller types
seller_options = df_encoded.filter(regex=r'^seller_type_', axis=1).columns
seller_selected = st.multiselect("Select Seller Types", seller_options, default=seller_options)

# Checkboxes for transmission types
transmission_options = df_encoded.filter(regex=r'^transmission_', axis=1).columns
transmission_selected = st.multiselect("Select Transmission Types", transmission_options, default=transmission_options)

# Checkboxes for owner types
owner_options = df_encoded.filter(regex=r'^owner_', axis=1).columns
owner_selected = st.multiselect("Select Owner Types", owner_options, default=owner_options)

if st.button("Predict"):
    # Create a DataFrame with selected features
    input_data = pd.DataFrame({
        "year": [year],
        "km_driven": [km_driven]
    })
    
    # Set selected features to 1 in the DataFrame
    for fuel in fuel_selected:
        input_data[fuel] = 1
    for seller in seller_selected:
        input_data[seller] = 1
    for transmission in transmission_selected:
        input_data[transmission] = 1
    for owner in owner_selected:
        input_data[owner] = 1
    
    # Scale the input data using the same MinMaxScaler
    input_data[["year", "km_driven"]] = scaler.transform(input_data[["year", "km_driven"]])
    
    # Make prediction using the loaded model
    prediction = loaded_model.predict(input_data)
    st.write(f"Predicted Selling Price: {prediction[0]:.2f} Lakh(s)")
