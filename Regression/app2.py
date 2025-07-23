import pandas as pd
import streamlit as st
import joblib
import numpy as np

# Load the model
with open('models/kc_house.pkl', 'rb') as file:
    model = joblib.load(file)

# Streamlit setup
st.set_page_config(page_title="House Price Predictor")
st.title("House Price Predict App")
st.subheader("Predict your house price")

# Sidebar inputs
st.sidebar.header("Enter your details 👍")
sqft_living = st.sidebar.slider("Square Footage of Living Space", 0.0, 10000.0, step=100.0)
sqft_lot = st.sidebar.slider("Square Footage of Lot", 0.0, 10000.0, step=100.0)
bedrooms = st.sidebar.slider("Number of Bedrooms", 0, 10, step=1)
bathrooms = st.sidebar.slider("Number of Bathrooms", 0.0, 10.0, step=0.5)
grade = st.sidebar.slider("House Grade", 1, 13, step=1)
sqft_above = st.sidebar.slider("Square Footage Above Ground", 0.0, 10000.0, step=100.0)
sqft_basement = st.sidebar.slider("Square Footage of Basement", 0.0, 10000.0, step=100.0)
yr_built = st.sidebar.slider("Year Built", 1900, 2023, step=1)
yr_renovated = st.sidebar.slider("Year Renovated", 0, 2023, step=1)
sqft_living15 = st.sidebar.slider("Square Footage of Living Space (15)", 0.0, 10000.0, step=100.0)
sqft_lot15 = st.sidebar.slider("Square Footage of Lot (15)", 0.0, 10000.0, step=100.0)

# Predict button
if st.sidebar.button("Predict House Price"):
    # Set default values for other required features
    floors = 1.0
    waterfront = 0
    view = 0
    condition = 3
    zipcode = 98103
    lat = 47.5
    long = -122.2
    id = 0
    date = 20141209000000

    # Define correct column names from training
    column_names = [
        'sqft_living', 'bedrooms', 'bathrooms', 'sqft_lot', 'floors',
        'waterfront', 'view', 'condition', 'grade', 'sqft_above',
        'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat',
        'long', 'sqft_living15', 'sqft_lot15', 'id', 'date'
    ]

    # Create values list (1 row of inputs)
    values = [[
        sqft_living, bedrooms, bathrooms, sqft_lot, floors, waterfront,
        view, condition, grade, sqft_above, sqft_basement, yr_built,
        yr_renovated, zipcode, lat, long, sqft_living15, sqft_lot15,
        id, date
    ]]

    # Create input DataFrame
    input_df = pd.DataFrame(values, columns=column_names)

    # Predict house price
    house_price = model.predict(input_df)[0]

    # Display result
    st.success(f"Predicted House Price: ₹{house_price:,.2f}")
    st.info("This prediction is based on a Multiple Linear Regression model.")

# Footer
st.markdown('-------')
st.markdown("Made with  using Streamlit")
