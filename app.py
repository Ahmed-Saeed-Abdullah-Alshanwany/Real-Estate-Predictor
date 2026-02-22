import streamlit as st
import pandas as pd
import joblib

# Load the trained model and expected columns
model = joblib.load('model.pkl')
model_columns = joblib.load('columns.pkl')

# Setup the Web App UI
st.set_page_config(page_title="Egypt Real Estate Predictor", page_icon="🏠")

st.title("🏠 Smart Real Estate Price Predictor")
st.markdown("Enter the apartment details below to get an estimated price in **EGP**.")
st.write("---")

# Create input fields for the user
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (in square meters)", min_value=50, max_value=500, value=120, step=10)
    bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=6, value=3)

with col2:
    location = st.selectbox("Location", ['Nasr City', 'Maadi', 'Zayed', 'New Cairo'])
    finishing = st.selectbox("Finishing Type", ['Without Finishing', 'Super Lux', 'Extra Super Lux'])

st.write("---")

# Prediction Logic
if st.button("Predict Price 💰"):
    # Map user inputs to the exact format the model was trained on
    input_data = {
        'Area': area,
        'Bedrooms': bedrooms,
        'Location_Maadi': 1 if location == 'Maadi' else 0,
        'Location_Nasr City': 1 if location == 'Nasr City' else 0,
        'Location_New Cairo': 1 if location == 'New Cairo' else 0,
        'Location_Zayed': 1 if location == 'Zayed' else 0,
        'Finishing_Extra Super Lux': 1 if finishing == 'Extra Super Lux' else 0,
        'Finishing_Super Lux': 1 if finishing == 'Super Lux' else 0,
        'Finishing_Without Finishing': 1 if finishing == 'Without Finishing' else 0
    }
    
    # Convert to DataFrame and align columns
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Display the result
    st.success(f"### 🏷️ Estimated Price: {prediction:,.0f} EGP")
    st.balloons()