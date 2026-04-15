import streamlit as st
#to run use-streamlit run app.py
import pickle
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Define feature names and their limits
feature_limits = {
    "mean radius": (6.0, 30.0),
    "mean texture": (9.0, 40.0),
    "mean perimeter": (40.0, 190.0),
    "mean area": (140.0, 2500.0),
    "area error": (6.0, 550.0),
    "worst texture": (10.0, 50.0),
    "worst perimeter": (50.0, 250.0),
    "worst area": (150.0, 3000.0)
}

st.title("Breast Cancer Prediction")
st.write("Enter the required features to predict if the tumor is benign or malignant.")

# Take user input
input_values = []
for feature, (min_val, max_val) in feature_limits.items():
    value = st.slider(f"{feature}", min_val, max_val, (min_val + max_val) / 2)
    input_values.append(value)

# Make prediction if user clicks the button
if st.button("Predict Cancer Type"):
    input_array = np.array(input_values).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    
    if prediction == 1:
        st.success("The tumor is **Benign** (Non-cancerous).")
    else:
        st.error("The tumor is **Malignant** (Cancerous).")
