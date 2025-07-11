import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("salary_model.pkl")
encoders = joblib.load("encoders.pkl")

st.title("💼 Employer Salary Prediction App")

# Collect input
age = st.number_input("Age", min_value=17, max_value=90)
education = st.selectbox("Education", encoders['education'].classes_)
occupation = st.selectbox("Occupation", encoders['occupation'].classes_)
hours = st.slider("Hours per week", 1, 99)
gender = st.selectbox("Gender", encoders['gender'].classes_)
marital_status = st.selectbox("Marital Status", encoders['marital-status'].classes_)
capital_gain = st.number_input("Capital Gain", min_value=0)
capital_loss = st.number_input("Capital Loss", min_value=0)

# Convert inputs using label encoders
education_encoded = encoders['education'].transform([education])[0]
occupation_encoded = encoders['occupation'].transform([occupation])[0]
gender_encoded = encoders['gender'].transform([gender])[0]
marital_encoded = encoders['marital-status'].transform([marital_status])[0]

# Prepare feature array (update based on your feature selection)
X_input = np.array([[age, education_encoded, occupation_encoded, hours, gender_encoded, marital_encoded,capital_gain,
    capital_loss]])

# Predict
prediction = model.predict(X_input)[0]

st.subheader("📊 Predicted Salary Class:")
st.success(">50K" if prediction == 1 else "≤50K")
