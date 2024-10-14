import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved Random Forest model
model = joblib.load('loan_eligibility_model.pkl')

# Define a function for making predictions
def predict_loan_eligibility(features):
    features = np.array(features).reshape(1, -1)  # Ensure the correct shape
    prediction = model.predict(features)
    return 'Eligible' if prediction == 1 else 'Not Eligible'

# Streamlit frontend
st.title('Loan Eligibility Prediction')

# User inputs for features
gender = st.selectbox('Gender', ['Male', 'Female'])
married = st.selectbox('Married', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
applicant_income = st.number_input('Applicant Income', min_value=0.0)
coapplicant_income = st.number_input('Coapplicant Income', min_value=0.0)
loan_amount = st.number_input('Loan Amount', min_value=0.0)
loan_amount_term = st.number_input('Loan Amount Term', min_value=0.0)
credit_history = st.selectbox('Credit History', ['Good', 'Bad'])
property_area = st.selectbox('Property Area', ['Urban', 'Rural', 'Semiurban'])

# Convert user inputs into model inputs
gender = 1 if gender == 'Male' else 0
married = 1 if married == 'Yes' else 0
education = 1 if education == 'Graduate' else 0
self_employed = 1 if self_employed == 'Yes' else 0
credit_history = 1 if credit_history == 'Good' else 0

# Manually encode 'Dependents' as one-hot encoding
dependents_1 = 1 if dependents == '1' else 0
dependents_2 = 1 if dependents == '2' else 0
dependents_3plus = 1 if dependents == '3+' else 0

# Manually encode 'Property_Area' as one-hot encoding
property_area_rural = 1 if property_area == 'Rural' else 0
property_area_semiurban = 1 if property_area == 'Semiurban' else 0

# Prepare the features for prediction
user_input = [
    gender, married, education, self_employed, applicant_income, 
    coapplicant_income, loan_amount, loan_amount_term, credit_history,
    dependents_1, dependents_2, dependents_3plus, property_area_rural, 
    property_area_semiurban
]

# Predict loan eligibility
if st.button('Predict'):
    result = predict_loan_eligibility(user_input)
    st.write(f'The applicant is **{result}** for the loan.')