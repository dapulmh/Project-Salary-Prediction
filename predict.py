import numpy as np
import pandas as pd
import streamlit as st
import pickle
import joblib


with open('regresi_steps.pkl', 'rb') as file:
    data = pickle.load(file)

model = data["model"]
encoder = data["encoder"]


def show_page():
    st.title("Annual Salary Prediction (USD)")
    st.write('''## Predict your salary in here !''')
    st.write('''### Insert your data in here to predict your salary''')

    age = st.slider("Age:", min_value=17, max_value=100, value=17, step=1)
    gender_opt = {
        'Male',
        'Female'
    }
    education_level = {
        "Bachelor's",
        "Master's",
        'PhD',
        'High School'
    }
    job_opt = {
        'Software Engineer', 'Data Scientist', 'Software Engineer Manager',
        'Data Analyst', 'Senior Project Engineer', 'Product Manager',
        'Full Stack Engineer', 'Marketing Manager', 'Back end Developer',
        'Senior Software Engineer', 'Front end Developer',
        'Marketing Coordinator', 'Junior Sales Associate',
        'Financial Manager', 'Marketing Analyst', 'Software Developer',
        'Operations Manager', 'Human Resources Manager'
    }
    gender = st.selectbox("Gender:", gender_opt)
    edu = st.selectbox("Education level:", education_level)
    job = st.selectbox("Job:", job_opt)
    year_experience = st.slider(
        "Year of experience:", min_value=0.0, max_value=50.0, value=0.0, step=0.5)

    submit = st.button('Calculate your annual salary')
    if submit:
        user_data = {'Age': age, 'Gender': gender, 'Education Level': edu,
                     'Job Title': job, 'Years of Experience': year_experience}
        df_user = pd.DataFrame([user_data])

        df_encoded_user = pd.get_dummies(
            df_user, columns=['Gender', 'Job Title', 'Education Level'])

        expected_columns = [
            'Gender_Male', 'Gender_Female',
            'Education Level_Bachelor\'s', 'Education Level_Master\'s', 'Education Level_PhD', 'Education Level_High School',
            'Job Title_Software Engineer', 'Job Title_Data Scientist', 'Job Title_Software Engineer Manager',
            'Job Title_Data Analyst', 'Job Title_Senior Project Engineer', 'Job Title_Product Manager',
            'Job Title_Full Stack Engineer', 'Job Title_Marketing Manager', 'Job Title_Back end Developer',
            'Job Title_Senior Software Engineer', 'Job Title_Front end Developer',
            'Job Title_Marketing Coordinator', 'Job Title_Junior Sales Associate',
            'Job Title_Financial Manager', 'Job Title_Marketing Analyst', 'Job Title_Software Developer',
            'Job Title_Operations Manager', 'Job Title_Human Resources Manager'
        ]

        for col in set(expected_columns) - set(df_encoded_user.columns):
            df_encoded_user[col] = 0

        X = df_encoded_user.values.astype(float)
        pred_salary = model.predict(X)

        st.subheader(f"Your prediction annual salary is ${pred_salary[0]:.2f}")
