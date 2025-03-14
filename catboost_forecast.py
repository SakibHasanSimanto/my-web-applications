import pickle
import shap
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from catboost import CatBoostRegressor  # Or whichever model type you're using

# Load the model
model = CatBoostRegressor()  # Initialize the model (adjust as necessary)
model.load_model('AQI_Forecaster.cbm')

# SHAP explainer
explainer = shap.TreeExplainer(model)

# Streamlit app
st.title('AQI Forecasting System')

if st.button('About'):
    st.write('This system predicts the AQI in your place after 6 hours.')

st.write('Enter your input in the following fields')

time = st.number_input('Time: (e.g. 0, 1, 2, ... 23)')
pm_twofive = st.number_input('PM2.5 in micrograms per cubic meter:')
pm_ten = st.number_input('PM10 in micrograms per cubic meter:')
co = st.number_input('CO in micrograms per cubic meter:')
so2 = st.number_input('SO2 in micrograms per cubic meter:')
no2 = st.number_input('NO2 in micrograms per cubic meter:')
o3 = st.number_input('O3 in micrograms per cubic meter:')
aqi = st.number_input('Current AQI:')
input_list = [time, pm_twofive, pm_ten, co, so2, no2, o3, aqi]

if st.button('Forecast'):
    try:
        prediction = model.predict([input_list])
        st.success(f'AQI after 6 hours: {prediction[0]}')

    except Exception as e:
        st.error(f"Error: {e}")

if st.button('XAI Interpretation'):
    try:
        input_df = pd.DataFrame([input_list], columns=['time', 'pmtwo', 'pmten', 'co', 'so2', 'no2', 'o3', 'aqi'])
        shap_values = explainer.shap_values(input_df)

        # Feature Importance Plot (Summary Plot)
        st.subheader("Feature Importance (Summary Plot)")
        shap.summary_plot(shap_values.values, input_df)
        st.pyplot(plt)

        # Bar plot
        st.subheader("Feature Impact (Bar Chart)")
        shap.summary_plot(shap_values.values, input_df, plot_type="bar")
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Error in SHAP interpretation: {e}")
