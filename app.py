import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.title('Gas Turbine Performance Predictor')

st.write("""
    Gas turbines operate by burning fuel—often natural gas—to produce high-pressure gas that drives a turbine to generate electricity.
    This helps industries plan maintenance and operations efficiently, saving costs and reducing environmental impact.
""")

st.write("""
    This model predicts gas turbine energy yield, carbon monoxide, and nitrogen oxides emissions based on input features.
    It can be used by industries to optimize fuel usage, plan maintenance schedules, and reduce environmental impact.
""")

# Input fields
AT = st.number_input('Ambient Temperature (AT) C [range : 0.28 - 34.92 ]', step=0.01)
AP = st.number_input('Ambient Pressure (AP) mbar [range : 985 - 1035 ]' , step=0.01)
AH = st.number_input('Ambient Humidity (AH) % [range : 27.5 - 100.2 ]', step=0.01)
AFDP = st.number_input('Air filter difference pressure (AFDP) mbar [range : 2.08 - 7.61 ]', step=0.01)
GTEP = st.number_input('Gas turbine exhaust pressure (GTEP) mbar [range : 17.87 - 37.40 ]', step=0.01)
TIT = st.number_input('Turbine inlet temperature (TIT) C [range : 1000 - 1100 ]', step=0.01)
TAT = st.number_input('Turbine after temperature (TAT) C [range : 512.45 - 550.61 ]', step=0.01)
CDP = st.number_input('Compressor discharge pressure (CDP) mbar [range : 9.87 - 15.08 ]', step=0.01)

# Predict button
if st.button('Predict'):
    data = CustomData(AT=AT, AP=AP, AH=AH, AFDP=AFDP, GTEP=GTEP, TIT=TIT, TAT=TAT, CDP=CDP)
    pred_df = data.get_data_as_data_frame()
    
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    
    # Output
    st.write('### Results')
    st.write(f'Energy Yield (TEY): {results[0,0]} MWH')
    st.write(f'CO Emission: {results[0,1]} mg/m3')
    st.write(f'NOX Emission: {results[0,2]} mg/m3')
