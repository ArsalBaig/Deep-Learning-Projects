# Importing Lib's

import numpy as np
import pickle
from tensorflow.keras.models import load_model
import streamlit as st

# Loading Models.
model = load_model('battery_model.h5')

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)
# Declaring Battery types.
all_battery_types = ['Li-ion', 'NiMH', 'Lead Acid']
label_encoder.fit(all_battery_types)

# Prediction Function
def predict_battery_life(model, Capacity, Re, Rct, type_value):
    input_data = np.array([[Capacity, Re, Rct, type_value]])
    input_data_scaled = scaler.transform(input_data)
    pred = model.predict(input_data_scaled)
    return pred[0][0]

# Streamlit UI
st.title('Battery Life Prediction')
type = st.selectbox('Select Battery Type', options=all_battery_types, key='type')
Capacity = st.selectbox('Select Battery Capacity (Ah)', options=[10, 20, 30, 40, 50], key='capacity')
Re = st.number_input('Enter Internal Resistance (Ohm)', value=0.1, key='Re')
Rct = st.number_input('Enter Contact Resistance (Ohm)', value=0.1, key='Rct')

# Prediction Button.
if st.button('Predict Battery Life'):
    transformed_type = label_encoder.transform([type])[0]
    prediction = predict_battery_life(model, Capacity, Re, Rct, transformed_type)
    st.success(f'Predicted Battery Life: {prediction:.2f} years')