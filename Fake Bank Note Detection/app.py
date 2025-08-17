# Note: In-order to run it: Run it in command line by printing 'streamlit run app.py'

# Import necessary libraries
import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model

# To deal with oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

st.title('Fake Bank Note Detection')
st.markdown('---')

# Load the Keras model
try:
    model = load_model('model.h5')
except Exception as e:
    st.error(f"Error loading model.h5: {e}.")
    model = None

# Load the StandardScaler
try:
    with open('my_model.pkl', 'rb') as f:
        scaler = pickle.load(f)
    st.success("Scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading my_model.pkl: {e}.")
    scaler = None

# Store resources in dictionary
resources = {'model': model, 'scaler': scaler}

# Sidebar inputs
st.sidebar.header('Input Features')
# Here, float(-7), float(+7) gives the range and 0.01 is step slider.
vwti = st.sidebar.slider('Variance of Wavelet Transformed Image (VWTI)', float(-7), float(7), float(0.0), 0.01)
swti = st.sidebar.slider('Skewness of Wavelet Transformed Image (SWTI)', float(-14), float(14), float(0.0), 0.01)
cwti = st.sidebar.slider('Curtosis of Wavelet Transformed Image (CWTI)', float(-7), float(7), float(0.0), 0.01)
ei = st.sidebar.slider('Entropy of Image (EI)', float(-4), float(4), float(0.0), 0.01)

# Prediction button
if st.sidebar.button('Predict'):
    if resources['model'] is not None:
        try:
            # Prepare input
            input_data = np.array([[vwti, swti, cwti, ei]])

            # Apply scaler if available and valid
            if resources['scaler'] is not None and hasattr(resources['scaler'], "transform"):
                input_data = resources['scaler'].transform(input_data)
            prediction_proba = resources['model'].predict(input_data)[0][0]

            if prediction_proba > 0.5:
                st.error("This bank note is likely **FAKE**.")
            else:
                st.success("This bank note is likely **REAL**.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Model not loaded. Cannot make a prediction.")