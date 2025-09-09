import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import librosa
import os
import io

MODEL_PATH = 'Audio_sentiment.h5'
CLASS_LABELS = ['positive', 'negative', 'neutral']
MFCC_COUNT = 40

@st.cache_resource # Keep the model in memory to be reused when re-run.
def get_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f'Error loading model: {e}')


def extract_mfcc_features(file_name):
    try:
        # Load directly from in-memory buffer
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=MFCC_COUNT)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

def main():
    st.set_page_config(page_title="Audio Sentiment Analysis", layout="centered")
    st.title("🎤 Audio Sentiment Analysis App")

    if not os.path.exists(MODEL_PATH):
        st.error(f'Model file not found at `{MODEL_PATH}`.')

    model = get_model()
    if model:
        st.success("Model loaded successfully!")
        
        uploaded_file = st.file_uploader(
            'Choose an audio file:', type=['wav', 'mp3'])
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav') # Display audio widget.

            if st.button('Analyze Sentiment'):
                with st.spinner('Analyzing...'): # io.BytesIO, so it behaves like a file.
                    features = extract_mfcc_features(io.BytesIO(uploaded_file.getbuffer()))
                    if features is not None:
                        try:
                            features = features.reshape(1, MFCC_COUNT, 1)
                            
                            prediction = model.predict(features)
                            predicted_index = np.argmax(prediction, axis=1)[0]
                            predicted_label = CLASS_LABELS[predicted_index]
                            
                            st.success('Analysis Complete!')
                            st.markdown(f"### Predicted Sentiment: **{predicted_label.capitalize()}**")
                        except Exception as e:
                            st.error(f"An error occurred during prediction: {e}")

if __name__ == '__main__': # Run the app only but not in-case of importing file.
    main()