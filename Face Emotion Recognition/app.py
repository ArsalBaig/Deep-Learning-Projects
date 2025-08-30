import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.title("🎭 Face Emotion Recognition App")
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


@st.cache_resource
def load_model():
    try:
        model_path = r'C:\Users\RANA LAPTOP\Desktop\App\Facial_Emotion_Detection.h5'
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error! Unable to load the model. Please check the file path: {e}")
        st.stop()

model = load_model()

uploaded_file = st.file_uploader("Choose an image: ", type=["jpg", "jpeg", "png"])

if uploaded_file != None:
# Reads the image data from uploaded_file.
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("### Analyzing Image...")
    try:
        # Convert to grayscale and resize
        grayscale_image = image.convert("L").resize((48, 48)) # 'L' means luminance.
        img_array = np.array(grayscale_image).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = np.expand_dims(img_array, axis=-1) # Add channel dimension

        prediction = model.predict(img_array)
        
        predicted_index = np.argmax(prediction[0])
        predicted_emotion = emotion_classes[predicted_index]
        confidence = prediction[0][predicted_index] * 100

        st.success(f"**Predicted Emotion:** {predicted_emotion.capitalize()} **(Confidence: {confidence:.2f}%)**")
        
        # Create a dictionary for the bar chart
        chart_data = {
            'Emotion': emotion_classes,
            'Probability': prediction[0]
        }
        
        st.bar_chart(chart_data, x='Emotion', y='Probability')

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please try uploading a different image.")
        
st.info("Note: This model works best with clear, front-facing gray-scale images of faces.")
