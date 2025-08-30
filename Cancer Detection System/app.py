import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model('Skin_Cancer_CNN.h5')
def predict_skin_cancer(image_path, img):
    img = image.load_img(image_path, target_size= (224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis= 0)

    prediction = model.predict(img_array)
    class_label = 'Maglignant' if prediction[0][0] > 0.5 else 'Benign'
    return class_label, img

st.title('Skin Cancer Detection.')

uploaded_img = st.file_uploader('Choose an image: ', type= ['jpg', 'jpeg', 'png'])
if uploaded_img != None:
    class_label, img = predict_skin_cancer(uploaded_img, model)
    if st.button('Predict'):
        st.write(f'Prediction: {class_label}')

    