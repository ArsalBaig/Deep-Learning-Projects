import pickle
import os
import streamlit as st
import tensorflow as tf

from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from PIL import Image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image


features_list = pickle.load(open('images_features.pkl', 'rb'))
filenames_list = pickle.load(open('images_file.pkl', 'rb'))


# Model
ResNet50_model = ResNet50(weights= 'imagenet', include_top=False, input_shape=(224, 224, 3))
ResNet50_model.trainable = False
ResNet50_model = Sequential([ResNet50_model, GlobalMaxPooling2D()])

# Streamlit UI
st.set_page_config(page_title="Fashion Recommender System", layout= 'centered')
st.title('Fashion Recommender System.')
st.text('Find similar fashion products from your image.')

def save_file(uploaded_file):
    try:
        with open(os.path.join('Uploader', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:
        return 0

def extract_features(image_path, ResNet50):
    image = image.load_img(image_path, target_size=(224, 224))
    image_array = image.img_to_array(image)
    image_array = tf.expand_dims(image_array, axis= 0)
    processed_image = preprocess_input(image_array)
    features = ResNet50.predict(processed_image).flatten()
    normalized_features = features / norm(features)
    return normalized_features

def recommend(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm= 'brute', metric= 'euclidean')
    neighbors.fit(features_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

uploaded_file = st.file_uploader('Choose an image', type= ['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    if save_file(uploaded_file):
        show_image = Image.open(uploaded_file)
        resized_image = show_image.resize((400, 400))
        st.image(resized_image)
        features = extract_features(os.path.join('Uploader', uploaded_file.name), ResNet50_model)
        img_indices = recommend(features, features_list)
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.header('Recommended 1')
            st.image(filenames_list[img_indices[0][0]])
        with col2:
            st.header('Recommended 2')
            st.image(filenames_list[img_indices[0][1]])
        with col3:
            st.header('Recommended 3')
            st.image(filenames_list[img_indices[0][2]])
        with col4:
            st.header('Recommended 4')
            st.image(filenames_list[img_indices[0][3]])
        with col5:
            st.header('Recommended 5')
            st.image(filenames_list[img_indices[0][4]])
    else:
        st.error('Error in file upload. Try again.')