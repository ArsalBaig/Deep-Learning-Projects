import pickle
import os
import numpy as np
import re
import html
import nltk
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Disable OneDNN optimization warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


model = load_model('sentiment_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


max_len = 50
nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text cleaning
def clean_review(review):
    review = html.unescape(review)
    review = re.sub(r'^\w\s', '', review)
    review = review.lower()
    review = re.sub(r'[^a-zA-Z0-9\s]', '', review)
    words = review.split()
    clean_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(clean_words)

# Prediction
def predict_sentiment(review):
    text = clean_review(review)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    prediction = model.predict(padded)
    if prediction < 0.5:
        return "The movie review is: Negative! 😞"
    else:
        return "The movie review is: Positive! 😊"

# Streamlit layout
st.set_page_config(page_title="Movie Review Sentiment Analysis", layout="centered")

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict if it is **Positive** or **Negative**.")

user_input = st.text_area("✍️ Write your review here:")

if st.button("🔍 Predict Sentiment"):
    if user_input.strip() != "":
        sentiment = predict_sentiment(user_input)
        st.success(sentiment)
    else:
        st.warning("⚠️ Please enter a review before predicting.")
