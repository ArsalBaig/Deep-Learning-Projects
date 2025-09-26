import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

# Streamlit UI
st.set_page_config(page_title="English to French Translator", page_icon="🌍", layout='centered')

st.title("🌍 English to French Translator")
st.write("Type in English, and I’ll translate it into French.")

@st.cache_resource
def load_model_and_tokenizer():
    # pretrained model for English to French translation from Hugging Face.
    model_name = "Helsinki-NLP/opus-mt-en-fr"
    tokenizer = AutoTokenizer.from_pretrained(model_name) # load tokenizer
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name) # load & translate model.
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# User input
english_text = st.text_area("Enter English text:")
if st.button("Translate"):
    if english_text.strip():
        
        inputs = tokenizer(english_text, return_tensors="tf", padding=True, truncation=True)
        
        outputs = model.generate(**inputs)
        
        # Decode the output tensor to get the French text
        french_text = tokenizer.decode(outputs[0], skip_special_tokens=True) # skip_special_tokens: Remove special elements.

        st.success("### French Translation:")
        st.write(french_text)
    else:
        st.warning("⚠️ Please enter some text before translating.")
