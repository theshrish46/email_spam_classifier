# app.py

import streamlit as st
import pickle
from utils.preprocess import clean_text

# Load model and vectorizer
model = pickle.load(open("model/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Streamlit UI
st.set_page_config(page_title="Email Spam Classifier", layout="centered")
st.title("📧 Email Spam Classifier")
st.write("Enter an email or message below to check if it's **Spam** or **Not Spam**.")

# Input
user_input = st.text_area("Your Email Text Here:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        # Preprocess
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        # Result
        if prediction == 1:
            st.error("🚨 This looks like **SPAM**!")
        else:
            st.success("✅ This looks like a **Legit Email**.")
