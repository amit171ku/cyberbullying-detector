
import streamlit as st
import pickle
import numpy as np

# Load model and vectorizer
model = pickle.load(open('cyberbullying_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("Cyberbullying Detection")

user_input = st.text_area("Enter a tweet for analysis:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a tweet!")
    else:
        vectorized_input = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_input)[0]

        if prediction == 1:
            st.error("⚠️ Cyberbullying detected!")
        else:
            st.success("✅ No cyberbullying detected.")
