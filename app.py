import streamlit as st
import joblib

# Load model and vectorizer correctly with joblib
model = joblib.load('cyberbullying_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("🛡️ Cyberbullying Detection App")

user_input = st.text_area("Enter a tweet for analysis:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a tweet!")
    else:
        vectorized_input = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_input)[0]

        if prediction in [1, 'cyberbullying', 'hate_speech', 'offensive_language']:
            st.error("🚫 Cyberbullying Detected!")
        else:
            st.success("✅ No Cyberbullying Detected.")
