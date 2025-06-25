import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open('cyberbullying_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("🛡️ Cyberbullying Detection App")

# Text input
user_input = st.text_area("Enter a tweet for analysis:")

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a tweet!")
    else:
        # Vectorize user input
        vectorized_input = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_input)[0]

        # Debug line (optional - show raw prediction)
        st.write("🔍 Raw Prediction:", prediction)

        # Handle prediction
        if prediction in [1, 'cyberbullying', 'hate_speech', 'offensive_language']:  # adjust if using strings
            st.error("🚫 Cyberbullying Detected!")
        else:
            st.success("✅ No Cyberbullying Detected.")
