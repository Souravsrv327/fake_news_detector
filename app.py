
import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="🕵️", layout="centered")

st.title("📰 Fake News Detector")
st.write("Enter a news headline or article text to check if it's REAL or FAKE.")

user_input = st.text_area("Paste the news content here:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        if prediction == "FAKE":
            st.error("🚨 This news appears to be FAKE!")
        else:
            st.success("✅ This news appears to be REAL.")
