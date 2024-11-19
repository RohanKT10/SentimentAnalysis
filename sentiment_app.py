# sentiment_app.py

import streamlit as st
import joblib
import re

# Load the saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


# Function to clean and preprocess user input text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.lower().strip()


# Streamlit app layout
st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet below to analyze its sentiment as Positive or Negative.")

# Text area for user input
user_input = st.text_area("Tweet text")

# Analyze button
if st.button("Analyze Sentiment"):
    # Clean and transform the input text
    clean_input = clean_text(user_input)
    input_vector = vectorizer.transform([clean_input])

    # Predict sentiment
    prediction = model.predict(input_vector)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"

    # Display the result
    st.write(f"Sentiment: {sentiment}")
