import streamlit as st
import joblib
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Load model and vectorizer
model = joblib.load('sentimentanalysis_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function (should match your training code)
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stemming(text):
    stemmed_text = re.sub('[^a-zA-z]', ' ', text)
    stemmed_text = stemmed_text.lower()
    stemmed_text = stemmed_text.split()
    stemmed_text = [port_stem.stem(word) for word in stemmed_text if word not in stop_words]
    stemmed_text = ' '.join(stemmed_text)
    return stemmed_text

st.title("Sentiment Analyzer")

user_input = st.text_input("Enter a sentence:")

if user_input:
    processed = stemming(user_input)
    vect = vectorizer.transform([processed])
    prediction = model.predict(vect)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    st.write("Sentiment:", sentiment)