# Sentiment Analysis Web App

This project is a sentiment analysis web application built with Python, Streamlit, and scikit-learn. It allows users to enter a sentence and predicts whether the sentiment is positive or negative.

## Features

- Text preprocessing (cleaning, stemming, stopword removal)
- TF-IDF vectorization
- Logistic Regression model for sentiment prediction
- Simple web interface using Streamlit

## Setup Instructions

1. **Clone the repository**
   ```sh
   git clone https://github.com/YOUR_USERNAME/sentiment-analysis-app.git
   cd sentiment-analysis-app
   ```

2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

3. **Download NLTK stopwords**
   ```python
   import nltk
   nltk.download('stopwords')
   ```

4. **Train the model (if not already trained)**
   ```sh
   python sentimentanalysis.py
   ```

5. **Run the Streamlit app**
   ```sh
   streamlit run sentimentapp.py
   ```

6. **Open your browser**
   - Go to [http://localhost:8501](http://localhost:8501)

## File Structure

- `sentimentanalysis.py` - Script for training and saving the model/vectorizer
- `sentimentapp.py` - Streamlit web app for sentiment prediction
- `sentimentanalysis_model.pkl` - Saved trained model
- `tfidf_vectorizer.pkl` - Saved TF-IDF vectorizer

## Requirements

- Python 3.x
- streamlit
- scikit-learn
- pandas
- numpy
- nltk
- joblib

## License

This project is for educational purposes.
