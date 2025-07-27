# import streamlit as st
# def predict_sentiment(text):
#     # Dummy implementation, replace with your model
#     if "good" in text.lower():
#         return "Positive"
#     else:
#         return "Negative"

# st.title("Sentiment Analyzer")
# user_input = st.text_input("Enter a sentence:")
# if user_input:
#     result = predict_sentiment(user_input)
#     st.write("Sentiment:", result)


#dowloading all the required librabries
import numpy as np #for numerical operations
import pandas as pd # for data manipulation and analysis
import re # for regular expressions remving the punctuations, links and symbols
#nltk for proceesing words performing stemming ,stopwords removal tokenization
from nltk.corpus import stopwords# from corpus of word librbary we are importing stopwords
from nltk.stem import PorterStemmer# for performing stemming 
from sklearn.feature_extraction.text import TfidfVectorizer # convert words->vectors
from sklearn.model_selection import train_test_split # for splitting the data into train and test
from sklearn.linear_model import LogisticRegression#building model
from sklearn.metrics import accuracy_score


#dowloading the stopwords
import nltk
nltk.download('stopwords')

#data preprocessing loading the dataset
columns=['target','id','date','flag','user','text']#1st row is read as header so we remove it by suing  columns=['target','id','date','flag','user','text']
dataset=pd.read_csv('twitterdata.csv',names=columns ,encoding='ISO-8859-1')
print(dataset.shape)
print(dataset.head(5))

print(dataset.isnull().sum()) #checking for null values
print(dataset['target'].value_counts()) #checking distribution of data should be even for good model 
dataset.replace({'target':{4:1}},inplace=True)
print(dataset['target'].value_counts()) #replacing 4 with 1 as 4 is positive sentiment and 0 is negative sentiment

#stemming the values
port_stem=PorterStemmer()
stop_words=set(stopwords.words('english'))
def stemming(text):
    stemmed_text=re.sub('[^a-zA-z]',' ',text)
    stemmed_text=stemmed_text.lower()
    stemmed_text=stemmed_text.split();
    stemmed_text=[port_stem.stem(text) for text in stemmed_text if text not in stop_words]
    stemmed_text=' '.join(stemmed_text)
    return stemmed_text

dataset['Stemmed_text']=dataset['text'].apply(stemming) #applying stemming function to the text column
print(dataset.head())
print(dataset['Stemmed_text'])
print(dataset['target'])

#defininf the independent and dependent variables
X=dataset['Stemmed_text'].values
Y=dataset['target'].values
print(X)
print(Y)

#splitting data into train and test
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,stratify=Y,random_state=13)
print(X_train.shape,X_test.shape)


#converting textual data into numerical data
vectorizer=TfidfVectorizer()
X_train=vectorizer.fit_transform(X_train)
X_test=vectorizer.transform(X_test)
print(X_train)
print(X_test)

#training the model using logistic regression
model=LogisticRegression()
model.fit(X_train,Y_train)


#evaluation
X_train_prediction=model.predict(X_train)
X_train_accuracy=accuracy_score(Y_train,X_train_prediction)
print('Accuracy on training data:',X_train_accuracy)

X_test_prediction=model.predict(X_test)
X_test_accuracy=accuracy_score(Y_test,X_test_prediction)
print("Accuracy on test data:",X_test_accuracy)


#saving the model
import joblib

# Save model and vectorizer
joblib.dump(model, 'sentimentanalysis_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')