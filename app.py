from keras import backend as K
from tensorflow.keras.models import Model, load_model
import streamlit as st
import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import string
from textblob import Word
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import numpy as np

MODEL_PATH = r"model_LSTM.h5"

max_words = 500
max_len=500
EMBEDDING_DIM = 32
tokenizer_file = 'tokenizer_LSTM.pkl'
wordnet = WordNetLemmatizer()
regex = re.compile('[%s]' % re.escape(string.punctuation))

with open(tokenizer_file,'rb') as handle:
    tokenizer = pickle.load(handle)

def text_cleaning(line_from_column):
    text = line_from_column.lower()
    # Replacing the digits/numbers
    text = text.replace('d', '')
    # remove stopwords
    words = [w for w in text if w not in stopwords.words("english")]
    # apply stemming
    words = [Word(w).lemmatize() for w in words]
    # merge words 
    words = ' '.join(words)
    return text

@st.cache(allow_output_mutation=True)
def Load_model():
    model = load_model(MODEL_PATH)
    model.summary() # included to make it visible when model is reloaded
    session = K.get_session()
    return model, session

if __name__ == '__main__':
    st.title('Political Reddit Threads Sentiment Classification app')
    st.write('A simple sentiment analysis classification app')
    st.subheader('Input the Reddit Thread below')
    sentence = st.text_area('Enter your thread here',height=200)
    predict_btt = st.button('predict')
    model, session = Load_model()
    if predict_btt:
        clean_text = []
        K.set_session(session)
        i = text_cleaning(sentence)
        clean_text.append(i)
        sequences = tokenizer.texts_to_sequences(clean_text)
        data = pad_sequences(sequences, maxlen =  max_len)
        # st.info(data)
        prediction = model.predict(data)

        prediction_prob_negative = prediction[0][0]
        prediction_prob_neutral = prediction[0][1]
        prediction_prob_positive= prediction[0][2]

        prediction_class = prediction.argmax(axis=-1)[0]
        print(prediction.argmax())
        st.header('Prediction using LSTM model')
        if prediction_class == 0:
          st.warning('Thread has negative sentiment')
        if prediction_class == 1:
          st.success('Thread has neutral sentiment')
        if prediction_class==2:
          st.success('Thread has positive sentiment')
