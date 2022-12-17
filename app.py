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
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import numpy as np

MODEL_PATH = r"model.h5"

max_words = 5000
max_len=50
EMBEDDING_DIM = 32
tokenizer_file = 'tokenizer.pkl'
wordnet = WordNetLemmatizer()
regex = re.compile('[%s]' % re.escape(string.punctuation))

with open(tokenizer_file,'rb') as handle:
   tokenizer = pickle.load(handle)

def text_cleaning(line_from_column):
    tokenized_doc = word_tokenize(line_from_column)
    new_review = []
    for token in tokenized_doc:
       new_token = regex.sub(u'', token)
       if not new_token == u'':
          new_review.append(new_token)
    new_term_vector = []
    for word in new_review:
      if not word in stopwords.words('english'):
        new_term_vector.append(word)
    final_doc = []
    for word in new_term_vector:
      final_doc.append(wordnet.lemmatize(word))
    return ''.join(final_doc)

@st.cache(allow_output_mutation=True)
def Load_model():
   model = load_model(MODEL_PATH)
#    model._make_predict_function()
   model.summary() # included to make it visible when model is reloaded
   session = K.get_session()
   return model, session

if __name__ == '__main__':
   st.title('Political Reddit Threads Sentiment Classification app')
   st.write('A simple sentiment analysis classification app')
#    st.info('Model and tokenizer loaded')
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
    data = pad_sequences(sequences, padding = 'post', maxlen =  max_len)
    # st.info(data)
    prediction = model.predict(data)

    prediction_prob_negative = prediction[0][0]
    prediction_prob_neutral = prediction[0][1]
    prediction_prob_positive= prediction[0][2]

    prediction_class = prediction.argmax(axis=-1)[0]
    print(prediction.argmax())
    st.header('Prediction using LSTM model')
    if prediction_class == -1:
      st.warning('Thread has negative sentiment')
    if prediction_class == 0:
      st.success('Thread has neutral sentiment')
    if prediction_class==1:
      st.success('Thread has positive sentiment')
