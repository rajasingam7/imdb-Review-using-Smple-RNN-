# Step 1: Import libraries and Load the Model
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import numpy as np


#Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

#load the pre trained model with ReLU activation
model = tf.keras.models.load_model('simplernn_imdb_model.h5')

# helper functions
# function to decode reviews back to words
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
# function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded = [word_index.get(word, 2) + 3 for word in words]  # 2 is for unknown words
    padded = sequence.pad_sequences([encoded], maxlen=500)
    return padded

#prediction function
def predict_sentiment(review):
    processed_review = preprocess_text(review)
    prediction = model.predict(processed_review)
    sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"
    return sentiment, prediction[0][0]


##Streamlit app
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive/Negative).")
user_review = st.text_area("Enter your movie review here:")

## Make prediction on button click
if st.button('Classify'):
    preprocessed_input = preprocess_text(user_review)

    #make Preduction
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"

    ## Display the result
    st.write(f"Predicted Sentiment: **{sentiment}** (Score: {prediction[0][0]:.4f})")
else:
    st.write("Please enter a review and click 'Classify' to see the sentiment prediction.")






