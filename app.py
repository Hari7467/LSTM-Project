## step 1: import libraries and load the model
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

#Load the tokenizer pickle file
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

#Load the pre trained lstm model 
model=load_model('lstm_model.h5')
model.summary()

# step:2 helper functions
#Predict the text
def predict_next_word(model,tokenizer,text,max_sequence_len):
  token_list=tokenizer.texts_to_sequences([text])[0]
  if len(token_list) >= max_sequence_len:
    token_list=token_list[-(max_sequence_len-1):]
  token_list=pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
  predicted=model.predict(token_list,verbose=0)
  predicted_word_index=np.argmax(predicted,axis=1)
  for word,index in tokenizer.word_index.items():
    if index==predicted_word_index:
      return word
  return None
  

#user Input and prediction
import streamlit as st
## streamlit app
st.title("Predict The Next Word")
st.write("Enter a text to predict")
#user Input
user_input=st.text_area('Text TO Predict:')
if st.button('Predict'):
    max_sequence_len=model.input_shape[1]+1
    next_word=predict_next_word(model,tokenizer,user_input,max_sequence_len)
    st.write(f'Next Word Prediction:   {next_word}') 
else:
    st.write("Please Enter text")
