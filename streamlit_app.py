# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 11:16:44 2022

@author: ASUS
"""
import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
#model filename
loaded_model= pickle.load(open("reviews_classifier.sav", 'rb'))
cv = CountVectorizer(max_features = 2000)
#loading the corpus
with open("corpus.data", 'rb') as filehandle:
    # read the data as binary data stream
    corpus = pickle.load(filehandle)
#fitting the count vectorizer with the corpus
cv.fit_transform(corpus)
# creating a function for Prediction

def review_prediction(input_data):
    test_vec = cv.transform(input_data)
    prediction = loaded_model.predict(test_vec)[0]
    print(prediction)
    
    if (prediction== 0):
      return 'The review entered was negative.\n The user did not like the restaurant.'
    else:
      return 'The review entered was positive. \n The user liked the restaurant.'

def main():
    
    
    # giving a title
    st.title("Restaurant Reviews Classifier using NLP")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Restaurant Reviews ML App </h2>
    </div>
    <div>
     <a href='https://medium.com/@shubhamparinati/restaurant-review-a-beginners-guide-to-nlp-sentimental-analysis-b2ded804b30b>To Know How It works Click Here!</a>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    # getting the input data from the user
    user_input=st.text_input('type your review below','review')
    
    # code for Prediction
    review_result = ''
    
    # creating a button for Prediction
    if st.button("Predict"):
        review_result=review_prediction([user_input])
        
    st.success(review_result)
    
if __name__=='__main__':
    main()        

