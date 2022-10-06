# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 11:16:44 2022

@author: ASUS
"""

import streamlit as st
import pickle as pkl
from sklearn.feature_extraction.text import CountVectorizer as cv


# loading the saved model
loaded_model = pkl.load(open('reviews_classifier.sav', 'rb'))

# creating a function for Prediction

def review_prediction(input_data):
    test = [input_data]
    test_vec = cv.transform(test)
    prediction = loaded_model.predict(test_vec)[0]
    print(prediction)
    
    if (prediction[0] == 0):
      return 'The review entered was negative.\n The user did not like the restaurant.'
    else:
      return 'The review entered was positive. \n The user liked the restaurant.'

def main():
    
    
    # giving a title
    st.title("Reviews Classifier")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Restaurant Reviews ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    # getting the input data from the user
    user_input=st.text_input('review','type your review here')
    
    # code for Prediction
    review_result = ''
    
    # creating a button for Prediction
    if st.button("Predict"):
        review_result=review_prediction([user_input])
        
    st.success(review_result)

if __name__=='__main__':
    main()        

