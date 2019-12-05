#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:47:44 2019

@author: Guy Girineza
"""

import sys
import os
import re
import string
import pandas as pd
pd.options.mode.chained_assignment = None
import nltk
import scipy as scipy


"""
Remove HTML entities from the comments
"""
def clean_html(comment):
    cleaner = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleaned = re.sub(cleaner, '', comment)
    return cleaned

"""
function to clean the word of any punctuation or special characters
"""
def clean_punc(comment): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',comment)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

"""
Remove non alphabetic characters
"""
def clean_non_alpha(comment):
    cleaned = ""
    for word in comment.split():
        clean_word = re.sub('[^a-z A-Z]+', ' ', word)
        cleaned = cleaned + clean_word
        cleaned = cleaned + " "
    cleaned = cleaned.strip()
    return cleaned

"""
List of stop words
""" 
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stop_words = set(ENGLISH_STOP_WORDS)
stop_words.update(['zero','one','two','three','four','five','six','seven',
                   'eight','nine','ten','may','also','across','among',
                   'beside','however','yet','within'])
list_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
    

"""
Cleaning function
"""
def corpus_cleaning(data):
    
    data['comment_text'] = data['comment_text'].str.lower()
    data['comment_text'] = data['comment_text'].apply(clean_html)
    data['comment_text'] = data['comment_text'].apply(clean_punc)
    data['comment_text'] = data['comment_text'].apply(clean_non_alpha)
    return data

"""
This function return a document-word matrix computed for TFIDF
"""
def function_tfidf(train, test):
    #In order to use TfidfVectorizer, pandas dataframe is converted in a list
    text_train = train["comment_text"].values.tolist()
    text_test = test["comment_text"].values.tolist()

    tfidf_vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word',
				       ngram_range=(1,3), norm='l2', max_features=3000)

    tfidf_vectorizer.fit(text_train)
    tfidf_vectorizer.fit(text_test)
				    
    return tfidf_vectorizer.transform(text_train), tfidf_vectorizer.transform(text_test)

from sklearn.preprocessing import MultiLabelBinarizer

y_mlb = MultiLabelBinarizer()
"""
Transform the labels format in order to use it in the predictions function.
"""
def function_labels(data):
    #In order to use TfidfVectorizer, pandas dataframe is converted in a list
    data = data.values.tolist()
    return y_mlb.fit_transform(data)


if __name__ == "__main__":

    input_train_path = sys.argv[1]
    if not os.path.exists(input_train_path):
        print("This 'train' file do not exit")
        sys.exit(1)
    
    data = pd.read_csv(input_train_path)

    from sklearn.model_selection import train_test_split

    train, test = train_test_split(data, random_state=42,
                                   test_size=0.30, shuffle=True)
    
    train_data = train.filter(["comment_text"],
                              axis=1).reset_index(drop=True)
    train_labels = function_labels(train.drop(labels = ['id','comment_text'],
                              axis=1).reset_index(drop=True))
    
    test_data = test.filter(["comment_text"],
                            axis=1).reset_index(drop=True)
    test_labels = function_labels(test.drop(labels = ['id','comment_text'],
                            axis=1).reset_index(drop=True))
    
    #Data cleaning for training and set
    train_data = corpus_cleaning(train_data)
    test_data = corpus_cleaning(test_data)

    # Tf-IdF corpus for the training and test set
    from sklearn.feature_extraction.text import TfidfVectorizer

    X_train_tfidf, X_test_tfidf = function_tfidf(train_data, test_data)

    from skmultilearn.problem_transform import BinaryRelevance
    from sklearn.naive_bayes import GaussianNB
    
    classifier = BinaryRelevance(GaussianNB())
    
    # train
    classifier.fit(X_train_tfidf, train_labels)
    
    # predict
    predictions = classifier.predict(X_test_tfidf)

    # accuracy
    from sklearn.metrics import accuracy_score

    print("Accuracy = ", accuracy_score(test_labels, predictions))

    # confusion matix
    pred = predictions.toarray()
    import sklearn.metrics as skm
    cm = skm.multilabel_confusion_matrix(test_labels, pred)
    print(cm)
    print(skm.classification_report(test_labels, pred))
                                