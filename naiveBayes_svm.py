#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:47:44 2019

@author: cj
"""
  
import re
import pandas as pd
pd.options.mode.chained_assignment = None
import string
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
def clean_punc(comment): 
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    cleaned = regex.sub('', comment)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

"""
Remove non alphabetic characters
"""
def clean_non_alpha(comment):
    cleaned = ""
    for word in comment.split():
        clean_word = re.sub('[^a-z A-Z _]+', ' ', word)
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
Remove stop word
"""
def remove_stop_words(comment):
    global list_stop_words
    return list_stop_words.sub(" ", comment)

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english")
def stemming(comment):
    stemSentence = ""
    for word in comment.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

    

"""
Cleaning function
"""
def corpus_cleaning(data):
    
    data['comment_text'] = data['comment_text'].str.lower()
    data['comment_text'] = data['comment_text'].apply(clean_html)
    data['comment_text'] = data['comment_text'].apply(clean_punc)
    data['comment_text'] = data['comment_text'].apply(clean_non_alpha)
    data['comment_text'] = data['comment_text'].apply(remove_stop_words)
    data['comment_text'] = data['comment_text'].apply(stemming)
  
    return data

from sklearn.feature_extraction.text import TfidfVectorizer

"""
This function return a document-word matrix computed for TFIDF
"""
def function_tfidf(data):
    #In order to use TfidfVectorizer, pandas dataframe is converted in a list
    text = data["comment_text"].values.tolist()
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=24626, min_df=5)

    return tfidf_vectorizer.fit_transform(text)

from sklearn.preprocessing import MultiLabelBinarizer
y_mlb = MultiLabelBinarizer()

"""
Transform the labels format in order to use it in the machine learning api.
"""
def function_labels(data):
    #In order to use TfidfVectorizer, pandas dataframe is converted in a list
    data = data.values.tolist()
    return y_mlb.fit_transform(data)


if __name__ == "__main__":

    import sys
    import os

    input_train_path = sys.argv[1]
    if not os.path.exists(input_train_path):
        print("This 'train' file do not exit")
        sys.exit(1)

    input_test_path = sys.argv[2]
    if not os.path.exists(input_test_path):
        print("This 'test' file do not exit")
        sys.exit(1)

    input_test_labels_path = sys.argv[3]
    if not os.path.exists(input_test_labels_path):
        print("This 'test_labels' file do not exit")
        sys.exit(1)
    
    train = pd.read_csv(input_train_path)
    train_text = train.filter(["comment_text"], axis=1)

    #Data cleaning and feature extraction for training set
    train_text = corpus_cleaning(train_text)
    X_train_tfidf = function_tfidf(train_text)
    #Save
    #scipy.sparse.save_npz('/home/cj/Bureau/X_train_tfidf.npz', X_train_tfidf)

    test = pd.read_csv(input_test_path)
    test_text = test.filter(["comment_text", ], axis=1)

    #Data cleaning and feature extraction for test set
    test_text = corpus_cleaning(test_text)
    X_test_tfidf = function_tfidf(test_text)
    #Save
    #scipy.sparse.save_npz('/home/cj/Bureau/X_test_tfidf.npz', X_test_tfidf)

    # Training set labels
    train_labels = train.drop(labels = ['id','comment_text'], axis=1)
    y_train = function_labels(train_labels)

    # Test set labels
    test_labels = pd.read_csv(input_test_labels_path)
    test_labels = test_labels.drop(labels = "id", axis=1)
    y_test = function_labels(test_labels)
    
    #%%
    print(X_train_tfidf.shape[1])
    print(X_test_tfidf.shape[1])
    #%%
    
    from skmultilearn.problem_transform import BinaryRelevance

    from sklearn.naive_bayes import GaussianNB
    classifier = BinaryRelevance(GaussianNB())
    
    from sklearn.svm import LinearSVC
    #classifier = BinaryRelevance(LinearSVC())
    
    # train
    classifier.fit(X_train_tfidf[0:20000], y_test[0:20000])
    
    # predict
    predictions = classifier.predict(X_test_tfidf[0:5000])

    # accuracy
    from sklearn.metrics import accuracy_score
    print("Accuracy = ", accuracy_score(y_test[0:5000], predictions))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
"""
The lemmatization done here is BASED on
'Introduction to Machine Learning with Python' book p.346

Tape this line in the terminal : python -m spacy download en

import spacy
from spacy.tokens import Doc

regexp = re.compile('(?u)\\b\\w\\w+\\b')

en_nlp = spacy.load('en')
old_tokenizer = en_nlp.tokenizer
en_nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(
        regexp.findall(string))

def custom_tokenizer(document):
    doc_spacy = spacy.load('en')
    doc = doc_spacy(document)
    doc_spacy = Doc(doc.vocab, words=[t.text for t in doc])
    token_lemma_data = [token.lemma_ for token in doc_spacy]
    return token_lemma_data

"""


"""
from sklearn.feature_extraction.text import CountVectorizer
lemma_vect = CountVectorizer(tokenizer=custom_tokenizer, min_df=5)
X_train_lemma = lemma_vect.fit_transform(data["comment_text"])
print("X_train_lemma.shape: {}".format(X_train_lemma.shape))
""" 
  

# TFIDF (BOOK)
# X_TRAIN, Y_TRAIN, X_TEST, Y_TEST
# MULTI-LABELS