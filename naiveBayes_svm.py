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
#import nltk

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

"""
Cleaning function
"""
def corpus_cleaning(data):
    
    data['comment_text'] = data['comment_text'].str.lower()
    data['comment_text'] = data['comment_text'].apply(clean_html)
    data['comment_text'] = data['comment_text'].apply(clean_punc)
    data['comment_text'] = data['comment_text'].apply(clean_non_alpha)
    data['comment_text'] = data['comment_text'].apply(remove_stop_words)
  
    return data


if __name__ == "__main__":
    
    train = pd.read_csv('/home/cj/Bureau/Master2/webAndText/project/data/train.csv')
    train_text = train.filter(["comment_text"], axis=1)
    train_labels = train.drop(labels = ['id','comment_text'], axis=1)
    
    test = pd.read_csv('/home/cj/Bureau/Master2/webAndText/project/data/test.csv')
    test_text = test.filter(["comment_text", ], axis=1)
    test_labels = pd.read_csv('/home/cj/Bureau/Master2/webAndText/project/data/test_labels.csv')
    test_labels = test_labels.drop(labels = "id", axis=1)
    
    """
    X_train_bis = X_train.iloc[0:10000, :]
    Y_train_bis = Y_train.iloc[0:10000, :]
    X_test_bis = X_test.iloc[0:10000, :]
    Y_test_bis = Y_test.iloc[0:10000, :]
    """
    
    train_text = corpus_cleaning(train_text)
    X_train = train_text["comment_text"].values.tolist()
    test_text = corpus_cleaning(test_text)
    X_test = test_text["comment_text"].values.tolist()
     
    from sklearn.feature_extraction.text import TfidfVectorizer 
 
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=5000)
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.fit_transform(X_test)
    
    # using binary relevance
    from skmultilearn.problem_transform import BinaryRelevance
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    # initialize binary relevance multi-label classifier
    # with a gaussian naive bayes base classifier
    
    classifier = BinaryRelevance(GaussianNB())
    
    from sklearn.svm import LinearSVC
    #classifier = BinaryRelevance(LinearSVC())
    
    from sklearn.preprocessing import MultiLabelBinarizer
    train_labels = train_labels.values.tolist()
    test_labels = test_labels.values.tolist()
    
    y_mlb = MultiLabelBinarizer()
    y_train = y_mlb.fit_transform(train_labels)
    y_test = y_mlb.fit_transform(test_labels)
    
    # train
    classifier.fit(X_train_tfidf[0:153163], y_test[0:153163])
    # predict
    predictions = classifier.predict(X_test_tfidf[0:1000])
    # accuracy
    print("Accuracy = ", accuracy_score(y_test[0:1000], predictions))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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