#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:47:44 2019

@author: cj
"""

import sys
import os
import re
import string
import pandas as pd
pd.options.mode.chained_assignment = None
import nltk
from nltk import word_tokenize
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

"""
Tokenization and pos of tag of the corpus
"""
def tokenization_pos_tag(corpus):
    corpus = corpus.tolist()
    X = []
    for sentence in corpus:
        tokenized = word_tokenize(sentence)
        X.append(nltk.pos_tag(tokenized))
    
    return X


"""
Bag of word of tagged corpus
"""
def dataset_creation(data):
    dataset = []
    for sentence in data:
        new = ''
        for word_tag in sentence:
            word, tag = word_tag
            new = new + word+'_'+tag+' '
        dataset.append(new.strip())
    
    return dataset


if __name__ == "__main__":

    input_train_path = sys.argv[1]
    if not os.path.exists(input_train_path):
        print("This 'train' file do not exit")
        sys.exit(1)

    print("Ficher charge")
    
    data = pd.read_csv(input_train_path)
    
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(data, random_state=42,
                                   test_size=0.30, shuffle=True)
    
    train_text = train.filter(["comment_text"],
                              axis=1).reset_index(drop=True)
    train_labels = train.drop(labels = ['id','comment_text'],
                              axis=1).reset_index(drop=True)
    
    test_text = test.filter(["comment_text"],
                            axis=1).reset_index(drop=True)
    test_labels = test.drop(labels = ['id','comment_text'],
                            axis=1).reset_index(drop=True)
    
    #Data cleaning for training and set
    train_text = corpus_cleaning(train_text)
    test_text = corpus_cleaning(test_text)

    train_text_tagged = tokenization_pos_tag(train_text["comment_text"])
    test_text_pos = tokenization_pos_tag(test_text["comment_text"])

    X_train = dataset_creation(train_text_tagged)
    X_test = dataset_creation(test_text_pos)
    
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    from sklearn.feature_extraction.text import CountVectorizer

    vect = CountVectorizer(min_df=1,  max_features=5000).fit(X_train)
    vect = CountVectorizer(min_df=1,  max_features=5000).fit(X_test)
    X_train = vect.transform(X_train)
    X_test = vect.transform(X_test)

    from skmultilearn.problem_transform import BinaryRelevance
    from sklearn.svm import LinearSVC
    
    classifier = BinaryRelevance(LinearSVC())
    # train
    classifier.fit(X_train, train_labels)
    # predict
    predictions = classifier.predict(X_test)

    # accuracy
    from sklearn.metrics import accuracy_score

    print("Accuracy = ", accuracy_score(test_labels, predictions))


    

