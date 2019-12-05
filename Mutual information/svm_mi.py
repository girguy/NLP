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
import numpy as np
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
def clean_punc(comment):
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
List of stop words.
"""
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stop_words = set(ENGLISH_STOP_WORDS)
stop_words.update(['zero','one','two','three','four','five','six','seven',
                   'eight','nine','ten','may','also','across','among',
                   'beside','however','yet','within'])
list_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
    
"""
Remove stop word in a corpus.
"""
def remove_stop_words(comment):
    global list_stop_words
    return list_stop_words.sub(" ", comment)

"""
Function that perfoms a stemming on words on that corpus.
"""
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
Cleaning of a corpus using different functions defined above.
"""
def corpus_cleaning(data):
    data['comment_text'] = data['comment_text'].str.lower()
    data['comment_text'] = data['comment_text'].apply(clean_html)
    data['comment_text'] = data['comment_text'].apply(clean_punc)
    data['comment_text'] = data['comment_text'].apply(clean_non_alpha)
    data['comment_text'] = data['comment_text'].apply(remove_stop_words)
    data['comment_text'] = data['comment_text'].apply(stemming)
    return data



if __name__ == "__main__":
    
    input_train_path = sys.argv[1]
    if not os.path.exists(input_train_path):
        print("This 'train' file do not exit")
        sys.exit(1)

    data = pd.read_csv(input_train_path)
    
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(data, random_state=45,
                                   test_size=0.30, shuffle=True)
    
    train_data = train.filter(["comment_text"],
                              axis=1).reset_index(drop=True)
    train_labels = train.drop(labels = ['id','comment_text'],
                              axis=1).reset_index(drop=True)
    
    test_data = test.filter(["comment_text"],
                            axis=1).reset_index(drop=True)
    test_labels = test.drop(labels = ['id','comment_text'],
                            axis=1).reset_index(drop=True)

    # Processing of the train labels y_train
    dic_labels = {0: 'toxic', 1: 'severe_toxic', 2: 'obscene', 3: 'threat', 4: 'insult', 5: 'identity_hate'}
    y = np.array(train_labels)

    y_train = []
    for i in range(len(y)):
        value_index = np.where(y[i] == 1)
        name = ''
        for j in value_index[0]:
            name = name + dic_labels[j] + '_'
        if name == '':
            y_train.append('non_toxic')
        else:
            y_train.append(name.strip('_').strip())

    
    # Processing of the train data x_train
    x_train = train_data['comment_text']

    from sklearn.feature_extraction.text import CountVectorizer

    vect = CountVectorizer(max_features = 10000)
    vect.fit(x_train)
    x_train = vect.transform(x_train)

    # Mutual information
    from sklearn.feature_selection import mutual_info_classif

    mi = mutual_info_classif(x_train, y_train)

    # We retreive the n features among 10000 features that contribute to
    # the making a correct classification
    n = 4250
    name = []
    for score, fname in sorted(zip(mi, vect.get_feature_names()), reverse=True)[:n]:
        name.append(fname)
        
    x_train = pd.DataFrame(data=x_train.toarray())
    x_train.columns = vect.get_feature_names()

    # Retreive the dataframe countaining the n best columns
    from scipy import sparse

    x_train = x_train[name]
    x_train = sparse.csr_matrix(np.array(x_train))

    from skmultilearn.problem_transform import BinaryRelevance
    from sklearn.svm import LinearSVC
    
    classifier = BinaryRelevance(LinearSVC())
    # Train the model
    classifier.fit(x_train, train_labels)

    x_test = test_data['comment_text'] 
    vect = CountVectorizer(min_df = 1,  max_features = n).fit(x_test)

    # create a vocabulary content for each document of the corpus then use the 
    x_test = vect.transform(x_test)

    # Predictions
    predictions = classifier.predict(x_test)

    # accuracy
    from sklearn.metrics import accuracy_score

    print("Accuracy = ", accuracy_score(test_labels, predictions))