#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 08:19:40 2019

@author: cj
"""

import pandas as pd
pd.options.mode.chained_assignment = None
import scipy as scipy

from sklearn.feature_extraction.text import TfidfVectorizer
"""
This function return a document-word matrix computed for TFIDF
"""
def function_tfidf(data):
    #In order to use TfidfVectorizer, pandas dataframe is converted in a list
    text = data["comment_text"].values.tolist()
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=5000, min_df=5)

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
    
    train = pd.read_csv('/home/cj/Bureau/Master2/webAndText/project/data/train.csv')
    #Training set
    X_train_tfidf = scipy.sparse.load_npz('/home/cj/Bureau/X_train_tfidf.npz')
    
    #Training set labels
    train_labels = train.drop(labels = ['id','comment_text'], axis=1)
    y_train = train_labels
    
    #Test set
    X_test_tfidf = scipy.sparse.load_npz('/home/cj/Bureau/X_test_tfidf.npz')
    
    #Test set labels
    test_labels = pd.read_csv('/home/cj/Bureau/Master2/webAndText/project/data/test_labels.csv')
    test_labels = test_labels.drop(labels = "id", axis=1)
    y_test = test_labels
    
    #%%
    print(X_train_tfidf.shape[1])
    print(X_test_tfidf.shape[1])
    #%%
    
    from skmultilearn.problem_transform import BinaryRelevance

    from sklearn.naive_bayes import GaussianNB
    #classifier = BinaryRelevance(GaussianNB())
    
    from sklearn.svm import LinearSVC
    classifier = BinaryRelevance(LinearSVC())
    
    # train
    classifier.fit(X_train_tfidf[0:30000], y_test[0:30000])
    
    # predict
    predictions = classifier.predict(X_test_tfidf[0:5000])

    # accuracy
    from sklearn.metrics import accuracy_score
    predictions = pd.DataFrame(predictions.toarray())
    predictions.columns = y_test.columns
    
    
    #%%
    print("Accuracy = ", accuracy_score(predictions["toxic"][0:5000],
                                        y_test["toxic"][0:5000]))
    
    """
    We have an accuracy of 50% with 20% of the training dataset -- can we imagine
    reaching > 80% when using 100% of the training dataset
    """
    
    """
    For mutual information is in function of a term and a class. Here, we are  
    in a multi-labels situation. What we can do is to compute the mutual
    mutual information of of each class and take the mean.
    have to ask to the teacher
    """
    
    """
    TO DO :
    TO DO POS tagging
    """
    
    
    
    
    
    