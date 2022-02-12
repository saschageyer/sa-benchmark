import re
import numpy as np
import pandas as pd
import spacy
import Utils
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as LogisticRegressionSklearn


class LogisticRegression:
    
    def __init__(self, ngram_range=(1,3), max_iter=100, random_state=42):
        self.ngram_range = ngram_range
        self.random_state = random_state
        self.model = LogisticRegressionSklearn(max_iter=max_iter, random_state=random_state,solver='liblinear')
        
    def get_tfidf_features(self, X_train, X_test):
        vectorizer = TfidfVectorizer(ngram_range=self.ngram_range)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        return X_train_tfidf, X_test_tfidf

    def fit(self, sets):
        X_train_tfidf, X_test_tfidf = self.get_tfidf_features(sets["X_train"], sets["X_test"])
        self.model.fit(X_train_tfidf, sets["y_train"])
        val_accuracy = self.model.score(X_test_tfidf, sets["y_test"])
        print(f'Logistic Regression validation accuracy: {val_accuracy}')
