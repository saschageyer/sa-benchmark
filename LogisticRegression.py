import re
import numpy as np
import pandas as pd
import spacy
import Utils
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as LogisticRegressionSklearn


class LogisticRegression:
    
    def __init__(self, df, ngram_range=(1,3), max_iter=100, random_state=42):
        self.X = df['text']
        self.y = df['sentiment']
        self.ngram_range = ngram_range
        self.random_state = random_state
        self.model = LogisticRegressionSklearn(max_iter=max_iter, random_state=random_state)

    
    def split_train_test(self, test_size):
        return train_test_split(
            self.X, self.y, test_size = test_size, random_state = self.random_state
        )
        
    def get_tfidf_features(self, X_train, X_test):
        vectorizer = TfidfVectorizer(ngram_range=self.ngram_range)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        return X_train_tfidf, X_test_tfidf

    def fit(self, test_size=0):
        X_train, X_test, y_train, y_test = self.split_train_test(test_size)
        X_train_tfidf, X_test_tfidf = self.get_tfidf_features(X_train, X_test)
        self.model.fit(X_train_tfidf, y_train)
        if (test_size):
            val_accuracy = self.model.score(X_test_tfidf, y_test)
            print(f'Logistic Regression validation accuracy: {val_accuracy}')