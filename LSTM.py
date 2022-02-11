import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM as KerasLSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split


class LSTM:

    def __init__(
        self,
        max_words=10000,
        embedding_vector_length=42,
        input_length=500,
        epochs=3,
        batch_size=256,
        dropout_rate=0.1,
        lstm_units=80,
        random_state=42,
        activation="sigmoid"
    ):
        self.max_words = max_words
        self.embedding_vector_length = embedding_vector_length
        self.input_length = input_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.lstm_units = lstm_units
        self.random_state = random_state
        self.activation = activation
        self.lstm = self.build_lstm()

    def build_lstm(self):
        lstm = Sequential()
        lstm.add(Embedding(self.max_words, self.embedding_vector_length, input_length=self.input_length))
        lstm.add(Dropout(self.dropout_rate))
        lstm.add(KerasLSTM(self.lstm_units))
        lstm.add(Dropout(self.dropout_rate))
        lstm.add(Dense(1, activation=self.activation))
        lstm.compile(loss = "binary_crossentropy", optimizer = "Adam", metrics = ["accuracy"])
        return lstm

    def fit(self, data):
        max_words = 10000
        tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        tokenizer.fit_on_texts(data["X_train"])
        seq_train = tokenizer.texts_to_sequences(data["X_train"])
        seq_test = tokenizer.texts_to_sequences(data["X_test"])
        X_train_pad = pad_sequences(seq_train, truncating='post', maxlen=self.input_length)
        X_test_pad = pad_sequences(seq_test, truncating='post', maxlen=self.input_length)
        X_train, X_val, y_train, y_val = train_test_split(X_train_pad, data["y_train"], test_size = 0.15, random_state = 42)
        self.lstm.fit(
            X_train,
            y_train,
            epochs = self.epochs,
            batch_size = self.batch_size,
            validation_data = (X_val, y_val)
        )
        scores_lstm = self.lstm.evaluate(X_test_pad, data["y_test"])
        print ("LSTM validation accuracy: %.2f%%" %(scores_lstm[1]*100))








