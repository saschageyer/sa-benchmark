from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.preprocessing import sequence
from tensorflow.keras.models import Model
from keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense,Input, Embedding,LSTM,Dropout,Conv1D, MaxPooling1D, GlobalMaxPooling1D,Dropout,Bidirectional,Flatten,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

import tokenizers

class LSTM:
    def __init__(self,epochs=3,random_state=42):
        self.random_state=random_state
        self.epochs=epochs
    
    def preprocessing_for_lstm (self,sets):
        X_train=sets['X_train']
        X_test=sets['X_test']
    
        tokenizer = Tokenizer(oov_token='<OOV>')
        tokenizer.fit_on_texts(X_train)
        word_index = tokenizer.word_index
        V = len(word_index) #vocabulary size
        print("Vocabulary of the dataset is : ",V)
        
        seq_train = tokenizer.texts_to_sequences(X_train)
        seq_test =  tokenizer.texts_to_sequences(X_test)
        
        input_len=max([len(x) for x in seq_train])
        print(input_len)
        X_train_pad = pad_sequences(seq_train,maxlen=input_len)
        X_test_pad = pad_sequences(seq_test,maxlen=input_len)
        print("Data padded!")
        return X_train_pad, X_test_pad, V, input_len, word_index
     #for pretrained embeddings
    def get_embed_matrix (self,pretrained_embeddings,voc_size,embedding_dim,word_index):
        embeddings_index = {}
        f = open(pretrained_embeddings,)
        for line in f:
            values = line.split()
            word = value = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
    
        print('Found %s word vectors.' %len(embeddings_index))

        embedding_matrix = np.zeros((voc_size+1, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
    
    #for pretrained emb
    def create_embedding_layer(self,embeddings_name,embedding_dim,V,max_seq_len,word_index):
        embedding_layer=Embedding(input_dim=V+1,
                                  output_dim=embedding_dim, 
                                  weights= [self.get_embed_matrix(embeddings_name,voc_size=V,
                                                embedding_dim=embedding_dim,word_index=word_index)],
                                  input_length=max_seq_len,
                                  trainable=False  
                                 )
        return embedding_layer
    
    def create_lstm_model(self,embedding_dim,embeddings_name,V,max_seq_len,word_index):
        dropout_rate=0.1
        lstm_units=100
    # Create the model
        if embeddings_name is None:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Embedding(V+1, embedding_dim,input_length = max_seq_len),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.LSTM(lstm_units),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(64,activation="relu"),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(1, activation = "sigmoid")
            ])
        else:
            model = tf.keras.models.Sequential([
                self.create_embedding_layer(embeddings_name,embedding_dim,V,max_seq_len,word_index),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.LSTM(lstm_units),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(64,activation="relu"),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(1, activation = "sigmoid")
            ])
                
        optimizer = "Adam"
        model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
        print(model.summary())
        return model
    
    def fit_lstm(self,sets,epochs,batch_size,embedding_dim,embeddings_name):
        X_train_pad, X_test_pad,V,max_seq_len, word_index = self.preprocessing_for_lstm(sets)
        #vocab_size=V
        y_train=sets['y_train']
        y_test=sets['y_test']
        model = self.create_lstm_model(embedding_dim=embedding_dim,embeddings_name=embeddings_name,
                                       V=V,max_seq_len=max_seq_len,word_index=word_index) 
        val_split=0.05
        model.fit(X_train_pad,y_train,epochs = epochs,batch_size = batch_size,verbose = 0,
                 validation_split=val_split) 
        scores_lstm = model.evaluate(X_test_pad,y_test)
        print ("LSTM Accuracy: %.2f%%" %(scores_lstm[1]*100))
