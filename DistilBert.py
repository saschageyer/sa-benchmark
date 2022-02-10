from transformers import TFDistilBertModel
from transformers import TFDistilBertForSequenceClassification
from transformers import DistilBertTokenizer



class DistilBert:
    def __init__(self,epochs=1,random_state=42):
        self.random_state=random_state
        self.epochs=epochs
    
    def train_val_split(self, X_train, y_train):
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                          test_size = 0.15, random_state = 42
                                                         )
        return X_train, X_val, y_train, y_val
    
    def bert_tokenize (self,X_train,X_val,X_test):
        print("Tokenizing")
        bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        Xtrain_enc = bert_tokenizer(X_train.tolist(), max_length=256, #50 for sent140
                                    truncation=True, padding='max_length', 
                                    add_special_tokens=True, return_tensors='np')
        Xval_enc = bert_tokenizer(X_val.tolist(), max_length=256,#50 for sent140 
                                  truncation=True, padding='max_length', 
                                  add_special_tokens=True, return_tensors='np')
        Xtest_enc = bert_tokenizer(X_test.tolist(), max_length=256, #50 for sent140
                                   truncation=True, padding='max_length', 
                                   add_special_tokens=True, return_tensors='np') #return numpy object
        print("Tokenizing completed")
        return Xtrain_enc,Xval_enc,Xtest_enc
    
    def prepare_datasets (self,Xtrain_enc,Xval_enc,Xtest_enc,y_train,y_val,y_test):
        train_dataset = tf.data.Dataset.from_tensor_slices((dict(Xtrain_enc),
                                                            y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((dict(Xval_enc),
                                                            y_val))
        test_dataset = tf.data.Dataset.from_tensor_slices((dict(Xtest_enc),
                                                            y_test))
        return train_dataset, val_dataset, test_dataset
    
    def bert_model(self,train_dataset,val_dataset,epochs):
        max_len=256 #50 for sent140
        epochs=self.epochs
        transformer = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
        print("----Building the model----")
        input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
        attention_mask = Input(shape=(max_len,),dtype=tf.int32,name = 'attention_mask') #attention mask
        sequence_output = transformer(input_ids,attention_mask)[0]
        cls_token = sequence_output[:, 0, :]
        x = Dense(512, activation='relu')(cls_token)
        #x = Dropout(0.5)(x)
        y = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=[input_ids,attention_mask], outputs=y)
        model.summary()
        model.compile(Adam(learning_rate=3e-5), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def fit_distil_bert(self, sets,epochs):
        X_train, X_val, y_train, y_val = self.train_val_split(sets['X_train'],sets['y_train'])
        Xtrain_enc,Xval_enc,Xtest_enc =self.bert_tokenize(X_train,X_val,sets['X_test'])
        train_dataset, val_dataset, test_dataset =self.prepare_datasets(Xtrain_enc,Xval_enc,Xtest_enc,
                                                                        y_train,y_val,sets['y_test'])
        model = self.bert_model(train_dataset,val_dataset,epochs=epochs)
        model.fit(train_dataset.batch(32),batch_size = 32,
                  epochs = epochs, validation_data = val_dataset.batch(32))
        print("Train score:", model.evaluate(train_dataset.batch(32)))
        print("Validation score:", model.evaluate(val_dataset.batch(32)))
        print("Test score:", model.evaluate(test_dataset.batch(32)))
    
    