import re
import numpy as np
import pandas as pd
import spacy
import Utils


class Preprocessor:
    
    def __init__(self, name, df, text_feature, target):
        '''
            Class used for text preprocessing
            If you don't have spaCy installed: https://spacy.io/usage
        '''
        self.name = name
        self.df = df
        self.text_feature = text_feature
        self.target = target
        self.messy_texts = df[text_feature]
        self.nlp = spacy.load('en_core_web_sm')
        self.stopwords = self.nlp.Defaults.stop_words
        self.regex = {
            # remove html tags
            'html': re.compile(r'<[^>]+>'),

            # remove non-alphabetic characters
            'non_alpha': re.compile('[^a-zA-Z]'),

            # custom regex to add a space between sticking words (e.g. 'endStart')
            'camel_case_space': re.compile('([a-z])([A-Z])'),

            # custom regex to remove single characters (also at beginning)
            'single_char': re.compile(r'(\s|^).(?=\s+)'),

            # regex to remove multiple spaces
            'multi_space': re.compile(r'\s\s+')
        }

        
    def run(self):
        '''
            If available, read preprocessed.parquet.gzip from cache
            Ohterwise create preprocessed.parquet.gzip and write it to cache
        '''
        output = Utils.parquet_caching(
            parquet_name = self.name + "_preprocessed",
            callback = self.run_nlp_pipeline
        )
        return output
        
        

    def lemmatize_and_remove_stopwords(self, text):
        '''
            We use spaCy to lemmatize our texts and remove stopwords.
            We skip pronouns which lemmatize '-PRON-' as well.
        '''
        doc = self.nlp(text)
        return ' '.join(token.lemma_ for token in doc if token.lemma_ != '-PRON-' and token.lemma_ not in self.stopwords)
    
    
    def get_factorized_targets(self):
        codes, uniques = self.df[self.target].factorize()
        return codes

    def run_nlp_pipeline(self):
        '''
            To generate a corpus of lemmatized documents we feed our text data
            through a nlp pipeline that performes lemmatization, removes stopwords
            and applies our regular expressions. 
        '''
        processed_texts = []

        for messy_text in self.messy_texts:

            text = self.lemmatize_and_remove_stopwords(messy_text.lower())
            text = self.regex['html'].sub(' ', text)
            text = self.regex['non_alpha'].sub(' ', text)
            text = self.regex['camel_case_space'].sub(r'\1 \2', text)
            text = self.regex['single_char'].sub(' ', text)
            text = self.regex['multi_space'].sub(' ', text)
            text = text.strip()

            processed_texts.append(text)
        
        processed_df = self.df.copy()
        processed_df[self.text_feature] = processed_texts
        processed_df[self.target] = self.get_factorized_targets()

        return processed_df