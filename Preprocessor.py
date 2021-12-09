import re
import numpy as np
import pandas as pd
import spacy
import Utils


class Preprocessor:
    
    def __init__(self, name, df, cache=True):
        '''
            Class used for text preprocessing
            If you don't have spaCy installed: https://spacy.io/usage
        '''
        self.name = name
        self.df = df
        self.cache = cache
        self.messy_texts = df['text']
        self.nlp = spacy.load('en_core_web_sm')
        self.stopwords = self.nlp.Defaults.stop_words
        self.regex = {
            
            # custom regex to add a space between sticking words (e.g. 'endStart')
            #'camel_case_space': re.compile('([a-z])([A-Z])'),
            
            # replace html with htmlregex
            'html': re.compile(r'<[^>]+>'),
            
            # replace url with urlregex
            'url': re.compile(r'((http:\/\/)[^ ]*|(https:\/\/)[^ ]*|(www\.)[^ ]*)'),

            # replace @user with USER
            "user": re.compile(r'@[^\s]+'),
            
            # replace #topic with topic (at least 3 chars)
            "hashtag": re.compile(r'#([^\s]{3,})'),
            
            # replace heyyyy' with 'heyy'
            "char_repetition": re.compile(r'(.)\1\1+'),
            
            "heart": re.compile(r'<3'),

            "happy_smile": re.compile(r"[8:=;]['`\-]?[)d]+"),
            
            "sad_smile": re.compile(r"[8:=;]['`\-]?\(+"),
            
            "neutral_smile": re.compile(r"[8:=;]['`\-]?[\/|]"),
            
            "laugh_smile": re.compile(r"[8:=;]['`\-]?[pD]+"),
            
            # remove non-alphabetic characters
            'non_alpha': re.compile('[^a-zA-Z]'),

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
        if self.cache:
            return Utils.parquet_caching(
                parquet_name = self.name + "_preprocessed",
                callback = self.run_nlp_pipeline
            )
        else:
            return self.run_nlp_pipeline()        
        

    def lemmatize_and_remove_stopwords(self, text):
        '''
            We use spaCy to lemmatize our texts and remove stopwords.
            We skip pronouns which lemmatize '-PRON-' as well.
        '''
        doc = self.nlp(text)
        return ' '.join(token.lemma_ for token in doc if token.lemma_ != '-PRON-' and token.lemma_ not in self.stopwords)
    
    
    def get_factorized_sentiments(self):
        codes, uniques = self.df['sentiment'].factorize()
        return codes

    def run_nlp_pipeline(self):
        '''
            To generate a corpus of lemmatized documents we feed our text data
            through a nlp pipeline that performes lemmatization, removes stopwords
            and applies our regular expressions. 
        '''
        processed_texts = []

        for messy_text in self.messy_texts:

            # text = self.regex['camel_case_space'].sub(r'\1 \2', messy_text)
            text = self.regex['html'].sub(' htmlregex ', messy_text)
            text = self.regex['url'].sub(' urlregex ', text)
            text = self.regex['user'].sub(' userregex ', text)
            text = self.regex['hashtag'].sub(r'\1', text)
            text = self.regex['char_repetition'].sub(r'\1\1', text)
            text = self.regex['heart'].sub('heartregex', text)
            text = self.regex['happy_smile'].sub('happyregex', text)
            text = self.regex['sad_smile'].sub('sadregex', text)
            text = self.regex['neutral_smile'].sub('neutralregex', text)
            text = self.regex['laugh_smile'].sub('laughregex', text)
            text = self.lemmatize_and_remove_stopwords(text.lower())
            text = self.regex['non_alpha'].sub(' ', text)
            text = self.regex['single_char'].sub(' ', text)
            text = self.regex['multi_space'].sub(' ', text)
            text = text.strip()

            processed_texts.append(text)
        
        processed_df = self.df.copy()
        processed_df['text'] = processed_texts
        processed_df['sentiment'] = self.get_factorized_sentiments()

        return processed_df