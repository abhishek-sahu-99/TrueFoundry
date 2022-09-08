#Loading Libraries
import pandas as pd
import re
import numpy as np
import emoji
import wordninja
import nltk
from nltk.corpus import stopwords
import spacy
import string
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

#Loading Components
nlp =spacy.load('en_core_web_sm')
stop_words=set(stopwords.words('english'))



#Logistic Regression Class
class logistic_regression:
    '''
    This is default constructor to initalise the model.
    '''
    def __init__(self):
        self.model=None
        self.tfidf_vector=None
    '''
    This cleans the text
    
    parameter
    ---------
    
    1) text (str)
    '''
    def clean(self,text):
        #remove_mentions, urls, hash_sign:
        mention_words_removed= re.sub(r'@\w+','',text)
        hash_sign_removed=re.sub(r'#','',mention_words_removed)
        url_removed=' '.join(word for word in hash_sign_removed.split(" ") if not word.startswith('http'))
        
        #Transform emoji to text
        demoj=emoji.demojize(url_removed)
        
        #Split compound words coming from hashtags
        splitted=wordninja.split(demoj)
        splitted=" ".join(word for word in splitted)
        
        # Implement lemmatization & remove punctuation
        lem = nlp(splitted)
        punctuations = string.punctuation
        punctuations=punctuations+'...'

        sentence=[]
        for word in lem:
            word = word.lemma_.lower().strip()
            if ((word != '-pron-') & (word not in punctuations)):
                sentence.append(word)    
                
        #Remove stopwords
        stop_words_removed=[word for word in sentence if word not in stop_words]
        
        return stop_words_removed
    
    '''
    This functions trains our model
    
    parameter
    ---------
    
    1) data (DataFrame)
    '''
    def train(self,data):
        data["clean_text_list"]=data["text"].apply(self.clean)
        data["clean_text"]=[" ".join(word) for word in data["clean_text_list"]]
        X=data["clean_text"].values
        Y=data["airline_sentiment"].map({"negative":-1,"positive":1})   
        self.tfidf_vector = TfidfVectorizer()
        X=self.tfidf_vector.fit_transform(X)
        self.model=LogisticRegression(random_state=0, solver='lbfgs',max_iter=500).fit(X,Y)
       
    '''
    This function predict the output based on given text
    
    parameter
    ---------
    
    1) text(str)
    
    Output: Positive/Negative
    ''' 
    def predict(self,text):
        text=self.clean(text)
        text=[" ".join(word for word in text)]
        X_test = self.tfidf_vector.transform(text)
        ans=self.model.predict(X_test)
        return "Positive" if ans[0]==1 else 'Negative'
        
        
        


