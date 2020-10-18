# This script is used for preprocessing the training data. 
# This script is also used for preprocessing the GET and POST requests in the flask_api before inferencing

import pandas as pd 
import numpy as np 
import string
import re
import nltk
import random
nltk.download('all')
# nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
stop_words_english = set(stopwords.words('english')) 
stop_words_french = set(stopwords.words('french')) 

def english_stop_word_remover(x):
    
    word_tokens = word_tokenize(x) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words_english] 
  
    filtered_sentence = [] 
    
    for w in word_tokens: 
        if w not in stop_words_english: 
            filtered_sentence.append(w) 
    return ' '.join(filtered_sentence)


def french_stop_word_remover(x):
    word_tokens = word_tokenize(x) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words_french] 
  
    filtered_sentence = [] 
    
    for w in word_tokens: 
        if w not in stop_words_french: 
            filtered_sentence.append(w) 
    return ' '.join(filtered_sentence)

def replace_escape_characters(x):
    
    try:
        x =  re.sub(r'\s', ' ', x)
        x = x.encode('ascii', 'ignore').decode('ascii')
        return x
    except:
        return x   

def preprocess(df):

    rawData=df
    cleanData = rawData.dropna()
    rawData["OriginalEmailBody"] = rawData["Subject"] + " " + rawData["OriginalEmailBody"]
    rawData['OriginalEmailBody'] = rawData['OriginalEmailBody'].str.lower()
    rawData['OriginalEmailBody'] = rawData.OriginalEmailBody.str.strip()
    rawData = rawData.dropna(subset=['OriginalEmailBody'])
    rawData['OriginalEmailBody'] = rawData['OriginalEmailBody'].apply(replace_escape_characters)
    rawData['OriginalEmailBody'] = rawData['OriginalEmailBody'].apply(english_stop_word_remover)
    rawData['OriginalEmailBody'] = rawData['OriginalEmailBody'].apply(french_stop_word_remover)
    
    cleanData['OriginalEmailBody'] = cleanData['OriginalEmailBody'].str.replace('@[A-Za-z0-9_]+', "")
    cleanData['OriginalEmailBody'] = cleanData['OriginalEmailBody'].str.replace('re:', "")
    cleanData['OriginalEmailBody'] = cleanData['OriginalEmailBody'].str.replace('graybar', "")
    cleanData['OriginalEmailBody'] = cleanData['OriginalEmailBody'].str.replace('\n', " ")
    cleanData['OriginalEmailBody'] = cleanData['OriginalEmailBody'].str.replace('#[A-Za-z0-9]+', "")
    cleanData['OriginalEmailBody'] = cleanData['OriginalEmailBody'].str.replace('#[ A-Za-z0-9]+', "")
    cleanData['OriginalEmailBody'] = cleanData['OriginalEmailBody'].str.replace('[0-9]+', "")
    cleanData['OriginalEmailBody'] = cleanData['OriginalEmailBody'].str.replace('[A-Za-z0-9_]+.com', "")
    cleanData['OriginalEmailBody'] = cleanData['OriginalEmailBody'].replace('[^a-zA-Z0-9 ]+', ' ', regex=True) #removes sp char
    
    cleanData['OriginalEmailBody'] = cleanData['OriginalEmailBody'].str.replace('please', "")
    cleanData['OriginalEmailBody'] = cleanData['OriginalEmailBody'].str.replace('com', "")
    cleanData['OriginalEmailBody'] = cleanData['OriginalEmailBody'].str.replace('saps', "")
    cleanData['OriginalEmailBody'] = cleanData['OriginalEmailBody'].str.replace('sent', "")
    cleanData['OriginalEmailBody'] = cleanData['OriginalEmailBody'].str.replace('subject', "")
    cleanData['OriginalEmailBody'] = cleanData['OriginalEmailBody'].str.replace('thank', "")
    cleanData['OriginalEmailBody'] = cleanData['OriginalEmailBody'].str.replace('www', "")
    cleanData['OriginalEmailBody'] = cleanData['OriginalEmailBody'].str.replace(' e ', "")
    cleanData['OriginalEmailBody'] = cleanData['OriginalEmailBody'].str.replace('email', "")
    cleanData['OriginalEmailBody'] = cleanData['OriginalEmailBody'].str.replace('cc', "")
    cleanData['OriginalEmailBody'] = cleanData['OriginalEmailBody'].str.replace('n t', "not")
    cleanData['OriginalEmailBody'] = cleanData.OriginalEmailBody.str.strip()
    
    return cleanData 
