
"""
Created on Thu Sep 17 01:23:14 2020

@author: anish.gupta
"""

from flask import Flask, request, jsonify
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
# from clean import preprocess 
from fastai.text import load_learner, DatasetType

from flasgger import APISpec, Schema, fields
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec_webframeworks.flask import FlaskPlugin

import operator

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


# Create an APISpec for documentation of this API using flassger
# spec = APISpec(
#     title='SAPS Email Classifier',
#     version='1.0.0',
#     openapi_version='2.0',
#     plugins=[
#         FlaskPlugin(),
#         MarshmallowPlugin(),
#     ],
# )

# # Optional marshmallow support
# class CategorySchema(Schema):
#     id = fields.Int()
#     name = fields.Str(required=True)


# class PetSchema(Schema):
#     category = fields.Nested(CategorySchema, many=True)
#     name = fields.Str()

# app=Flask(__name__)

from azureml.core.model import Model
#from azureml.monitoring import ModelDataCollector

def init():
    global learn
    print ("model initialized" + time.strftime("%H:%M:%S"))
    model_path = Model.get_model_path(model_name = 'saps_classification')
#     model = load(model_path)
    learn = load_learner(model_path,'')

# model_path = Model.get_model_path(model_name = 'saps_classification')
# # Load fastai model trained using model_train.py
# learn = load_learner(model_path,'')

# @app.route('/')
# def welcome():
#     return "Welcome All"

# GET method
# @app.route('/predict',methods=["Get"])
# def predict_email():
    
#     # GET specification for flassger 
#     """GET Method 
#     This is using docstrings for specifications.
#     ---
#     tags:
#       - Email Classification Model API
#     parameters:  
#       - name: subject
#         in: query
#         type: string
#         required: true
#       - name: body
#         in: query
#         type: string
#         required: true
#     responses:
#         200:
#             description: The output values
        
#     """

#     # GET email subject and body
#     subject=request.args.get("subject")
#     body=request.args.get("body")
#     # Create a dataframe
#     data = {'Subject' : [subject], 'OriginalEmailBody' : [body]}
#     df_test=pd.DataFrame(data)

#     # Clean the data
#     df_clean = preprocess(df_test)

#     # Make prediction and return top 3 probabilities and RequestType
#     learn.data.add_test(df_clean['OriginalEmailBody'])
#     prob_preds = learn.get_preds(ds_type=DatasetType.Test, ordered=True)
#     keys = ['Chargeback Request', 'Chargeback Workflow Follow-up', 'Freight Deductions Issue',
#             'Invoice Submission','Miscellaneous Deduction Information','Other',
#             'Payment status on Invoice','Secondary','Statement']
#     values = prob_preds[0].numpy().tolist()[0]
#     # Create a dict with keys as RequestType names and values as their corresponding probabilities
#     pred = dict(zip(keys, values))
#     # Select maximum probability
#     max_prob = max(pred.items(), key=operator.itemgetter(1))[1]
#     # Sort in descending and select top 3 values
#     res = dict(sorted(pred.items(), key = itemgetter(1), reverse = True)[:3]) 
#     res['max_prob'] = max_prob
#     # Create final response with top 3 probabilities and their RequestTypes
#     predictedRequestType1 = list(res.keys())[0]
#     predictedRequestType2 = list(res.keys())[1]
#     predictedRequestType3 = list(res.keys())[2]
#     predictedProbability1 = list(res.values())[0]
#     predictedProbability2 = list(res.values())[1]
#     predictedProbability3 = list(res.values())[2]
#     predictedMaxProb = max_prob
#     response = {
#                 'PredictedRequestType1':predictedRequestType1,
#                 'PredictedRequestType2':predictedRequestType2,
#                 'PredictedRequestType3':predictedRequestType3,
#                 'PredictedProbability1':predictedProbability1,
#                 'PredictedProbability2':predictedProbability2,
#                 'PredictedProbability3':predictedProbability3,
#                 'MaxProb':predictedMaxProb
#                 }
#     # Return response dict            
#     return response

# @app.route('/random')
# def random_pet():
#     """
#     A cute furry animal endpoint.
#     Get a random pet
#     ---
#     description: Get a random pet
#     responses:
#         200:
#             description: A pet to be returned
#             schema:
#                 $ref: '#/definitions/Pet'
#     """
#     pet = {'category': [{'id': 1, 'name': 'rodent'}], 'name': 'Mickey'}
#     return jsonify(PetSchema().dump(pet).data)

# template = spec.to_flasgger(
#     app,
#     definitions=[CategorySchema, PetSchema],
#     paths=[random_pet]
# )

# # POST method
# @app.route('/predict_file',methods=["POST"])
# def predict_email_file():

#     # POST spec for flassger
#     """POST Method
#     This is using docstrings for specifications.
#     ---
#     tags:
#       - Email Classification Model API
#     parameters:
#       - name: file
#         in: formData
#         type: file
#         required: true
      
#     responses:
#         200:
#             description: The output values
        
#     """

#     # Read csv sent as POST request 
#     df_test=pd.read_csv(request.files.get("file"))
#     # Preprocess the csv file
#     df_clean = preprocess(df_test)
#     learn.data.add_test(df_clean['OriginalEmailBody'])
#     # Get predictions
#     prob_preds = learn.get_preds(ds_type=DatasetType.Test, ordered=True)
#     # Return predictions
#     return str(list(prob_preds))

# template = spec.to_flasgger(
#     app
# )

# # start Flasgger using a template from apispec
# swag = Swagger(app, template=template)

# if __name__=='__main__':
#     # app.run(host='0.0.0.0',port=8000)
#     app.run(debug=False)
    

import pickle
import json
import numpy 
import time

# from sklearn.linear_model import Ridge
from joblib import load

# from azureml.core.model import Model
# #from azureml.monitoring import ModelDataCollector

# def init():
#     global model

#     print ("model initialized" + time.strftime("%H:%M:%S"))
#     model_path = Model.get_model_path(model_name = 'saps_classification')
#     model = load(model_path)
    
def run(raw_data):
    try:
        data = json.loads(raw_data)["data"]
        data = numpy.array(data)
        learn.data.add_test(data)
        result = learn.get_preds(ds_type=DatasetType.Test, ordered=True)
#         result = model.predict(data)
        return result
#         return json.dumps({"result": result})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
