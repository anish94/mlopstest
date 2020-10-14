import torch 
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
# Reference:- https://course.fast.ai/videos/?lesson=8
# importing fastai dependencies
from fastai.text import TextLMDataBunch, language_model_learner, text_classifier_learner, AWD_LSTM, TextClasDataBunch, DatasetType
from fastai.callbacks import SaveModelCallback

# importing preprocessing script
from clean import preprocess 

from azureml.core import Workspace, Datastore, Dataset
from azureml.core.run import Run
from azureml.core.model import Model

run = Run.get_context()
exp = run.experiment
ws = run.experiment.workspace

print("Loading training data...")
# datastore = ws.get_default_datastore()
# datastore_paths = [(datastore, 'latest_10052020/latest_10052020.csv')]
# traindata = Dataset.Tabular.from_delimited_files(path=datastore_paths)
# print(traindata.take(1).to_pandas_dataframe())
df = pd.read_csv('https://mlopswsestoragee95af9d45.blob.core.windows.net/azureml-blobstore-dffebbb2-d010-49e5-8d64-023ab918a731/latest_10052020/latest_10052020.csv?sp=r&st=2020-10-14T17:04:59Z&se=2020-10-15T01:04:59Z&spr=https&sv=2019-12-12&sr=b&sig=OppxbeEScaIXWS22qs57YT%2FqE7QWBukVm%2BJHA1aypxk%3D')
# df = traindata.to_pandas_dataframe()
print("Columns:", df.columns) 
print("Diabetes data set dimensions : {}".format(df.shape))

# Read training data
# df = pd.read_excel('training-data/08302020.xlsx')
# Preprocessing Subject and OriginalEmailBody (Cleaned result is stored back into OriginalEmailBody)
clean_df = preprocess(df)

# Grouping categories with less than 3% data into 'Secondary'
value_list = [ 'Other', 'Chargeback Request', 'Chargeback Workflow Follow-up',  'Freight Deductions Issue', 'Invoice Submission', 'Miscellaneous Deduction Information','Payment status on Invoice',  'Statement' ]

clean_df.loc[~clean_df["RequestType"].isin(value_list), "RequestType"] = "Secondary"

# Creating dataframe with only RequestType and OriginalEmailBody. To be used for modelling.
train = pd.concat([clean_df['RequestType'], clean_df['OriginalEmailBody']], axis = 1)
train=train[~(train.OriginalEmailBody=='')]
# train.to_csv('training-data/train.csv', index = False)

# split data into training and validation set
df_trn, df_val = train_test_split(train, test_size = 0.3, random_state = 12)

# Loading data with fastai TextLMDataBunch
# data_lm = TextLMDataBunch.from_csv('training-data/', 'train.csv', valid_pct = 0.25, bs = 32)
data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, path = "")

# Loading pretrained Wikitext103 language model in fastai
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)

# Unfreeze the weights and biases to fine tune the model using our corpus
learn.unfreeze()
# Fit using learning rate 1e-2 (This learning rate is obtained from learn.recorder.plot(skip_end=15,suggestion=True) )
learn.fit_one_cycle(10, 1e-2, moms=(0.8,0.7))
# Save the encoder (model trained above)
encoder_folder = './encoder_files'
os.makedirs(encoder_folder, exist_ok=True)
encoder_filename = "fine_tuned_enc"
encoder_path = os.path.join(encoder_folder, encoder_filename)
learn.save_encoder(encoder_path)

# Classifier:
# We train the 
# Create a new data object that only grabs the labelled data and keeps those labels:-
data_clas = TextClasDataBunch.from_csv('training-data/', 'train.csv', valid_pct = 0.25, vocab = data_lm.vocab)

# Create a model to classify those emails and load the encoder saved before.
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn.load_encoder(encoder_path)

callbacks = SaveModelCallback(learn,monitor="accuracy", mode="max", name="best_lang_model")

# Train the model
learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))

learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))

learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))

learn.unfreeze()
learn.fit_one_cycle(15, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7), callbacks=[callbacks])

# Save Final model
# learn.export(file = 'nlp-lang-1.pkl')

# Save model as part of the run history
print("Exporting the model as pickle file...")
outputs_folder = './model'
os.makedirs(outputs_folder, exist_ok=True)

model_filename = "nlp-lang-1.pkl"
model_path = os.path.join(outputs_folder, model_filename)
learn.save(model_path)

# upload the model file explicitly into artifacts
print("Uploading the model into run artifacts...")
run.upload_file(name="./outputs/models/" + model_filename, path_or_stream=model_path)
print("Uploaded the model {} to experiment {}".format(model_filename, run.experiment.name))
dirpath = os.getcwd()
print(dirpath)
print("Following files are uploaded ")
print(run.get_file_names())

run.complete()
