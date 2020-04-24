# coding: utf-8
"""
Generic template for new model.
Check parameters template in models_config.json

"model_pars":   { "learning_rate": 0.001, "num_layers": 1, "size": 6, "size_layer": 128, "output_size": 6, "timestep": 4, "epoch": 2 },
"data_pars":    { "data_path": "dataset/GOOG-year.csv", "data_type": "pandas", "size": [0, 0, 6], "output_size": [0, 6] },
"compute_pars": { "distributed": "mpi", "epoch": 10 },
"out_pars":     { "out_path": "dataset/", "data_type": "pandas", "size": [0, 0, 6], "output_size": [0, 6] }



"""
import inspect
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd

import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
from keras.datasets import imdb
####################################################################################################

######## Logs
from mlmodels.util import os_package_root_path, log, path_norm, get_model_uri



#### Import EXISTING model and re-map to mlmodels
from mlmodels.model_keras.raw.textcnn_.text_cnn import TextCNN


VERBOSE = False
MODEL_URI = get_model_uri(__file__)



####################################################################################################
class Model:
  def __init__(self, model_pars=None, data_pars=None, compute_pars=None
               ):
    ### Model Structure        ################################
    maxlen         = model_pars['maxlen']
    max_features   = model_pars['max_features']
    embedding_dims = model_pars['embedding_dims']

    self.model = TextCNN(maxlen, max_features, embedding_dims).get_model()

    self.model.compile(compute_pars['engine'],  # adam 
    	               compute_pars['loss'], 
    	               metrics= compute_pars['metrics'])
    self.model.summary()
    


def fit(model, data_pars=None, compute_pars=None, out_pars=None,   **kw):
  """
  """
  
  batch_size = compute_pars['batch_size']
  epochs = compute_pars['epochs']
  
  sess = None # 
  Xtrain, Xtest, ytrain, ytest = get_dataset(data_pars)


  early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
  model.model.fit(Xtrain, ytrain,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[early_stopping],
          validation_data=(Xtest, ytest))

  return model, sess



def fit_metrics(model, data_pars=None, compute_pars=None, out_pars=None,  **kw):
    """
       Return metrics ofw the model when fitted.
    """
    ddict = {}
    
    return ddict

    

def predict(model, sess=None, data_pars=None, out_pars=None, compute_pars=None, **kw):
  ##### Get Data ###############################################
  data_pars['train'] = False
  X, ytrue = get_dataset(data_pars)

  #### Do prediction
  ypred = model.model.predict(X)

  ### Return val
  if kw.get("return_ytrue"):
    return ypred, ytrue
  else:
      return ypred, None


  
  
def reset_model():
  pass





def save(model=None, session=None, save_pars={}):
    from mlmodels.util import save_keras
    print(save_pars)
    save_keras(model, session, save_pars)
     


def load(load_pars={}):
    from mlmodels.util import load_keras
    print(load_pars)
    model =  load_keras(load_pars)
    return model, None




####################################################################################################
def get_dataset(data_pars=None, **kw):
  """
    JSON data_pars to get dataset
    "data_pars":    { "data_path": "dataset/GOOG-year.csv", "data_type": "pandas",
    "size": [0, 0, 6], "output_size": [0, 6] },
  """


  print('Loading data...')
  max_features = data_pars['max_features']
  maxlen       = data_pars['maxlen']

  if data_pars['train'] :

    ### Remove Keras download --> csv on disk
    (Xtrain, ytrain), (Xtest, ytest) = imdb.load_data(num_words=max_features)

    print('Pad sequences (samples x time)...')
    Xtrain = sequence.pad_sequences(Xtrain, maxlen=maxlen)
    Xtest  = sequence.pad_sequences(Xtest, maxlen=maxlen)


    return Xtrain, Xtest, ytrain, ytest 


  else :
     (Xtrain, ytrain), (Xtest, ytest) = imdb.load_data(num_words=max_features)
     Xtest = sequence.pad_sequences(Xtest, maxlen=maxlen)
     return Xtest, ytest 




def get_params(param_pars={}, **kw):
    import json
    choice      = param_pars['choice']
    config_mode = param_pars['config_mode']
    data_path   = param_pars['data_path']


    if choice == "json":
       data_path = path_norm(data_path)
       cf = json.load(open(data_path, mode='r'))
       cf = cf[config_mode]
       return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']


    if choice == "test01":
        log("#### Path params   ##########################################")
        data_path  = path_norm( "dataset/text/imdb.csv"  )   
        out_path   = path_norm( "ztest/model_keras/textcnn/model.h5" )
        model_path = out_path


        data_pars    = {"path" : data_path, "train": 1, "maxlen":40, "max_features": 5, }

        model_pars   = {"maxlen":40, "max_features": 5, "embedding_dims":50,

                       }
                       
        compute_pars = {"engine": "adam", "loss": "binary_crossentropy", "metrics": ["accuracy"] ,
                        "batch_size": 1000, "epochs":1
                       }

        out_pars     = {"path": out_path,  "model_path": model_path}

        return model_pars, data_pars, compute_pars, out_pars

    else:
        raise Exception(f"Not support choice {choice} yet")




################################################################################################
########## Tests are  ##########################################################################
def test(data_path="dataset/", pars_choice="json", config_mode="test"):
    ### Local test

    log("#### Loading params   ##############################################")
    param_pars = {"choice":pars_choice,  "data_path":data_path,  "config_mode": config_mode}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)

    log("#### Loading dataset   #############################################")
    Xtuple = get_dataset(data_pars)


    log("#### Model init, fit   #############################################")
    session = None
    model = Model(model_pars, data_pars, compute_pars)
    model, session = fit(model, data_pars, compute_pars, out_pars)


    log("#### save the trained model  #######################################")
    save(model, session,  save_pars= out_pars)


    log("#### Predict   #####################################################")
    data_pars["train"] = 0
    ypred, _ = predict(model, session, data_pars, compute_pars, out_pars)


    log("#### metrics   #####################################################")
    metrics_val = fit_metrics(model, data_pars, compute_pars, out_pars)
    print(metrics_val)


    log("#### Plot   ########################################################")


    log("#### Save/Load   ###################################################")
    save(model, None, out_pars)
    model2 = load(out_pars)
    #     ypred = predict(model2, data_pars, compute_pars, out_pars)
    #     metrics_val = metrics(model2, ypred, data_pars, compute_pars, out_pars)
    print(model2)



if __name__ == '__main__':
    VERBOSE = True
    test_path = os.getcwd() + "/mytest/"
    
    ### Local fixed params
    test(pars_choice="test01")


    ### Local json file
    # test(pars_choice="json")


    ####    test_module(model_uri="model_xxxx/yyyy.py", param_pars=None)
    from mlmodels.models import test_module
    param_pars = {'choice': "test01", 'config_mode' : 'test', 'data_path' : '/dataset/' }
    test_module(model_uri = MODEL_URI, param_pars= param_pars)

    ##### get of get_params
    # choice      = pp['choice']
    # config_mode = pp['config_mode']
    # data_path   = pp['data_path']


    ####    test_api(model_uri="model_xxxx/yyyy.py", param_pars=None)
    from mlmodels.models import test_api
    param_pars = {'choice': "test01", 'config_mode' : 'test', 'data_path' : '/dataset/' }
    test_api(model_uri = MODEL_URI, param_pars= param_pars)






"""
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
epochs = 10

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)...')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = TextCNN(maxlen, max_features, embedding_dims).get_model()
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[early_stopping],
          validation_data=(x_test, y_test))

print('Test...')
result = model.predict(x_test)
"""

