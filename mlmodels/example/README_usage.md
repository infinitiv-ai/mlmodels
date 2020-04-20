## Usage with Online Editor
  Pre-Installed setup

https://github.com/arita37/mlmodels/issues/101



#
#

## Steps to add a new Colab notebook /Jupyter notebook :

```
0) Check existing examples
    https://github.com/arita37/mlmodels/tree/dev/mlmodels/example


1) Create a branch from DEV branch called : notebook_

2) Create Jupyter Notebook in  mlmodels/example/           
            
3) Create mymodel.json in  mlmodels/example/
 
4)  Do Pull Request to dev Branch.

```



## Existing JSON definition for models :

  https://github.com/arita37/mlmodels/tree/dev/mlmodels/dataset/json

  https://github.com/arita37/mlmodels/tree/dev/mlmodels/example/


  #### Access to json files inside mlmodels :
```python

from mlmodels.util import path_norm_dict, path_norm

print( path_norm( 'example/hyper_titanic_randomForest.json'  ) )
  ###  --> /home/ubuntu/mlmodels/example/hyper ...

print( path_norm( 'dataset/text/mytest.txt'  ) )
  ###  --> /home/ubuntu/mlmodels/dataset/text ...

```



## Example of sample cripts
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example/arun_hyper.py

https://github.com/arita37/mlmodels/blob/dev/mlmodels/example/arun_model.py



#
#




## Example of notebooks



### Progressive GAN , Image Generation with mlmodels

https://github.com/arita37/mlmodels/issues/168






### Hyper-Parameter with LightGBM, Ex 1

https://github.com/arita37/mlmodels/blob/dev/mlmodels/example/model_lightgbm_home_retail.ipynb

#

---

### Hyper-Parameter with LightGBM , Ex 2

https://github.com/arita37/mlmodels/blob/dev/mlmodels/example/model_lightgbm_glass.ipynb


#
---
### Hyper-Parameter with LightGBM , Ex 3

https://github.com/arita37/mlmodels/blob/dev/mlmodels/example/model_lightgbm_titanic.ipynb



#
---



### LSTM example in TensorFlow ([Example notebook](mlmodels/example/1_lstm.ipynb))

#### Define model and data definitions
```python
# import library
import mlmodels


model_uri    = "model_tf.1_lstm.py"
model_pars   =  {  "num_layers": 1,
                  "size": ncol_input, "size_layer": 128, "output_size": ncol_output, "timestep": 4,
                }
data_pars    =  {"data_path": "/folder/myfile.csv"  , "data_type": "pandas" }
compute_pars =  { "learning_rate": 0.001, }

out_pars     =  { "path": "ztest_1lstm/", "model_path" : "ztest_1lstm/model/"}
save_pars = { "path" : "ztest_1lstm/model/" }
load_pars = { "path" : "ztest_1lstm/model/" }



#### Load Parameters and Train
from mlmodels.models import module_load

module        =  module_load( model_uri= model_uri )                           # Load file definition
model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)


#### Inference
metrics_val   =  module.fit_metrics( model, sess, data_pars, compute_pars, out_pars) # get stats
ypred         = module.predict(model, sess,  data_pars, compute_pars, out_pars)     # predict pipeline


```


---

### AutoML example in Gluon ([Example notebook](mlmodels/example/gluon_automl.ipynb))
```python
# import library
import mlmodels
import autogluon as ag

#### Define model and data definitions
model_uri = "model_gluon.gluon_automl.py"
data_pars = {"train": True, "uri_type": "amazon_aws", "dt_name": "Inc"}

model_pars = {"model_type": "tabular",
              "learning_rate": ag.space.Real(1e-4, 1e-2, default=5e-4, log=True),
              "activation": ag.space.Categorical(*tuple(["relu", "softrelu", "tanh"])),
              "layers": ag.space.Categorical(
                          *tuple([[100], [1000], [200, 100], [300, 200, 100]])),
              'dropout_prob': ag.space.Real(0.0, 0.5, default=0.1),
              'num_boost_round': 10,
              'num_leaves': ag.space.Int(lower=26, upper=30, default=36)
             }

compute_pars = {
    "hp_tune": True,
    "num_epochs": 10,
    "time_limits": 120,
    "num_trials": 5,
    "search_strategy": "skopt"
}

out_pars = {
    "out_path": "dataset/"
}



#### Load Parameters and Train
from mlmodels.models import module_load

module        =  module_load( model_uri= model_uri )                           # Load file definition
model         =  module.Model(model_pars=model_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, model_pars=model_pars, compute_pars=compute_pars, out_pars=out_pars)      


#### Inference
ypred       = module.predict(model, data_pars, compute_pars, out_pars)     # predict pipeline


```

---

### RandomForest example in Scikit-learn ([Example notebook](mlmodels/example/sklearn.ipynb))
```python
# import library
import mlmodels

#### Define model and data definitions
model_uri    = "model_sklearn.sklearn.py"

model_pars   = {"model_name":  "RandomForestClassifier", "max_depth" : 4 , "random_state":0}

data_pars    = {'mode': 'test', 'path': "../mlmodels/dataset", 'data_type' : 'pandas' }

compute_pars = {'return_pred_not': False}

out_pars    = {'path' : "../ztest"}


#### Load Parameters and Train
from mlmodels.models import module_load

module        =  module_load( model_uri= model_uri )                           # Load file definition
model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model


#### Inference
ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline
```


---

### TextCNN example in keras ([Example notebook](example/textcnn.ipynb))

```python
# import library
import mlmodels

#### Define model and data definitions
model_uri    = "model_keras.textcnn.py"

data_pars    = {"path" : "../mlmodels/dataset/text/imdb.csv", "train": 1, "maxlen":400, "max_features": 10}

model_pars   = {"maxlen":400, "max_features": 10, "embedding_dims":50}
                       
compute_pars = {"engine": "adam", "loss": "binary_crossentropy", "metrics": ["accuracy"] ,
                        "batch_size": 32, "epochs":1, 'return_pred_not':False}

out_pars     = {"path": "ztest/model_keras/textcnn/"}



#### Load Parameters and Train
from mlmodels.models import module_load

module        =  module_load( model_uri= model_uri )                           # Load file definition
model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model


#### Inference
data_pars['train'] = 0
ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
```

---

### Using json config file for input ([Example notebook](example/1_lstm_json.ipynb), [JSON file](mlmodels/mlmodels/example/1_lstm.json))

#### Import library and functions
```python
# import library
import mlmodels

#### Load model and data definitions from json
from mlmodels.models import module_load
from mlmodels.util import load_config

model_uri    = "model_tf.1_lstm.py"
module        =  module_load( model_uri= model_uri )                           # Load file definition

model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars={
    'choice':'json',
    'config_mode':'test',
    'data_path':'../mlmodels/example/1_lstm.json'
})

#### Load parameters and train
model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model

#### Check inference
ypred       = module.predict(model, sess=sess,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline


```

---

### Using Scikit-learn's SVM for Titanic Problem from json file ([Example notebook](mlmodels/example/sklearn_titanic_svm.ipynb), [JSON file](mlmodels/example/sklearn_titanic_svm.json))

#### Import library and functions
```python
# import library
import mlmodels

#### Load model and data definitions from json
from mlmodels.models import module_load
from mlmodels.util import load_config

model_uri    = "model_sklearn.sklearn.py"
module        =  module_load( model_uri= model_uri )                           # Load file definition

model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars={
    'choice':'json',
    'config_mode':'test',
    'data_path':'../mlmodels/example/sklearn_titanic_svm.json'
})

#### Load Parameters and Train

model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model


#### Inference
ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline
ypred


#### Check metrics
import pandas as pd
from sklearn.metrics import roc_auc_score

y = pd.read_csv('../mlmodels/dataset/tabular/titanic_train_preprocessed.csv')['Survived'].values
roc_auc_score(y, ypred)


```

---

### Using Scikit-learn's Random Forest for Titanic Problem from json file ([Example notebook](mlmodels/example/sklearn_titanic_randomForest.ipynb), [JSON file](mlmodels/example/sklearn_titanic_randomForest.json))

#### Import library and functions
```python
# import library
import mlmodels

#### Load model and data definitions from json
from mlmodels.models import module_load
from mlmodels.util import load_config

model_uri    = "model_sklearn.sklearn.py"
module        =  module_load( model_uri= model_uri )                           # Load file definition

model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars={
    'choice':'json',
    'config_mode':'test',
    'data_path':'../mlmodels/example/sklearn_titanic_randomForest.json'
})


#### Load Parameters and Train
model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model


#### Inference

ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline
ypred

#### Check metrics
import pandas as pd
from sklearn.metrics import roc_auc_score

y = pd.read_csv('../mlmodels/dataset/tabular/titanic_train_preprocessed.csv')['Survived'].values
roc_auc_score(y, ypred)

```

---

### Using Autogluon for Titanic Problem from json file ([Example notebook](mlmodels/example/gluon_automl_titanic.ipynb), [JSON file](mlmodels/example/gluon_automl.json))

#### Import library and functions
```python
# import library
import mlmodels

#### Load model and data definitions from json
from mlmodels.models import module_load
from mlmodels.util import load_config

model_uri    = "model_gluon.gluon_automl.py"
module        =  module_load( model_uri= model_uri )                           # Load file definition

model_pars, data_pars, compute_pars, out_pars = module.get_params(
    choice='json',
    config_mode= 'test',
    data_path= '../mlmodels/example/gluon_automl.json'
)


#### Load Parameters and Train
model         =  module.Model(model_pars=model_pars, compute_pars=compute_pars)             # Create Model instance
model   =  module.fit(model, model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model
model.model.fit_summary()


#### Check inference
ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline

#### Check metrics
model.model.model_performance

import pandas as pd
from sklearn.metrics import roc_auc_score

y = pd.read_csv('../mlmodels/dataset/tabular/titanic_train_preprocessed.csv')['Survived'].values
roc_auc_score(y, ypred)


```

---
---

### Using hyper-params (optuna) for Titanic Problem from json file ([Example notebook](mlmodels/example/sklearn_titanic_randomForest_example2.ipynb), [JSON file](mlmodels/example/hyper_titanic_randomForest.json))

#### Import library and functions
```python
# import library
from mlmodels.models import module_load
from mlmodels.optim import optim
from mlmodels.util import params_json_load


#### Load model and data definitions from json

###  hypermodel_pars, model_pars, ....
model_uri   = "model_sklearn.sklearn.py"
config_path = path_norm( 'example/hyper_titanic_randomForest.json'  )
config_mode = "test"  ### test/prod



#### Model Parameters
hypermodel_pars, model_pars, data_pars, compute_pars, out_pars = params_json_load(config_path, config_mode= config_mode)
print( hypermodel_pars, model_pars, data_pars, compute_pars, out_pars)


module            =  module_load( model_uri= model_uri )                      
model_pars_update = optim(
    model_uri       = model_uri,
    hypermodel_pars = hypermodel_pars,
    model_pars      = model_pars,
    data_pars       = data_pars,
    compute_pars    = compute_pars,
    out_pars        = out_pars
)


#### Load Parameters and Train
model         =  module.Model(model_pars=model_pars_update, data_pars=data_pars, compute_pars=compute_pars)y
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)

#### Check inference
ypred         = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline
ypred


#### Check metrics
import pandas as pd
from sklearn.metrics import roc_auc_score

y = pd.read_csv( path_norm('dataset/tabular/titanic_train_preprocessed.csv') )
y = y['Survived'].values
roc_auc_score(y, ypred)


```


---

### Using LightGBM for Titanic Problem from json file ([Example notebook](mlmodels/example/model_lightgbm.ipynb), [JSON file](mlmodels/example/lightgbm_titanic.json))

#### Import library and functions
```python
# import library
import mlmodels
from mlmodels.models import module_load
from mlmodels.util import path_norm_dict, path_norm
import json

#### Load model and data definitions from json
# Model defination
model_uri   = "model_sklearn.model_lightgbm.py"
data_path   =  path_norm('dataset/json/lightgbm_titanic.json' ) 


# Model Parameters
pars = json.load(open( data_path , mode='r'))
for key, pdict in  pars.items() :
  globals()[key] = path_norm_dict( pdict   )   ###Normalize path

model_pars      = test['model_pars']
data_pars       = test['data_pars']
compute_pars    = test['compute_pars']
out_pars        = test['out_pars']



#### Load Parameters and Train
module        =  module_load( model_uri= model_uri)
model = module.Model(model_pars, data_pars, compute_pars) # create model instance
model, session = module.fit(model, data_pars, compute_pars, out_pars) # fit model


#### Check inference
ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # get predictions
ypred


#### Check metrics
metrics_val = module.fit_metrics(model, data_pars, compute_pars, out_pars)
metrics_val 

```

---




### Using Vision CNN RESNET18 for MNIST dataset  ([Example notebook](mlmodels/example/model_restnet18.ipynb), [JSON file](mlmodels/model_tch/torchhub_cnn.json))



```python
# import library
import mlmodels
from mlmodels.models import module_load
from mlmodels.util import path_norm_dict, path_norm, params_json_load
import json


#### Model URI and Config JSON
model_uri   = "model_tch.torchhub.py"
config_path = path_norm( 'model_tch/torchhub_cnn.json'  )
config_mode = "test"  ### test/prod





#### Model Parameters
hypermodel_pars, model_pars, data_pars, compute_pars, out_pars = params_json_load(config_path, config_mode= config_mode)
print( hypermodel_pars, model_pars, data_pars, compute_pars, out_pars)


#### Setup Model 
module         = module_load( model_uri)
model          = module.Model(model_pars, data_pars, compute_pars) 

#### Fit
model, session = module.fit(model, data_pars, compute_pars, out_pars)           #### fit model
metrics_val    = module.fit_metrics(model, data_pars, compute_pars, out_pars)   #### Check fit metrics
print(metrics_val)


#### Inference
ypred          = module.predict(model, session, data_pars, compute_pars, out_pars)   
print(ypred)




```
---



### Using ARMDN Time Series : Ass for MNIST dataset  ([Example notebook](mlmodels/example/model_timeseries_armdn.ipynb), [JSON file](mlmodels/model_keras/armdn.json))



```python
# import library
import mlmodels
from mlmodels.models import module_load
from mlmodels.util import path_norm_dict, path_norm, params_json_load
import json


#### Model URI and Config JSON
model_uri   = "model_keras.ardmn.py"
config_path = path_norm( 'model_keras/ardmn.json'  )
config_mode = "test"  ### test/prod




#### Model Parameters
hypermodel_pars, model_pars, data_pars, compute_pars, out_pars = params_json_load(config_path, config_mode= config_mode)
print( hypermodel_pars, model_pars, data_pars, compute_pars, out_pars)


#### Setup Model 
module         = module_load( model_uri)
model          = module.Model(model_pars, data_pars, compute_pars) 

#### Fit
model, session = module.fit(model, data_pars, compute_pars, out_pars)           #### fit model
metrics_val    = module.fit_metrics(model, data_pars, compute_pars, out_pars)   #### Check fit metrics
print(metrics_val)


#### Inference
ypred          = module.predict(model, session, data_pars, compute_pars, out_pars)   
print(ypred)



#### Save/Load
module.save(model, save_pars ={ 'path': out_pars['path'] +"/model/"})

model2 = module.load(load_pars ={ 'path': out_pars['path'] +"/model/"})



```
---


















