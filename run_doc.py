# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""


"""
import io
import os
import subprocess
import sys

from setuptools import find_packages, setup

######################################################################################
root = os.path.abspath(os.path.dirname(__file__))



##### Version  #######################################################################
# from setup import version, entry_points
print("start Doc")



######################################################################################
#with open("README.md", "r") as fh:
#    long_description = fh.read()


#################################################################################################
des1 = """


#######################################################################################
### ④ Interface

models.py 
```
   module_load(model_uri)
   model_create(module)
   fit(model, module, session, data_pars, out_pars   )
   metrics(model, module, session, data_pars, out_pars)
   predict(model, module, session, data_pars, out_pars)
   save(model, path)
   load(model)
```

optim.py
```
   optim(modelname="model_tf.1_lstm.py",  model_pars= {}, data_pars = {}, compute_pars={"method": "normal/prune"}
       , save_folder="/mymodel/", log_folder="", ntrials=2) 

   optim_optuna(modelname="model_tf.1_lstm.py", model_pars= {}, data_pars = {}, compute_pars={"method" : "normal/prune"},
                save_folder="/mymodel/", log_folder="", ntrials=2) 
```

#### Generic parameters 
```
   Define in models_config.json
   model_params      :  Relative to model definition 
   compute_pars      :  Relative to  the compute process
   data_pars         :  Relative to the input data
   out_pars          :  Relative to outout data
```
   Sometimes, data_pars is required to setup the model (ie CNN with image size...)
   

## In Jupyter 

#### Model, data, ... definition
```python
model_uri    = "model_tf.1_lstm.py"
model_pars   = {  "num_layers": 1,
                  "size": ncol_input, "size_layer": 128, "output_size": ncol_output, "timestep": 4,
                }
data_pars    = {"data_path": "/folder/myfile.csv"  , "data_type": "pandas" }
compute_pars = { "learning_rate": 0.001, }
out_pars     = { "path": "ztest_1lstm/", "model_path" : "ztest_1lstm/model/"}
save_pars    = { "path" : "ztest_1lstm/model/" }
load_pars    = { "path" : "ztest_1lstm/model/" }


```


#### Using local module (which contain the model)
```python

from mlmodels.models import module_load

module        =  module_load( model_uri= model_uri )                           # Load file definition
model         =  module.Model(model_pars, data_pars, compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars, compute_pars, out_pars)          # fit the model
metrics_val   =  module.fit_metrics( model, sess, data_pars, compute_pars, out_pars) # get stats
module.save(model, sess, save_pars)



#### Inference
model, sess = module.load(load_pars)    #Create Model instance
ypred       = module.predict(model, sess,  data_pars, compute_pars, out_pars)     # predict pipeline


```




###### Using Generic API : Common to all models, models.py methods
```python

from mlmodels.models import module_load, create_model, fit, predict, stats
from mlmodels.models import load  # Load model weights

module        =  module_load( model_uri= model_uri )                           # Load file definition
model         =  model_create(module, model_pars, data_pars, compute_pars)     # Create Model instance
model, sess   =  fit(model, data_pars, compute_pars, out_pars)                 # fit the model
metrics_val   =  fit_metrics( model, sess, data_pars, compute_pars, out_pars)  # get stats

save(model, sess, save_pars)



#### Inference
load_pars = { "path" : "ztest_1lstm/model/", "model_type": "model_tf" }

model, sess  = load( load_pars )       # Create Model instance
ypred        = predict(model, module, sess,  data_pars, compute_pars, out_pars)     





```

"""



##########################################################################################
### Packages  ####################################################
packages = ["mlmodels"] + ["mlmodels." + p for p in find_packages("mlmodels")]




#########################################################################################
def os_package_root_path(add_path="",n=0):
  from pathlib import Path
  add_path = os.path.join(Path(__file__).parent.absolute(), add_path)
  # print("os_package_root_path,check", add_path)
  return add_path


def get_recursive_files(folderPath, ext='/*model*/*.py'):
  import glob
  files = glob.glob( folderPath + ext, recursive=True) 
  return files


# Get all the model.py into folder  
folder = None
folder = os_package_root_path() if folder is None else folder
# print(folder)
module_names = get_recursive_files(folder, r'/*model*//*model*/*.py' )                       




des = """
#### Model list 

```
--model_uri


"""
for t in module_names :
    t = t.replace(folder, "").replace("\\", ".")

    if "__init__.py" in t  :
      des = des  + "\n\n"
    else  :    
      if  not 'util' in  t and not 'preprocess' in t :
        des = des + str(t).replace("mlmodels.", "" ) + "\n" 

des = des + """
```

"""

   





#########################################################################################
################ Print on file
with open("README_model_list.md", mode="w") as f :
  f.writelines(des1)
  f.writelines(des)
