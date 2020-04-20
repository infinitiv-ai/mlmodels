
## In Jupyter 



##########################################################################################
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
   

   #######################################################################################
   ### ⑥ Naming convention
   
   ### Function naming
   ```
   pd_   :  input is pandas dataframe
   np_   :  input is numpy
   sk_   :  inout is related to sklearn (ie sklearn model), input is numpy array
   plot_


   col_ :  function name for column list related.
   ```
   
   



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


#### Model list 

```
--model_uri






mlmodels.model_gluon.gluon_automl.py
mlmodels.model_gluon.gluon_deepar.py
mlmodels.model_gluon.gluon_ffn.py
mlmodels.model_gluon.gluon_prophet.py


mlmodels.model_keras.01_deepctr.py
mlmodels.model_keras.02_cnn.py
mlmodels.model_keras.armdn.py
mlmodels.model_keras.charcnn.py
mlmodels.model_keras.charcnn_zhang.py
mlmodels.model_keras.keras_gan.py
mlmodels.model_keras.namentity_crm_bilstm.py
mlmodels.model_keras.namentity_crm_bilstm_dataloader.py
mlmodels.model_keras.nbeats.py
mlmodels.model_keras.textcnn.py
mlmodels.model_keras.textvae.py




mlmodels.model_sklearn.model_lightgbm.py
mlmodels.model_sklearn.model_sklearn.py


mlmodels.model_tch.02_mlp.py
mlmodels.model_tch.Autokeras.py
mlmodels.model_tch.matchzoo_models.py
mlmodels.model_tch.nbeats.py
mlmodels.model_tch.pplm.py
mlmodels.model_tch.pytorch_vae.py
mlmodels.model_tch.textcnn.py
mlmodels.model_tch.textcnn_dataloader.py
mlmodels.model_tch.torchhub.py
mlmodels.model_tch.transformer_classifier.py
mlmodels.model_tch.transformer_sentence.py


mlmodels.model_tf.1_lstm.py



```

