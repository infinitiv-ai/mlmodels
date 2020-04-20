# CLI tools: package provide below tools 

- ml_models
- ml_optim    
- ml_test



### How to use Command Line
```bash
ml_models --do  
    model_list  :  list all models in the repo                            
    testall     :  test all modules inside model_tf
    test        :  test a certain module inside model_tf
    fit         :  wrap fit generic method
    predict     :  predict  using a pre-trained model and some data
    generate_config  :  generate config file from code source
```

> **ml_models**

#### --do fit
```bash
  --model_uri     model_tf.1_lstm
  --save_folder   myfolder/
  --config_file   myfile.json
  --config_mode   "test"
```    

#### --do predict
```bash
    --load_folder   mymodel_folder/
    "testall"     :  test all modules inside model_tf
    "test"        :  test a certain module inside model_tf
    
    
    "model_list"  :  #list all models in the repo          
    "fit"         :  wrap fit generic m    ethod
    "predict"     :  predict  using a pre-trained model and some data
    "generate_config"  :  generate config file from code source
```

> **ml_optim**

```bash
ml_optim --do
    test      :  Test the hyperparameter optimization for a specific model
    test_all  :  TODO, Test all
    search    :  search for the best hyperparameters of a specific model
```

> **ml_test**

```bash
ml_test
  "search"    :  search for the best hyperparameters of a specific model

```


### Command line tool sample

#### generate config file
```bash
ml_models  --do generate_config  --model_uri model_tf.1_lstm.py  --save_folder "c:\myconfig"
```
    

#### TensorFlow LSTM model
```bash
ml_models  --model_uri model_tf/1_lstm.py  --do test
```
    

#### PyTorch models
```bash
ml_models  --model_uri model_tch/mlp.py  --do test
```
    
#### Custom  Models
```bash
ml_models --do test  --model_uri "D:\_devs\Python01\gitdev\mlmodels\mlmodels\model_tf_lstm.py"
```

#### Model param search test
```bash
ml_optim --do test
```

#### For normal optimization search method
```bash
ml_optim --do search --ntrials 1  --config_file optim_config.json --optim_method normal
ml_optim --do search --ntrials 1  --config_file optim_config.json --optim_method prune  ###### for pruning method

ml_optim --modelname model_tf.1_lstm.py  --do test
ml_optim --modelname model_tf.1_lstm.py  --do search
```



# Distributed training on Pytorch Horovod

#### Distributed Pytorch on CPU (using Horovod and MPI on Linux, 4 processes)  in model_tch/mlp.py
```bash
mlmodels/distri_torch_mpirun.sh   4    model_tch.mlp    mymodel.json
```


# Using Generic API : Common to all models
```python

from mlmodels.models import module_load, create_model, fit, predict, stats
from mlmodels.models import load #Load model weights

module        =  module_load( model_uri= model_uri )                           # Load file definition
model         =  model_create(module, model_pars, data_pars, compute_pars)     # Create Model instance
model, sess   =  fit(model, data_pars, compute_pars, out_pars)                 # fit the model
metrics_val   =  fit_metrics( model, sess, data_pars, compute_pars, out_pars)  # get stats

save(save_pars)



#### Inference
load_pars = { "path" : "ztest_1lstm/model/" }

module      = module_load( model_uri= model_uri )     # Load file definition
model,sess  = load(folder, model_type="model_tf")      # Create Model instance
ypred       = predict(model, module, sess,  data_pars, compute_pars, out_pars)     
```


