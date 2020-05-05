# CLI tools: package provide below tools 

### How to use Command Line
```bash
### ml_models --do  
    model_list  :  list all models in the repo                            
    testall     :  test all modules inside model_tf
    test        :  test a certain module inside model_tf
    fit         :  wrap fit generic m    ethod
    predict     :  predict  using a pre-trained model and some data
    generate_config  :  generate config file from code source
    
  ## do fit  
  --model_uri     model_tf.1_lstm
  --save_folder   myfolder/
  --config_file   myfile.json
  --config_mode   "test"


  ## do predict  
  --load_folder   mymodel_folder/
    "testall"     :  test all modules inside model_tf
    "test"        :  test a certain module inside model_tf


    "model_list"  :  #list all models in the repo          
    "fit"         :  wrap fit generic m    ethod
    "predict"     :  predict  using a pre-trained model and some data
    "generate_config"  :  generate config file from code source


### ml_optim --do
    test      :  Test the hyperparameter optimization for a specific model
    test_all  :  TODO, Test all
    search    :  search for the best hyperparameters of a specific model


### ml_test
  "search"    :  search for the best hyperparameters of a specific model




### Command line tool sample

#### generate config file
    ml_models  --do generate_config  --model_uri model_tf.1_lstm.py  --save_folder "c:\myconfig"

#### TensorFlow LSTM model
    ml_models  --model_uri model_tf/1_lstm.py  --do test

#### PyTorch models
    ml_models  --model_uri model_tch/mlp.py  --do test
    
    
#### Custom  Models
    ml_models --do test  --model_uri ("D:\_devs\Python01\gitdev\mlmodels\mlmodels\model_tf_lstm.py")



#### Model param search test
    ml_optim --do test


#### For normal optimization search method
    ml_optim --do search --ntrials 1  --config_file optim_config.json --optim_method normal
    ml_optim --do search --ntrials 1  --config_file optim_config.json --optim_method prune  ###### for pruning method

    ml_optim --modelname model_tf.1_lstm.py  --do test
    ml_optim --modelname model_tf.1_lstm.py  --do search


```


# Distributed training on Pytorch Horovod
```
#### Distributed Pytorch on CPU (using Horovod and MPI on Linux, 4 processes)  in model_tch/mlp.py
    mlmodels/distri_torch_mpirun.sh   4    model_tch.mlp    mymodel.json


```



