# -*- coding: utf-8 -*-
"""
 ml_test --do test_benchmark

 
For all json in Json_path_list :
   Load_json, 
   Load_model, 
   run_model,
   get_metrics, 
   add_to_dataframe
"""
import argparse
import glob
import inspect
import json
import os
import re
import sys
import numpy as np
import pandas as pd
import json
import importlib
from importlib import import_module
from pathlib import Path
from warnings import simplefilter
from datetime import datetime

####################################################################################################
from mlmodels.models import module_load
from mlmodels.util import path_norm_dict,  params_json_load
from mlmodels.util import (get_recursive_files, load_config, log, os_package_root_path, path_norm)

####################################################################################################
def get_all_json_path(json_path):
    return get_recursive_files(json_path, ext='/*.json')

def config_model_list(folder=None):
    # Get all the model.py into folder
    folder = os_package_root_path() if folder is None else folder
    # print(folder)
    module_names = get_recursive_files(folder, r'/*model*/*.py')
    mlist = []
    for t in module_names:
        mlist.append(t.replace(folder, "").replace("\\", "."))
        print(mlist[-1])

    return mlist

####################################################################################################
def metric_eval(actual=None, pred=None, metric_name="mean_absolute_error"):
    metric = getattr(importlib.import_module("sklearn.metrics"), metric_name)
    return metric(actual, pred)

def preprocess_timeseries_m5(data_path=None, dataset_name=None, pred_length=10, item_id=None):
    data_path = path_norm(data_path)
    df         = pd.read_csv(data_path + dataset_name)
    col_to_del = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    temp_df    = df.drop(columns=col_to_del).copy()

    # 1, -1 are hardcoded because we have to explicitly mentioned days column 
    temp_df    = pd.melt(temp_df, id_vars=["id"], value_vars=temp_df.columns[1: -1])

    log("# select one itemid for which we have to forecast")
    i_id       = item_id
    temp_df    = temp_df.loc[temp_df["id"] == i_id]
    temp_df.rename(columns={"variable": "Day", "value": "Demand"}, inplace=True)

    log("# making df to compatible 3d shape, otherwise cannot be reshape to 3d compatible form")
    pred_length = pred_length
    temp_df     = temp_df.iloc[:pred_length * (temp_df.shape[0] // pred_length)]
    temp_df.to_csv( f"{data_path}/{i_id}.csv", index=False)



####################################################################################################
def benchmark_run(bench_pars=None, args=None, config_mode="test"):
      
    dataset_uri  = args.data_path + f"{args.item_id}.csv"
    json_path    = path_norm( args.path_json )
    output_path  = path_norm( args.path_out )

    metric_list  = bench_pars['metric_list']
    bench_df     = pd.DataFrame(columns=["date_run", "model_uri", "json",
                                         "dataset_uri", "metric", "metric_name"])

    if ".json" in json_path :
       ### All config in ONE BIG JSON 
       ddict     = json.load(open(json_path, mode='r'))
       json_list = [  x for k,x in ddict.items() ]


    else :    
       ### All configs in Separate files 
       json_list = []
       json_list_tmp = get_all_json_path(json_path)
       for jsonf in json_list_tmp :
          ddict = json.load(open( path_norm(jsonf), mode='r'))[config_mode]
          json_list.append(ddict) 



    if len(json_list) < 1 :
        raise Exception("empty model list json")
    
    log("Model List", json_list)
    ii = -1
    for js in json_list :
        log ( f"### Running {jsonf} #####")
        try : 
            log("#### Model URI and Config JSON")
            #config_path = path_norm(jsonf)
            #model_pars, data_pars, compute_pars, out_pars = params_json_load(config_path, config_mode= config_mode)

            model_pars, data_pars, compute_pars, out_pars = js['model_pars'], js['data_pars'], js['compute_pars'], js['out_pars'] 

            model_uri    =  model_pars['model_uri']            
            print(model_pars)

            log("#### Setup Model   ############################################# ")
            module    = module_load(model_uri)   # "model_tch.torchhub.py"
            model     = module.Model(model_pars, data_pars, compute_pars)
            
            log("#### Fit  #######################################################")
            data_pars["train"] = True
            print(">>>model: ", model, type(model))
            model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          


            log("#### Inference Need return ypred, ytrue #########################")
            data_pars["train"] = False
            ypred, ytrue = module.predict(model=model, session=session,
                                          data_pars=data_pars, compute_pars=compute_pars, 
                                          out_pars=out_pars, return_ytrue=1)   

            ytrue = np.array(ytrue).reshape(-1, 1)
            ypred = np.array(ypred).reshape(-1, 1)
            log("### Calculate Metrics          ")
            for metric in metric_list:
                ii = ii + 1
                metric_val = metric_eval(actual=ytrue, pred=ypred,  metric_name=metric)
                bench_df.loc[ii, "date_run"]    = str(datetime.now())
                bench_df.loc[ii, "model_uri"]   = model_uri
                bench_df.loc[ii, "json"]        = str([model_pars, data_pars, compute_pars ])
                bench_df.loc[ii, "dataset_uri"] = dataset_uri
                bench_df.loc[ii, "metric_name"] = metric
                bench_df.loc[ii, "metric"]      = metric_val
                log( bench_df.loc[ii,:])
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            log( jsonf, e)

    log( f"benchmark file saved at {output_path}")  
    os.makedirs( output_path, exist_ok=True)
    bench_df.to_csv( f"{output_path}/benchmark.csv", index=False)
    return bench_df




####################################################################################################
############CLI Command ############################################################################
def cli_load_arguments(config_file=None):
    """
        Load CLI input, load config.toml , overwrite config.toml by CLI Input
    """
    if config_file is None:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        config_file = os.path.join(cur_path, "config/benchmark_config.json")
    p = argparse.ArgumentParser()

    def add(*w, **kw):
        p.add_argument(*w, **kw)
    
    add("--config_file", default=config_file, help="Params File")
    add("--config_mode", default="test", help="test/ prod /uat")
    add("--log_file",    default="ztest/benchmark/mlmodels_log.log", help="log.log")

    add("--do",          default="vision_fashion_mnist", help="do ")

    ### Benchmark config
    add("--benchmark_json", default="dataset/json/benchmark.json", help=" benchmark config")
    add("--path_json",      default="dataset/json/benchmark_cnn/", help=" list of json")
    add("--path_out",       default="example/benchmark/", help=".")


    #### Input dataset
    add("--data_path",   default="dataset/timeseries/", help="Dataset path")
    add("--dataset_name",default="sales_train_validation.csv", help="dataset name")   


    #### Specific to timeseries
    add("--item_id",     default="HOBBIES_1_001_CA_1_validation", help="forecast for which item")

    arg = p.parse_args()
    return arg



def main():
    arg = cli_load_arguments()

    if arg.do == "preprocess_v1":
        arg.data_path    = "dataset/timeseries/"
        arg.dataset_name = "sales_train_validation.csv"
        preprocess_timeseries_m5(data_path    = arg.data_path, 
                                 dataset_name = arg.dataset_name, 
                                 pred_length  = 100, item_id=arg.item_id)   


    elif arg.do == "timeseries":
        log("Time series model")
        bench_pars = {"metric_list": ["mean_absolute_error", "mean_squared_error",
                                       "median_absolute_error",  "r2_score"], 
                      "pred_length": 100,
                      
                      #### Over-ride data
                      "data_pars" : {
                         "train_data_path": "dataset/timeseries/stock/qqq_us_train.csv",
                         "test_data_path": "dataset/timeseries/stock/qqq_us_test.csv",
                         "col_Xinput": ["Close"],
                         "col_ytarget": "Close"
                      }

                      }

        arg.data_path    = ""
        arg.dataset_name = ""
        arg.path_json    = "dataset/json/benchmark_timeseries/test01/"
        arg.path_out     = "example/benchmark/timeseries/test01/"

        log(benchmark_run(bench_pars, arg)) 


    elif arg.do == "vision_mnist":
        log("Vision models")

        arg.data_path    = ""
        arg.dataset_name = ""
        arg.path_json    = "dataset/json/benchmark_cnn/mnist"
        arg.path_out     = "example/benchmark/cnn/mnist"

        bench_pars = {"metric_list": ["accuracy_score"]}
        log(benchmark_run(bench_pars=bench_pars, args=arg))


    elif arg.do == "vision_fashion_mnist":
        log("Vision models")

        arg.data_path    = ""
        arg.dataset_name = ""
        arg.path_json    = "dataset/json/benchmark_cnn/fashion_mnist"
        arg.path_out     = "example/benchmark/cnn/fashion_mnist/"

        bench_pars = {"metric_list": ["accuracy_score"]}
        log(benchmark_run(bench_pars=bench_pars, args=arg))


    elif arg.do == "nlp_reuters":
        """
           User Reuters datasts
           config files in  "dataset/json/benchmark_text/"

        """
        log("NLP Reuters")
        arg.data_path    = ""
        arg.dataset_name = ""
        arg.path_json    = "dataset/json/benchmark_text/"
        arg.path_out     = "example/benchmark/text/"

        bench_pars = {"metric_list": ["accuracy, f1_score"]}
        log(benchmark_run(bench_pars=bench_pars, args=arg))


    elif arg.do == "custom":
        log("Custom benchmark")
        bench_pars = json.load(open( arg.benchmark_json, mode='r'))
        log(bench_pars['metric_list'])
        log(benchmark_run(bench_pars=bench_pars, args=arg))


    elif arg.do == "text_classification":
        log("text_classification")
        arg.data_path = ""
        arg.dataset_name = ""
        arg.path_json = "dataset/json/benchmark_text_classification/"
        arg.path_out = "example/benchmark/text_classification/"

        bench_pars = {"metric_list": ["accuracy_score"]}
        log(benchmark_run(bench_pars=bench_pars, args=arg))


    else :
        raise Exception("No options")

if __name__ == "__main__":
    main()




