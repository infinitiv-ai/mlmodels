# -*- coding: utf-8 -*-
"""
All test are located here

ml_test --do test_json --config_file test/pullrequest.json

ml_test --do test_all  --config_file  mlmdodels/config/test_config.json"

"""
import copy
import math
import os
from collections import Counter, OrderedDict
import json

import numpy as np
####################################################################################################


####################################################################################################
import mlmodels

from mlmodels.util import get_recursive_files, log, os_package_root_path, model_get_list, os_get_file
from mlmodels.util import get_recursive_files2, path_norm, path_norm_dict



           
####################################################################################################
def log_git_push() :
 return " ; git config --local user.email 'noelkev0@gmail.com' &&   git config --local user.name 'arita37'  &&  cd /home/runner/work/mlmodels/mlmodels_store/   && ls &&  git add --all &&  git commit -m 'log'   && git push --all   && cd /home/runner/work/mlmodels/mlmodels/ "


def to_logfile(prefix="", dateformat='+%Y-%m-%d_%H:%M:%S,%3N' ) : 
    ### On Linux System
    if dateformat == "" :
           return  f"  2>&1 | tee -a  cd log_{prefix}.txt"     

    return  f"  2>&1 | tee -a  cd log_{prefix}_$(date {dateformat}).txt"



def os_file_current_path():
    import inspect
    val = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # return current_dir + "/"
    # Path of current file
    # from pathlib import Path

    # val = Path().absolute()
    val = str(os.path.join(val, ""))
    # print(val)
    return val


def os_system(cmd, dolog=1, prefix="", dateformat='+%Y-%m-%d_%H:%M:%S,%3N') :
    if dolog :
        cmd = cmd + to_logfile(prefix, dateformat)
    os.system(cmd)




def json_load(path) :
  try :
    return json.load(open( path, mode='r'))
  except :
    return {}  

####################################################################################################
def log_remote_start(arg=None):
   ## Download remote log on disk 
   s = """ cd /home/runner/work/mlmodels/  && git clone git@github.com:arita37/mlmodels_store.git  &&  ls && pwd
       """

   cmd = " ; ".join(s.split("\n"))
   print(cmd, flush=True)
   os.system(cmd)



def log_remote_push(arg=None):
   ### Pushing to mlmodels_store 
   s = """ cd /home/runner/work/mlmodels/mlmodels_store/
           git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"        
           git pull --all   
           ls &&  git add --all &&  git commit -m "cli_store" 
           git push --all
           cd /home/runner/work/mlmodels/mlmodels/
       """

   cmd = " ; ".join(s.split("\n"))
   print(cmd, flush=True)
   os.system(cmd)





####################################################################################################
def test_model_structure():
    print("os.getcwd", os.getcwd())
    print(mlmodels)

    path = mlmodels.__path__[0]

    print("############Check structure ############################")
    cmd = f"ztest_structure.py"
    os.system(cmd)


def test_import(arg=None):
    #import tensorflow as tf
    #import torch

    #print(np, np.__version__)
    #print(tf, tf.__version__)    #### Import internally Create Issues
    #print(torch, torch.__version__)
    #print(mlmodels)

    from importlib import import_module

    block_list = ["raw"]

    file_list = os_get_file(folder=None, block_list=[], pattern=r"/*.py")
    print(file_list)
    for f in file_list:
        try:
            f = "mlmodels." + f.replace("\\", ".").replace(".py", "").replace("/", ".")

            import_module(f)
            print(f)
        except Exception as e:
            print("Error", f, e)




def test_jupyter(arg=None, config_mode="test_all"):
    print("os.getcwd", os.getcwd())

    root = os_package_root_path()
    root = root.replace("\\", "//")
    print(root)

    print("############Check model ################################")
    model_list = get_recursive_files2(root, r'/*/*.ipynb')
    print(model_list)


    ## Block list
    cfg = json.load(open( f"{arg.config_file}", mode='r'))[ config_mode ]
    block_list = cfg.get('jupyter_blocked', [])
    model_list = [t for t in model_list if t not in block_list]
    print("Used", model_list)

    test_list = [f"ipython {root}/{t}"  for  t in model_list]

    for cmd in test_list:
        print("\n\n\n", flush=True)
        print(cmd, flush=True)
        os.system(cmd)





def test_benchmark(arg=None):
    print("os.getcwd", os.getcwd())

    path = mlmodels.__path__[0]
    print("############Check model ################################")
    path = path.replace("\\", "//")
    test_list = [ f"python {path}/benchmark.py --do timeseries "   ,
                  f"python {path}/benchmark.py --do vision_mnist "   ,
                  f"python {path}/benchmark.py --do fashion_vision_mnist "   ,
                  f"python {path}/benchmark.py --do text_classification "   ,
                  f"python {path}/benchmark.py --do nlp_reuters "   ,

    ]

    for cmd in test_list:
        print("\n\n\n", flush=True)
        print(cmd, flush=True)
        os.system(cmd)





def test_cli(arg=None):
    print("# Testing Command Line System  ")

    import mlmodels, os
    path = mlmodels.__path__[0]   ### Root Path
    # if arg is None :           
    #  fileconfig = path_norm( f"{path}/config/cli_test_list.md" ) 
    # else :
    #  fileconfig = path_norm(  arg.config_file )

    # fileconfig = path_norm( f"{path}/../config/cli_test_list.md" ) 
    fileconfig = path_norm( f"{path}/../README_usage_CLI.md" ) 
    print(fileconfig)

    def is_valid_cmd(cmd) :
       cmd = cmd.strip() 
       if len(cmd) > 8 :
          if cmd.startswith("ml_models ") or cmd.startswith("ml_benchmark ") or cmd.startswith("ml_optim ")  :
              return True
       return False 
             

    with open( fileconfig, mode="r" ) as f:
        cmd_list = f.readlines()
    print(cmd_list[:5])



    #### Parse the CMD from the file .md and Execute
    for ss in cmd_list:                      
        cmd = ss.strip()
        if is_valid_cmd(cmd):
          cmd =  cmd  + to_logfile("cli", '+%Y-%m-%d_%H')
          print("\n\n\n", cmd ,  flush=True)
          os.system(cmd)



def test_pullrequest(arg=None):
    """
      Scan files in /pullrequest/ and run test on it.


    """
    from pathlib import Path
    print("os.getcwd", os.getcwd())
    path = str( os.path.join(Path(mlmodels.__path__[0] ).parent , "pullrequest/") )
    print(path)

    print("############Check model ################################")
    file_list = get_recursive_files(path , r"*.py" )
    print(file_list)

    ## Block list
    block_list = []
    test_list = [t for t in file_list if t not in block_list]
    print("Used", test_list)
    

    print("########### Run Check ##############################")
    test_import(arg=None)
    os.system("ml_optim")
    os.system("ml_mlmodels")


   
    for file in test_list:
        file = file +  to_logfile(prefix="", dateformat='' ) 
        cmd = f"python {file}"
        print("\n\n\n",cmd, flush=True)
        os.system(cmd)

    #### Check the logs
    with open("log_.txt", mode="r")  as f :
       lines = f.readlines()

    for x in lines :
        if "Error" in x :
           raise Exception(f"Unknown dataset type", x)
  




def test_dataloader(arg=None):
    print("os.getcwd", os.getcwd())
    path = mlmodels.__path__[0]
    cfg  = json_load(path_norm(arg.config_file))

    print("############Check model ################################")
    path = path.replace("\\", "//")
    test_list = [ f"python {path}/dataloader.py --do test "   ,
    ]

    for cmd in test_list:
        print("\n\n\n", cmd, flush=True)
        os.system(cmd)






def test_all(arg=None):
    print("os.getcwd", os.getcwd())

    path = mlmodels.__path__[0]
    print("############Check model ################################")
    model_list = model_get_list(folder=None, block_list=[])
    print(model_list)

    ## Block list
    # root = os_package_root_path()
    cfg = json.load(open( path_norm(arg.config_file), mode='r'))['test_all']
    block_list = cfg['model_blocked']
    model_list = [t for t in model_list if t not in block_list]
    print("Used", model_list)

    path = path.replace("\\", "//")
    test_list = [f"python {path}/" + t.replace(".", "//").replace("//py", ".py") for t in model_list]

    for cmd in test_list:
        cmd = cmd + log_git_push()
        print("\n\n\n",cmd, flush=True)
        os.system(cmd)



def test_json(arg):
    print("os.getcwd", os.getcwd())
    print("############Check model ################################")
    path = mlmodels.__path__[0]
    cfg = json.load(open(arg.config_file, mode='r'))

    mlist = cfg['model_list']
    print(mlist)
    test_list = [f"python {path}/{model}" for model in mlist]

    for cmd in test_list:
        print("\n\n\n", flush=True)
        print(cmd, flush=True)
        os.system(cmd)


def test_list(mlist):
    print("os.getcwd", os.getcwd())
    print("############Check model ################################")
    path = mlmodels.__path__[0]
    # mlist = str_list.split(",")
    test_list = [f"python {path}/{model}" for model in mlist]

    for cmd in test_list:
        print("\n\n\n", flush=True)
        print(cmd, flush=True)
        os.system(cmd)


def test_custom():
    test_list0 = [
        ### Tflow
        f"model_tf/namentity_crm_bilstm.py",

        ### Keras
        f"model_keras/01_deepctr.py",
        f"model_keras/charcnn.py",

        f"model_keras/01_deepctr.py",
        f"model_keras/textcnn.py",

        ### SKLearn
        f"model_sklearn/sklearn.py",

        ### Torch
        f"model_tch/03_nbeats.py",
        f"model_tch/textcnn.py",
        f"model_tch/transformer_classifier.py",

        ### Glueon
        f"model_gluon/gluon_deepar.py",
        f"model_glufon/gluon_ffn.py",
    ]
    test_list(test_list0)




#################################################################################################################
#################################################################################################################
def cli_load_arguments(config_file=None):
    #Load CLI input, load config.toml , overwrite config.toml by CLI Input
    import argparse
    from mlmodels.util import load_config, path_norm, os_package_root_path
    if config_file is None  :
      config_file =  path_norm( "config/test_config.json" )
    print(config_file)

    p = argparse.ArgumentParser()

    def add(*w, **kw):
        p.add_argument(*w, **kw)

    add("--do"            , default="test_all"  , help="  Action to do")
    add("--config_file" , default=config_file , help="Params File")
    add("--config_mode" , default="test"      , help="test/ prod /uat")
    add("--log_file"    , help="log.log")
    add("--folder"      , default=None        , help="test")

    ##### model pars

    ##### data pars

    ##### compute pars

    ##### out pars
    add("--save_folder", default="ztest/", help=".")

    arg = p.parse_args()
    # arg = load_config(arg, arg.config_file, arg.config_mode, verbose=0)
    return arg



def main():
    arg = cli_load_arguments()
    print(arg.do)

    #### Input is String list of model name
    if ".py" in arg.do:
        s = arg.do
        test_list(s.split(","))

    else:
        print("Running command", arg.do, flush=True)
        globals()[arg.do](arg)


if __name__ == "__main__":
    main()


