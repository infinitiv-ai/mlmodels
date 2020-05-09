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
from pathlib import Path
import numpy as np
from time import sleep
####################################################################################################


####################################################################################################
import mlmodels

from mlmodels.util import get_recursive_files, log, os_package_root_path, model_get_list, os_get_file
from mlmodels.util import get_recursive_files2, path_norm, path_norm_dict



           
####################################################################################################
def os_bash(cmd):
  # os_bash("dir ")  
  import subprocess  
  try :
    l = subprocess.run( cmd, stdout=subprocess.PIPE, shell=True, ).stdout.decode('utf-8')
    return l
  except :
    return ""



def log_separator():
   print("\n" * 5, "*" * 120, flush=True )


def log_info_repo(arg=None):
   """
      Grab Github Variables
      https://help.github.com/en/actions/configuring-and-managing-workflows/using-environment-variables    

   """ 
   #print( "Check", os_bash(  "echo $GITHUB_REF" ),  os_bash(  "echo $GITHUB_REPOSITORY" ),  os_bash(  "echo $GITHUB_SHA" )  )
   # repo = arg.repo
   # sha  = arg.sha 

   repo      =  os_bash(  "echo $GITHUB_REPOSITORY" )
   sha       =  os_bash(  "echo $GITHUB_SHA" )
   workflow  =  os_bash(  "echo $GITHUB_WORKFLOW" )
   branch    =  os_bash(  "echo $GITHUB_REF" )


   repo     = repo.replace("\n", "").replace("\r", "").strip()
   workflow = workflow.replace("\n", "").replace("\r", "").strip()
   sha      = sha.replace("\n", "").replace("\r", "").strip()
   branch   = branch.replace("\n", "").replace("\r", "").strip()

   github_repo_url = f"https://github.com/{repo}/tree/{sha}"
   url_branch_file = "https://github.com/{repo}/blob/{branch}/" 

   log_separator()
   print("\n" * 1, "******** TAG : ", workflow, repo, sha, branch , flush=True)
   print("\n" * 1, "******** GITHUB_WOKFLOW : " + f"https://github.com/{repo}/actions?query=workflow%3A{workflow}" , flush=True)
   print("\n" * 1, "******** GITHUB_REPO_URL : "   + github_repo_url , flush=True)
   print("\n" * 1, "******** GITHUB_COMMIT_URL : " + f"https://github.com/{repo}/commit/{sha}" , flush=True)
   print("\n" * 1, "*" * 120 )

   ### Export
   dd = { k: locals()[k]  for k in [ "github_repo_url", "url_branch_file", "repo", "branch", "sha", "workflow"  ]}
   return dd



def to_logfile(prefix="", dateformat='+%Y-%m-%d_%H:%M:%S,%3N' ) : 
    ### On Linux System
    if dateformat == "" :
           return  f"  2>&1 | tee -a  cd log_{prefix}.txt"     

    return  f"  2>&1 | tee -a  cd log_{prefix}_$(date {dateformat}).txt"



def os_file_current_path():
    import inspect, os
    val = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    val = str(val)
    # val = Path().absolute()
    # val = str(os.path.join(val, ""))
    # log(val)
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
   log(cmd)
   os.system(cmd)



def log_remote_push(arg=None):
   ### Pushing to mlmodels_store 
   tag ="ml_store"
   s = f""" cd /home/runner/work/mlmodels/mlmodels_store/
           git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"        
           git pull --all   
           ls &&  git add --all &&  git commit -m "{tag}" 
           git push --all
           cd /home/runner/work/mlmodels/mlmodels/
       """

   cmd = " ; ".join(s.split("\n"))
   log(cmd)
   os.system(cmd)



####################################################################################################
def test_model_structure():
    log("os.getcwd", os.getcwd())
    log(mlmodels)

    path = mlmodels.__path__[0]

    log("############Check structure ############################")
    cmd = f"ztest_structure.py"
    os.system(cmd)


def test_import(arg=None):
    #import tensorflow as tf
    #import torch

    #log(np, np.__version__)
    #log(tf, tf.__version__)    #### Import internally Create Issues
    #log(torch, torch.__version__)
    #log(mlmodels)

    from importlib import import_module

    log_info_repo(arg)

    block_list = ["raw"]
    log_separator()
    log("test_import")

    file_list = os_get_file(folder=None, block_list=[], pattern=r"/*.py")
    print(file_list)

    for f in file_list:
        try:
            f = "mlmodels." + f.replace("\\", ".").replace(".py", "").replace("/", ".")

            import_module(f)
            print(f)
        except Exception as e:
            log("Error", f, e)



def test_jupyter(arg=None, config_mode="test_all"):
    """
      Tests files in mlmodels/example/

    """
    #log("os.getcwd", os.getcwd())
    git = log_info_repo(arg)

    root = os_package_root_path()
    root = root.replace("\\", "//")
    #log(root)


    path = str( os.path.join(root, "example/") )
    print(path)

    print("############ List of files ################################")
    #model_list = get_recursive_files2(root, r'/*/*.ipynb')
    model_list  = get_recursive_files2(path, r'*.ipynb')
    model_list2 = get_recursive_files2(path, r'*.py')
    model_list  = model_list + model_list2
    # log(model_list)


    ## Block list
    #cfg = json.load(open( path_norm(arg.config_file) , mode='r'))[ config_mode ]
    #block_list = cfg.get('jupyter_blocked', [])
    
    block_list = [ "ipynb_checkpoints" ] 
    model_list = [t for t in model_list if t not in block_list]

    test_list = [f"ipython {t}"  for  t in model_list]
    print(test_list, flush=True) 

    log_separator()
    print("############ Running Jupyter files ################################")
    for cmd in test_list:
        log_separator()
        print( cmd.replace("/home/runner/work/mlmodels/mlmodels/", git.get("url_branch_file", "")), "\n", flush=True)
        os.system(cmd)






def test_benchmark(arg=None):
    log_info_repo(arg)
    # log("os.getcwd", os.getcwd())

    path = mlmodels.__path__[0]
    log("############Check model ################################")
    path = path.replace("\\", "//")
    test_list = [ f"python {path}/benchmark.py --do timeseries "   ,
                  f"python {path}/benchmark.py --do vision_mnist "   ,
                  f"python {path}/benchmark.py --do fashion_vision_mnist "   ,
                  f"python {path}/benchmark.py --do text_classification "   ,
                  f"python {path}/benchmark.py --do nlp_reuters "   ,

    ]

    for cmd in test_list:
        log_separator()
        print( cmd)
        os.system(cmd)




def test_cli(arg=None):
    log("# Testing Command Line System  ")
    log_info_repo(arg)

    import mlmodels, os
    path = mlmodels.__path__[0]   ### Root Path
    # if arg is None :           
    #  fileconfig = path_norm( f"{path}/config/cli_test_list.md" ) 
    # else :
    #  fileconfig = path_norm(  arg.config_file )

    # fileconfig = path_norm( f"{path}/../config/cli_test_list.md" ) 
    fileconfig = path_norm( f"{path}/../README_usage_CLI.md" ) 
    log(fileconfig)

    def is_valid_cmd(cmd) :
       cmd = cmd.strip() 
       if len(cmd) > 8 :
          if cmd.startswith("ml_models ") or cmd.startswith("ml_benchmark ") or cmd.startswith("ml_optim ")  :
              return True
       return False 
             

    with open( fileconfig, mode="r" ) as f:
        cmd_list = f.readlines()
    log(cmd_list[:5])


    #### Parse the CMD from the file .md and Execute
    for ss in cmd_list:                      
        cmd = ss.strip()
        if is_valid_cmd(cmd):
          cmd =  cmd  + to_logfile("cli", '+%Y-%m-%d_%H')
          log_separator()
          print( cmd)
          os.system(cmd)


def test_pullrequest(arg=None):
    """
      Scan files in /pullrequest/ and run test on it.


    """
    log_info_repo(arg)

    from pathlib import Path
    # log("os.getcwd", os.getcwd())
    path = str( os.path.join(Path(mlmodels.__path__[0] ).parent , "pullrequest/") )
    log(path)

    log("############Check model ################################")
    file_list = get_recursive_files(path , r"*.py" )
    log(file_list)

    ## Block list
    block_list = []
    test_list = [t for t in file_list if t not in block_list]
    log("Used", test_list)
    

    log("########### Run Check ##############################")
    test_import(arg=None)
    sleep(20)
    os.system("ml_optim")
    os.system("ml_mlmodels")


   
    for file in test_list:
        file = file +  to_logfile(prefix="", dateformat='' ) 
        cmd = f"python {file}"
        log_separator()
        log( cmd)
        os.system(cmd)
        sleep(5)

    
    #### Check the logs   ###################################
    with open("log_.txt", mode="r")  as f :
       lines = f.readlines()

    for x in lines :
        if "Error" in x :
           raise Exception(f"Unknown dataset type", x)
  




def test_dataloader(arg=None):
    log_info_repo(arg)
    # log("os.getcwd", os.getcwd())
    path = mlmodels.__path__[0]
    cfg  = json_load(path_norm(arg.config_file))

    log("############Check model ################################")
    path = path.replace("\\", "//")
    test_list = [ f"python {path}/dataloader.py --do test "   ,
    ]

    for cmd in test_list:
          log_separator()
          log( cmd)
          os.system(cmd)





def test_json_all(arg):
    log_info_repo(arg)
    # log("os.getcwd", os.getcwd())
    root = os_package_root_path()
    root = root.replace("\\", "//")
    log(root)
    path = str( os.path.join(root, "dataset/json/") )
    log(path)

    log("############ List of files ################################")
    #model_list = get_recursive_files2(root, r'/*/*.ipynb')
    model_list  = get_recursive_files2(path, r'/*/.json')
    model_list2 = get_recursive_files2(path, r'/*/*.json')
    model_list  = model_list + model_list2
    print("List of JSON Files", model_list)


    for js_file in model_list:
        log("\n\n\n", "************", "JSON File", js_file)
        cfg = json.load(open(js_file, mode='r'))
        for kmode, ddict in cfg.items():
            cmd = f"ml_models --do fit --config_file {js_file}  --config_mode {kmode} "   
            log_separator()
            log( cmd)
            os.system(cmd)






def test_all(arg=None):
    log_info_repo(arg)
    from time import sleep
    # log("os.getcwd", os.getcwd())

    path = mlmodels.__path__[0]
    log("############Check model ################################")
    model_list = model_get_list(folder=None, block_list=[])
    log(model_list)

    ## Block list
    # root = os_package_root_path()
    cfg = json.load(open( path_norm(arg.config_file), mode='r'))['test_all']
    block_list = cfg['model_blocked']
    model_list = [t for t in model_list if t not in block_list]
    log("Used", model_list)

    path = path.replace("\\", "//")
    test_list = [f"python {path}/" + t.replace(".", "//") + ".py" for t in model_list]

    for cmd in test_list:
        log_separator()
        log( cmd)
        os.system(cmd)
        log_remote_push()
        sleep(5)




def test_json(arg):
    log_info_repo(arg)
    log("os.getcwd", os.getcwd())

    path = mlmodels.__path__[0]
    cfg = json.load(open(arg.config_file, mode='r'))

    mlist = cfg['model_list']
    log(mlist)
    test_list = [f"python {path}/{model}" for model in mlist]

    for cmd in test_list:
          log_separator()
          log( cmd)
          os.system(cmd)



def test_list(mlist):
    #log("os.getcwd", os.getcwd())

    path = mlmodels.__path__[0]
    # mlist = str_list.split(",")
    test_list = [f"python {path}/{model}" for model in mlist]

    for cmd in test_list:
          log_separator()
          log( cmd)
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
    from mlmodels.util import load_config, path_norm
    
    config_file =  path_norm( "config/test_config.json" ) if config_file is None  else config_file
    log(config_file)

    p = argparse.ArgumentParser()
    def add(*w, **kw):
        p.add_argument(*w, **kw)

    add("--do"          , default="test_all"  , help="  Action to do")
    add("--config_file" , default=config_file , help="Params File")
    add("--config_mode" , default="test"      , help="test/ prod /uat")
    add("--log_file"    , help="log.log")
    add("--folder"      , default=None        , help="test")

    ##### model pars

    ##### data pars

    ##### compute pars

    ##### out pars
    add("--save_folder", default="ztest/", help=".")

    #### Env Vars :
    """
     https://help.github.com/en/actions/configuring-and-managing-workflows/using-environment-variables
    """
    add("--repo" , default="GITHUB_REPOSITORT"      , help="test/ prod /uat")
    add("--sha" , default="GITHUB_SHA"      , help="test/ prod /uat")
    add("--ref" , default="GITHUB_REF"      , help="test/ prod /uat")
    add("--workflow" , default="GITHUB_WORKFLOW"      , help="test/ prod /uat")


    # add("--event_name" , default="test"      , help="test/ prod /uat")
    #add("--event_path" , default="test"      , help="test/ prod /uat")
    # add("--workspace" , default="test"      , help="test/ prod /uat")


    arg = p.parse_args()
    # arg = load_config(arg, arg.config_file, arg.config_mode, verbose=0)
    return arg



def main():
    arg = cli_load_arguments()
    log(arg.do, arg.repo, arg.sha)

    #### Input is String list of model name
    if ".py" in arg.do:
        s = arg.do
        test_list(s.split(","))

    else:
        log("Running command", arg.do)
        globals()[arg.do](arg)


if __name__ == "__main__":
    main()


