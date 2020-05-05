# -*- coding: utf-8 -*-
import copy
import math
import os
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import scipy as sci
import sklearn as sk

####################################################################################################
import mlmodels


####################################################################################################
def get_recursive_files(folderPath, ext='/*model*/*.py'):
  import glob
  files = glob.glob( folderPath + ext, recursive=True) 
  return files


def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    print(sjump, sspace, s, sspace, flush=True)



####################################################################################################
def os_package_root_path(filepath, sublevel=0, path_add=""):
    """
       get the module package root folder
    """
    from pathlib import Path
    path = Path(os.path.realpath(filepath)).parent
    for i in range(1, sublevel + 1):
        path = path.parent

    path = os.path.join(path.absolute(), path_add)
    return path



def os_file_current_path():
  val = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
  # val = Path().absolute()

  val = str(os.path.join(val, ""))
  # print(val)
  return val



def model_get_list(folder=None, block_list=[]):
  # Get all the model.py into folder  
  folder = os_package_root_path(__file__) if folder is None else folder
  # print(folder)
  module_names = get_recursive_files(folder, r'/*model*/*.py' )                       
  print(module_names)

  NO_LIST = [  "init", "util", "preprocess" ]
  NO_LIST = NO_LIST + block_list

  list_select = []
  for t in module_names :
      #t = t.replace(folder, "").replace("\\", ".")
      flag = False     
      for x in NO_LIST :
        if x in t: 
          flag = True
          break

      if not flag  :
       list_select.append( t )
  return list_select

 
def find_in_list(x, llist) :
   flag = False
   for l in llist :
     if x in l : return True
   return False



def code_check(sign_list=None, model_list=None) :
  """
    Signatures check
  """
  flag0 = None
  for m in model_list :
    print( "\n", m )
    with open(m, mode="r") as f :
      lines = f.readlines()
      flag = False
      for s in sign_list :
        flag = find_in_list(s, lines)
        if not flag :
           print("Error", s)
           flag0= False  if flag0 is None else flag0

  return flag0


def main():
  print("os.getcwd", os.getcwd())
  print(np, np.__version__) 
  print(mlmodels) 


  path = mlmodels.__path__[0]
  
  model_list = model_get_list(folder=None, block_list=[])
  print(model_list)


  ##### Signature Check
  sign_list = [
     "def fit(model, data_pars=None, compute_pars=None, out_pars=None, out_pars=None,",
     "def predict(model, sess=None, data_pars=None, out_pars=None, compute_pars=None",
     "def save(model, path)",
     "def load(path)",
     "def get_dataset(data_pars=None, **kw)",
     "def get_params(choice=''",
     "def test(data_path=",
  ]

  flag =   code_check(sign_list, model_list)
  print(flag)



if __name__ == "__main__":
    main()



