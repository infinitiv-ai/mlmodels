# -*- coding: utf-8 -*-
"""
Template for testing your code
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
from mlmodels.util import log, os_package_root_path, model_get_list, os_get_file
from mlmodels.util import path_norm, path_norm_dict




#### Import your models or
# from mlmodels.model_keras import textcnn

           
####################################################################################################
def os_file_current_path():
    import inspect
    val = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    val = str(os.path.join(val, ""))
    return val





##### Create a function called test()   ###########################################################
def test(arg=None):
    print("os.getcwd", os.getcwd())
    root = mlmodels.__path__[0]   ## Root Folder
    root = root.replace("\\", "//")
    print("############ Your custom code ################################")





    test_list = [ 
       f"python {root}/optim.py  "   
      ,f"python {root}/model_keras/textcnn.py    "   

    ]







    ##### End of Custom Code   ###########################################
    for cmd in test_list:
        print("\n\n\n",cmd, flush=True)
        os.system(cmd)







####################################################################################################
if __name__ == "__main__":
    test()





