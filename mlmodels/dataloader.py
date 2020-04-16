# -*- coding: utf-8 -*-
"""



https://www.tensorflow.org/api_docs/python/tf/compat/v1
tf.compat.v1   IS ALL TF 1.0

tf.compat.v2    iS TF 2.0



Typical user workflow

def get_dataset(data_pars):
    loader = DataLoader(data_pars)
    loader.compute()
    data = loader.get_data()
    [print(x.shape) for x in data]
    return data




"""
#### System utilities
import os
import sys
import inspect
from urllib.parse import urlparse
import json
from importlib import import_module
import pandas as pd
import numpy as np
from collections.abc import MutableMapping
from functools import partial

#possibly replace with keras.utils.get_file down the road?
#### It dowloads from HTTP from Dorpbox, ....  (not urgent)
from cli_code.cli_download import Downloader 

from sklearn.model_selection import train_test_split
import cloudpickle as pickle


#########################################################################
#### mlmodels-internal imports
from util import load_callable_from_dict



#########################################################################
#### Specific packages   ##### Be ware of tensorflow version
#### I fpossible, we dont use to have dependance on tensorflow, torch, ...


"""  Not used
import tensorflow as tf
import torch
import torchtext
import keras

import tensorflow.data
"""






#########################################################################
def pickle_load(file):
    return pickle.load(open(f, " r"))


def pickle_dump(t,path):
    pickle.dump(t, open(path, "wb" ))


def image_dir_load(path):
    return ImageDataGenerator().flow_from_directory(path)


def batch_generator(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]



def _interpret_input_pars(self, input_pars):
    try:
        path = input_pars["path"]
    except KeyError:
        raise Exception('Missing path key in the dataloader.')

    path_type = input_pars.get("path_type", None)
    if path_type is None:
        if os.path.isfile(path):
            path_type = "file"
        if os.path.isdir(path):
            path_type = "dir"

        if urlparse(path).scheme != "":
            path_type = "url"
            download_path = input_pars.get("download_path", "./")

        if path_type == "dropbox":
            dropbox_download(path)
            path_type = "file"

        if path_type is None:
            raise Exception(f'Path type for {path} is undeterminable')

    elif path_type != "file" and path_type != "dir" and path_type != "url":
        raise Exception('Unknown location type')

    file_type = input_pars.get("file_type", None)
    if file_type is None:
        if path_type == "dir":
            file_type = "image_dir"
        elif path_type == "file":
            file_type = os.path.splitext(path)[1]
        else:
            if path[-1] == "/":
                raise Exception('URL must target a single file.')
            file_type = os.path.splittext(path.split("/")[-1])[1]

    self.path = path
    self.path_type = path_type
    self.file_type = file_type
    self.test_size = input_pars.get("test_size", None)
    self.generator = input_pars.get("generator", False)
    if self.generator:
       self.batch_size = int(input_pars.get("batch_size", 1))

    self._names        = input_pars.get("names", None) #None by default. (Possibly rename for clarity?)
    self.col_Xinput    = input_pars.get('col_Xinput',None)
    self.col_Yinput    = input_pars.get('col_Yinput',None)
    self.col_miscinput = input_pars.get('col_miscinput',None)
    validation_split_function = [
        {"uri": "sklearn.model_selection::train_test_split", "args": {}},
        "test_size",
    ]
    self.validation_split_function = input_pars.get(
        "split_function", validation_split_function
    )
        

def _check_output_shape(self, inter_output, shape, max_len):
    case = 0
    if isinstance(inter_output, tuple):
        if not isinstance(inter_output[0], dict):
            case = 1
        else:
            case = 2
    if isinstance(inter_output, dict):
        if not isinstance(tuple(inter_output.values())[0], dict):
            case = 3
        else:
            case = 4
    #max_len enforcement
    if max_len is not None:
        try:
            if case == 0:
                inter_output = inter_output[0:max_len]
            if case == 1:
                inter_output = [o[0:max_len] for o in inter_output]
            if case == 3:
                inter_output = {
                    k: v[0:max_len] for k, v in inter_output.items()
                }
        except:
            pass
    # shape check
    if shape is not None:
        if (
            case == 0
            and hasattr(inter_output, "shape")
            and tuple(shape) != inter_output.shape
        ):
            raise Exception(f'Expected shape {tuple(shape)} does not match shape data shape {inter_output.shape[1:]}')
        if case == 1:
            for s, o in zip(shape, inter_output):
                if hasattr(o, "shape") and tuple(s) != o.shape[1:]:
                    raise Exception(f'Expected shape {tuple(shape)} does not match shape data shape {inter_output.shape[1:]}')
        if case == 3:
            for s, o in zip(shape, tuple(inter_output.values())):
                if hasattr(o, "shape") and tuple(s) != o.shape[1:]:
                    raise Exception(f'Expected shape {tuple(shape)} does not match shape data shape {inter_output.shape[1:]}')
    self.output_shape = shape
    return inter_output



class DataLoader:

    default_loaders = {
        ".csv": {"uri": "pandas::read_csv"},
        ".npy": {"uri": "numpy::load"},
        ".npz": {"uri": "np:load", "arg": {"allow_pickle": True}},
        ".pkl": {"uri": "dataloader::pickle_load"},
        "image_dir": {"uri": "dataloader::image_dir_load"},
    }
    _interpret_input_pars = _interpret_input_pars
    _check_output_shape   = _check_output_shape
    
    def __init__(self, data_pars):
        self.input_pars                = data_pars['input_pars']
        
        self.inter_output       = None
        self.inter_output_split = None
        self.final_output              = None
        self.final_output_split        = None

        self.loader                    = data_pars['loader']
        self.preprocessor              = data_pars.get('preprocessor',{})
        self.split_xy                  = data_pars.get('split_xy',{})
        self.split_train_test          = data_pars.get('split_train_test',{})
        self.save_inter_output  = data_pars.get('save_inter_output',{})
        self.output                    = data_pars.get('output',{})
        self.data_pars                 = data_pars
        
               
    def compute(self):
        #Interpret input_pars
        self._interpret_input_pars(self.input_pars)

        #Delegate loading data
        if self.loader == {}:
            self.loader = self.default_loaders[self.file_type]
        data_loader, data_loader_args, other_keys = load_callable_from_dict(self.loader,return_other_keys=True)
        if other_keys.get('pass_data_pars', True):
            loaded_data = data_loader(self.path, self.data_pars, **data_loader_args)
        else:
            loaded_data = data_loader(self.path, **data_loader_args)

            
        #Delegate data preprocessing
        preprocessor_class, preprocessor_class_args, other_keys = load_callable_from_dict(self.preprocessor, return_other_keys=True)
        if other_keys.get('pass_data_pars', True):
            self.preprocessor = preprocessor_class(self.data_pars, **preprocessor_class_args)
        else:
            preprocessor = preprocessor_class(**preprocessor_class_args)
        preprocessor.compute(loaded_data)
        self.inter_output = preprocessor.get_data()


        #Delegate data splitting
        if self.split_xy != {}:
            split_xy, split_xy_args, other_keys = load_callable_from_dict(self.split_xy,return_other_keys=True)
            if other_keys.get('pass_data_pars', True):
                self.inter_output = split_xy(self.inter_output,self.data_pars,**split_xy_args)
            else:
                self.inter_output = split_xy(self.inter_output,**split_xy_args)
                
        #Check output shape, trim to max_len
        shape = self.output.get('shape', None)
        max_len = self.output.get('max_len',None)
        self.inter_output = self._check_output_shape(self.inter_output,shape,max_len)
        

        #Delegate train-test splitting
        if self.split_train_test != {}:
            outputs_to_split = self.inter_output if self.col_miscinput is None else self.inter_output[:-1]
            split_train_test, split_train_test_args, other_keys = load_callable_from_dict(self.split_train_test, return_other_keys=True)
            if 'testsize_keyword' in other_keys.keys():
                split_train_test_args[other_keys.get('testsize_keyword','test_size')] = self.test_size
            if other_keys.get('pass_data_pars', True):
                split_out = split_train_test(*outputs_to_split, self.data_pars, **split_train_test_args)
            else:
                split_out = split_train_test(*outputs_to_split, **split_train_test_args)
            i = len(self.inter_output)
            self.inter_output_split = (split_out[0:i], split_out[i:])
            if self.col_miscinput is not None:
                self.inter_output_split = self.inter_output_split + ([self.inter_output[-1]],) #add back misc outputs
            else:
                self.inter_output_split = self.inter_output_split + ([],)
        
                
        #delegate output saving
        if self.save_inter_output != {}:
            if self.inter_output_split is None:
                outputs_to_save = self.inter_output
            else:
                outputs_to_save = self.inter_output_split
            path = self.save_inter_output['path']
            save_inter_output, save_inter_output_args, other_keys = load_callable_from_dict(self.save_inter_output['save_function'], return_other_keys=True)
            if other_keys.get('pass_data_pars', True):
                save_inter_output(outputs_to_save,path,self.data_pars,**save_inter_output_args)
            else:
                save_inter_output(outputs_to_save,path,**save_inter_output_args)
        
        #Delegate output formatting
        format_dict = self.output.get('format_func',{})
        format_func = lambda x: x
        if format_dict != {}:
            format_func, format_func_args, other_keys = load_callable_from_dict(format_dict,return_other_keys=True)
            if other_keys.get('pass_data_pars',True):
                format_func = partial(format_func, data_pars=self.data_pars, **format_func_args) 
            else:
                format_func = partial(format_func, **format_func_args) 
        if self.split_train_test is not None:
            self.final_output_split = tuple(
                format_func(o)
                for o in self.inter_output_split)
            
        else:
            self.final_output = format_func(self.inter_output)
     
    def get_data(self, intermediate=False):
        if intermediate or self.final_output is None:
            if self.inter_output_split is not None:
                return (
                    *self.inter_output_split[0],
                    *self.inter_output_split[1],
                    *self.inter_output_split[2],
                )
            
            if isinstance(self.inter_output, dict):
                return tuple(self.inter_output.values())
            return self.inter_output

        if self.final_output_split is not None:
            return (
                *self.final_output_split[0],
                *self.final_output_split[1],
                *self.final_output_split[2],
            )
        return self.final_output




### Test functions
def split_xy_from_dict(out,data_pars):
    X_c    = data_pars['input_pars'].get('col_Xinput',[])
    y_c    = data_pars['input_pars'].get('col_yinput',[])
    misc_c = data_pars['input_pars'].get('col_miscinput',[])
    X      = [out[n] for n in X_c]
    y      = [out[n] for n in y_c]
    misc   = [out[n] for n in misc_c]
    return (*X,*y,*misc)


    
    
if __name__ == "__main__":
    from models import test_module

    param_pars = {
        "choice": "json",
        "config_mode": "test",
        "data_path": "dataset/json/refactor/03_nbeats_dataloader.json",
    }
    test_module("model_tch/03_nbeats_dataloader.py", param_pars)
    


    param_pars = {
        "choice": "json",
        "config_mode": "test",
        "data_path": "dataset/json/refactor/namentity_crm_bilstm_dataloader.json",
    }
    test_module("model_keras/namentity_crm_bilstm_dataloader.py", param_pars)
    
    




