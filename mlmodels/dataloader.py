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

# possibly replace with keras.utils.get_file down the road?
#### It dowloads from HTTP from Dorpbox, ....  (not urgent)
from cli_code.cli_download import Downloader

from sklearn.model_selection import train_test_split
import cloudpickle as pickle


#########################################################################
#### mlmodels-internal imports
from mlmodels.util import load_callable_from_dict, load_callable_from_uri, path_norm


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



DATASET_TYPES = ["csv_dataset", "text_dataset"]


#########################################################################
def pickle_load(file):
    return pickle.load(open(f, " r"))


def pickle_dump(t, **kwargs):
    with open(kwargs["path"], "wb") as fi:
        pickle.dump(t, fi)
    return t


def image_dir_load(path):
    return ImageDataGenerator().flow_from_directory(path)


def batch_generator(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def _validate_data_info(self, data_info):
    try:
        path = data_info["path"]
    except KeyError:
        raise KeyError("Missing path key in the dataloader.")

    dataset_type = data_info.get("dataset_type", None)

    if dataset_type and dataset_type not in DATASET_TYPES:
        raise Exception(f"Unknown dataset type {dataset_type}")

    self.path = path
    self.dataset_type = dataset_type
    self.test_size = data_info.get("test_size", None)
    self.generator = data_info.get("generator", False)
    self.batch_size = int(data_info.get("batch_size", 1))

    self.col_Xinput = data_info.get("col_Xinput", None)
    self.col_Yinput = data_info.get("col_Yinput", None)
    self.col_miscinput = data_info.get("col_miscinput", None)


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
    # max_len enforcement
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
            raise Exception(
                f"Expected shape {tuple(shape)} does not match shape data shape {inter_output.shape[1:]}"
            )
        if case == 1:
            for s, o in zip(shape, inter_output):
                if hasattr(o, "shape") and tuple(s) != o.shape[1:]:
                    raise Exception(
                        f"Expected shape {tuple(shape)} does not match shape data shape {inter_output.shape[1:]}"
                    )
        if case == 3:
            for s, o in zip(shape, tuple(inter_output.values())):
                if hasattr(o, "shape") and tuple(s) != o.shape[1:]:
                    raise Exception(
                        f"Expected shape {tuple(shape)} does not match shape data shape {inter_output.shape[1:]}"
                    )
    self.output_shape = shape
    return inter_output



def get_dataset_type(x) :
    from mlmodel.process.generic import PandasDataset, NumpyDataset, Dataset, kerasDataset  #Pytorch


    from mlmodel.process.generic import DataLoader


    if isinstance(x, PandasDataset  ) : return "PandasDataset"
    if isinstance(x, NumpyDataset  ) : return "NumpyDataset"
    if isinstance(x, Dataset  ) : return "pytorchDataset"



class DataLoader:

    default_loaders = {
        ".csv": {"uri": "pandas::read_csv", "pass_data_pars":False},
        ".npy": {"uri": "numpy::load", "pass_data_pars":False},
        ".npz": {"uri": "np:load", "arg": {"allow_pickle": True}, "pass_data_pars":False},
        ".pkl": {"uri": "dataloader::pickle_load", "pass_data_pars":False},

        "image_dir": {"uri": "dataloader::image_dir_load", "pass_data_pars":False},
    }
    _validate_data_info = _validate_data_info
    _check_output_shape = _check_output_shape
    
    def __init__(self, data_pars):
        self.final_output             = {}
        self.internal_states          = {}
        self.data_info                = data_pars['data_info']
        self.preprocessors            = data_pars.get('preprocessors', [])
        self.final_output_type        = data_pars['output_type'] 



    def check(self):
        # Validate data_info
        self._validate_data_info(self.data_info)

        input_tmp = self.data_info["path"]
        input_type_prev = "file"   ## HARD CODE , Bad

        for preprocessor in self.preprocessors:
            uri = preprocessor.get("uri", None)
            if not uri:
                print(f"Preprocessor {preprocessor} missing uri")


            ### Compare date type for COMPATIBILITY
            input_type = preprocessor.get("input_type", None)   ### Automatic ???
            if input_type != input_type_prev :
                print(f"Preprocessor {preprocessor} missing uri")                  

            input_type_prev = preprocessor.get('output_type', None)
       


    def compute(self, docheck=1):
        # Validate data_info
        self._validate_data_info(self.data_info)

        if docheck :
            self.check()


        input_tmp = self.data_info["path"]

        for preprocessor in self.preprocessors:
            uri = preprocessor.get("uri", None)
            if not uri:
                raise Exception(f"Preprocessor {preprocessor} missing uri")
            preprocessor_func = load_callable_from_uri(uri)

        
            args = preprocessor.get("args", {})

            ### Should match PytorchDataloader, KerasDataloader, PandasDataset, ....
            if inspect.isclass(preprocessor_func):
                obj_preprocessor = preprocessor_func(args, self.data_info)
                obj_preprocessor.compute(input_tmp)
                out_tmp = obj_preprocessor.get_data()

            else:
                pos_params = inspect.getfullargspec(preprocessor_func)[0]
                if isinstance(input_tmp, (tuple, list)) and len(input_tmp) > 0 and len(pos_params) == 0:
                   out_tmp = preprocessor_func(args, self.data_info, *input_tmp)
                else:
                    out_tmp = preprocessor_func(args, self.data_info, input_tmp) 


            ## Be vareful of Very Large Dataset, not to save ALL 
            if preprocessor.get("internal_states", None):
                for internal_state in preprocessor.get("internal_states", None):
                    if isinstance(out_tmp, dict):
                        self.internal_states[internal_state] = out_tmp[internal_state]

            input_tmp = out_tmp
        self.final_output = out_tmp

    def get_data(self):
        return self.final_output, self.internal_states


### Test functions
def split_xy_from_dict(out, **kwargs):
    X_c    = kwargs.get('col_Xinput',[])
    y_c    = kwargs.get('col_yinput',[])
    X      = [out[n] for n in X_c]
    y      = [out[n] for n in y_c]
    return (*X,*y)


if __name__ == "__main__":
    from mlmodels.models import test_module

    # param_pars = {
    #     "choice": "json",
    #     "config_mode": "test",
    #     "data_path": "dataset/json/refactor/03_nbeats_dataloader.json",
    # }
    # test_module("model_tch/03_nbeats_dataloader.py", param_pars)

    param_pars = {
        "choice": "json",
        "config_mode": "test",
        "data_path": "dataset/json/refactor/namentity_crm_bilstm_dataloader_new.json",
    }
    test_module("model_keras/namentity_crm_bilstm_dataloader.py", param_pars)
    
    
    
    
    
    
    
    
    
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
        raise Exception("Missing path key in the dataloader.")

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
            raise Exception(f"Path type for {path} is undeterminable")

    elif path_type != "file" and path_type != "dir" and path_type != "url":
        raise Exception("Unknown location type")

    file_type = input_pars.get("file_type", None)
    if file_type is None:
        if path_type == "dir":
            file_type = "image_dir"
        elif path_type == "file":
            file_type = os.path.splitext(path)[1]
        else:
            if path[-1] == "/":
                raise Exception("URL must target a single file.")
            file_type = os.path.splittext(path.split("/")[-1])[1]

    self.path = path
    self.path_type = path_type
    self.file_type = file_type
    self.test_size = input_pars.get("test_size", None)
    self.generator = input_pars.get("generator", False)
    if self.generator:
        try:
            self.batch_size = int(input_pars.get("batch_size", 1))
        except:
            raise Exception("Batch size must be an integer")
    self.col_Xinput = input_pars.get("col_Xinput", None)
    self.col_Yinput = input_pars.get("col_Yinput", None)
    self.col_miscinput = input_pars.get("col_miscinput", None)
    validation_split_function = [
        {"uri": "sklearn.model_selection::train_test_split", "arg": {}},
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
    # max_len enforcement
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
            raise Exception(
                f"Expected shape {tuple(shape)} does not match shape data shape {inter_output.shape[1:]}"
            )
        if case == 1:
            for s, o in zip(shape, inter_output):
                if hasattr(o, "shape") and tuple(s) != o.shape[1:]:
                    raise Exception(
                        f"Expected shape {tuple(shape)} does not match shape data shape {inter_output.shape[1:]}"
                    )
        if case == 3:
            for s, o in zip(shape, tuple(inter_output.values())):
                if hasattr(o, "shape") and tuple(s) != o.shape[1:]:
                    raise Exception(
                        f"Expected shape {tuple(shape)} does not match shape data shape {inter_output.shape[1:]}"
                    )
    self.output_shape = shape
    return inter_output


class DataLoader:

    default_loaders = {
        ".csv": {"uri": "pandas::read_csv", "pass_data_pars":False},
        ".npy": {"uri": "numpy::load", "pass_data_pars":False},
        ".npz": {"uri": "np:load", "arg": {"allow_pickle": True}, "pass_data_pars":False},
        ".pkl": {"uri": "dataloader::pickle_load", "pass_data_pars":False},
        "image_dir": {"uri": "dataloader::image_dir_load", "pass_data_pars":False},
    }
    _validate_data_info = _validate_data_info
    _check_output_shape = _check_output_shape
    
    def __init__(self, data_pars):
        self.final_output             = {}
        self.internal_states          = {}
        self.data_info                = data_pars['data_info']
        self.preprocessors            = data_pars.get('preprocessors', [])

    def compute(self):
        # Validate data_info
        self._validate_data_info(self.data_info)

        input_tmp = None
        for preprocessor in self.preprocessors:
            uri = preprocessor.get("uri", None)
            if not uri:
                raise Exception(f"Preprocessor {preprocessor} missing uri")

            name = preprocessor.get("name", None)
            args = preprocessor.get("args", {})

            preprocessor_func = load_callable_from_uri(uri)
            if name == "loader":
                out_tmp = preprocessor_func(self.path, **args)
            elif name == "saver":  # saver do not return output
                preprocessor_func(self.path, **args)
            else:
                if inspect.isclass(preprocessor_func):
                    obj_preprocessor = preprocessor_func(**args)
                    obj_preprocessor.compute(input_tmp)
                    out_tmp = obj_preprocessor.get_data()
                else:
                    if isinstance(input_tmp, (tuple, list)):
                        out_tmp = preprocessor_func(*input_tmp[:2], **args)
                    else:
                        out_tmp = preprocessor_func(input_tmp, **args)
            if preprocessor.get("internal_states", None):
                for internal_state in preprocessor.get("internal_states", None):
                    if isinstance(out_tmp, dict):
                        self.internal_states[internal_state] = out_tmp[internal_state]

            input_tmp = out_tmp
        self.final_output = out_tmp

    def get_data(self):
        return self.final_output, self.internal_states



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
    from mlmodels.models import test_module

    # param_pars = {
    #     "choice": "json",
    #     "config_mode": "test",
    #     "data_path": "dataset/json/refactor/03_nbeats_dataloader.json",
    # }
    # test_module("model_tch/03_nbeats_dataloader.py", param_pars)

    param_pars = {
        "choice": "json",
        "config_mode": "test",
        "data_path": "dataset/json/refactor/namentity_crm_bilstm_dataloader_new.json",
    }
    test_module("model_keras/namentity_crm_bilstm_dataloader.py", param_pars)

    """
    
    
    
    
    