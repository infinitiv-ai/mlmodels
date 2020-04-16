# -*- coding: utf-8 -*-
"""
GluonTS
# First install package from terminal:  pip install mxnet autogluon
https://autogluon.mxnet.io/tutorials/tabular_prediction/tabular-quickstart.html



"""

"""
Things to do for benchamrking

1. In mlmodels.model_gluon.util(if this is use in this model)
    i- explicitly mentioned Features columns and Target columns, incase for 
       multivariate time series forecasting

2. Fit function should return two values
    i. first is trained model
    ii. second is session, though if their is no session use in 
        your model then return session=None

3. Run and test this model for choice="json" and prepare prod ready json file
   which contain all params_info to run this model
   JSON should have following parameters
    i. data_pars
    ii. model_pars
    iii. compute_pars 
    iv. out_pars (output param like model output path, plot save path, use some dummy
                  value incase if their is no need of out_pars) 

4. predict should return two arrays
    i. first should be prediction (only array/list/numpy array)
    ii. second would be actual (only array/list/numpy array)
    iii. prediction and actual should have same length
         so to easily calculate metrics using third party
         code
"""



import json
import os
from pathlib import Path

import pandas as pd

from gluonts.model.prophet import ProphetPredictor
from gluonts.trainer import Trainer
from mlmodels.model_gluon.util import (
    _config_process, fit, get_dataset, load,  metrics,
    plot_predict, plot_prob_forecasts, predict, save)


from mlmodels.util import path_norm, os_package_root_path, log

VERBOSE = False




########################################################################################################################
#### Model defintion
class Model(object) :
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None) :
        ## Empty model for Seaialization
        if model_pars is None and compute_pars is None :
            self.model = None
        else :
            m = model_pars
            self.model = ProphetPredictor(prediction_length=m['prediction_length'],freq=m['freq'])





########################################################################################################################
def get_params(choice="", data_path="dataset/", config_mode="test", **kw):
    if choice == "json":
       return _config_process(data_path, config_mode=config_mode)


    if choice == "test01":
        from mlmodels.util import path_norm
        log("#### Path params   ################################################")
        data_path  = path_norm("dataset/timeseries/")
        out_path   = path_norm("ztest/model_gluon/gluon_prophet/" )
        model_path = os.path.join(out_path, "model")

        train_data_path = data_path + "GLUON-train.csv"
        test_data_path  = data_path + "GLUON-test.csv"
        start           = pd.Timestamp("01-01-1750", freq='1H')
        data_pars = {"train_data_path": train_data_path, "test_data_path": test_data_path, 
                     "train": False,
                     "prediction_length": 48, "freq": '1H', "start": start, "num_series": 37,
                     "save_fig": "./series.png","modelpath":model_path}

        log("## Model params   ################################################")
        model_pars = {"prediction_length": data_pars["prediction_length"], "freq": data_pars["freq"]}
        compute_pars = {}
        outpath = out_path + "result"

        out_pars = {"outpath": outpath, "plot_prob": True, "quantiles": [0.1, 0.5, 0.9]}

    return model_pars, data_pars, compute_pars, out_pars



########################################################################################################################
def test(data_path="dataset/", choice=""):
    ### Local test

    log("#### Loading params   ##############################################")
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=choice, data_path=data_path)


    log("#### Loading dataset   #############################################")
    gluont_ds = get_dataset(data_pars)


    log("#### Model init, fit   ###########################################")
    from mlmodels.models import module_load_full, fit, predict
    module, model = module_load_full("model_gluon.gluon_prophet", model_pars, data_pars, compute_pars)
    print(module, model)

    # model=m.model    ### WE WORK WITH THE CLASS (not the attribute GLUON )
    model = fit(module, model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)


    # model = Model(model_pars, compute_pars)
    # #model=m.model    ### WE WORK WITH THE CLASS (not the attribute GLUON )
    # # Requires sess as 2nd aprametetr
    # model=fit(model, None, data_pars, model_pars, compute_pars)

    log("#### save the trained model  #############################################")
    save(model, data_pars["modelpath"])


    log("#### Predict   ###################################################")
    ypred = predict(model, data_pars, compute_pars, out_pars)
    print(ypred)


    log("#### metrics   ################################################")
    metrics_val, item_metrics = metrics(ypred, data_pars, compute_pars, out_pars)
    print(metrics_val)

    log("#### Plot   #######################################################")
    plot_prob_forecasts(ypred, out_pars)
    plot_predict(item_metrics, out_pars)



if __name__ == '__main__':
    VERBOSE = True
    test(data_path="dataset/timeseries/" , choice="test01" )
