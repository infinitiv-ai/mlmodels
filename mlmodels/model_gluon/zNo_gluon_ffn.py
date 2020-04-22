# -*- coding: utf-8 -*-
"""
GluonTS
# First install package from terminal:  pip install mxnet autogluon
https://autogluon.mxnet.io/tutorials/tabular_prediction/tabular-quickstart.html



"""
import json
import os
from pathlib import Path

import pandas as pd

from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer
from mlmodels.model_gluon.util import (
    _config_process, fit, get_dataset, load, metrics,
    plot_predict, plot_prob_forecasts, predict, save)


from mlmodels.util import path_norm, os_package_root_path, log

VERBOSE = False





########################################################################################################################
#### Model defintion
class Model(object) :
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None, **kwargs) :
        ## Empty model for Seaialization
        if model_pars is None and compute_pars is None :
           self.model = None
        else :
           m = compute_pars
           trainer = Trainer(ctx=m["ctx"], epochs=m["epochs"], learning_rate=m["learning_rate"],
                      hybridize=m["hybridize"], num_batches_per_epoch=m["num_batches_per_epoch"])

           ##set up the model
           m = model_pars
           self.model = SimpleFeedForwardEstimator(num_hidden_dimensions=m['num_hidden_dimensions'],
                                           prediction_length= m["prediction_length"],
                                           context_length= m["context_length"],
                                           freq=m["freq"],trainer=trainer)





def get_params(choice="", data_path="dataset/timeseries/", config_mode="test", **kw):
    import json
    #pp = param_pars
    #choice = pp['choice']
    #config_mode = pp['config_mode']
    #data_path = pp['data_path']


    if choice == "json":
        data_path = path_norm(data_path)
        cf = json.load(open(data_path, 'rb'))
        cf = cf[config_mode]
        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']


    if choice == "test01":
        log("#### Path params   #####################################################")
        data_path  = path_norm("dataset/timeseries/")
        out_path   = path_norm("ztest/model_gluon/gluon_prophet/" )
        model_path = os.path.join(out_path, "model")


        train_data_path = data_path + "GLUON-train.csv"
        test_data_path = data_path + "GLUON-test.csv"
        start = pd.Timestamp("01-01-1750", freq='1H')

        data_pars = {"train_data_path": train_data_path, 
                     "test_data_path": test_data_path, 
                     "train": False,
                     'prediction_length': 48, 'freq': '1H', "start": start, "num_series": 37,
                     "save_fig": "./series.png","modelpath":model_path}


        log("#### Model params   ###################################################")
        model_pars = {"num_hidden_dimensions": [10], "prediction_length": data_pars["prediction_length"],
                      "context_length":2*data_pars["prediction_length"],"freq":data_pars["freq"]
                     }

        compute_pars = {"ctx":"cpu","epochs":5,"learning_rate":1e-3,"hybridize":False,
                      "num_batches_per_epoch":100,'num_samples':100}

        outpath = out_path+"result"
        out_pars = {"outpath": outpath, "plot_prob": True, "quantiles": [0.1, 0.5, 0.9]}

    return model_pars, data_pars, compute_pars, out_pars




########################################################################################################################
def test(data_path="dataset/", choice="test01"):
    ### Local test

    log("#### Loading params   ##############################################")
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=choice, data_path=data_path)
    print(model_pars, data_pars, compute_pars, out_pars)

    log("#### Loading dataset   #############################################")
    gluont_ds = get_dataset(data_pars)


    log("#### Model init, fit   ###########################################")
    from mlmodels.models import module_load_full, fit, predict
    module, model = module_load_full("model_gluon.gluon_ffn", model_pars, data_pars, compute_pars)
    #model=m.model    ### WE WORK WITH THE CLASS (not the attribute GLUON )
    model= fit(module, model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)

    log("#### save the trained model  ######################################")
    save(model, data_pars["modelpath"])


    log("#### Predict   ###################################################")
    ypred = predict(module, model, data_pars=data_pars, out_pars=out_pars, compute_pars=compute_pars)
    print(ypred)


    log("#### metrics   ################################################")
    metrics_val, item_metrics = metrics(ypred, data_pars, compute_pars, out_pars)
    print(metrics_val)

    log("#### Plot   #######################################################")
    plot_prob_forecasts(ypred, out_pars)
    plot_predict(item_metrics, out_pars)
                        


                        
if __name__ == '__main__':
    VERBOSE=True
    test(data_path="dataset/timeseries/", choice="test01")
    # test(data_path="dataset/", choice="json")



