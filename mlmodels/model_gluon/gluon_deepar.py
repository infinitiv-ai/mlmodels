# -*- coding: utf-8 -*-
import os
import pandas as pd

from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.predictor import Predictor
import matplotlib.pyplot as plt
from pathlib import Path
import json


#########################################################################################################
from mlmodels.util import os_package_root_path, log, path_norm, get_model_uri


VERBOSE = False
MODEL_URI = get_model_uri(__file__)








#########################################################################################################
class Model(object):
    def __init__(self, model_pars=None, data_pars=None, 
                 compute_pars=None, **kwargs):
        ## Empty model for Seaialization
        if model_pars is None :
            self.model = None

        else:
            self.compute_pars = compute_pars
            self.model_pars = model_pars

            m = self.compute_pars
            trainer = Trainer(batch_size=m['batch_size'], clip_gradient=m['clip_gradient'], 
                              ctx=m["ctx"],
                              epochs=m["epochs"],
                              learning_rate=m["learning_rate"], init=m['init'],
                              learning_rate_decay_factor=m['learning_rate_decay_factor'],
                              minimum_learning_rate=m['minimum_learning_rate'], hybridize=m["hybridize"],
                              num_batches_per_epoch=m["num_batches_per_epoch"],
                              patience=m['patience'], weight_decay=m['weight_decay']
                              )

            ##set up the model
            m = self.model_pars
            self.model = DeepAREstimator(prediction_length=m['prediction_length'], freq=m['freq'],
                                         num_layers=m['num_layers'],
                                         num_cells=m["num_cells"],
                                         cell_type=m["cell_type"], dropout_rate=m["dropout_rate"],
                                         use_feat_dynamic_real=m["use_feat_dynamic_real"],
                                         use_feat_static_cat=m['use_feat_static_cat'],
                                         use_feat_static_real=m['use_feat_static_real'],
                                         scaling=m['scaling'], num_parallel_samples=m['num_parallel_samples'],
                                         trainer=trainer)


def _config_process(data_path, config_mode="test"):
    if choice=='test01':
        data_path = path_norm(  "model_gluon/deepar_run.json" )

    elif choice=='test02':
        data_path = path_norm( "model_gluon/deepar.json")


    with open(data_path, encoding='utf-8') as config_f:
        config = json.load(config_f)
        config = config[config_mode]

    return config["model_pars"], config["data_pars"], config["compute_pars"], config["out_pars"]



def get_params(choice="", data_path="dataset/timeseries/", config_mode="test", **kw):
    if choice == "json":
        return _config_process(data_path, config_mode=config_mode, choice=choice)

    log("#### Path params   ###################################################")
    data_path  = path_norm( "dataset/timeseries" )   
    out_path   = path_norm( "ztest/model_gluon/gluon_deepar/" )   
    model_path = os.path.join(out_path , "model")


    if choice == "test01" :        
        log("#### Data params   ###################################################")
        data_pars = {"train_data_path": data_path + "/train_deepar.csv",
                     "test_data_path":  data_path + "/test_deepar.csv", 
                     "train": True,
                     'prediction_length': 12, 'freq': '5min', 
                     "save_fig": "./series.png", "modelpath": model_path,
                     "choice":choice}

        log("#### output params   ###################################################")
        out_pars = {"outpath": out_path + "result", 
                    "plot_prob": True, "quantiles": [0.5]}


    elif choice == "test02" :
        log("#### Data params   ###################################################")
        data_pars = {"train_data_path": data_path + "/GLUON-train.csv",
                    "test_data_path":  data_path + "/GLUON-test.csv", 
                    "train": False,
                    'prediction_length': 48, 'freq': '1H', 
                    "start": pd.Timestamp("01-01-1750", freq='1H'), 
                    "num_series":37,
                    "save_fig": "./series.png", "modelpath": model_path,
                    "choice":choice}

        log("#### output params   ###################################################")
        out_pars = {"outpath": out_path + "result", 
                    "plot_prob": True, "quantiles": [0.1, 0.5, 0.9]}

    log("#### Model params   ################################################")
    model_pars = {"prediction_length": data_pars["prediction_length"], "freq": data_pars["freq"],
                "num_layers": 2, "num_cells": 40, "cell_type": 'lstm', "dropout_rate": 0.1,
                "use_feat_dynamic_real": False, "use_feat_static_cat": False, "use_feat_static_real": False,
                "scaling": True, "num_parallel_samples": 100}

    log("#### Compute params   ################################################")
    compute_pars = {"batch_size": 32, "clip_gradient": 100, "ctx": None, "epochs": 10, "init": "xavier",
                    "learning_rate": 1e-3,
                    "learning_rate_decay_factor": 0.5, "hybridize": False, "num_batches_per_epoch": 10,
                    'num_samples': 100,
                    "minimum_learning_rate": 5e-05, "patience": 10, "weight_decay": 1e-08}
    
    return model_pars, data_pars, compute_pars, out_pars


def get_dataset(data_pars):    
    if data_pars["choice"]=='test01':
        data_path = data_pars['train_data_path']
        df = pd.read_csv(data_path, header=0, index_col=0)  
        gluonts_ds = ListDataset([{"start": df.index[0],"target": df.value[:"2015-04-05 00:00:00"]}],
                                freq="5min")

    elif data_pars["choice"]=='test02':
        data_path = data_pars['train_data_path'] if data_pars['train'] else data_pars['test_data_path']
        #### read from csv file
        if data_pars.get("uri_type") == "pickle":
            data_set = pd.read_pickle(data_path)
        else:
            data_set = pd.read_csv(data_path)
        ### convert to gluont format
        gluonts_ds = ListDataset([{FieldName.TARGET: data_set.iloc[i].values, FieldName.START: data_pars['start']}
                              for i in range(data_pars['num_series'])], freq=data_pars['freq'])


    if VERBOSE:
        entry = next(iter(gluonts_ds))
        train_series = to_pandas(entry)
        train_series.plot()
        save_fig = data_pars['save_fig']
        # plt.savefig(save_fig)
    return gluonts_ds


def fit(modeule,model, sess=None, data_pars=None, model_pars=None, compute_pars=None, out_pars=None, session=None, **kwargs):
        ##loading dataset
        """
          Classe Model --> model,   model.model contains thte sub-model
        """
        model_gluon = model.model
        gluont_ds = get_dataset(data_pars)
        predictor = model_gluon.train(gluont_ds)
        return predictor


class Model_empty(object):
    def __init__(self, model_pars=None, compute_pars=None):
        # Empty model for Seaialization
        self.model = None


def save(model, path):
    if os.path.exists(path):
        model.model.serialize(Path(path))


def load(path):
    if os.path.exists(path):
        predictor_deserialized = Predictor.deserialize(Path(path))

    model = Model_empty()
    model.model = predictor_deserialized
    #### Add back the model parameters...

    return model


def predict(model, sess=None, data_pars=None, compute_pars=None, out_pars=None, **kwargs):
        ##  Model is class
        ## load test dataset
    if data_pars['choice']=='test01':
        data_path = data_pars['test_data_path'] 

        df = pd.read_csv(data_path, header=0, index_col=0)
    
        test_ds = ListDataset([{"start": df.index[0],
                                "target": df.value[:"2015-04-15 00:00:00"]}],
                            freq="5min")
    elif data_pars['choice']=='test02':
        data_pars['train'] = False
        test_ds = get_dataset(data_pars)
    
    forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_ds,  # test dataset
            predictor=model,  # predictor
            num_samples=compute_pars['num_samples'],  # number of sample paths we want for evaluation
        )

    forecasts, tss = list(forecast_it), list(ts_it)
    forecast_entry, ts_entry = forecasts[0], tss[0]

    print("forcast:",forecasts)
    print("tss:", tss)

    if VERBOSE:
        print(f"Number of sample paths: {forecast_entry.num_samples}")
        print(f"Dimension of samples: {forecast_entry.samples.shape}")
        print(f"Start date of the forecast window: {forecast_entry.start_date}")
        print(f"Frequency of the time series: {forecast_entry.freq}")
        print(f"Mean of the future window:\n {forecast_entry.mean}")
        print(f"0.5-quantile (median) of the future window:\n {forecast_entry.quantile(0.5)}")

    dd = {"forecasts": forecasts, "tss": tss}
    return dd

def metrics(ypred, data_pars, compute_pars=None, out_pars=None, **kwargs):
        ## load test dataset
        data_pars['train'] = False
        test_ds = get_dataset(data_pars)

        forecasts = ypred["forecasts"]
        tss = ypred["tss"]

        ## evaluate
        evaluator = Evaluator(quantiles=out_pars['quantiles'])
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
        metrics_dict = json.dumps(agg_metrics, indent=4)
        return metrics_dict, item_metrics


def plot_prob_forecasts(ypred, out_pars=None):
    forecast_entry = ypred["forecasts"][0]
    ts_entry = ypred["tss"][0]

    plot_length = 150
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()


def plot_predict(item_metrics, out_pars=None):
    item_metrics.plot(x='MSIS', y='MASE', kind='scatter')
    plt.grid(which="both")
    outpath = out_pars['outpath']
    if not os.path.exists(outpath): os.makedirs(outpath, exist_ok=True)
    plt.savefig(outpath)
    plt.clf()
    print('Saved image to {}.'.format(outpath))


def test(data_path="dataset/", choice=""):
    ### Local test
    log("#### Loading params   ##############################################")
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=choice, data_path=data_path)
    print(model_pars, data_pars, compute_pars, out_pars)

    log("#### Loading dataset   #############################################")
    gluont_ds = get_dataset(data_pars)

    log("#### Model init, fit   #############################################")
    from mlmodels.models import module_load_full
    module, model = module_load_full("model_gluon.gluon_deepar", model_pars, data_pars, compute_pars)
    print(module, model)

    model = fit(module, model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
    print(model)

    log("#### save the trained model  ######################################")
    save(model, data_pars["modelpath"])

    log("#### Predict   ####################################################")
    ypred = predict(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
    # print(ypred)

    log("#### metrics   ####################################################")
    metrics_val, item_metrics = metrics(ypred, data_pars, compute_pars, out_pars)
    print(metrics_val)

    log("#### Plot   #######################################################")
    plot_prob_forecasts(ypred, out_pars)
    plot_predict(item_metrics, out_pars)


if __name__ == '__main__':
    VERBOSE = True
    test(data_path="dataset/timeseries", choice="test01")

    test(data_path="dataset/timeseries", choice="test02")


    test(data_path="model_gluon/gluon_deepar,json", choice="json")



