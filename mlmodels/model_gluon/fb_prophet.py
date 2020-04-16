import pandas as pd
import os, copy
import matplotlib.pyplot as plt
import json
from fbprophet import Prophet
from mlmodels.util import log, path_norm, save_pkl, load_pkl


#####################################################################################################
class Model:
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        self.model_pars = copy.deepcopy(model_pars)
        self.fit_metrics = {}

        if model_pars is None :
            self.model = None
            return None

        self.model = Prophet()



def get_dataset(data_pars):
    train_col = data_pars["col_Xinput"]
    date_col = data_pars["date_col"]
    col = [train_col] + [date_col]
    pred_length = data_pars["prediction_length"]

    # if both train and test are provided
    if data_pars["test_data_path"]:
        # ds and y are hardcoded because it is restriction in model
        train_df = pd.read_csv(data_pars["train_data_path"], parse_dates=True)[col]
        train_df.rename(columns={date_col: "ds", train_col: "y"}, inplace=True)
        test_df = pd.read_csv(data_pars["test_data_path"], parse_dates=True)[col]
        test_df.rename(columns={date_col: "ds", train_col: "y"}, inplace=True)
        return train_df, test_df

    # when only train is provided
    df = pd.read_csv(data_pars["train_data_path"], parse_dates=True)[col]
    df.rename(columns={date_col: "ds", train_col: "y"}, inplace=True)
    train_df = df.iloc[:-pred_length].copy()
    test_df = df.iloc[-pred_length:].copy()
    return train_df, test_df


def get_params(param_pars={}, **kw):
    data_path = param_pars["data_path"]
    config_mode = param_pars["config_mode"]

    if param_pars["choice"] == "json":
        data_path = path_norm(data_path)
        cf = json.load(open(data_path, mode='r'))
        cf = cf[config_mode]
        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']


    if param_pars["choice"] == "test01":
        log("#### Path params   ##########################################")
        data_path       = path_norm(data_path)
        out_path        = path_norm("ztest/model_fb/prophet/")
        train_data_path = data_path
        test_data_path  = None
        
        os.makedirs(out_path, exist_ok=True)        
        log(data_path, out_path)
        
        log("#### Data params ####")
        data_pars = {"train_data_path": train_data_path,
                     "test_data_path": test_data_path,
                     "prediction_length": 12,
                     "date_col": "month",
                     "freq": "M",
                     "col_Xinput": "milk_production_pounds", }

        log("#### Model params ####")
        model_pars = {}

        log("#### Compute params ####")
        compute_pars = {}
        

        outpath = out_path + "result"
        out_pars = {"outpath": outpath}
    return model_pars, data_pars, compute_pars, out_pars


def fit(model=None, data_pars={}, compute_pars={}, out_pars={}, **kw):
    sess = None
    train_df, test_df = get_dataset(data_pars)
    model.model.fit(train_df)
    return model, None


def predict(model=None, model_pars=None, sess=None, data_pars=None,
            compute_pars=None, out_pars=None, **kwargs):
    train_df, test_df = get_dataset(data_pars)
    future = model.model.make_future_dataframe(periods=data_pars["prediction_length"],
                                               include_history=False, freq=data_pars["freq"])
    pred = model.model.predict(future)
    return pred["yhat"].values, test_df["y"].values


def save(model=None, session=None, save_pars={}):
    path = save_pars["outpath"]
    os.makedirs(path, exist_ok=True)
    save_pkl(model=model, session=None, save_pars={"path": path + "/fbprophet.pkl"})


def load(load_pars={}, **kw):
    path = load_pars["outpath"]
    model0 = load_pkl({"path": path + "/fbprophet.pcikle"})
    session = None
    return model0, session


def metrics_plot(metrics_params):
    os.makedirs(metrics_params["outpath"], exist_ok=True)
    if metrics_params["plot_type"] == "line":
        plt.plot(metrics_params["actual"], label="Actual",    color="blue")
        plt.plot(metrics_params["pred"], label="Prediction", color="red")
        plt.legend(loc="best")
        plt.savefig(metrics_params["outpath"] + "/prophet.png")


def test(data_path="dataset/", pars_choice="test0", config_mode="test"):
    log("#### Loading params   ##############################################")
    param_pars = {"choice": pars_choice, "config_mode": config_mode,
                  "data_path": data_path}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)


    log("#### Model init, fit   #############################################")
    model = Model(model_pars=model_pars, data_pars=data_pars,
                  compute_pars=compute_pars)

    log("### Model created ###")
    fit(model=model, data_pars=data_pars, compute_pars=compute_pars)

    # for prediction

    log("#### Predict   ####################################################")
    y_pred, y_test = predict(model=model, model_pars=model_pars, data_pars=data_pars)


    log("### Plot ###########################################################")
    data_pars["predict"] = True
    metrics_params = {"plot_type": "line", "pred": y_pred,
                      "outpath": out_pars["outpath"],
                      "actual": y_test}
    metrics_plot(metrics_params)

    log("#### Save/Load   ###################################################")
    save(model=model, session=None, save_pars=out_pars)
    model2, session2 = load(load_pars=out_pars, model_pars=model_pars,
                            data_pars=data_pars, compute_pars=compute_pars)


if __name__ == "__main__":
    VERBOSE = True
    test(data_path = "model_fb/fbprophet.json", choice="json" )


    test(data_path = "dataset/timeseries/milk.csv", choice="test01" )



