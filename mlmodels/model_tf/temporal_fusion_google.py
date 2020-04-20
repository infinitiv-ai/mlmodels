# coding: utf-8
"""
LSTM Time series predictions
python  model_tf/1_lstm.py
"""
import os
from warnings import simplefilter

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

####################################################################################################
from mlmodels.util import os_package_root_path, log, params_json_load, path_norm

from mlmodels.mode_tf.raw  import temporal_fusion_google




### Tf 2.0
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()




simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)
tf.compat.v1.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # **** change the warning level ****



####################################################################################################





####################################################################################################
class Model:
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None, **kwargs):
        reset_model()

        epoch         = model_pars.get('epoch', 5)
        learning_rate = model_pars.get('learning_rate', 0.001)
        num_layers    = model_pars.get('num_layers', 2)
        size_layer    = model_pars.get('size_layer', 128)
        forget_bias   = model_pars.get('forget_bias', 0.1)
        timestep      = model_pars.get('timestep', 5)

         #### Depends on input data !!!!!
        size          = model_pars.get('size', None)
        output_size   = model_pars.get('output_size', None)


        self.epoch = epoch
        self.stats = {"loss": 0.0,
                      "loss_history": []}

        self.X = tf.compat.v1.placeholder(tf.compat.v1.float32, (None, None, size))
        self.Y = tf.compat.v1.placeholder(tf.compat.v1.float32, (None, output_size))

        ### Model Structure        ################################

        ## drop = tf.compat.v1.contrib.rnn.DropoutWrapper(rnn_cells, output_keep_prob=forget_bias)
        drop = tf.compat.v1.nn.rnn_cell.DropoutWrapper(rnn_cells, output_keep_prob=forget_bias)

        self.hidden_layer = tf.compat.v1.placeholder(tf.compat.v1.float32, (None, self.hidden_layer_size))
        self.outputs, self.last_state = tf.compat.v1.nn.dynamic_rnn(
            drop, self.X, initial_state=self.hidden_layer, dtype=tf.compat.v1.float32
        )
        self.logits = tf.compat.v1.layers.dense(self.outputs[-1], output_size)

        ### Loss    ##############################################
        self.cost = tf.compat.v1.reduce_mean(tf.compat.v1.square(self.Y - self.logits))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(self.cost)


def fit(model, data_pars=None, compute_pars=None,  out_pars=None, **kwarg):
    df = get_dataset(data_pars)
    print(df.head(5))
    msample = df.shape[0]
    nlog_freq = compute_pars.get("nlog_freq", 100)

    ######################################################################
    # sess = tf.compat.v1.compat.v1.InteractiveSession()
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(model.epoch):
        total_loss = 0.0

        ######## Model specific  ########################################
        init_value = np.zeros((1, model.hidden_layer_size))
        for k in range(0, df.shape[0] - 1, model.timestep):
            index   = min(k + model.timestep, df.shape[0] - 1)
            batch_x = np.expand_dims(df.iloc[k:index, :].values, axis=0)
            batch_y = df.iloc[k + 1: index + 1, :].values
            last_state, _, loss = sess.run(
                [model.last_state, model.optimizer, model.cost],
                feed_dict={model.X: batch_x, model.Y: batch_y, model.hidden_layer: init_value},
            )
            init_value = last_state
            total_loss += loss
        ####### End Model specific    ##################################

        total_loss /= msample // model.timestep
        model.stats["loss"] = total_loss

        if (i + 1) % nlog_freq == 0:
            print("epoch:", i + 1, "avg loss:", total_loss)
    return model,sess


def fit_metrics(model, sess=None, data_pars=None, compute_pars=None, out_pars=None):
    """
       Return metrics of the model stored
    """

    return model.stats



def metrics(model, sess=None, data_pars=None, compute_pars=None, out_pars=None):
    """
       Return metrics of the model stored
    #### SK-Learn metrics
    # Compute stats on training
    #df = get_dataset(data_pars)
    #arr_out = predict(model, sess, df, get_hidden_state=False, init_value=None)
    :param model:
    :param sess:
    :param data_pars:
    :param out_pars:
    :return:
    """

    return model.stats


def predict(model, sess=None, data_pars=None,  compute_pars=None, out_pars=None,
            get_hidden_state=False, init_value=None):
    df = get_dataset(data_pars)
    print(df, flush=True)

    #############################################################
    if init_value is None:
        init_value = np.zeros((1, model.hidden_layer_size))
    output_predict = np.zeros((df.shape[0], df.shape[1]))
    upper_b = (df.shape[0] // model.timestep) * model.timestep

    if upper_b == model.timestep:
        out_logits, init_value = sess.run(
            [model.logits, model.last_state],
            feed_dict={
                model.X: np.expand_dims(df.values, axis=0),
                model.hidden_layer: init_value,
            },
        )
        output_predict[1:  model.timestep + 1] = out_logits

    else:
        for k in range(0, (df.shape[0] // model.timestep) * model.timestep, model.timestep):
            out_logits, last_state = sess.run(
                [model.logits, model.last_state],
                feed_dict={model.X: np.expand_dims(df.iloc[k: k + model.timestep].values, axis=0),
                           model.hidden_layer: init_value,
                           },
            )
            init_value = last_state
            output_predict[k + 1: k + model.timestep + 1] = out_logits

    if get_hidden_state:
        return output_predict, init_value
    return output_predict




def reset_model():
    tf.compat.v1.reset_default_graph()


def save(model, session=None, save_pars=None):
    from mlmodels.util import save_tf
    print(save_pars)
    save_tf(model, session, save_pars)
     


def load(load_pars=None):
    from mlmodels.util import load_tf
    print(load_pars)
    return load_tf(load_pars)








####################################################################################################
def get_dataset(data_pars=None):
    """
              "path"            : "dataset/text/ner_dataset.csv",
              "location_type"   :  "repo",
              "data_type"   :   "text",


              "data_loader" :  "pandas.read_csv",
              "data_loader_pars" :  {""},


              "data_preprocessor" : "mlmodels.model_keras.prepocess:process",
              "data_preprocessor_pars" : "mlmodels.model_keras.prepocess:process",              

              "size" : [0,1,2],
              "output_size": [0, 6]  


    """
    print(data_pars)
    filename = path_norm( data_pars["data_path"])  #

    ##### Specific   ######################################################
    df = pd.read_csv(filename)
    df = df.iloc[:10,:]
    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
    print(filename)
    print(df.head(5))

    minmax = MinMaxScaler().fit(df.iloc[:, 1:].astype("float32"))
    df_log = minmax.transform(df.iloc[:, 1:].astype("float32"))
    df_log = pd.DataFrame(df_log)
    return df_log



def get_params(param_pars={}, **kw):
    import json
    pp          = param_pars
    choice      = pp['choice']
    config_mode = pp['config_mode']
    data_path   = pp['data_path']

    if choice == "json":
       cf = params_json_load(data_path) 
       #  cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']
       return cf


    if choice == "test01":
        log("############# Data, Params preparation   #################")        
        data_path  = path_norm( "dataset/timeseries/GOOG-year.csv"  )   
        out_path   = path_norm( "ztest/model_tf/1_lstm/" )
        model_path = os.path.join(out_path , "model")


        model_pars   = {"learning_rate": 0.001, "num_layers": 1, "size": 6, "size_layer": 128,
                         "timestep": 4, "epoch": 2,
                         "output_size" : 6 }

        data_pars    = {"data_path": data_path, "data_type": "pandas"}
        compute_pars = {}
        out_pars     = {"path": out_path, "model_path": model_path}

        return model_pars, data_pars, compute_pars, out_pars
    else:
        raise Exception(f"Not support choice {choice} yet")


####################################################################################################
def test(data_path="dataset/", pars_choice="test01", config_mode="test"):
    ### Local test

    log("#### Loading params   ##############################################")
    param_pars = {"choice":pars_choice,  "data_path":data_path,  "config_mode": config_mode}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
    log( model_pars, data_pars, compute_pars, out_pars )


    log("#### Loading dataset   #############################################")
    Xtuple = get_dataset(data_pars)


    log("#### Model init  #############################################")
    session = None
    model = Model(model_pars, data_pars, compute_pars)

    log("#### Model fit   #############################################")
    model, session = fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)


    log("#### Predict   #####################################################")
    data_pars["train"] = 0
    ypred = predict(model, session, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)


    log("#### metrics   #####################################################")
    metrics_val = fit_metrics(model, data_pars, compute_pars, out_pars)
    print(metrics_val)


    log("#### Plot   ########################################################")


    log("#### Save/Load   ###################################################")
    save(model, session, out_pars)
    session = load(out_pars)
    #     ypred = predict(model2, data_pars, compute_pars, out_pars)
    #     metrics_val = metrics(model2, ypred, data_pars, compute_pars, out_pars)
    # print(model2)






if __name__ == "__main__":
    print("start")
    test(data_path="", pars_choice="test01", config_mode="test")






