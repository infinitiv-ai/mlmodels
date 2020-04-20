# coding: utf-8
"""


%tensorflow_version 2.x
import tensorflow as tf
print("Tensorflow version " + tf.__version__)

try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
  print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
  raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)


def create_model():
  pretrained_model = tf.keras.applications.Xception(input_shape=[*IMAGE_SIZE, 3], include_top=False)
  pretrained_model.trainable = True
  model = tf.keras.Sequential([
    pretrained_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(5, activation='softmax')
  ])
  model.compile(
    optimizer='adam',
    loss = 'categorical_crossentropy',
    metrics=['accuracy']
  )
  return model

with tpu_strategy.scope(): # creating the model in the TPUStrategy scope means we will train the model on the TPU
  model = create_model()
model.summary()


with tpu_strategy.scope(): # creating the model in the TPUStrategy scope means we will train the model on the TPU
   model = Model()




###########
    batch_size = compute_pars['batch_size']
    epochs = compute_pars['epochs']

    sess = None  #
    Xtrain, Xtest, ytrain, ytest = get_dataset(data_pars)

    # This address identifies the TPU we'll use when configuring TensorFlow.
    TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']


    keras.backend.clear_session()

    resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)
    tf.contrib.distribute.initialize_tpu_system(resolver)
    strategy = tf.contrib.distribute.TPUStrategy(resolver)

    with strategy.scope():
    
      model0.compile(
          optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
          loss='sparse_categorical_crossentropy',
          metrics=['sparse_categorical_accuracy'])

    model.model.fit(Xtrain, ytrain,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  callbacks=[early_stopping],
                                  validation_data=(Xtest, ytest))




"""
import os
import numpy as np
from keras.callbacks import EarlyStopping


#### Import EXISTING model and re-map to mlmodels
from mlmodels.model_keras.raw.char_cnn.data_utils import Data
from mlmodels.model_keras.raw.char_cnn.models.char_cnn_kim import CharCNNKim




####################################################################################################
from mlmodels.util import os_package_root_path, log, path_norm, get_model_uri

VERBOSE = False
MODEL_URI = get_model_uri(__file__)
# print( path_norm("dataset") )



####################################################################################################
class Model:
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None
                 ):
        ### Model Structure        ################################
        if model_pars is None :
            self.model = None
            return None

        self.model = CharCNNKim(input_size=data_pars["input_size"],
                                alphabet_size          = data_pars["alphabet_size"],
                                embedding_size         = model_pars["embedding_size"],
                                conv_layers            = model_pars["conv_layers"],
                                fully_connected_layers = model_pars["fully_connected_layers"],
                                num_of_classes         = data_pars["num_of_classes"],
                                dropout_p              = model_pars["dropout_p"],
                                optimizer              = model_pars["optimizer"],
                                loss                   = model_pars["loss"]).model




def fit(model, data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """

    batch_size = compute_pars['batch_size']
    epochs = compute_pars['epochs']

    sess = None  #
    Xtrain, Xtest, ytrain, ytest = get_dataset(data_pars)

    early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
    model.model.fit(Xtrain, ytrain,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  callbacks=[early_stopping],
                                  validation_data=(Xtest, ytest))

    return model, sess





def fit_metrics(model, data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
       Return metrics ofw the model when fitted.
    """
    from sklearn.metrics import accuracy_score
    _,Xval,_, yval = get_dataset(data_pars)
    ypred = model.model.predict(Xval)
    metric_score_name = compute_pars.get('metric_score') 
    if metric_score_name is None :
        return {}
    ddict = {}
    if metric_score_name == "accuracy_score":
        ypred = ypred.argmax(axis=1)
        yval = np.argmax(yval, axis=1)
        score = accuracy_score(yval, ypred)
        ddict[metric_score_name] = score
    return ddict


def predict(model, session=None, data_pars=None, out_pars=None, compute_pars=None, **kw):
    ##### Get Data ###############################################
    data_pars['train'] = False
    Xpred, ypred = get_dataset(data_pars)

    #### Do prediction
    ypred = model.model.predict(Xpred)

    ### Save Results

    ### Return val
    if compute_pars.get("return_pred_not") is  None:
        return ypred


def reset_model():
    pass


def save(model=None,  save_pars=None, session=None):
    from mlmodels.util import save_keras
    print(save_pars)
    save_keras(model, session, save_pars=save_pars)


def load(load_pars=None):
    from mlmodels.util import load_keras
    model0 = load_keras(load_pars)

    model = Model()
    model.model = model0.model
    session = None
    return model, session


####################################################################################################
def get_dataset(data_pars=None, **kw):
    """
      JSON data_pars to get dataset
      "data_pars":    { "data_path": "dataset/GOOG-year.csv", "data_type": "pandas",
      "size": [0, 0, 6], "output_size": [0, 6] },
    """
    from mlmodels.util import path_norm
    
    if data_pars['train']:

        print('Loading data...')
        train_data = Data(data_source= path_norm( data_pars["train_data_source"]) ,
                             alphabet       = data_pars["alphabet"],
                             input_size     = data_pars["input_size"],
                             num_of_classes = data_pars["num_of_classes"])
        if data_pars['type'] == "npz":
            train_inputs,train_labels, val_inputs, val_labels = train_data.get_all_data_npz()
        else: 
            train_data.load_data()
            train_inputs, train_labels = train_data.get_all_data()

            # Load val data
            val_data = Data(data_source = path_norm( data_pars["val_data_source"]) ,
                                   alphabet=data_pars["alphabet"],
                                   input_size=data_pars["input_size"],
                                   num_of_classes=data_pars["num_of_classes"])
            val_data.load_data()
            val_inputs, val_labels = val_data.get_all_data()

        return train_inputs, val_inputs, train_labels, val_labels


    else:
        val_data = Data(data_source = path_norm( data_pars["val_data_source"]) ,
                               alphabet=data_pars["alphabet"],
                               input_size=data_pars["input_size"],
                               num_of_classes=data_pars["num_of_classes"])
        val_data.load_data()
        Xtest, ytest = val_data.get_all_data()
        return Xtest, ytest


def get_params(param_pars={}, **kw):
    import json
    pp = param_pars
    choice = pp['choice']
    config_mode = pp['config_mode']
    data_path = pp['data_path']


    if choice == "json":
        data_path = path_norm(data_path)
        cf = json.load(open(data_path, mode='r'))
        cf = cf[config_mode]
        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']


    if choice == "test01":
        log("#### Path params   ##########################################")
        root       = path_norm()
        data_path  = path_norm( "dataset/text/imdb.npz"  )   
        out_path   = path_norm( "ztest/model_keras/charcnn/" )
        model_path = os.path.join(out_path , "model")


        model_pars = {
            "embedding_size": 128,
            "conv_layers": [[256, 10 ], [256, 7 ], [256, 5 ], [256, 3 ] ], 
            "fully_connected_layers": [
                1024,
                1024
            ],
            "threshold": 1e-6,
            "dropout_p": 0.1,
            "optimizer": "adam",
            "loss": "categorical_crossentropy"
        }

        data_pars = {
            "train": True,
            "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
            "alphabet_size": 69,
            "input_size": 1014,
            "num_of_classes": 4,
            "train_data_source": path_norm("dataset/text/ag_news_csv/train.csv") ,
            "val_data_source": path_norm("dataset/text/ag_news_csv/test.csv")
        }


        compute_pars = {
            "epochs": 1,
            "batch_size": 128
        }

        out_pars = {
            "path":  path_norm( "ztest/ml_keras/charcnn/charcnn.h5"),
            "data_type": "pandas",
            "size": [0, 0, 6],
            "output_size": [0, 6]
        }

        return model_pars, data_pars, compute_pars, out_pars

    else:
        raise Exception(f"Not support choice {choice} yet")


################################################################################################
########## Tests ###############################################################################
def test(data_path="dataset/", pars_choice="json", config_mode="test"):
    ### Local test
    from mlmodels.util import path_norm
    data_path = path_norm(data_path)

    log("#### Loading params   ##############################################")
    param_pars = {"choice": pars_choice, "data_path": data_path, "config_mode": config_mode}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)

    log("#### Loading dataset   #############################################")
    Xtuple = get_dataset(data_pars)
    print(len(Xtuple))

    log("#### Model init, fit   #############################################")
    session = None
    model = Model(model_pars, data_pars, compute_pars)
    model, session = fit(model, data_pars, compute_pars, out_pars)


    log("#### Predict   #####################################################")
    data_pars["train"] = 0
    ypred = predict(model, session, data_pars, compute_pars, out_pars)

    log("#### metrics   #####################################################")
    metrics_val = fit_metrics(model, session, data_pars, compute_pars, out_pars)
    print(metrics_val)

    log("#### Plot   ########################################################")


    log("#### Save/Load   ###################################################")
    save_pars = {"path": out_pars['path']}
    save(model, session, save_pars=save_pars)
    model2, session2 = load(save_pars)

    log("#### Save/Load - Predict   #########################################")
    print(model2, session2)
    ypred = predict(model2, session2, data_pars, compute_pars, out_pars)



if __name__ == '__main__':
    VERBOSE = True
    test_path = os.getcwd() + "/mytest/"
    root_path = os_package_root_path()

    ### Local fixed params
    test(pars_choice="test01")

    #### Local json file
    test(pars_choice="json", data_path= f"model_keras/charcnn.json")


    # ####    test_module(model_uri="model_xxxx/yyyy.py", param_pars=None)
    # from mlmodels.models import test_module
    #
    # param_pars = {'choice': "json", 'config_mode': 'test', 'data_path': "model_keras/charcnn.json"}
    # test_module(model_uri=MODEL_URI, param_pars=param_pars)
    #
    # #### get of get_params
    # # choice      = pp['choice']
    # # config_mode = pp['config_mode']
    # # data_path   = pp['data_path']
    #
    # ####    test_api(model_uri="model_xxxx/yyyy.py", param_pars=None)
    # from mlmodels.models import test_api
    #
    # param_pars = {'choice': "json", 'config_mode': 'test', 'data_path': "model_keras/charcnn.json"}
    # test_api(model_uri=MODEL_URI, param_pars=param_pars)



