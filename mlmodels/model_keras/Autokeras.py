
# -*- coding: utf-8 -*-
"""
https://autokeras.com/examples/imdb/
"""
from mlmodels.util import os_package_root_path, log, path_norm, get_model_uri, path_norm_dict
import os
import json

import numpy as np

import autokeras as ak

from keras.models import load_model

from tensorflow.keras.datasets import imdb




MODEL_URI = get_model_uri(__file__)


def get_config_file():
    return os.path.join(os_package_root_path(__file__, 1), 'config', 'model_tch', 'Imagecnn.json')


###########################################################################################################
###########################################################################################################
class Model:
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None, out_pars=None):
        ### Model Structure        ################################

        if model_pars is None:
            self.model = None
            return self

        # Initialize the text classifier.
        # It tries n different models.
        self.model = ak.TextClassifier(max_trials=model_pars['max_trials'])


def get_params(param_pars=None, **kw):
    pp = param_pars
    choice = pp['choice']
    config_mode = pp['config_mode']
    data_path = pp['data_path']

    if choice == "json":
        data_path = path_norm(data_path)
        cf = json.load(open(data_path, mode='r'))
        cf = cf[config_mode]

        ####Normalize path  : add /models/dataset/
        cf['data_pars'] = path_norm_dict(cf['data_pars'])
        cf['out_pars'] = path_norm_dict(cf['out_pars'])

        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']

    else:
        raise Exception(f"Not support choice {choice} yet")


def get_dataset_imbd(data_pars):

    # Load the integer sequence the IMDB dataset with Keras.
    index_offset = 3  # word index offset
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=data_pars["num_words"],
                                                          index_from=index_offset)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    # Prepare the dictionary of index to word.
    word_to_id = imdb.get_word_index()
    word_to_id = {k: (v + index_offset) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    id_to_word = {value: key for key, value in word_to_id.items()}
    # Convert the word indices to words.
    x_train = list(map(lambda sentence: ' '.join(
        id_to_word[i] for i in sentence), x_train))
    x_test = list(map(lambda sentence: ' '.join(
        id_to_word[i] for i in sentence), x_test))
    x_train = np.array(x_train, dtype=np.str)
    x_test = np.array(x_test, dtype=np.str)
    return x_train, y_train, x_test, y_test


def get_dataset(data_pars=None):
    if data_pars['dataset'] == 'IMDB':
        x_train, y_train, x_test, y_test = get_dataset_imbd(data_pars)
        return x_train, y_train, x_test, y_test

    else:
        raise Exception("Dataloader not implemented")
        exit


def fit(model, data_pars=None, compute_pars=None, out_pars=None, **kwargs):
    x_train, y_train, _, _ = get_dataset(data_pars)
    print(type(x_train))
    print(type(y_train))
    os.makedirs(out_pars["checkpointdir"], exist_ok=True)
    model.model.fit(x_train,
              y_train,
              epochs=compute_pars["epochs"],
              # Split the training data and use the last 15% as validation data.
              validation_split=data_pars["validation_split"])

    return model, None


def predict(model, session=None, data_pars=None, compute_pars=None, out_pars=None, **kwargs):
    # get a batch of data
    _, _, x_test, y_test = get_dataset(data_pars)
    predicted_y = model.model.predict(x_test)
    if kwargs.get("return_ytrue"):
        return predicted_y, y_test
    else:
        return predicted_y, None


def fit_metrics(model, data_pars=None, compute_pars=None, out_pars=None):
    _, _, x_test, y_test = get_dataset(data_pars)
    return model.model.evaluate(x_test, y_test)


def save(model, session=None, save_pars=None):
    model.save(os.path.join(save_pars["checkpointdir"],'autokeras_classifier_model.h5'))


def load(load_pars):
    return load_model(os.path.join(load_pars["checkpointdir"],'autokeras_classifier_model.h5'), custom_objects=ak.CUSTOM_OBJECTS)


###########################################################################################################
###########################################################################################################
def test(data_path="dataset/", pars_choice="json", config_mode="test"):
    ### Local test

    log("#### Loading params   ##############################################")
    param_pars = {"choice": pars_choice, "data_path": data_path, "config_mode": config_mode}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
    log(data_pars, out_pars)

    log("#### Loading dataset   #############################################")
    #xtuple = get_dataset(data_pars)

    log("#### Model init, fit   #############################################")
    model = Model(model_pars, data_pars, compute_pars)
    fitted_model = fit(model, data_pars, compute_pars, out_pars)

    log("#### Predict   #####################################################")
    ypred, _ = predict(fitted_model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
    print(ypred[:10])

    log("#### metrics   #####################################################")
    metrics_val = fit_metrics(fitted_model, data_pars, compute_pars, out_pars)
    print(metrics_val)

    log("#### Plot   ########################################################")

    log("#### Save/Load   ###################################################")
    ## Export as a Keras Model.
    fitted_model = fitted_model.model
    save_model = fitted_model.export_model()
    save(model=save_model, save_pars=out_pars)
    loaded_model = load( out_pars )
    ypred = predict(loaded_model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
    print(ypred[:10])


if __name__ == "__main__":
    test(data_path="model_keras/autokeras_classifier.json", pars_choice="json", config_mode="test")
