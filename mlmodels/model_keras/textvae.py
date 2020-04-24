# coding: utf-8
"""
Generic template for new model.
Check parameters template in models_config.json

"model_pars":   { "learning_rate": 0.001, "num_layers": 1, "size": 6, "size_layer": 128, "output_size": 6, "timestep": 4, "epoch": 2 },
"data_pars":    { "data_path": "dataset/GOOG-year.csv", "data_type": "pandas", "size": [0, 0, 6], "output_size": [0, 6] },
"compute_pars": { "distributed": "mpi", "epoch": 10 },
"out_pars":     { "out_path": "dataset/", "data_type": "pandas", "size": [0, 0, 6], "output_size": [0, 6] }



"""

import codecs
import csv
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed, Layer, \
    Dropout
from keras.layers.advanced_activations import ELU
from keras.models import Model as KModel
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer




####################################################################################################
from mlmodels.util import  os_package_root_path, log, path_norm, get_model_uri

VERBOSE = False
MODEL_URI = get_model_uri(__file__)




####################################################################################################
class Model:
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        ### Model Structure        ################################
        if model_pars is None:
            self.model = None

        else:
            texts, embeddings_index = get_dataset(data_pars)

            self.tokenizer = Tokenizer(model_pars["MAX_NB_WORDS"])
            self.tokenizer.fit_on_texts(texts)
            word_index = self.tokenizer.word_index  # the dict values start from 1 so this is fine with zeropadding
            index2word = {v: k for k, v in word_index.items()}
            
            print('Found %s unique tokens' % len(word_index))
            sequences = self.tokenizer.texts_to_sequences(texts)
            data_1    = pad_sequences(sequences, maxlen=data_pars["MAX_SEQUENCE_LENGTH"])
            
            print('Shape of data tensor:', data_1.shape)
            NB_WORDS   = (min(self.tokenizer.num_words, len(word_index)) + 1)  # +1 for zero padding
            data_1_val = data_1[801000:807000]  # select 6000 sentences as validation data
            data_pars["data_1"]     = data_1
            data_pars["data_1_val"] = data_1_val

            max_len    = data_pars["MAX_SEQUENCE_LENGTH"]
            emb_dim    = model_pars["EMBEDDING_DIM"]
            latent_dim = model_pars["latent_dim"]
            intermediate_dim = model_pars["intermediate_dim"]
            epsilon_std = model_pars["epsilon_std"]
            num_sampled = model_pars["num_sampled"]
            act = ELU()

            print('Found %s word vectors.' % len(embeddings_index))

            glove_embedding_matrix = np.zeros((NB_WORDS, model_pars["EMBEDDING_DIM"]))
            for word, i in word_index.items():
                if i < NB_WORDS:
                    embedding_vector = embeddings_index.get(word)
                    if embedding_vector is not None:
                        # words not found in embedding index will be the word embedding of 'unk'.
                        glove_embedding_matrix[i] = embedding_vector
                    else:
                        glove_embedding_matrix[i] = embeddings_index.get('unk')
            print('Null word embeddings: %d' % np.sum(np.sum(glove_embedding_matrix, axis=1) == 0))

            
            # y = Input(batch_shape=(None, max_len, NB_WORDS))
            x       = Input(batch_shape=(None, max_len))
            x_embed = Embedding(NB_WORDS, emb_dim, weights=[glove_embedding_matrix],
                                input_length=max_len, trainable=False)(x)
            h = Bidirectional(LSTM(intermediate_dim, return_sequences=False, recurrent_dropout=0.2),
                              merge_mode='concat')(x_embed)
            # h = Bidirectional(LSTM(intermediate_dim, return_sequences=False), merge_mode='concat')(h)
            h = Dropout(0.2)(h)
            h = Dense(intermediate_dim, activation='linear')(h)
            h = act(h)
            h = Dropout(0.2)(h)
            z_mean    = Dense(latent_dim)(h)
            z_log_var = Dense(latent_dim)(h )

            def sampling(args):
                z_mean, z_log_var = args
                epsilon = K.random_normal(shape=(compute_pars["batch_size"], latent_dim), mean=0.,
                                          stddev=epsilon_std)
                return z_mean + K.exp(z_log_var / 2) * epsilon

            # note that "output_shape" isn't necessary with the TensorFlow backend
            z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

            # we instantiate these layers separately so as to reuse them later
            repeated_context = RepeatVector(max_len)
            decoder_h        = LSTM(intermediate_dim, return_sequences=True, recurrent_dropout=0.2)
            decoder_mean     = TimeDistributed( Dense(NB_WORDS, activation='linear'))  # softmax is applied in the seq2seqloss by tf
            h_decoded        = decoder_h(repeated_context(z))
            x_decoded_mean   = decoder_mean(h_decoded)

            # placeholder loss
            def zero_loss(y_true, y_pred):
                return K.zeros_like(y_pred)

            # =========================== Necessary only if you want to use Sampled Softmax =======================#
            # Sampled softmax
            logits  = tf.constant(np.random.randn(compute_pars["batch_size"], max_len, NB_WORDS), tf.float32)
            targets = tf.constant(np.random.randint(NB_WORDS, size=(compute_pars["batch_size"], max_len)), tf.int32)
            proj_w  = tf.constant(np.random.randn(NB_WORDS, NB_WORDS), tf.float32)
            proj_b  = tf.constant(np.zeros(NB_WORDS), tf.float32)

            def _sampled_loss(labels, logits):
                labels = tf.cast(labels, tf.int64)
                labels = tf.reshape(labels, [-1, 1])
                logits = tf.cast(logits, tf.float32)
                return tf.cast( tf.nn.sampled_softmax_loss(
                        proj_w,  proj_b,   labels,  logits, num_sampled=num_sampled, num_classes=NB_WORDS),
                    tf.float32)

            softmax_loss_f = _sampled_loss

            # ====================================================================================================#
            # Custom VAE loss layer
            class CustomVariationalLayer(Layer):
                def __init__(self, **kwargs):
                    self.is_placeholder = True
                    super(CustomVariationalLayer, self).__init__(**kwargs)
                    self.target_weights = tf.constant(np.ones((compute_pars["batch_size"], max_len)), tf.float32)

                def vae_loss(self, x, x_decoded_mean):
                    # xent_loss = K.sum(metrics.categorical_crossentropy(x, x_decoded_mean), axis=-1)
                    labels = tf.cast(x, tf.int32)
                    xent_loss = K.sum(tf.contrib.seq2seq.sequence_loss(x_decoded_mean, labels,
                                                                       weights                  = self.target_weights,
                                                                       average_across_timesteps = False,
                                                                       average_across_batch     = False), axis=-1)
                    # softmax_loss_function=softmax_loss_f), axis=-1)#, uncomment for sampled doftmax
                    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
                    return K.mean(xent_loss + kl_loss)

                def call(self, inputs):
                    x = inputs[0]
                    x_decoded_mean = inputs[1]
                    print(x.shape, x_decoded_mean.shape)
                    loss = self.vae_loss(x, x_decoded_mean)
                    self.add_loss(loss, inputs=inputs)
                    # we don't use this output, but it has to have the correct shape:
                    return K.ones_like(x)

            loss_layer = CustomVariationalLayer()([x, x_decoded_mean])
            self.model = KModel(x, [loss_layer])
            
            #opt = Adam(lr=0.01)  # SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
            self.model.compile(optimizer=model_pars['optimizer'], loss=[zero_loss])
            self.model.summary()


def fit(model, data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    batch_size = compute_pars['batch_size']
    epochs     = compute_pars['epochs']
    n_steps    = (800000 / 2) / batch_size
    sess       = None  

    def sent_generator(TRAIN_DATA_FILE, chunksize):
        import pandas as pd
        reader = pd.read_csv(TRAIN_DATA_FILE, chunksize=chunksize, iterator=True)
        for df in reader:
            val3  = df.iloc[:, 3:4].values.tolist()
            val4  = df.iloc[:, 4:5].values.tolist()
            flat3 = [item for sublist in val3 for item in sublist]
            flat4 = [str(item) for sublist in val4 for item in sublist]
            texts = []
            texts.extend(flat3[:])
            texts.extend(flat4[:])

            sequences  = model.tokenizer.texts_to_sequences(texts)
            data_train = pad_sequences(sequences, maxlen=data_pars["MAX_SEQUENCE_LENGTH"])
            yield [data_train, data_train]

    model.model.fit_generator(sent_generator(data_pars["train_data_path"], batch_size / 2),
                    epochs          = epochs,
                    steps_per_epoch = n_steps,
                    validation_data = (data_pars["data_1_val"], data_pars["data_1_val"]))

    return model, sess


def fit_metrics(model, data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
       Return metrics of the model when fitted.
    """
    ddict = {}

    return ddict


def predict(model, sess=None, data_pars=None, compute_pars=None, out_pars=None, **kw):
    ##### Get Data ###############################################
    sentence1 = ['where can i find a book on machine learning']

    def sent_parse(sentence, mat_shape):
        sequence    = model.tokenizer.texts_to_sequences(sentence)
        padded_sent = pad_sequences(sequence, maxlen=data_pars["MAX_SEQUENCE_LENGTH"])
        return padded_sent  # [padded_sent, sent_one_hot]

    mysent = sent_parse(sentence1, [15])
    #### Do prediction
    ypred = model.model.predict(mysent, batch_size=16)

    ### Save Results

    ### Return val
    if kw.get("return_ytrue"):
        return ypred, sentence1
    else:
        return ypred, None


def reset_model():
    pass


def save(model=None, session=None, save_pars={}):
    from mlmodels.util import save_tf
    print(save_pars)
    save_tf(model, session, save_pars['path'])


def load(load_pars={}):
    from mlmodels.util import load_tf
    print(load_pars)
    input_tensors, output_tensors = load_tf( load_pars['path'],
                                             filename = load_pars['model_uri'])

    model       = Model()
    model.model = None
    session     = None
    return model, session


####################################################################################################
def get_dataset(data_pars=None, **kw):
    """
      JSON data_pars to get dataset
    """

    texts = []
    with codecs.open(data_pars["train_data_path"], encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for values in reader:
            texts.append(values[3])
            texts.append(values[4])
    print('Found %s texts in train.csv' % len(texts))

    embeddings_index = {}
    f = open(data_pars["glove_embedding"], encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    return texts, embeddings_index


def get_params(param_pars={}, **kw):
    pp = param_pars
    choice = pp['choice']
    config_mode = pp['config_mode']
    data_path = pp['data_path']

    if choice == "test01":
        log("#### Path params   ###################################################")
        data_path  = path_norm("dataset/text/quora/train.csv")
        glove_path = path_norm("dataset/text/glove/glove.6B.50d.txt")    #Big file
        out_path   = path_norm("ztest/model_keras/textvae/")

        data_pars  = {"MAX_SEQUENCE_LENGTH": 15, "train_data_path": data_path, "glove_embedding": glove_path}

        log("#### Model params   #################################################")
        model_pars   = {"MAX_NB_WORDS": 12000,  "EMBEDDING_DIM": 50, "latent_dim": 32,
                        "intermediate_dim": 96, "epsilon_std": 0.1,  "num_sampled": 500, "optimizer": "adam"}
        compute_pars = {"batch_size": 100, "epochs": 1, "VALIDATION_SPLIT": 0.2}
        out_pars     = {"path": out_path}

        return model_pars, data_pars, compute_pars, out_pars

    else:
        raise Exception(f"Not support choice {choice} yet")


################################################################################################
########## Tests are normalized Do not Change ##################################################
def test(data_path="dataset/", pars_choice="json", config_mode="test"):
    ### Local test

    log("#### Loading params   ##############################################")
    param_pars = {"choice": pars_choice, "data_path": data_path, "config_mode": config_mode}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)

    log("#### Loading dataset   #############################################")
    xtuple = get_dataset(data_pars)

    log("#### Model init, fit   #############################################")
    session = None
    model = Model(model_pars, data_pars, compute_pars)
    model, session = fit(model, data_pars, compute_pars, out_pars)

    log("#### Predict   #####################################################")
    ypred, _ = predict(model, session, data_pars, compute_pars, out_pars)

    log("#### metrics   #####################################################")
    metrics_val = fit_metrics(model, data_pars, compute_pars, out_pars)
    print(metrics_val)

    log("#### Plot   ########################################################")

    log("#### Save/Load   ###################################################")
    # save(model, session, out_pars)
    # model2 = load(out_pars)
    #     ypred = predict(model2, data_pars, compute_pars, out_pars)
    #     metrics_val = metrics(model2, ypred, data_pars, compute_pars, out_pars)
    # print(model2)


if __name__ == '__main__':
    VERBOSE = True
    test_path = os.getcwd() + "/mytest/"

    ### Local fixed params
    test(pars_choice="test01")

    ### Local json file
    # test(pars_choice="json")

    ####    test_module(model_uri="model_xxxx/yyyy.py", param_pars=None)
    from mlmodels.models import test_module

    param_pars = {'choice': "test01", 'config_mode': 'test', 'data_path': '/dataset/'}
    test_module(model_uri=MODEL_URI, param_pars=param_pars)

    ##### get of get_params
    # choice      = pp['choice']
    # config_mode = pp['config_mode']
    # data_path   = pp['data_path']

    ####    test_api(model_uri="model_xxxx/yyyy.py", param_pars=None)
    from mlmodels.models import test_api

    param_pars = {'choice': "test01", 'config_mode': 'test', 'data_path': '/dataset/'}
    test_api(model_uri=MODEL_URI, param_pars=param_pars)
