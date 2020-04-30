# -*- coding: utf-8 -*-
"""
Advanded GlutonTS models

"""
import os, copy
import pandas as pd, numpy as np


import matplotlib.pyplot as plt
from pathlib import Path
import json


from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.gp_forecaster import GaussianProcessEstimator
from gluonts.model.seq2seq import Seq2SeqEstimator
from gluonts.model.transformer import TransformerEstimator
from gluonts.model.simple_feedforward import  SimpleFeedForwardEstimator
from gluonts.model.wavenet import WaveNetEstimator, WaveNetSampler, WaveNet



from gluonts.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.predictor import Predictor


#### Only for SeqtoSeq
from gluonts.block.encoder import (
    HierarchicalCausalConv1DEncoder,
    RNNCovariateEncoder,
    MLPEncoder,
    Seq2SeqEncoder,  # Buggy, not implemented
)


####################################################################################################
from mlmodels.util import os_package_root_path, log, path_norm, get_model_uri, json_norm


VERBOSE = False
MODEL_URI = get_model_uri(__file__)


MODELS_DICT = {
"deepar"         : DeepAREstimator
,"deepstate"     : DeepStateEstimator
,"deepfactor"    : DeepFactorEstimator
,"gp_forecaster" : GaussianProcessEstimator
,"seq2seq"       : Seq2SeqEstimator
,"feedforward"   : SimpleFeedForwardEstimator
,"transformer"   : TransformerEstimator
,"wavenet"       : WaveNetEstimator
}


####################################################################################################
class Model(object):
    def __init__(self, model_pars=None, data_pars=None,  compute_pars=None, **kwargs):
        self.compute_pars = compute_pars
        self.model_pars   = model_pars
        self.data_pars    = data_pars
        

        ##### Empty model for Seiialization
        if model_pars is None :
            self.model = None

        else:
            mpars = json_norm(model_pars['model_pars'] )      #"None" to None
            cpars = json_norm(compute_pars['compute_pars'])
            
            if model_pars["model_name"] == "seq2seq" :
                mpars['encoder'] = MLPEncoder()   #bug in seq2seq
            
            
            ### Setup the compute
            trainer = Trainer( **cpars  )

            ### Setup the model
            self.model = MODELS_DICT[model_pars["model_name"]]( trainer=trainer, **mpars )


def get_params(choice="", data_path="dataset/timeseries/", config_mode="test", **kw):
    if choice == "json":
      data_path = path_norm( data_path )
      config    = json.load(open(data_path, encoding='utf-8'))
      config    = config[config_mode]
      
      return config["model_pars"], config["data_pars"], config["compute_pars"], config["out_pars"]
  
    else :
        raise Exception("Error no JSON FILE") 



def get_dataset(data_pars):    

    from mlmodels.preprocess.timeseries import pandas_to_gluonts, pd_clean_v1

    data_path  = data_pars['train_data_path'] if data_pars['train'] else data_pars['test_data_path']
    data_path  = path_norm( data_path )

    df = pd.read_csv(data_path)
    df = df.set_index( data_pars['col_date'] )
    df = pd_clean_v1(df)

    # start_date = pd.Timestamp( data_pars['start'], freq=data_pars['freq'])
    pars = { "start" : data_pars['start'], 
             "cols_target" : data_pars['col_ytarget'],
             "freq"        : data_pars['freq'],
             "cols_cat"    : data_pars["cols_cat"],
             "cols_num"    : data_pars["cols_num"]
        }    
    gluonts_ds = pandas_to_gluonts(df, pars=pars) 
 
    if VERBOSE:
        entry        = next(iter(gluonts_ds))
        train_series = to_pandas(entry)
        train_series.plot()
        save_fig     = data_pars.get('save_fig', "save_fig.png")
        # plt.savefig(save_fig)
    return gluonts_ds



def fit(model, sess=None, data_pars=None, model_pars=None, compute_pars=None, out_pars=None, session=None, **kwargs):
        """
          Classe Model --> model,   model.model contains thte sub-model
        """
        data_pars['train'] = True
        model_gluon        = model.model
        gluont_ds          = get_dataset(data_pars)
        predictor          = model_gluon.train(gluont_ds)
        model.model        = predictor
        return model


def predict(model, sess=None, data_pars=None, compute_pars=None, out_pars=None, **kw):
    
    data_pars['train'] = False
    test_ds            = get_dataset(data_pars)
    model_gluon        = model.model
    
    forecast_it, ts_it = make_evaluation_predictions(
            dataset     = test_ds,      # test dataset
            predictor   = model_gluon,  # predictor
            num_samples = compute_pars['num_samples'],  # number of sample paths we want for evaluation
        )

    forecasts, tss = list(forecast_it), list(ts_it)
    forecast_entry, ts_entry = forecasts[0], tss[0]

    ### External benchmark.py evaluation
    if kw.get("return_ytrue") :
        ypred, ytrue = forecasts, tss
        return ypred, ytrue

    if VERBOSE:
        print(f"Number of sample paths: {forecast_entry.num_samples}")
        print(f"Dimension of samples: {forecast_entry.samples.shape}")
        print(f"Start date of the forecast window: {forecast_entry.start_date}")
        print(f"Frequency of the time series: {forecast_entry.freq}")
        print(f"Mean of the future window:\n {forecast_entry.mean}")
        print(f"0.5-quantile (median) of the future window:\n {forecast_entry.quantile(0.5)}")

    dd = {"forecasts": forecasts, "tss": tss}
    return dd



def metrics(ypred, data_pars, compute_pars=None, out_pars=None, **kw):
        ## load test dataset
        data_pars['train'] = False
        test_ds = get_dataset(data_pars)

        forecasts = ypred["forecasts"]
        tss = ypred["tss"]

        ## Evaluate
        evaluator = Evaluator(quantiles=out_pars['quantiles'])
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
        metrics_dict = json.dumps(agg_metrics, indent=4)
        return metrics_dict, item_metrics



def fit_metrics(ypred, data_pars, compute_pars=None, out_pars=None, **kw):
        ### load test dataset
        data_pars['train'] = False
        test_ds = get_dataset(data_pars)

        forecasts = ypred["forecasts"]
        tss = ypred["tss"]

        ### Evaluate
        evaluator = Evaluator(quantiles=out_pars['quantiles'])
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
        metrics_dict = json.dumps(agg_metrics, indent=4)
        return metrics_dict, item_metrics



def save(model, path):
    import pickle
    path = path_norm(path + "/gluonts_model/")
    os.makedirs(path, exist_ok = True)

    model.model.serialize(Path(path) )   
    d = {"model_pars"  :  model.model_pars, 
         "compute_pars":  model.compute_pars,
         "data_pars"   :  model.data_pars
        }
    pickle.dump(d, open(path + "/glutonts_model_pars.pkl", mode="wb"))
    log(os.listdir(path))


def load(path):
    import pickle
    path = path_norm(path  + "/gluonts_model/" )

    predictor_deserialized = Predictor.deserialize(Path(path))
    d = pickle.load( open(path + "/glutonts_model_pars.pkl", mode="rb")  )
    
    ### Setup Model
    model = Model(model_pars= d['model_pars'], compute_pars= d['compute_pars'],
                  data_pars= d['data_pars'])  

    model.model = predictor_deserialized

    return model


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
    outpath = out_pars['path']
    os.makedirs(outpath, exist_ok=True)
    plt.savefig(outpath)
    plt.clf()
    print('Saved image to {}.'.format(outpath))



####################################################################################################
def test_single(data_path="dataset/", choice="", config_mode="test"):
    model_uri = MODEL_URI
    log("#### Loading params   ##############################################")
    log( MODEL_URI)
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=choice, data_path=data_path, config_mode=config_mode)
    print(model_pars, data_pars, compute_pars, out_pars)

    log("#### Loading dataset   #############################################")
    gluonts_ds = get_dataset(data_pars)

    log("#### Model init, fit   #############################################")
    from mlmodels.models import module_load_full
    module, model = module_load_full(model_uri, model_pars, data_pars, compute_pars)
    print(module, model)

    model = fit(model, sess=None, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
    print(model)

    log("#### Save the trained model  ######################################")
    save(model, out_pars["path"])


    log("#### Load the trained model  ######################################")
    model = load(out_pars["path"])

    log("#### Predict   ####################################################")
    ypred = predict(model, sess=None, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
    # print(ypred)

    log("#### metrics   ####################################################")
    metrics_val, item_metrics = metrics(ypred, data_pars, compute_pars, out_pars)
    print(metrics_val)

    log("#### Plot   #######################################################")
    if VERBOSE :
      plot_prob_forecasts(ypred, out_pars)
      plot_predict(item_metrics, out_pars)



def test() :
    ll = [ "deepar" , "deepfactor" , "transformer"  ,"wavenet", "feedforward",
           "gp_forecaster", "deepstate" ]

    ## Not yet  Implemented, error in Glutonts
    ll2 = [   "seq2seq"  ]
    
    for t in ll  :
      test_single(data_path="model_gluon/gluonts_model.json", choice="json", config_mode= t )



if __name__ == '__main__':
    VERBOSE = False

    test()






INFO = """
DeepStateEstimator,
    This implements the deep state space model described in
    [RSG+18]_.


    freq
        Frequency of the data to train on and predict
    prediction_length
        Length of the prediction horizon
    cardinality
        Number of values of each categorical feature.
        This must be set by default unless ``use_feat_static_cat``
        is set to `False` explicitly (which is NOT recommended).
    add_trend
        Flag to indicate whether to include trend component in the
        state space model
    past_length
        This is the length of the training time series;
        i.e., number of steps to unroll the RNN for before computing 
        predictions.
        Set this to (at most) the length of the shortest time series in the 
        dataset.
        (default: None, in which case the training length is set such that 
        at least
        `num_seasons_to_train` seasons are included in the training.
        See `num_seasons_to_train`)
    num_periods_to_train
        (Used only when `past_length` is not set)
        Number of periods to include in the training time series. (default: 4)
        Here period corresponds to the longest cycle one can expect given 
        the granularity of the time series.
        See: https://stats.stackexchange.com/questions/120806/frequency
        -value-for-seconds-minutes-intervals-data-in-r

WaveNetEstimator
        Model with Wavenet architecture and quantized target.

        freq
            Frequency of the data to train on and predict
        prediction_length
            Length of the prediction horizon
        trainer
            Trainer object to be used (default: Trainer())
        cardinality
            Number of values of the each categorical feature (default: [1])
        embedding_dimension
            Dimension of the embeddings for categorical features (the same
            dimension is used for all embeddings, default: 5)
        num_bins
            Number of bins used for quantization of signal (default: 1024)
        hybridize_prediction_net
            Boolean (default: False)
        n_residue
            Number of residual channels in wavenet architecture (default: 24)
        n_skip
            Number of skip channels in wavenet architecture (default: 32)
        dilation_depth
            Number of dilation layers in wavenet architecture.
            If set to None (default), dialation_depth is set such that the receptive length is at least
            as long as typical seasonality for the frequency and at least 2 * prediction_length.
        n_stacks
            Number of dilation stacks in wavenet architecture (default: 1)
        temperature
            Temparature used for sampling from softmax distribution.
            For temperature = 1.0 (default) sampling is according to estimated probability.
        act_type
            Activation type used after before output layer (default: "elu").
            Can be any of 'elu', 'relu', 'sigmoid', 'tanh', 'softrelu', 'softsign'.
        num_parallel_samples
            Number of evaluation samples per time series to increase parallelism during inference.
            This is a model optimization that does not affect the accuracy (default: 200)



DeepFactorEstimator(GluonEstimator):

    DeepFactorEstimator is an implementation of the 2019 ICML paper "Deep Factors for Forecasting"
    https://arxiv.org/abs/1905.12417.  It uses a global RNN model to learn patterns across multiple related time series
    and an arbitrary local model to model the time series on a per time series basis.  In the current implementation,
    the local model is a RNN (DF-RNN).

    freq
        Time series frequency.
    prediction_length
        Prediction length.
    num_hidden_global
        Number of units per hidden layer for the global RNN model (default: 50).
    num_layers_global
        Number of hidden layers for the global RNN model (default: 1).
    num_factors
        Number of global factors (default: 10).
    num_hidden_local
        Number of units per hidden layer for the local RNN model (default: 5).
    num_layers_local
        Number of hidden layers for the global local model (default: 1).
    cell_type
        Type of recurrent cells to use (available: 'lstm' or 'gru';
        default: 'lstm').
    trainer
        Trainer object to be used (default: Trainer()).
    context_length
        Training length (default: None, in which case context_length = prediction_length).
    num_parallel_samples
        Number of evaluation samples per time series to increase parallelism during inference.
        This is a model optimization that does not affect the accuracy (default: 100).
    cardinality
        List consisting of the number of time series (default: list([1]).
    embedding_dimension
        Dimension of the embeddings for categorical features (the same
        dimension is used for all embeddings, default: 10).
    distr_output
        Distribution to use to evaluate observations and sample predictions
        (default: StudentTOutput()).



GaussianProcessEstimator shows how to build a local time series model using
Gaussian Processes (GP).
    Each time series has a GP with its own
    hyper-parameters.  For the radial basis function (RBF) Kernel, the
    learnable hyper-parameters are the amplitude and lengthscale. The periodic
    kernel has those hyper-parameters with an additional learnable frequency
    parameter. The RBFKernel is the default, but either kernel can be used by
    inputting the desired KernelOutput object. The noise sigma in the model is
    another learnable hyper-parameter for both kernels. These parameters are
    fit using an Embedding of the integer time series indices (each time series
    has its set of hyper-parameter that is static in time). The observations
    are the time series values. In this model, the time features are hour of
    the day and day of the week.

    freq
        Time series frequency.
    prediction_length
        Prediction length.
    cardinality
        Number of time series.
    trainer
        Trainer instance to be used for model training (default: Trainer()).
    context_length
        Training length (default: None, in which case context_length = prediction_length).
    kernel_output
        KernelOutput instance to determine which kernel subclass to be
        instantiated (default: RBFKernelOutput()).
    params_scaling
        Determines whether or not to scale the model parameters (default: True).
    float_type
        Determines whether to use single or double precision (default: np.float64).
    max_iter_jitter
        Maximum number of iterations for jitter to iteratively make the matrix positive definite (default: 10).
    jitter_method
        Iteratively jitter method or use eigenvalue decomposition depending on problem size (default: "iter").
    sample_noise
        Boolean to determine whether to add :math:`\sigma^2I` to the predictive covariance matrix (default: True).
    time_features
        Time features to use as inputs of the model (default: None, in which
        case these are automatically determined based on the frequency).
    num_parallel_samples
        Number of evaluation samples per time series to increase parallelism during inference.
        This is a model optimization that does not affect the accuracy (default: 100).


Seq2SeqEstimator(GluonEstimator):
    Quantile-Regression Sequence-to-Sequence Estimator
        freq: str,
        prediction_length: int,
        cardinality: List[int],
        embedding_dimension: int,
        encoder: Seq2SeqEncoder,
        decoder_mlp_layer: List[int],
        decoder_mlp_static_dim: int,
        scaler: Scaler = NOPScaler(),
        context_length: Optional[int] = None,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        trainer: Trainer = Trainer(),
        num_parallel_samples: int = 100,
    ) -> None:
        


TransformerEstimator(GluonEstimator):
        Construct a Transformer estimator.
        This implements a Transformer model, close to the one described in
        [Vaswani2017]_.
        .. [Vaswani2017] Vaswani, Ashish, et al. "Attention is all you need."
            Advances in neural information processing systems. 2017.
        freq
            Frequency of the data to train on and predict
        prediction_length
            Length of the prediction horizon
        context_length
            Number of steps to unroll the RNN for before computing predictions
            (default: None, in which case context_length = prediction_length)
        trainer
            Trainer object to be used (default: Trainer())
        dropout_rate
            Dropout regularization parameter (default: 0.1)
        cardinality
            Number of values of the each categorical feature (default: [1])
        embedding_dimension
            Dimension of the embeddings for categorical features (the same
            dimension is used for all embeddings, default: 5)
        distr_output
            Distribution to use to evaluate observations and sample predictions
            (default: StudentTOutput())
        model_dim
            Dimension of the transformer network, i.e., embedding dimension of the input
            (default: 32)
        inner_ff_dim_scale
            Dimension scale of the inner hidden layer of the transformer's
            feedforward network (default: 4)
        pre_seq
            Sequence that defined operations of the processing block before the main transformer
            network. Available operations: 'd' for dropout, 'r' for residual connections
            and 'n' for normalization (default: 'dn')
        post_seq
            seq
            Sequence that defined operations of the processing block in and after the main
            transformer network. Available operations: 'd' for dropout, 'r' for residual connections
            and 'n' for normalization (default: 'drn').
        act_type
            Activation type of the transformer network (default: 'softrelu')
        num_heads
            Number of heads in the multi-head attention (default: 8)
        scaling
            Whether to automatically scale the target values (default: true)
        lags_seq
            Indices of the lagged target values to use as inputs of the RNN
            (default: None, in which case these are automatically determined
            based on freq)
        time_features
            Time features to use as inputs of the RNN (default: None, in which
            case these are automatically determined based on freq)
        num_parallel_samples
            Number of evaluation samples per time series to increase parallelism during inference.
            This is a model optimization that does not affect the accuracy (default: 100)

        
"""