  
# -*- coding: utf-8 -*-
"""
https://github.com/NTMC-Community/MatchZoo-py/tree/master/tutorials
https://matchzoo.readthedocs.io/en/master/model_reference.html

https://github.com/NTMC-Community/MatchZoo-py/blob/master/tutorials/classification/esim.ipynb

Match ZOO Architecture :

   Trainer : Core , gathers all components.
   Model :   BERT, ...
   Task :  Classification, Ranking,....




"""
import os, json
import importlib
from copy import deepcopy
import numpy as np
import pandas as pd


import torch
import matchzoo as mz
from mlmodels.util import os_package_root_path, log, path_norm, get_model_uri, path_norm_dict
from mlmodels.util import json_norm



###########################################################################################################
MODEL_URI = get_model_uri(__file__)

MODELS = {
    'DRMM'         : mz.models.DRMM,
    'DRMMTKS'      : mz.models.DRMMTKS,
    'ARC-I'        : mz.models.ArcI,
    'ARC-II'       : mz.models.ArcII,
    'DSSM'         : mz.models.DSSM,
    'CDSSM'        : mz.models.CDSSM,
    'MatchLSTM'    : mz.models.MatchLSTM,
    'DUET'         : mz.models.DUET,
    'KNRM'         : mz.models.KNRM,
    'ConvKNRM'     : mz.models.ConvKNRM,
    'ESIM'         : mz.models.ESIM,
    'BiMPM'        : mz.models.BiMPM,
    'MatchPyramid' : mz.models.MatchPyramid,
    'Match-SRNN'   : mz.models.MatchSRNN,
    'aNMM'         : mz.models.aNMM,
    'HBMP'         : mz.models.HBMP,
    'BERT'         : mz.models.Bert
}

TASKS = {
    'ranking' : mz.tasks.Ranking,
    'classification' : mz.tasks.Classification,
}

METRICS = {
    'NormalizedDiscountedCumulativeGain' : mz.metrics.NormalizedDiscountedCumulativeGain,
    'MeanAveragePrecision' : mz.metrics.MeanAveragePrecision,
    'acc' : 'acc'
}

LOSSES = {
    'RankHingeLoss'        : mz.losses.RankHingeLoss,
    'RankCrossEntropyLoss' : mz.losses.RankCrossEntropyLoss
}

from pytorch_transformers import AdamW
from torch.optim import Adadelta
OPTIMIZERS = {
    'ADAMW' : lambda prm, cp : AdamW(prm, lr=cp["lr"], betas=(cp["beta1"],cp["beta2"]), eps=cp["eps"]),
    'ADADELTA' : lambda prm, cp : Adadelta(prm, lr=cp["lr"], rho=cp["rho"], eps=cp["eps"], weight_decay=cp["weight_decay"])
}

CALLBACKS = {
    'PADDING' : lambda mn : MODELS[mn].get_default_padding_callback()
}


###########################################################################################################
def get_task(model_pars):
    _task = model_pars['task']
    assert _task in TASKS.keys()

    #### Task  #######################################
    if _task == "ranking":
        _loss = list(model_pars["loss"].keys())[0]
        _loss_params = model_pars["loss"][_loss]
        if _loss == 'RankHingeLoss':
            loss =  LOSSES[_loss]()

        elif _loss == 'RankCrossEntropyLoss':
            loss =  LOSSES[_loss](num_neg=_loss_params["num_neg"])
        task = mz.tasks.Ranking(losses=loss)

    elif _task == "classification" :
        task = mz.tasks.Classification(num_classes=model_pars["num_classes"])

    else:
        raise Exception(f"No support task {task} yet")

    #### Metrics  ####################################
    _metrics = model_pars['metrics']
    task.metrics = []
    for metric in _metrics.keys():
        metric_params = _metrics[metric]

        # Find a better way later to apply params for metric, for now hardcode.
        if metric == 'NormalizedDiscountedCumulativeGain' and metric_params != {}:
            task.metrics.append(METRICS[metric](k=metric_params["k"]))

        elif metric in METRICS:
            task.metrics.append(METRICS[metric]())
        else:
            raise Exception(f"No support of metric {metric} yet")
    return task


def get_glove_embedding_matrix(term_index, dimension):
    glove_embedding  = mz.datasets.embeddings.load_glove_embedding(dimension=dimension)
    embedding_matrix = glove_embedding.build_matrix(term_index)
    l2_norm          = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
    embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]
    return embedding_matrix



def get_data_loader(model_name, preprocessor, preprocess_pars, raw_data):

    pp = preprocess_pars

    if "transform" in pp:
        pack_processed = preprocessor.transform(raw_data)

    elif "fit_transform" in pp:
        pack_processed = preprocessor.fit_transform(raw_data)
    
    mode                       = pp.get("mode", "point")
    num_dup                    = pp.get("num_dup", 1)
    num_neg                    = pp.get("num_neg", 1)
    dataset_callback           = pp.get("dataset_callback")
    glove_embedding_matrix_dim = pp.get("glove_embedding_matrix_dim")

    if glove_embedding_matrix_dim:
        # Make sure you've transformed data before generating glove embedding,
        # else, term_index would be 0 and embedding matrix would be None.
        term_index = preprocessor.context['vocab_unit'].state['term_index']
        embedding_matrix = get_glove_embedding_matrix(term_index, glove_embedding_matrix_dim)


    if dataset_callback == "HISTOGRAM":
        # For now, hardcode callback. Hard to generalize
        dataset_callback = [mz.dataloader.callbacks.Histogram(
            embedding_matrix, bin_size=30, hist_mode='LCH'
        )]

    resample   = pp.get("resample")
    sort       = pp.get("sort")
    batch_size = pp.get("batch_size", 1)
    dataset = mz.dataloader.Dataset(
        data_pack  = pack_processed,
        mode       = mode,
        num_dup    = num_dup,
        num_neg    = num_neg,
        batch_size = batch_size,
        resample   = resample,
        sort       = sort,
        callbacks  = dataset_callback
    )

    stage               = pp.get("stage")
    dataloader_callback = pp.get("dataloader_callback")
    dataloader_callback = CALLBACKS[dataloader_callback](model_name)
    dataloader = mz.dataloader.DataLoader(
        device   = 'cpu',
        dataset  = dataset,
        stage    = stage,
        callback = dataloader_callback
    )
    return dataloader


"""
def update_model_param(params, model, task, preprocessor):
    model.params['task'] = task
    glove_embedding_matrix_dim = params.get("glove_embedding_matrix_dim")

    if glove_embedding_matrix_dim:
        term_index                = preprocessor.context['vocab_unit'].state['term_index']
        embedding_matrix          = get_glove_embedding_matrix(term_index, glove_embedding_matrix_dim)
        model.params['embedding'] = embedding_matrix
        # Remove those entried in JSON which not directly feeded to model as params
        del params["glove_embedding_matrix_dim"]

    # Feed rest all params directly to the model
    for key, value in params.items():
        model.params[key] = value
"""


def get_config_file():
    return os.path.join(os_package_root_path(__file__, 1), 'config', 'model_tch', 'Imagecnn.json')


def get_raw_dataset(data_pars, task):
    if data_pars["dataset"] == "WIKI_QA":
        filter_train_pack_raw = data_pars.get("preprocess").get("train").get("filter", False)
        filter_test_pack_raw  = data_pars.get("preprocess").get("test").get("filter", False)
        train_pack_raw        = mz.datasets.wiki_qa.load_data('train', task=task, filtered=filter_train_pack_raw)
        test_pack_raw         = mz.datasets.wiki_qa.load_data('test', task=task, filtered=filter_test_pack_raw)
        return train_pack_raw, test_pack_raw
    else:
        dataset_name = data_pars["dataset"]
        raise Exception(f"Not support choice {dataset_name} dataset yet")

###########################################################################################################
###########################################################################################################
class Model:
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None, out_pars=None):
        self.model_pars   = deepcopy( model_pars)
        self.compute_pars = deepcopy( compute_pars)
        self.data_pars    = deepcopy( data_pars)

        ### Model empty      ################################
        if model_pars is None :
            self.model = None
            return None
 


        ### Model Build : Expose directly the JSON to the MatchZoo Core API  ##############
        _model = model_pars['model']
        # assert _model in MODELS.keys()
        self.model = MODELS[_model]()

        ### Add static params
        mpars =json_norm(model_pars['model_pars'])
        for key, value in mpars.items():
            self.model.params[key] = value


        ### Add Task
        self.task = get_task(model_pars)
        self.model.params['task'] = self.task


        ### Add PreProcessor
        _preprocessor_pars = data_pars["preprocess"]
        if "basic_preprocessor" in _preprocessor_pars:
            pars = _preprocessor_pars["basic_preprocessor"]
            preprocessor = mz.preprocessors.BasicPreprocessor(
                truncated_length_left  = pars["truncated_length_left"],
                truncated_length_right = pars["truncated_length_right"],
                filter_low_freq        = pars["filter_low_freq"]
            )
        else:
            preprocessor = MODELS[_model].get_default_preprocessor()


        ### Add Embedding
        glove_embedding_matrix_dim = model_pars.get("glove_embedding_matrix_dim")
        if glove_embedding_matrix_dim:
            term_index                = preprocessor.context['vocab_unit'].state['term_index']
            embedding_matrix          = get_glove_embedding_matrix(term_index, glove_embedding_matrix_dim)
            self.model.params['embedding'] = embedding_matrix
            
            ## No need : we seprate Pure MatchZoo parameters
            # Remove those entried in JSON which not directly feeded to model as params
            # del params["glove_embedding_matrix_dim"]
        # update_model_param(model_pars["model_pars"], self.model, self.task, preprocessor)        

        self.model.build()        
        


        ### Data Loader        #####################################  : part of traimer
        train_pack_raw, test_pack_raw = get_raw_dataset(data_pars, self.task)
        self.trainloader = get_data_loader(_model, preprocessor, _preprocessor_pars["train"], train_pack_raw)
        self.testloader  = get_data_loader(_model, preprocessor, _preprocessor_pars["test"], test_pack_raw)









def fit(model, data_pars=None, compute_pars=None, out_pars=None, **kwargs):
    model0 = model.model
    #epochs = compute_pars["epochs"]


    #######  Add optimizer
    optimize_parameters = compute_pars.get("optimizie_parameters", False)
    if optimize_parameters:
        # Currently hardcode optimized parameters for Bert,
        # Hard to generalize.
        no_decay = ['bias', 'LayerNorm.weight']
        model_parameters = [
            {'params': [p for n, p in model0.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 5e-5},
            {'params': [p for n, p in model0.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    else:
        model_parameters = model0.parameters()

    
    optimizer_ = list(compute_pars["optimizer"].keys())[0]
    optimizer = OPTIMIZERS[optimizer_](model_parameters, compute_pars["optimizer"][optimizer_])


    ### Expose all Trainer class : Static pars, dynamic pars  #####################################
    #### Static params from JSON
    train_pars = compute_pars.get('compute_pars', {})
    train_pars = json_norm(train_pars)
    # train_pars['epoch']       = epochs   # implciit in the JSON


    #### Dynamic params
    train_pars['model']       = model.model
    train_pars['optimizer']   = optimizer
    train_pars['trainloader'] = model.trainloader
    train_pars['validloader'] = model.testloader
 
    trainer = mz.trainers.Trainer( ** train_pars)

    """
    trainer = mz.trainers.Trainer(
                model             = model.model,
                optimizer         = optimizer,
                trainloader       = model.trainloader,
                validloader       = model.testloader,
                validate_interval = None,
                epochs            = epochs
            )
    """        
    trainer.run()

    #### trainer Acts as trainer (like in tensorflow)
    return model, trainer


def predict(model, session=None, data_pars=None, compute_pars=None, out_pars=None):
    """
        https://github.com/NTMC-Community/MatchZoo-py/blob/master/matchzoo/trainers/trainer.py#L341

       Trainer:
        def predict(
        self,
        dataloader: DataLoader
    ) -> np.array:
        
        Generate output predictions for the input samples.
        dataloader: input DataLoader
        :return: predictions
    
        with torch.no_grad():
            self._model.eval()
            predictions = []
            for batch in dataloader:
                inputs = batch[0]
                outputs = self._model(inputs).detach().cpu()
                predictions.append(outputs)
            self._model.train()
            return torch.cat(predictions, dim=0).numpy()

    """

    ### Data Loader        #####################################
    data_pars['train'] = 0
    test_pack_raw = get_raw_dataset(data_pars, model.task)
    
    _preprocessor_pars = data_pars["preprocess"]
    if "basic_preprocessor" in _preprocessor_pars:
        pars = _preprocessor_pars["basic_preprocessor"]
        preprocessor = mz.preprocessors.BasicPreprocessor(
            truncated_length_left  = pars["truncated_length_left"],
            truncated_length_right = pars["truncated_length_right"],
            filter_low_freq        = pars["filter_low_freq"]
        )
    else:
        preprocessor = model.model.get_default_preprocessor()

    testloader  = get_data_loader(model.model, preprocessor, _preprocessor_pars["test"], test_pack_raw)


    ### Model Predict applied on session = Trainer()
    ypred = session.predict(testloader)
    return ypred



def fit_metrics(model, data_pars=None, compute_pars=None, out_pars=None):
    pass


def save(model, session=None, save_pars=None):
    """
      trainer == session
          save_dir: Directory to save trainer.
`       save_all: Bool. If True, save `Trainer` instance; If False,
        only save model. Defaults to False.

     https://github.com/NTMC-Community/MatchZoo-py/blob/master/matchzoo/trainers/trainer.py#L369   

    """
    session.save_dir = save_pars['path']   # save_dir: Directory to save trainer.
    session.save()
    session.save_model()



def load(load_pars):
    """
     need trainer instance
     https://github.com/NTMC-Community/MatchZoo-py/blob/master/matchzoo/trainers/trainer.py#L415
 
    """
    pass




def get_params(param_pars=None, **kw):
    pp          = param_pars
    choice      = pp['choice']
    model_name = pp['model_name']
    data_path   = pp['data_path']

    if choice == "json":
        data_path = path_norm(data_path)
        cf = json.load(open(data_path, mode='r'))
        cf = cf[model_name]

        ####Normalize path  : add /models/dataset/
        cf['data_pars'] = path_norm_dict(cf['data_pars'])
        cf['out_pars']  = path_norm_dict(cf['out_pars'])

        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']

    else:
        raise Exception(f"Not support choice {choice} yet")

###########################################################################################################
###########################################################################################################
def test_train(data_path, pars_choice, model_name):
    ### Local test

    log("#### Loading params   ##############################################")
    param_pars = {"choice":pars_choice,  "data_path":data_path,  "model_name": model_name}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
    log(  data_pars, out_pars )

    log("#### Loading dataset   #############################################")
    #xtuple = get_dataset(data_pars)


    log("#### Model init   ##################################################")
    session = None
    model = Model(model_pars, data_pars, compute_pars)



    log("#### Model  fit   #############################################")
    model, session = fit(model, data_pars, compute_pars, out_pars)


    log("#### Predict   #####################################################")
    #ypred = predict(model, session, data_pars, compute_pars, out_pars)


    log("#### metrics   #####################################################")
    #metrics_val = fit_metrics(model, data_pars, compute_pars, out_pars)
    # print(metrics_val)


    log("#### Plot   ########################################################")


    log("#### Save   ########################################################")
    save_pars = { "path": out_pars["path"]  }
    save(model=model, save_pars=save_pars)


    log("#### Load   ###################################################")
    model2 = load( save_pars )


    log("#### Predict after Load   ###########################################")
    ypred = predict(model2, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
    print(model2)





if __name__ == "__main__":
    test_train(data_path="model_tch/matchzoo_models.json", pars_choice="json", model_name="BERT_RANKING")

    # test_train(data_path="model_tch/matchzoo_models.json", pars_choice="json", model_name="BERT_RANKING")




"""
class Trainer:
            model: BaseModel,
        optimizer: optim.Optimizer,
        trainloader: DataLoader,
        validloader: DataLoader,
        device: typing.Union[torch.device, int, list, None] = None,
        start_epoch: int = 1,
        epochs: int = 10,
        validate_interval: typing.Optional[int] = None,
        scheduler: typing.Any = None,
        clip_norm: typing.Union[float, int] = None,
        patience: typing.Optional[int] = None,
        key: typing.Any = None,
        checkpoint: typing.Union[str, Path] = None,
        save_dir: typing.Union[str, Path] = None,
        save_all: bool = False,
        verbose: int = 1,

    MatchZoo tranier.
    model: A :class:`BaseModel` instance.
    optimizer: A :class:`optim.Optimizer` instance.
    trainloader: A :class`DataLoader` instance. The dataloader
        is used for training the model.
    validloader: A :class`DataLoader` instance. The dataloader
        is used for validating the model.
    device: The desired device of returned tensor. Default:
        if None, use the current device. If `torch.device` or int,
        use device specified by user. If list, use data parallel.
    start_epoch: Int. Number of starting epoch.
    epochs: The maximum number of epochs for training.
        Defaults to 10.
    validate_interval: Int. Interval of validation.
    scheduler: LR scheduler used to adjust the learning rate
        based on the number of epochs.
    clip_norm: Max norm of the gradients to be clipped.
    patience: Number fo events to wait if no improvement and
        then stop the training.
    key: Key of metric to be compared.
    checkpoint: A checkpoint from which to continue training.
        If None, training starts from scratch. Defaults to None.
        Should be a file-like object (has to implement read, readline,
        tell, and seek), or a string containing a file name.
    save_dir: Directory to save trainer.
    save_all: Bool. If True, save `Trainer` instance; If False,
        only save model. Defaults to False.
    verbose: 0, 1, or 2. Verbosity mode. 0 = silent,
        1 = verbose, 2 = one log line per epoch.


"""