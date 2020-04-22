"""
Ensemble of preprocessor for time series, generic and re-usable

https://docs-time.giotto.ai/


https://pypi.org/project/tslearn/#documentation


https://pypi.org/project/skits/



https://github.com/awslabs/gluon-ts/issues/695


Gluon TS



"""
import os, sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict


from mlmodels.util import path_norm


import gluonts



####################################################################################################
def gluonts_dataset_to_pandas(dataset_name_list=["m4_hourly", "m4_daily", "m4_weekly", "m4_monthly", "m4_quarterly", "m4_yearly", ]):
    """
     n general, the datasets provided by GluonTS are objects that consists of three main members:

    dataset.train is an iterable collection of data entries used for training. Each entry corresponds to one time series
    dataset.test is an iterable collection of data entries used for inference. The test dataset is an extended version of the train dataset that contains a window in the end of each time series that was not seen during training. This window has length equal to the recommended prediction length.
    dataset.metadata contains metadata of the dataset such as the frequency of the time series, a recommended prediction horizon, associated features, etc.
    In [5]:
    entry = next(iter(dataset.train))
    train_series = to_pandas(entry)
    train_series.plot()

    # datasets = ["m4_hourly", "m4_daily", "m4_weekly", "m4_monthly", "m4_quarterly", "m4_yearly", ]


    """
    from gluonts.dataset.repository.datasets import get_dataset
    from gluonts.dataset.util import to_pandas

    ds_dict = {}
    for t in dataset_name_list :
      ds1 = get_dataset(t)
      print(ds1.train)
      ds_dict[t] = {}
      ds_dict[t]['train'] = to_pandas( next(iter(ds1.train)) )
      ds_dict[t]['test'] = to_pandas( next(iter(ds1.test))  ) 
      ds_dict[t]['metadata'] = ds1.metadata 


    return ds_dict



def gluonts_to_pandas(ds):
   from gluonts.dataset.util import to_pandas    
   ll =  [ to_pandas( t ) for t in ds ]
   return ll



def pandas_to_gluonts(df, pars=None) :
    """
       df.index : Should timestamp
       start date : part of index
       freq: Multiple of TimeStamp
          
    N = 10  # number of time series
    T = 100  # number of timesteps
    prediction_length = 24
    freq = "1H"
    custom_dataset = np.random.normal(size=(N, T))
    start = pd.Timestamp("01-01-2019", freq=freq)  # can be different for each time series
    
    from gluonts.dataset.common import ListDataset

    # train dataset: cut the last window of length "prediction_length", add "target" and "start" fields
    train_ds = ListDataset([{'target': x, 'start': start}
                            for x in custom_dataset[:, :-prediction_length]],
                           freq=freq)

    # test dataset: use the whole dataset, add "target" and "start" fields
    test_ds = ListDataset([{'target': x, 'start': start}
                           for x in custom_dataset],
                          freq=freq)

    test_target_values = train_target_values.copy()
    train_target_values = [ts[:-single_prediction_length] for ts in train_df.values]

    m5_dates = [pd.Timestamp("2011-01-29", freq='1D') for _ in range(len(sales_train_validation))]

    train_ds = ListDataset([
    {
        FieldName.TARGET: target,
        FieldName.START: start,
        FieldName.FEAT_DYNAMIC_REAL: fdr,
        FieldName.FEAT_STATIC_CAT: fsc
    }
    for (target, start, fdr, fsc) in zip(train_target_values,
                                         m5_dates,
                                         train_cal_features_list,
                                         stat_cat)
    ], freq="D")

   data = common.ListDataset([{"start": df.index[0],
                            "target": df.value[:"2015-04-05 00:00:00"]}],
                          freq="5min")

    #ds = ListDataset([{FieldName.TARGET: df.iloc[i].values,  
    #    FieldName.START:  pars['start']}  
    #                   for i in range(cols)], 
    #                   freq = pars['freq'])

    class gluonts.dataset.common.ListDataset(data_iter: Iterable[Dict[str, Any]], freq: str, one_dim_target: bool = True)[source]¶
    Bases: gluonts.dataset.common.Dataset
    
    Dataset backed directly by an array of dictionaries.
    
    data_iter Iterable object yielding all items in the dataset. Each item should be a dictionary mapping strings to values. For instance: {“start”: “2014-09-07”, “target”: [0.1, 0.2]}.
    freq Frequency of the observation in the time series. Must be a valid Pandas frequency.
    one_dim_target  Whether to accept only univariate target time series.

    """
    ### convert to gluont format
    from gluonts.dataset.common import ListDataset
    from gluonts.dataset.field_names import FieldName       

    cols_num     = pars.get('cols_num', [])
    cols_cat     = pars.get('cols_cat', [])
    cols_target  = pars.get('cols_target', [])
    freq         = pars.get("freq", "1d")

    m_series = len(cols_target)  #Nb of timeSeries
    
    
    y_list      = [ df[coli].values for coli in cols_target ]    # Actual Univariate Time Series
    dfnum_list  = [ df[coli].values for coli in cols_num   ]     # Time moving Category
    dfcat_list  = [ df[coli].values for coli in cols_cat   ]     # Static Category
    
    ### One start date per timeseries col
    sdate   = pars.get('start_date') 
    sdate   = df.index[0]  if sdate is None or len(sdate) == 0   else sdate  
    start_dates  = [ pd.Timestamp(sdate, freq=freq) if isinstance(sdate, str) else sdate for _ in range(len(y_list)) ]

    

    print(y_list, start_dates, dfnum_list, dfcat_list ) 
    ds_percol = [] 
    for i in range( m_series) :
        d = {  FieldName.TARGET             : y_list[i],       # start Timestamps
               FieldName.START            : start_dates[i],  # Univariate time series
            }
        if i < len(dfnum_list) :  d[ FieldName.FEAT_DYNAMIC_REAL ] = dfnum_list[i]  # Moving with time series
        if i < len(dfcat_list) :  d[ FieldName.FEAT_STATIC_CAT ] = dfnum_list[i]    # Static over time, like product_id
   
        ds_percol.append( d)
        print(d)

    ds = ListDataset(ds_percol,freq = freq)
    return ds 


def tofloat(x):
    try :
        return float(x)
    except :
        return np.nan

def tests():    
    df = pd.read_csv(path_norm("dataset/timeseries/TSLA.csv "))
    df = df.set_index("Date")
    pars = { "start" : "", "cols_target" : [ "High", "Low" ],
             "freq" :"1d",
             "cols_cat" : [],
             "cols_num" : []
            }    
    gts = pandas_to_gluonts(df, pars=pars) 
    print(gluonts_to_pandas( gts ) )    
    #for t in gts :
    #   print( to_pandas(t)[:10] )



    df = pd.read_csv(path_norm("dataset/timeseries/train_deepar.csv "))
    df = df.set_index("timestamp")
    df = pd_clean_v1(df)
    pars = { "start" : "", "cols_target" : [ "value" ],
             "freq" :"5min",
             "cols_cat" : [],
             "cols_num" : []
            }    
    gts = pandas_to_gluonts(df, pars=pars) 
    print(gluonts_to_pandas( gts ) )    


    #### To_
    dict_df = gluonts_dataset_to_pandas(dataset_name_list=["m4_hourly"])
    a = dict_df['m4_hourly']['train']




class Preprocess_nbeats:
    """
      it should go to nbeats.py BECAUSE Specialized code.
    """

    def __init__(self,backcast_length, forecast_length):
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
    def compute(self,df):
        df = df.values  # just keep np array here for simplicity.
        norm_constant = np.max(df)
        df = df / norm_constant
        
        x_train_batch, y = [], []
        for i in range(self.backcast_length, len(df) - self.forecast_length):
            x_train_batch.append(df[i - self.backcast_length:i])
            y.append(df[i:i + self.forecast_length])
    
        x_train_batch = np.array(x_train_batch)[..., 0]
        y = np.array(y)[..., 0]
        self.data = x_train_batch,y
        
    def get_data(self):
        return self.data
        
class SklearnMinMaxScaler:

    def __init__(self, **args):
        self.preprocessor = MinMaxScaler(**args)
    def compute(self,df):
        self.preprocessor.fit(df)
        self.data = self.preprocessor.transform(df)
        
    def get_data(self):
        return self.data



def pd_load(path) :
   return pd.read_csv(path_norm(path ))




def pd_interpolate(df, cols, pars={"method": "linear", "limit_area": "inside"  }):
    """
        Series.interpolate(self, method='linear', axis=0, limit=None, inplace=False, limit_direction='forward', limit_area=None, downcast=None, **kwargs)[source]¶
        Please note that only method='linear' is supported for DataFrame/Series with a MultiIndex.

        ‘linear’: Ignore the index and treat the values as equally spaced. This is the only method supported on MultiIndexes.
        ‘time’: Works on daily and higher resolution data to interpolate given length of interval.
        ‘index’, ‘values’: use the actual numerical values of the index.
        ‘pad’: Fill in NaNs using existing values.
        ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘spline’, ‘barycentric’, ‘polynomial’: Passed to scipy.interpolate.interp1d. These methods use the numerical values of the index. Both ‘polynomial’ and ‘spline’ require that you also specify an order (int), e.g. df.interpolate(method='polynomial', order=5).
        ‘krogh’, ‘piecewise_polynomial’, ‘spline’, ‘pchip’, ‘akima’: Wrappers around the SciPy interpolation methods of similar names. See Notes.
        ‘from_derivatives’: Refers to scipy.interpolate.BPoly.from_derivatives which replaces ‘piecewise_polynomial’ interpolation method in scipy 0.18.

        axis{0 or ‘index’, 1 or ‘columns’, None}, default None

        limitint, optional Maximum number of consecutive NaNs to fill. Must be greater than 0.

        limit_direction{‘forward’, ‘backward’, ‘both’}, default ‘forward’
        If limit is specified, consecutive NaNs will be filled in this direction.

        limit_area{None, ‘inside’, ‘outside’}, default None
        If limit is specified, consecutive NaNs will be filled with this restriction.
        None: No fill restriction.
        ‘inside’: Only fill NaNs surrounded by valid values (interpolate).
        ‘outside’: Only fill NaNs outside valid values (extrapolate).

        New in version 0.23.0.
        downcastoptional, ‘infer’ or None, defaults to None
        Downcast dtypes if possible.
    """
    for t in cols :
        df[t] = df[t].interpolate( **pars)

    return df



def pd_clean_v1(df, cols=None,  pars=None) :
  if pars is None :
     pars = {"method" : "linear", "axis": 0,
             }

  cols = df.columns if cols is None else cols
  for t in cols :
    df[t] = df[t].apply( tofloat )  
    df[t] = df[t].interpolate( **pars )
  return df


def pd_reshape(test, features, target, pred_len, m_feat) :
    x_test = test[features]
    x_test = x_test.values.reshape(-1, pred_len, m_feat)
    y_test = test[target]
    y_test = y_test.values.reshape(-1, pred_len, 1)        
    return x_test, y_test



def pd_clean(df, cols=None, pars=None ):
  cols = df.columns if cols is None else cols

  if pars is None :
     pars = {"method" : "linear", "axis": 0,}

  for t in cols :
    df[t] = df[t].fillna( **pars )
  
  return df




def time_train_test_split2(df , **kw):
    """
       train_data_path
       test_data_path
       predict_only 

    """
    d = kw
    pred_len = d["prediction_length"]
    features = d["col_Xinput"]
    target   = d["col_ytarget"]
    m_feat   = len(features)


    # when train and test both are provided
    if d["test_data_path"]:
        test   = pd_load(d["test_data_path"])
        test   = pd_clean(test)
        x_test, y_test = pd_reshape(test, features, target, pred_len, m_feat) 
        if d["predict_only"]:
            return x_test, y_test


        train   = pd_load( d["train_data_path"])
        #train   = pd_clean(train)
        x_train, y_train = pd_reshape(train, features, target, pred_len, m_feat) 

        return x_train, y_train, x_test, y_test
    

    # for when only train is provided
    df      = pd_load(d["train_data_path"])
    train   = df.iloc[:-pred_len]
    #train   = pd_clean(train)
    x_train, y_train = pd_reshape(train, features, target, pred_len, m_feat) 


    test   = df.iloc[-pred_len:]
    test   = pd_clean(test)
    x_test, y_test = pd_reshape(test, features, target, pred_len, m_feat) 
    if d["predict_only"]:
        return x_test, y_test

    return x_train, y_train, x_test, y_test


def time_train_test_split(data_pars):
    """
       train_data_path
       test_data_path
       predict_only

    """
    d = data_pars
    pred_len = d["prediction_length"]
    features = d["col_Xinput"]
    target   = d["col_ytarget"]
    m_feat   = len(features)


    # when train and test both are provided
    if d["test_data_path"]:
        test   = pd_load(d["test_data_path"])
        test   = pd_clean(test)
        x_test, y_test = pd_reshape(test, features, target, pred_len, m_feat) 
        if d["predict_only"]:
            return x_test, y_test


        train   = pd_load( d["train_data_path"])
        #train   = pd_clean(train)
        x_train, y_train = pd_reshape(train, features, target, pred_len, m_feat) 

        return x_train, y_train, x_test, y_test
    

    # for when only train is provided
    df      = pd_load(d["train_data_path"])
    train   = df.iloc[:-pred_len]
    #train   = pd_clean(train)
    x_train, y_train = pd_reshape(train, features, target, pred_len, m_feat) 


    test   = df.iloc[-pred_len:]
    test   = pd_clean(test)
    x_test, y_test = pd_reshape(test, features, target, pred_len, m_feat) 
    if d["predict_only"]:
        return x_test, y_test

    return x_train, y_train, x_test, y_test





def benchmark_m4() :
    # This example shows how to fit a model and evaluate its predictions.
    import pprint
    from functools import partial
    import pandas as pd

    from gluonts.dataset.repository.datasets import get_dataset
    from gluonts.distribution.piecewise_linear import PiecewiseLinearOutput
    from gluonts.evaluation import Evaluator
    from gluonts.evaluation.backtest import make_evaluation_predictions
    from gluonts.model.deepar import DeepAREstimator
    from gluonts.model.seq2seq import MQCNNEstimator
    from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
    from gluonts.trainer import Trainer

    datasets = ["m4_hourly", "m4_daily", "m4_weekly", "m4_monthly", "m4_quarterly", "m4_yearly", ]
    epochs = 100
    num_batches_per_epoch = 50

    estimators = [
        partial(  SimpleFeedForwardEstimator, trainer=Trainer(epochs=epochs, num_batches_per_epoch=num_batches_per_epoch ), ),
        
        partial(  DeepAREstimator, trainer=Trainer(epochs=epochs, num_batches_per_epoch=num_batches_per_epoch ), ),
        
        partial(  DeepAREstimator, distr_output=PiecewiseLinearOutput(8), trainer=Trainer(epochs=epochs, num_batches_per_epoch=num_batches_per_epoch ), ), 

        partial(  MQCNNEstimator, trainer=Trainer(epochs=epochs, num_batches_per_epoch=num_batches_per_epoch ), ), 
        ]


    def evaluate(dataset_name, estimator):
        dataset = get_dataset(dataset_name)
        estimator = estimator( prediction_length=dataset.metadata.prediction_length, freq=dataset.metadata.freq, use_feat_static_cat=True, 
                   cardinality=[ feat_static_cat.cardinality  for feat_static_cat in dataset.metadata.feat_static_cat
                   ],
        )

        print(f"evaluating {estimator} on {dataset}")
        predictor = estimator.train(dataset.train)

        forecast_it, ts_it = make_evaluation_predictions( dataset.test, predictor=predictor, num_samples=100 )
        agg_metrics, item_metrics = Evaluator()(ts_it, forecast_it, num_series=len(dataset.test) )
        pprint.pprint(agg_metrics)

        eval_dict = agg_metrics
        eval_dict["dataset"] = dataset_name
        eval_dict["estimator"] = type(estimator).__name__
        return eval_dict


    #if __name__ == "__main__":
    results = []
    for dataset_name in datasets:
        for estimator in estimators:
            # catch exceptions that are happening during training to avoid failing the whole evaluation
            try:
                results.append(evaluate(dataset_name, estimator))
            except Exception as e:
                print(str(e))


    df = pd.DataFrame(results)
    sub_df = df[ ["dataset", "estimator", "RMSE", "mean_wQuantileLoss", "MASE", "sMAPE", "OWA", "MSIS", ] ]
    print(sub_df.to_string())




####################################################################################################
if __name__ == '__main__':
   VERBOSE = True
   tests()
    














