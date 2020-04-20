"""
M5 Forecasting Competition GluonTS Template¶
This notebook can be used as a starting point for participating in the M5 forecasting competition using GluonTS-based tooling.

"""

########################
# %matplotlib inline
import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from tqdm.autonotebook import tqdm
from pathlib import Path


"""
We also define globally accessible variables, such as the pred length and the input path for the M5 data. Note that single_pred_length corresponds to the length of the val/evaluation periods, while submission_pred_length corresponds to the length of both these periods combined.

By default the notebook is configured to run in submission mode (submission will be True), which means that we use all of the data for training and predict new values for a total length of submission_pred_length for which we don't have ground truth values available (performance can be assessed by submitting pred results to Kaggle). In contrast, setting submission to False will instead use the last single_pred_length-many values of our training set as val points (and hence these values will not be used for training), which enables us to validate our model's performance offline.
"""
########################
single_pred_length = 28
submission_pred_length = single_pred_length * 2
m5_input_path="./m5-forecasting-accuracy"
submission=True

if submission:
    pred_length = submission_pred_length
else:
    pred_length = single_pred_length


"""    
Reading the M5 data into GluonTS
First we need to convert the provided M5 data into a format that is readable by GluonTS. At this point we assume that the M5 data, which can be downloaded from Kaggle, is present under m5_input_path.
"""

########################
calendar               = pd.read_csv(f'{m5_input_path}/calendar.csv')
sales_train_val        = pd.read_csv(f'{m5_input_path}/sales_train_val.csv')
sample_submission      = pd.read_csv(f'{m5_input_path}/sample_submission.csv')
sell_prices            = pd.read_csv(f'{m5_input_path}/sell_prices.csv')






"""
We start the data convertion process by building dynamic features 
(features that change over time, just like the target values). 
Here, we are mainly interested in the event indicators event_type_1 and event_type_2. 
We will mostly drop dynamic time features as GluonTS will automatically add 
some of these as part of many models' transformation chains.


"""
########################
cal_features = calendar.drop(
    ['date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year', 'event_name_1', 'event_name_2', 'd'], 
    axis=1
)
cal_features['event_type_1'] = cal_features['event_type_1'].apply(lambda x: 0 if str(x)=="nan" else 1)
cal_features['event_type_2'] = cal_features['event_type_2'].apply(lambda x: 0 if str(x)=="nan" else 1)

test_cal_features = cal_features.values.T
if submission:
    train_cal_features = test_cal_features[:,:-submission_pred_length]
else:
    train_cal_features = test_cal_features[:,:-submission_pred_length-single_pred_length]
    test_cal_features = test_cal_features[:,:-submission_pred_length]

test_cal_features_list = [test_cal_features] * len(sales_train_val)
train_cal_features_list = [train_cal_features] * len(sales_train_val)





"""
# We then go on to build static features (features which are constant and series-specific).
 Here, we make use of all categorical features that are provided to us as part of the M5 data.
"""
########################
state_ids = sales_train_val["state_id"].astype('category').cat.codes.values
state_ids_un , state_ids_counts = np.unique(state_ids, return_counts=True)

store_ids = sales_train_val["store_id"].astype('category').cat.codes.values
store_ids_un , store_ids_counts = np.unique(store_ids, return_counts=True)

cat_ids = sales_train_val["cat_id"].astype('category').cat.codes.values
cat_ids_un , cat_ids_counts = np.unique(cat_ids, return_counts=True)

dept_ids = sales_train_val["dept_id"].astype('category').cat.codes.values
dept_ids_un , dept_ids_counts = np.unique(dept_ids, return_counts=True)

item_ids = sales_train_val["item_id"].astype('category').cat.codes.values
item_ids_un , item_ids_counts = np.unique(item_ids, return_counts=True)

stat_cat_list = [item_ids, dept_ids, cat_ids, store_ids, state_ids]

stat_cat = np.concatenate(stat_cat_list)
stat_cat = stat_cat.reshape(len(stat_cat_list), len(item_ids)).T

stat_cat_cardinalities = [len(item_ids_un), len(dept_ids_un), len(cat_ids_un), len(store_ids_un), len(state_ids_un)]

# Finally, we can build both the training and the testing set from target values and both static and dynamic features.
########################
from gluonts.dataset.common import load_datasets, ListDataset
from gluonts.dataset.field_names import FieldName

train_df = sales_train_val.drop(["id","item_id","dept_id","cat_id","store_id","state_id"], axis=1)
train_target_values = train_df.values

if submission == True:
    test_target_values = [np.append(ts, np.ones(submission_pred_length) * np.nan) for ts in train_df.values]
else:
    test_target_values = train_target_values.copy()
    train_target_values = [ts[:-single_pred_length] for ts in train_df.values]

m5_dates = [pd.Timestamp("2011-01-29", freq='1D') for _ in range(len(sales_train_val))]

train_ds = ListDataset([
    {
        FieldName.TARGET: target,
        FieldName.START: start,
        FieldName.FEAT_DYNAMIC_REAL: fdr,
        FieldName.FEAT_STATIC_CAT: fsc
    } for (target, start, fdr, fsc) in zip(train_target_values, m5_dates, train_cal_features_list, stat_cat)
    ],     freq="D")

test_ds = ListDataset([
    {
        FieldName.TARGET: target,
        FieldName.START: start,
        FieldName.FEAT_DYNAMIC_REAL: fdr,
        FieldName.FEAT_STATIC_CAT: fsc
    }
    for (target, start, fdr, fsc) in zip(test_target_values,
                                         m5_dates,
                                         test_cal_features_list,
                                         stat_cat)
], freq="D")

#Just to be sure, we quickly verify that dataset format is correct and that our dataset does indeed 
# contain the correct target values as well as dynamic and static features.
########################
next(iter(train_ds))
"""
Define the estimator
Having obtained our training and testing data, we can now create a GluonTS estimator. In our example we will use the DeepAREstimator, an autoregressive RNN which was developed primarily for the purpose of time series forecasting. Note however that you can use a variety of different estimators. Also, since GluonTS is mainly target at probabilistic time series forecasting, lots of different output distributions can be specified. In the M5 case, we think that the NegativeBinomialOutput distribution best describes the output.

For a full list of available estimators and possible initialization arguments see https://gluon-ts.mxnet.io/api/gluonts/gluonts.model.html.

For a full list of available output distributions and possible initialization arguments see https://gluon-ts.mxnet.io/api/gluonts/gluonts.distribution.html.
"""

########################
from gluonts.model.deepar import DeepAREstimator
from gluonts.distribution.neg_binomial import NegativeBinomialOutput
from gluonts.trainer import Trainer

estimator = DeepAREstimator(
    pred_length     = pred_length,
    freq                  = "D",
    distr_output          = NegativeBinomialOutput(),
    use_feat_dynamic_real = True,
    use_feat_static_cat   = True,
    cardinality           = stat_cat_cardinalities,
    trainer               = Trainer(
    learning_rate         = 1e-3,
    epochs                = 100,
    num_batches_per_epoch = 50,
    batch_size            = 32
    )
)

predictor = estimator.train(train_ds)

"""
Generating forecasts
Once the estimator is fully trained, we can generate preds from it for the test values.
"""
########################
from gluonts.evaluation.backtest import make_evaluation_preds

forecast_it, ts_it = make_evaluation_preds(
    dataset=test_ds,
    predictor=predictor,
    num_samples=100
)

print("Obtaining time series conditioning values ...")
tss = list(tqdm(ts_it, total=len(test_ds)))
print("Obtaining time series preds ...")
forecasts = list(tqdm(forecast_it, total=len(test_ds)))

"""
Local performance val (if submission is False)
Since we don't want to constantly submit our results to Kaggle, it is important to being able to evaluate performace on our own val set offline. To do so, we create a custom evaluator which, in addition to GluonTS's standard performance metrics, also returns MRMSSE (corresponding to the mean RMSSE). Note that the official score for the M5 competition, the WRMSSE, is not yet computed. A future version of this notebook will replace the MRMSSE by the WRMSSE.
"""

########################
if submission == False:
    
    from gluonts.evaluation import Evaluator
    
    class M5Evaluator(Evaluator):
        
        def get_metrics_per_ts(self, time_series, forecast):
            successive_diff  = np.diff(time_series.values.reshape(len(time_series)))
            successive_diff  = successive_diff ** 2
            successive_diff  = successive_diff[:-pred_length]
            denom            = np.mean(successive_diff)
            pred_values      = forecast.samples.mean(axis=0)
            true_values      = time_series.values.reshape(len(time_series))[-pred_length:]
            num              = np.mean((pred_values - true_values)**2)
            rmsse            = num / denom
            metrics          = super().get_metrics_per_ts(time_series, forecast)
            metrics["RMSSE"] = rmsse
            return metrics
        
        def get_aggregate_metrics(self, metric_per_ts):
            wrmsse = metric_per_ts["RMSSE"].mean()
            agg_metric , _ = super().get_aggregate_metrics(metric_per_ts)
            agg_metric["MRMSSE"] = wrmsse
            return agg_metric, metric_per_ts
        
    
    evaluator = M5Evaluator(quantiles=[0.5, 0.67, 0.95, 0.99])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
    print(json.dumps(agg_metrics, indent=4))


"""
Converting forecasts back to M5 submission format (if submission is True)
Since GluonTS estimators return a sample-based probabilistic forecasting predictor, we first need to reduce these results to a single pred per time series. This can be done by computing the mean or median over the predicted sample paths.
"""
########################
if submission == True:
    forecasts_acc = np.zeros((len(forecasts), pred_length))
    for i in range(len(forecasts)):
        forecasts_acc[i] = np.mean(forecasts[i].samples, axis=0)
We then reshape the forecasts into the correct data shape for submission ...

########################
if submission == True:
    forecasts_acc_sub = np.zeros((len(forecasts)*2, single_pred_length))
    forecasts_acc_sub[:len(forecasts)] = forecasts_acc[:,:single_pred_length]
    forecasts_acc_sub[len(forecasts):] = forecasts_acc[:,single_pred_length:]

"""
.. and verfiy that reshaping is consistent.
"""
########################
if submission == True:
    np.all(np.equal(forecasts_acc[0], np.append(forecasts_acc_sub[0], forecasts_acc_sub[30490])))


## Then, we save our submission into a timestamped CSV file which can subsequently be uploaded to Kaggle.
########################
if submission == True:
    import time
    sample_submission            = pd.read_csv(f'{m5_input_path}/sample_submission.csv')
    sample_submission.iloc[:,1:] = forecasts_acc_sub
    submission_id                = 'submission_{}.csv'.format(int(time.time()))
    sample_submission.to_csv(submission_id, index=False)


"""
Plotting sample preds
Finally, we can also visualize our preds for some of the time series.
"""


########################
plot_log_path = "./plots/"
directory = os.path.dirname(plot_log_path)
if not os.path.exists(directory):
    os.makedirs(directory)
    
def plot_prob_forecasts(ts_entry, forecast_entry, path, sample_id, inline=True):
    plot_length = 150
    pred_intervals = (50, 67, 95, 99)
    legend = ["observations", "median pred"] + [f"{k}% pred interval" for k in pred_intervals][::-1]

    _, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)
    forecast_entry.plot(pred_intervals=pred_intervals, color='g')
    ax.axvline(ts_entry.index[-pred_length], color='r')
    plt.legend(legend, loc="upper left")
    if inline:
        plt.show()
        plt.clf()
    else:
        plt.savefig('{}forecast_{}.pdf'.format(path, sample_id))
        plt.close()

print("Plotting time series preds ...")
for i in tqdm(range(5)):
    ts_entry = tss[i]
    forecast_entry = forecasts[i]
    plot_prob_forecasts(ts_entry, forecast_entry, plot_log_path, i)






