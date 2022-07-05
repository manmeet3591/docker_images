import pandas as pd

import numpy as np
import optuna

import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import lightgbm as lgb

df = pd.read_csv('study_trials_lightgbm.csv')



#print(df.loc[df.value.idxmax()])
df_ = df.loc[df.value.idxmax()]


print(df_["params_bagging_fraction"])#                       0.577731
print(df_["params_bagging_freq"])#                                  3
print(df_["params_feature_fraction"])#                       0.802626
print(df_["params_lambda_l1"])#                              0.000513
print(df_["params_lambda_l2"])#                              0.000024
print(df_["params_min_child_samples"])#                            37
print(df_["params_num_leaves"])#                                   45

#    lambda_l1: 0.0005129057432649834
#    lambda_l2: 2.3759528723441945e-05
#    num_leaves: 45
#    feature_fraction: 0.8026264632486393
#    bagging_fraction: 0.5777307765249753
#    bagging_freq: 3
#    min_child_samples: 37


(data, target) = sklearn.datasets.load_breast_cancer(return_X_y=True)
train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
dtrain = lgb.Dataset(train_x, label=train_y)
dvalid = lgb.Dataset(valid_x, label=valid_y)

param = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "lambda_l1": df_["params_lambda_l1"], #trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
    "lambda_l2": df_["params_lambda_l2"], # trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    "num_leaves": df_["params_num_leaves"], #  trial.suggest_int("num_leaves", 2, 256),
    "feature_fraction": df_["params_feature_fraction"], #trial.suggest_float("feature_fraction", 0.4, 1.0),
    "bagging_fraction": df_["params_bagging_fraction"], #trial.suggest_float("bagging_fraction", 0.4, 1.0),
    "bagging_freq": df_["params_bagging_freq"], #trial.suggest_int("bagging_freq", 1, 7),
    "min_child_samples": df_["params_min_child_samples"], #trial.suggest_int("min_child_samples", 5, 100),
    "device":"gpu",
    "gpu_platform_id": 4,
    "gpu_device_id": 4,
}

import time
n_ens  = 100
preds_ = []
acc_ = []
for ens in range(n_ens):
    param['seed'] = ens
    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    preds_.append(pred_labels)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    acc_.append(accuracy)
    print(accuracy)
#    time.sleep(3)

print(np.asarray(acc_).shape, np.asarray(preds_).shape)
#print(np.asarray(acc_)* np.asarray([preds_]).shape)
preds_ens = np.asarray([acc_]).T* np.asarray(preds_)
print(np.rint(preds_ens.mean(axis=0)))


final_accuracy = sklearn.metrics.accuracy_score(valid_y, np.rint(preds_ens.mean(axis=0)))
print(final_accuracy)
