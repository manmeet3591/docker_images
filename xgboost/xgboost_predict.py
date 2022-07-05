import pandas as pd

import numpy as np
import optuna

import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd



df = pd.read_csv('study_trials_xgboos.csv')

#print(df.loc[df.value.idxmax()])
df_ = df.loc[df.value.idxmax()]

print(df_['params_alpha'])                  #               0.420554
print(df_['params_booster'])                #                 gbtree
print(df_['params_colsample_bytree'])       #               0.236673
print(df_['params_eta'])                    #                    0.0
print(df_['params_gamma'])                  #               0.051345
print(df_['params_grow_policy'])            #              lossguide
print(df_['params_lambda'])                 #                    0.0
print(df_['params_max_depth'])              #                    9.0
print(df_['params_min_child_weight'])       #                    2.0
print(df_['params_normalize_type'])         #                    NaN
print(df_['params_rate_drop'])              #                    NaN
print(df_['params_sample_type'])            #                    NaN
print(df_['params_skip_drop'])              #                    NaN
print(df_['params_subsample'])              #               0.998424


(data, target) = sklearn.datasets.load_breast_cancer(return_X_y=True)
train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
dtrain = xgb.DMatrix(train_x, label=train_y)
dvalid = xgb.DMatrix(valid_x, label=valid_y)


param = {
     "verbosity": 0,
     "objective": "binary:logistic",
     # use exact for small dataset.
    # "tree_method": "exact",
     "tree_method": "gpu_hist",
     # defines booster, gblinear for linear functions.
     "booster": df_['params_booster'],   # trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
     # L2 regularization weight.
     "lambda": df_['params_lambda'],   # trial.suggest_float("lambda", 1e-8, 1.0, log=True),
     # L1 regularization weight.
     "alpha": df_['params_alpha'], # trial.suggest_float("alpha", 1e-8, 1.0, log=True),
     # sampling ratio for training data.
     "subsample": df_['params_subsample'], #  trial.suggest_float("subsample", 0.2, 1.0),
     # sampling according to each tree.
     "colsample_bytree": df_['params_colsample_bytree'], # trial.suggest_float("colsample_bytree", 0.2, 1.0),
     "eta":   df_['params_eta'],                    #                    0.0
     "gamma":   df_['params_gamma'],                  #               0.051345
     "grow_policy":   df_['params_grow_policy'],            #              lossguide
     "max_depth":   int(df_['params_max_depth']),              #                    9.0
     "min_child_weight":   df_['params_min_child_weight'],       #                    2.0
     "normalize_type":   df_['params_normalize_type'],         #                    NaN
     "rate_drop":  df_['params_rate_drop'],             #                    NaN
     "sample_type":   df_['params_sample_type'],            #                    NaN
     "skip_drop":   df_['params_skip_drop'],              #                    NaN
     }

import time
n_ens  = 100
preds_ = []
acc_ = []
for ens in range(n_ens):
    param['seed'] = ens
    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
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
    


