
"""
Optuna example that optimizes a classifier configuration for cancer dataset using
Catboost.

In this example, we optimize the validation accuracy of cancer detection using
Catboost. We optimize both the choice of booster model and their hyperparameters.

"""

import numpy as np
import optuna

import catboost as cb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

data, target = load_breast_cancer(return_X_y=True)
train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.3)

df = pd.read_csv('study_trials_catboost.csv')



#print(df.loc[df.value.idxmax()])
df_ = df.loc[df.value.idxmax()]



param = {
    "objective": df_["params_objective"], # trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
    "colsample_bylevel":df_["params_colsample_bylevel"], # trial.suggest_float("colsample_bylevel", 0.01, 0.1),
    "depth": df_["params_depth"], #  trial.suggest_int("depth", 1, 12),
    "boosting_type": df_["params_boosting_type"], # trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
    "bootstrap_type": df_["params_bootstrap_type"], #   trial.suggest_categorical(
#        "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
#    ),
    "used_ram_limit": "6gb",
#     "task_type": "GPU",
#     "devices": "6:7",
}

if param["bootstrap_type"] == "Bayesian":
    param["bagging_temperature"] = df_["params_bagging_temperature"] #  trial.suggest_float("bagging_temperature", 0, 10)
elif param["bootstrap_type"] == "Bernoulli":
    param["subsample"] = df_["params_subsample"]#  trial.suggest_float("subsample", 0.1, 1)

n_ens = 100
preds_ = []
acc_ = []
for ens in range (n_ens):
    param["random_seed"] = ens
    gbm = cb.CatBoostClassifier(**param)

    gbm.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=0, early_stopping_rounds=100)

    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    preds_.append(pred_labels)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(valid_y, pred_labels)
    acc_.append(accuracy)
    print(accuracy)


print(np.asarray(acc_).shape, np.asarray(preds_).shape)
#print(np.asarray(acc_)* np.asarray([preds_]).shape)
preds_ens = np.asarray([acc_]).T* np.asarray(preds_)
print(np.rint(preds_ens.mean(axis=0)))


final_accuracy = accuracy_score(valid_y, np.rint(preds_ens.mean(axis=0)))
print(final_accuracy)
