from glob import glob
import os
import csv
    
import pandas as pd
import numpy as np

from clinical_preprocess import preprocess_clinical_df
from data_loader import get_pids_split

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn_pandas import DataFrameMapper
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer

from sksurv.preprocessing import OneHotEncoder
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (concordance_index_censored,
                            concordance_index_ipcw,
                            cumulative_dynamic_auc)
from sksurv.ensemble import RandomSurvivalForest
import sksurv.datasets as skds

from lifelines import CoxPHFitter
from lifelines.utils.sklearn_adapter import sklearn_adapter
from lifelines.utils import concordance_index

import eli5
from eli5.sklearn import PermutationImportance

import matplotlib.pyplot as plt
import matplotlib

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams['figure.figsize'] = [12, 7]
matplotlib.rcParams.update({'font.size': 18})
random_state = 20
n_jobs=None
cv=None
hyper_name = "hyperparameters"

def cox_ph(x, y, df, dir_name, use_case, train_indices, val_indices):
    # print("Starting CPH_lifelines model")
    x_lifelines = x.join(df[['days_1stpos_death', 'death_cancer']])
      
    x_lifelines_copy = x_lifelines.copy()
    y_ll = x_lifelines_copy.pop('days_1stpos_death')
    x_ll = x_lifelines_copy
    
    #import ipdb
    #ipdb.set_trace()

    base_class = sklearn_adapter(CoxPHFitter, event_col='death_cancer')
    bclass = base_class()
    param_grid = {"penalizer": 10.0 ** np.arange(-3,4),
                  "l1_ratio": [0]}
    
    gcv = GridSearchCV(bclass, param_grid, cv=[(train_indices, val_indices)], return_train_score=True, \
                       refit=False)

    warnings.simplefilter("ignore")
    gcv.fit(x_ll, y_ll)
    warnings.resetwarnings()
    warnings.filterwarnings("ignore", category=FutureWarning)

    print(f"CPH_lifelines best model: lifelines.CoxPHFitter(penalizer={gcv.best_params_['penalizer']}, \
          l1_ratio={gcv.best_params_['l1_ratio']})")
    
    #############################
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    path = dir_name + "/" + hyper_name
    if not os.path.exists(path):
        os.mkdir(path)
    f_tmp = open(path + "/" + "best_parameters.txt", "a")
    f_tmp.write(use_case + "\n")
    f_tmp.write(f"CPH_lifelines best model: lifelines.CoxPHFitter(penalizer={gcv.best_params_['penalizer']}, \
                l1_ratio={gcv.best_params_['l1_ratio']})\n")
    f_tmp.write("best test score: " + str(gcv.best_score_) + "\n")
    f_tmp.close()
    
    np.save(path + '/' + use_case + "_penalizer", gcv.cv_results_['param_penalizer'])
    np.save(path + '/' + use_case + "_l1_ratio", gcv.cv_results_['param_l1_ratio'])
    np.save(path + '/' + use_case + "_mean_test", gcv.cv_results_['mean_test_score'])
    np.save(path + '/' + use_case + "_mean_train", gcv.cv_results_['mean_train_score'])
    np.save(path + '/' + use_case + "_mean_fit_time", gcv.cv_results_['mean_fit_time'])
    np.save(path + '/' + use_case + "_mean_score_time", gcv.cv_results_['mean_score_time'])
    
    ##################################
    
    cph = CoxPHFitter(l1_ratio=gcv.best_params_['l1_ratio'],
              penalizer=gcv.best_params_['penalizer'])
    
    x_train_lifelines = x_lifelines.iloc[train_indices]
    y_train = y[train_indices]
    
    x_val_lifelines = x_lifelines.iloc[val_indices]
    y_val = y[val_indices]
    
    #take away train subset and validate on validation subset
    cph.fit(x_train_lifelines, duration_col='days_1stpos_death', event_col='death_cancer')
    np.save(path + '/' + use_case + "_coefficients", cph.params_)
        
    prediction_cph_train = -cph.predict_expectation(x_train_lifelines)
    
    cindex_cph_train = concordance_index_censored(y_train["death_cancer"], y_train["days_1stpos_death"], \
                                                  prediction_cph_train)
    
    assert (cindex_cph_train[0] == gcv.cv_results_['mean_train_score'][np.argmax(gcv.cv_results_['mean_test_score'])])
    
    prediction_cph_val = -cph.predict_expectation(x_val_lifelines)
    
    cindex_cph_val = concordance_index_censored(y_val["death_cancer"], y_val["days_1stpos_death"], prediction_cph_val)
    
    assert (cindex_cph_val[0] == gcv.best_score_)
    
    return cph, prediction_cph_train, prediction_cph_val, cindex_cph_train, cindex_cph_val

def cox_ph_sksurv(x_train, x_test, y_train, y_test):
    '''
    This code uses sksurv instead of lifelines.
    '''
    # print("Starting CPH_sksurv model")
    cph = CoxPHSurvivalAnalysis()
    param_grid = {"alpha":[1e-4, 1e-3, 1e-2, 1e-1, 0, 1e4, 1e3, 1e2, 10]}
    gcv = GridSearchCV(cph, param_grid, cv=cv)

    warnings.simplefilter("ignore")
    gcv.fit(x_train, y_train)
    warnings.resetwarnings()

    cph = cph.set_params(**gcv.best_params_)

    print(f"CPH_sksurv best model: sksurv.{cph}")
    cph.fit(x_train, y_train)
    
    prediction_cph_train = cph.predict(x_train)
    prediction_cph_test = cph.predict(x_test)

    cindex_cph_train = concordance_index_censored(y_train["death_cancer"], 
                                                          y_train["days_1stpos_death"],
                                                          prediction_cph_train)
    cindex_cph_test = concordance_index_censored(y_test["death_cancer"], 
                                                         y_test["days_1stpos_death"], 
                                                         prediction_cph_test)
    return cph, prediction_cph_train, prediction_cph_test, cindex_cph_train, cindex_cph_test

def rsf(x, y, dir_name, use_case, train_indices, val_indices):
    # print("Starting RSF model")
    rsf = RandomSurvivalForest(max_features="sqrt", random_state=random_state)
    param_grid = {"n_estimators":[10, 100, 1000],
                 "min_samples_split":[2, 4, 6, 8, 10],
                 "min_samples_leaf":[1, 4, 15]}
    gcv = GridSearchCV(rsf, param_grid, n_jobs=n_jobs, cv=[(train_indices, val_indices)], return_train_score=True, refit=False)

    warnings.simplefilter("ignore")
    gcv.fit(x, y)
    warnings.resetwarnings()
    
    rsf.set_params(**gcv.best_params_)
    
    print(f"RSF best model: {gcv.best_params_}")
    
    #############################
    
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    path = dir_name + "/" + "rsf" + hyper_name
    if not os.path.exists(path):
        os.mkdir(path)
    f_tmp = open(path + "/" + "best_parameters.txt", "a")
    f_tmp.write(use_case + "\n")
    f_tmp.write(f"RSF best model: {rsf}\n")
    f_tmp.write("best test score: " + str(gcv.best_score_) + "\n")
    f_tmp.close()
    
    np.save(path + '/' + use_case + "_n_estimators", gcv.cv_results_['param_n_estimators'])
    np.save(path + '/' + use_case + "_min_samples_split", gcv.cv_results_['param_min_samples_split'])
    np.save(path + '/' + use_case + "_min_samples_leaf", gcv.cv_results_['param_min_samples_leaf'])
    np.save(path + '/' + use_case + "_mean_test", gcv.cv_results_['mean_test_score'])
    np.save(path + '/' + use_case + "_mean_train", gcv.cv_results_['mean_train_score'])
    np.save(path + '/' + use_case + "_mean_fit_time", gcv.cv_results_['mean_fit_time'])
    np.save(path + '/' + use_case + "_mean_score_time", gcv.cv_results_['mean_score_time'])
    
    ##################################
    
    x_train = x.iloc[train_indices]
    y_train = y[train_indices]
    
    x_val = x.iloc[val_indices]
    y_val = y[val_indices]

    rsf.fit(x_train, y_train)

    prediction_rsf_train = rsf.predict(x_train)
    prediction_rsf_val = rsf.predict(x_val)

    cindex_rsf_train = concordance_index_censored(y_train["death_cancer"], y_train["days_1stpos_death"], \
                                                  prediction_rsf_train)
    
    assert (cindex_rsf_train[0] == gcv.cv_results_['mean_train_score'][np.argmax(gcv.cv_results_['mean_test_score'])])
    
    cindex_rsf_val = concordance_index_censored(y_val["death_cancer"], y_val["days_1stpos_death"], \
                                                prediction_rsf_val)
    
    assert (cindex_rsf_val[0] == gcv.best_score_)
    
    return rsf, prediction_rsf_train, prediction_rsf_val, cindex_rsf_train, cindex_rsf_val


def compare_td_aucs(y_train, y_test, prediction_cph_test, prediction_rsf_test, title_suffix, fig_suffix, dir_name, label_suffix='', reset=False):
    
    assert (len(y_train) > len(y_test))
    assert (len(y_test) == len(prediction_cph_test) and len(y_test) == len(prediction_rsf_test))
    
    if reset is True:
        plt.figure()
    
    test_min = np.inf
    test_max = 0
    for (e, d) in y_test: 
        if d > test_max:
            test_max = d
        if d < test_min:
            test_min = d
    
    va_times = np.arange(test_min, test_max)

    cph_auc, cph_mean_auc = cumulative_dynamic_auc(
                y_train, y_test, prediction_cph_test, va_times,
                )

    rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(
                y_train, y_test, prediction_rsf_test, va_times,
                )

    plt.ylim(0.0,1.0)
    locs, labels = plt.yticks()  # Get the current locations and labels.
    plt.yticks(np.arange(0.0, 1, step=0.1))  # Set label locations.
    plt.plot(va_times, cph_auc, "o-", label=("CoxPH" + label_suffix + "(mean AUC = {:.2f}%)").format(cph_mean_auc*100))
    plt.plot(va_times, rsf_auc, "x-", label=("RSF" + label_suffix + "(mean AUC = {:.2f}%)").format(rsf_mean_auc*100))
    plt.xlabel("Days from enrollment")
    plt.ylabel("AUC")
    plt.title("Time-dependent AUC " + title_suffix)
    plt.legend(loc="lower center")
    plt.grid(True)
    
    np.save(dir_name + '/' + fig_suffix + '_cpc_auc', cph_auc)
    np.save(dir_name + '/' + fig_suffix + '_rsf_auc', rsf_auc)
    np.save(dir_name + '/' + fig_suffix + '_va_times', va_times)
    
    if not os.path.exists(dir_name + "/figs"):
        os.mkdir(dir_name + "/figs")
    plt.savefig(dir_name + '/figs/'+ fig_suffix + ".png", bbox_inches="tight")

    return cph_mean_auc, rsf_mean_auc

def get_feature_importances_rsf(model, x_train, x_test, y_train,
        feature_names, weight_suffix, dir_name): 
    
    assert (x_train.shape[0] == len(y_train))
    assert (x_train.shape[1] == len(feature_names))
    assert (x_train.shape[0] > len(x_test))
    
    perm = PermutationImportance(model, n_iter=15, random_state=random_state)
    perm.fit(x_train, y_train)
    w = eli5.show_weights(perm, feature_names=feature_names)
    
    if not os.path.exists(dir_name + "/results"):
        os.mkdir(dir_name + "/results")
        
    html = w.data
    with open(dir_name + '/results/' + weight_suffix +'.html', 'w') as f:
        f.write(html)

    thresh = np.median(list(perm.feature_importances_))
    mask = perm.feature_importances_ > thresh
    features = x_test.columns[mask]
    features = list(features)

    return features

def get_feature_importances_cph(model, x_train, x_test, y_train, df_train, df_test, feature_names, weight_suffix, dir_name): 
    
    assert (x_train.shape[0] == len(y_train) and x_train.shape[0] == df_train.shape[0])
    assert (x_test.shape[0] == df_test.shape[0])
    assert (x_train.shape[1] == len(feature_names) and x_test.shape[1] == len(feature_names))
    assert (x_train.shape[0] > len(x_test))
    
    x_train_lifelines = x_train.join(df_train[['days_1stpos_death', 'death_cancer']])
    x_test_lifelines = x_test.join(df_test[['days_1stpos_death', 'death_cancer']])

    model_df = model.summary[['coef', 'p']]
    model_df.index = feature_names
    
    if not os.path.exists(dir_name + "/results"):
        os.mkdir(dir_name + "/results")
    model_df.to_html(dir_name + '/results/' + weight_suffix + '.html')

    thresh = np.median(list(model_df['p']))
    mask = model_df['p'] < thresh
    features = x_train.columns[mask]
    features = list(features)

    return features 

