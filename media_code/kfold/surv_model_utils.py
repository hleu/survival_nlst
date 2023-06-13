import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from glob import glob
import os
import csv
import sys
    
import pandas as pd
import numpy as np

sys.path.append('..')
from clinical_preprocess import preprocess_clinical_df
from data_loader import get_pids_split

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn_pandas import DataFrameMapper
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, KFold

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

import eli5
from eli5.sklearn import PermutationImportance

import matplotlib.pyplot as plt
import matplotlib

### This file is updated for k-fold based on media_code/surv_model_utils.py

plt.rcParams['figure.figsize'] = [12, 7]
matplotlib.rcParams.update({'font.size': 18})
random_state = 20
n_jobs=None
cv=None
n = 1000
# cv = KFold(n_splits=10, random_state=random_state, shuffle=True)

def cox_ph(x_train, x_test, y_train, y_test, df_train, df_test):
    # print("Starting CPH_lifelines model")
    x_train_lifelines = x_train.join(df_train[['days_1stpos_death', 'death_cancer']])
    x_test_lifelines = x_test.join(df_test[['days_1stpos_death', 'death_cancer']])

    df_trains = x_train_lifelines.copy()
    y_train_ll = df_trains.pop('days_1stpos_death')
    x_train_ll = df_trains
    
    df_tests = x_test_lifelines.copy()
    y_test_ll = df_tests.pop('days_1stpos_death')
    x_test_ll = df_tests

    #import ipdb
    #ipdb.set_trace()

    base_class = sklearn_adapter(CoxPHFitter, event_col='death_cancer')
    bclass = base_class()
    param_grid = {"penalizer": 10.0 ** np.arange(-3, 4),
                  "l1_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    gcv = GridSearchCV(bclass, param_grid, cv=cv, return_train_score=True)

    warnings.simplefilter("ignore")
    gcv.fit(x_train_ll, y_train_ll)
    warnings.resetwarnings()

    print(f"CPH_lifelines best model: lifelines.CoxPHFitter(penalizer={gcv.best_params_['penalizer']}, l1_ratio={gcv.best_params_['l1_ratio']})")
    pr = gcv.predict(x_train_ll)   
    cph = CoxPHFitter(l1_ratio=gcv.best_params_['l1_ratio'],
                      penalizer=gcv.best_params_['penalizer'])
    cph.fit(x_train_lifelines, duration_col='days_1stpos_death', event_col='death_cancer')

    prediction_cph_train = -cph.predict_expectation(x_train_lifelines)
    prediction_cph_test = -cph.predict_expectation(x_test_lifelines)
    
    cindex_cph_train = concordance_index_censored(y_train["death_cancer"], 
                                                          y_train["days_1stpos_death"],
                                                          prediction_cph_train)
    cindex_cph_test = concordance_index_censored(y_test["death_cancer"], 
                                                         y_test["days_1stpos_death"], 
                                                         prediction_cph_test)
    return cph, prediction_cph_train, prediction_cph_test, cindex_cph_train, cindex_cph_test, gcv

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

def rsf(x_train, x_test, y_train, y_test):
    # print("Starting RSF model")
    rsf = RandomSurvivalForest(max_features="sqrt", random_state=random_state)
    param_grid = {"n_estimators":[10, 100, 1000],
                 "min_samples_split":[2, 4, 6, 8, 10],
                 "min_samples_leaf":[1, 4, 15]}
    gcv = GridSearchCV(rsf, param_grid, n_jobs=n_jobs, cv=cv, return_train_score=True)

    warnings.simplefilter("ignore")
    gcv.fit(x_train, y_train)
    warnings.resetwarnings()

    rsf.set_params(**gcv.best_params_)

    print(f"RSF best model: {rsf}")
    rsf.fit(x_train, y_train)

    prediction_rsf_train = rsf.predict(x_train)
    prediction_rsf_test = rsf.predict(x_test)

    cindex_rsf_train = concordance_index_censored(y_train["death_cancer"], 
                                                          y_train["days_1stpos_death"],
                                                          prediction_rsf_train)
    cindex_rsf_test = concordance_index_censored(y_test["death_cancer"], 
                                                         y_test["days_1stpos_death"],
                                                         prediction_rsf_test)
    return rsf, prediction_rsf_train, prediction_rsf_test, cindex_rsf_train, cindex_rsf_test, gcv

def compare_td_aucs(y_train, y_test, prediction_cph_test, prediction_rsf_test, title_suffix, fig_suffix, label_suffix='', reset=False, outer_dir=''):
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
    
    if not os.path.exists(os.path.join(outer_dir, 'figs')) and outer_dir != '':
        os.mkdir(os.path.join(outer_dir, 'figs'))
    
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
    plt.savefig(os.path.join(outer_dir, 'figs', fig_suffix), bbox_inches="tight")

    return cph_mean_auc, rsf_mean_auc

def get_feature_importances_rsf(model, x_train, x_test, y_train,
        feature_names, weight_suffix, outer_dir=''): 

    perm = PermutationImportance(model, n_iter=15, random_state=random_state)
    perm.fit(x_train, y_train)
    w = eli5.show_weights(perm, feature_names=feature_names)
    
    if not os.path.exists(os.path.join(outer_dir, 'results')) and outer_dir != '':
        os.mkdir(os.path.join(outer_dir, 'results'))
        
    html = w.data
    with open(os.path.join(outer_dir, 'results', weight_suffix +'.html'), 'w') as f:
        f.write(html)

    thresh = np.median(list(perm.feature_importances_))
    mask = perm.feature_importances_ > thresh
    features = x_test.columns[mask]
    features = list(features)

    return features

def get_feature_importances_cph(model, x_train, x_test, y_train, df_train, df_test, feature_names, weight_suffix, outer_dir=''): 
    x_train_lifelines = x_train.join(df_train[['days_1stpos_death', 'death_cancer']])
    x_test_lifelines = x_test.join(df_test[['days_1stpos_death', 'death_cancer']])

    model_df = model.summary[['coef', 'p']]
    model_df.index = feature_names
    
    if not os.path.exists(os.path.join(outer_dir, 'results')) and outer_dir != '':
        os.mkdir(os.path.join(outer_dir, 'results'))
              
    model_df.to_html(os.path.join(outer_dir, 'results', weight_suffix + '.html'))

    thresh = np.median(list(model_df['p']))
    mask = model_df['p'] < thresh
    features = x_train.columns[mask]
    features = list(features)

    return features 

