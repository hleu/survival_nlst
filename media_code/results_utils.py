import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

from sksurv.preprocessing import OneHotEncoder
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (concordance_index_censored,
                            concordance_index_ipcw,
                            cumulative_dynamic_auc)
from sksurv.ensemble import RandomSurvivalForest
import sksurv.datasets as skds

import eli5
from eli5.sklearn import PermutationImportance

import matplotlib.pyplot as plt
import matplotlib

data_root = '/mnt/nlst/'
csv_path = '/mnt/nlst/package-nlst-512.2019-08-13/CT_and_Path_w_CT/'
file_list = glob('/mnt/data0-nfs/shared-datasets/cancer_multimodal/nlst/pyradiomics_features/'+'/*')

matplotlib.rcParams.update({'font.size': 18})
random_state = 20

def get_clinical_arr(df, surv_type='1stslice_death'):
	"""Preprocess clinical dataframe and return as numpy array.
    surv_type: string. Type of surival analysis - 
    	'1stslice_death': from 1st scan to death
    Returns: 
	clinical array: np array of float. Array of clinical information
		with numerical, categorical, boolean, event, time order
	clinical array categories: list of clinical categories based on clinical array
	"""
	# categories: list of catergorical column names (perform OneHotEncoder later)
	# numericals: list of numerical column names (perform Standardize later)
	# cols_leave: list of Boolean column names to leave out from further processing 
	# categories_dict: dict of categorical columns category 
	# 	(key: column name, values: array of categories). For example, 
	# 	categories_dict['educat'] = array([ 1,  2,  3,  4,  5,  6,  7,  8, 95, 98, 99])
	# Init categories, numericals, cols_leave, categories_vals dict
	categories = ['educat', 'ethnic', 'gender', 'race', 'cigsmok']
	numericals = ['age', 'BMI', 'pkyr', 'smokeage', 'smokeday', 'smokeyr']
	cols_leave = ['smokelive', 'smokework', 'personal_cancer', 'fam_cancer']
	categories_dict = {}
	for cat in categories:
		categories_dict[cat] = np.sort(pd.unique(df[cat].values.ravel()) )
        
	# Add survival information depending on survival analysis type
	if (surv_type == '1stslice_death'):
		cols_leave.extend(['days_1stpos_death', 'death_cancer'])

	# Preprocess clinical data: 
	# OneHotEncoder for Categorical
	# Median SimpleImputer and StandardScaler for Numerical
	# Leave Boolean column
	onehot = [([col], None) for col in categories] #categories = [categories_dict[col]]
	standardize = [([col], [SimpleImputer(missing_values=np.nan, strategy='mean'), StandardScaler()]) for col in numericals]
	leave = [([col], None) for col in cols_leave]
	data_mapper = DataFrameMapper(standardize + onehot + leave) 

	# Get categories based on categories_dict
	categories_ret = []
	for k in categories_dict.keys():
		categories_tmp =[k+'_'+str(val) for val in categories_dict[k]]
		categories_ret.append(categories_tmp)
	categories_ret = [item for sublist in categories_ret for item in sublist]

	arr = data_mapper.fit_transform(df).astype('float32')
    
    # fill nan boolean with 0
	mask = np.isnan(arr)
	idx = np.where(~mask,np.arange(mask.shape[1]),0)
	np.maximum.accumulate(idx,axis=1, out=idx)
	out = arr[np.arange(idx.shape[0])[:,None], idx]
    
	clinical = out[:,:-2]
	duration = out[:,-2]
	event = out[:,-1].astype('int32')
    
	df_clin_prep = pd.DataFrame(out, columns=numericals+categories+cols_leave)
	return df_clin_prep# numericals+categories_ret+cols_leave, df,clinical, duration, event, 
    
def parse_clinical(pids, prsndf):
    # preprocess prsndf and get only pid we care
    survdf = preprocess_clinical_df(list(prsndf['pid']), prsndf)
    survdf = survdf[survdf['pid'].isin(pids)]
    
    # fill na, get x y
    df_clin_prep = get_clinical_arr(survdf)
    data_x, data_y = skds.get_x_y(df_clin_prep, attr_labels=['death_cancer', 'days_1stpos_death'], pos_label=1)

    # onehotencoder sksurv
    categories = ['educat', 'ethnic', 'gender', 'race', 'cigsmok']
    numericals = ['age', 'BMI', 'pkyr', 'smokeage', 'smokeday', 'smokeyr']
    cols_leave = ['smokelive', 'smokework', 'personal_cancer', 'fam_cancer']
    data_x[categories+cols_leave] = data_x[categories+cols_leave].astype("category")
    data_x_numeric = OneHotEncoder(allow_drop=False).fit_transform(data_x)
        
    # remove sparse data
    cats_have = list(data_x_numeric)
    race_posible = ['race=2.0','race=3.0','race=4.0','race=5.0','race=6.0','race=99.0']
    race_have = [value for value in race_posible if value in cats_have]
    drop_posible = ['educat=2.0','educat=8.0','ethnic=99.0']
    drop_have = [value for value in drop_posible if value in cats_have]

    df_clin_prep2 = data_x_numeric.copy()
    df_clin_prep2.insert(loc=16, column='race=other', value = df_clin_prep2[race_have].sum(axis=1))
    df_clin_prep2 = df_clin_prep2.drop(columns=drop_have+race_have)

    return df_clin_prep2, data_y, df_clin_prep

def plot_cumulative_dynamic_auc(risk_score, label, color=None):
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_score, times)
    plt.ylim(0.2,1.0)
    locs, labels = plt.yticks()  # Get the current locations and labels.
    plt.yticks(np.arange(0.2, 1, step=0.1))  # Set label locations.

    plt.plot(times, auc, marker="o", color=color, label=label)
    plt.xlabel("days from first scan")
    plt.ylabel("time-dependent AUC")
    plt.axhline(mean_auc, color=color, linestyle="--")
    plt.legend(loc='upper right')

def get_radiomics_arr(df, surv_type='1stslice_death'):
	"""Preprocess clinical dataframe and return as numpy array.
    surv_type: string. Type of surival analysis - 
    	'1stslice_death': from 1st scan to death
    Returns: 
	clinical array: np array of float. Array of clinical information
		with numerical, categorical, boolean, event, time order
	clinical array categories: list of clinical categories based on clinical array
	"""
	# categories: list of catergorical column names (perform OneHotEncoder later)
	# numericals: list of numerical column names (perform Standardize later)
	# cols_leave: list of Boolean column names to leave out from further processing 
	# categories_dict: dict of categorical columns category 
	# 	(key: column name, values: array of categories). For example, 
	# 	categories_dict['educat'] = array([ 1,  2,  3,  4,  5,  6,  7,  8, 95, 98, 99])
	# Init categories, numericals, cols_leave, categories_vals dict
	cols_leave = []

	# Add survival information depending on survival analysis type
	if (surv_type == '1stslice_death'):
		cols_leave.extend(['days_1stpos_death', 'death_cancer'])
	numericals = [str(i) for i in range (107)]
	# Preprocess clinical data: 
	# OneHotEncoder for Categorical
	# Median SimpleImputer and StandardScaler for Numerical
	# Leave Boolean column
	standardize = [([col], [SimpleImputer(missing_values=np.nan, strategy='mean'), StandardScaler()]) for col in numericals]
	leave = [([col], None) for col in cols_leave]
	data_mapper = DataFrameMapper(standardize + leave) 

	arr = data_mapper.fit_transform(df).astype('float32')
    
    # fill nan boolean with 0
	mask = np.isnan(arr)
	idx = np.where(~mask,np.arange(mask.shape[1]),0)
	np.maximum.accumulate(idx,axis=1, out=idx)
	out = arr[np.arange(idx.shape[0])[:,None], idx]
    
	clinical = out[:,:-2]
	duration = out[:,-2]
	event = out[:,-1].astype('int32')
	df_clin_prep = pd.DataFrame(out, columns=numericals+cols_leave)
	return df_clin_prep

def parse_radiomics(pids, prsndf):
    raddf =  prsndf.iloc[:,-107:]

    # preprocess prsndf and get only pid we care
    df_clin_prep = preprocess_clinical_df(list(prsndf['pid']), prsndf)
    df_clin_prep = df_clin_prep[df_clin_prep['pid'].isin(pids)]
    df_clin_prep = pd.concat([raddf, df_clin_prep[['death_cancer', 'days_1stpos_death']].copy()], axis=1, join="inner")  
    survdf = get_radiomics_arr(df_clin_prep)
    
    data_x, data_y = skds.get_x_y(survdf, attr_labels=['death_cancer', 'days_1stpos_death'], pos_label=1)
    return data_x, data_y, df_clin_prep.reset_index()

