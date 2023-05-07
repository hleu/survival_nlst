import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from glob import glob
import os
import csv
    
import pandas as pd
import numpy as np

from clinical_preprocess import preprocess_clinical_df
from data_loader import get_pids_split
from surv_model_utils import *
from results_utils import *

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

### UPDATE THESE BEFORE RUNNING
data_root = '../data/' # Location of PIDS (pids.txt)
csv_path = '../data/' # Location of nlst_15kpct_prsn_062119.csv'
file_list = glob('../data/pyradiomics_features/'+'/*') # Location of the pyradiomics features
radiomics_feature_name_file = '../data/' #Location of pyradiomics_features.txt
result_file = './results.txt'
subset_file = './subset.txt'

#data_root = '/mnt/nlst/' # Location of PIDS (pids.txt)
#csv_path = '/mnt/nlst/package-nlst-512.2019-08-13/CT_and_Path_w_CT/' # Location of nlst_15kpct_prsn_062119.csv'
#file_list = glob('/mnt/data0-nfs/shared-datasets/cancer_multimodal/nlst/pyradiomics_features/'+'/*') # Location of the pyradiomics features
#radiomics_feature_name_file = '/mnt/data0-nfs/shared-datasets/cancer_multimodal/nlst/'

matplotlib.rcParams.update({'font.size': 18})
random_state = 20
np.random_state = 20

def run_surv(use_case, run_type, x_train, x_test, y_train, y_test, df_train, df_test, kfold_idx):
    dir_name = "split" + str(kfold_idx)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        
    f = open(dir_name + "/" + result_file, "a")

    cph_model, cph_pred_train, cph_pred_test, cph_cindex_train, cph_cindex_test = \
    cox_ph(x_train, x_test, y_train, y_test, df_train, df_test)

    rsf_model, rsf_pred_train, rsf_pred_test, rsf_cindex_train, rsf_cindex_test = \
    rsf(x_train, x_test, y_train, y_test)

    cph_mtdauc, rsf_mtdauc = \
    compare_td_aucs(y_train, y_test, cph_pred_test, rsf_pred_test, f"- {use_case}-{run_type}", f"{use_case}_{run_type}", dir_name, reset=True)

    CPH_liflines_res = f"CPH_lifelines-{use_case}-{run_type} \t {round(cph_cindex_test[0]*100,2)} \t {round(cph_mtdauc*100,2)}"
    RSF_res = f"RSF-{use_case}-{run_type} \t {round(rsf_cindex_test[0]*100,2)} \t {round(rsf_mtdauc*100,2)}"
    f.write(f"{CPH_liflines_res}\n")
    f.write(f"{RSF_res}\n")
    f.close()
    return cph_model, rsf_model

def perform_all_test(use_case, x_train, x_test, y_train, y_test, df_train, df_test, feature_names, kfold_idx, subset=[]):
    cph_model, rsf_model = run_surv(use_case, 'all', x_train, x_test, y_train, y_test, df_train, df_test, kfold_idx)

    # TODO: overwrite if directory already exists
    dir_name = "split" + str(kfold_idx)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        
    cph_subset = get_feature_importances_cph(cph_model, x_train, x_test, y_train, \
    df_train, df_test, feature_names, f'cph_{use_case}_all', dir_name)

    rsf_subset = get_feature_importances_rsf(rsf_model, x_train, x_test, y_train, \
        feature_names, f'rsf_{use_case}_all', dir_name)

    print(f"Most important CPH-{use_case} subset names:", cph_subset)
    print(f"Most important RSF-{use_case} subset names:", rsf_subset)
        
    f_tmp = open(dir_name + "/" + subset_file, "a")
    f_tmp.write(use_case+'\n')
    f_tmp.write("CPH: \n")
    for v in cph_subset:
        f_tmp.write(v+'\n')
    f_tmp.write("RSF: \n")
    for v in rsf_subset:
        f_tmp.write(v+'\n')
    f_tmp.write('\n')
    f_tmp.close()

    if len(subset) > 0:
        x_train_subset = x_train[subset]
        x_test_subset = x_test[subset]
       
        _,_ = run_surv(use_case, 'subset', x_train_subset, x_test_subset, y_train, y_test, df_train, df_test, kfold_idx)

def main():
    train_pos, val_pos, test_pos, _, _, _ = get_pids_split(data_root+'pids.txt')
    pos_pids = np.array(train_pos + val_pos + test_pos)

    kf = KFold(5, shuffle=True, random_state=1234)

    for split_idx, (train_index, test_index) in enumerate(kf.split(pos_pids)):
        
        if split_idx != 1:
            continue

        train_pids = pos_pids[list(train_index)]
        test_pids = pos_pids[list(test_index)]

        prsndf = pd.read_csv(csv_path+'nlst_15kpct_prsn_062119.csv')
        prsndf = prsndf[prsndf['scr_group']== 1] # cancer positive

        pids_radiomics = []
        radiomics = []
        for f in file_list:
            pids_radiomics.append(int(f.split('/')[-1].split('.')[0]))
            radiomics.append(np.load(f)['arr_0'])
        radiomics = np.array(radiomics)
        df_radiomics = pd.DataFrame(radiomics, columns=[str(i) for i in range(np.shape(radiomics)[1])], index=pids_radiomics)

        prsndf = prsndf[prsndf['pid'].isin(pids_radiomics)]
        prsndf_rad = prsndf.join(df_radiomics, on='pid')
        raddf = prsndf_rad.iloc[:,-107:]
        numericals = [str(i) for i in range (107)]

        #########################################################
        print("Clinical only:")
        x_train, y_train, df_train = parse_clinical(train_pids, prsndf)
        x_test, y_test, df_test = parse_clinical(test_pids, prsndf)

        clin_feature_names = [
            'Age',
            'BMI',
            'Pack-year',
            'Smoking-start-age',
            'Cigarettes-per-day',
            'Number-of-smoking-years',
            'High-school-graduate',
            'Post-HS-training',
            'Associate-degree',
            'Bachelors-degree',
            'Graduate-school',
            'Female',
            'Non-white',
            'Smoking-at-the-start-of-trial',
            'Lived-with-smoker',
            'Worked-with-smoker',
            'Cancer-prior-to-trial',
            'Family-member-had-cancer']

        x_test = x_test.reindex(columns=list(x_train))

        #import ipdb
        #ipdb.set_trace()
        # NOTE: Commenting out subset model until feature analysis complete
        # clin_subset = ['BMI', 'pkyr','gender=2.0','personal_cancer=1.0','smokeage', 'smokework=1.0']
        clin_subset = []
        perform_all_test('clinical', x_train, x_test, y_train, y_test, df_train, df_test, clin_feature_names, split_idx, clin_subset)

        ###################################################################
        print("\nRadiomics features only:")
        rad_feature_names = []
        with open(radiomics_feature_name_file
                 + 'pyradiomics_features.txt') as f:
            for line in f.readlines():
                rad_feature_names.append(line.strip())

        ###################################################################
        x_train_rad, y_train, df_train = parse_radiomics(train_pids, prsndf_rad)
        x_test_rad, y_test, df_test = parse_radiomics(test_pids, prsndf_rad)

        f1 = [str(i) for i in range (14)]
        x_train_rad1 = x_train_rad[f1].copy()    
        x_train_rad1 = x_train_rad1.drop(columns=['8']) 
        x_test_rad1 = x_test_rad[f1].copy()
        x_test_rad1 = x_test_rad1.drop(columns=['8']) 

        f1 = [rad_feature_names[i] for i in range(14)]
        f1.remove(rad_feature_names[8])

        x_train_rad1.columns = f1
        x_test_rad1.columns = f1
        rad1_feature_names = f1 

        f2 = [str(i) for i in range (14, 32)]
        x_train_rad2 = x_train_rad[f2].copy()    
        x_train_rad2 = x_train_rad2.drop(columns=['21', '22', '25', '26']) 
        x_test_rad2 = x_test_rad[f2].copy()
        x_test_rad2 = x_test_rad2.drop(columns=['21', '22', '25', '26'])

        f2_indices = [14, 15, 16, 17, 18, 19, 20, 23, 24, 27, 28, 29, 30, 31]
        f2 = [rad_feature_names[i] for i in f2_indices]

        x_train_rad2.columns = f2
        x_test_rad2.columns = f2
        rad2_feature_names = f2 

        f3 = ['50', '102', '53', '27']
        x_train_rad3 = x_train_rad[f3].copy()
        x_test_rad3 = x_test_rad[f3].copy()

        f3 = rad_feature_names.copy()
        f3 = [rad_feature_names[i] for i in [50, 102, 53, 27]] #.remove(rad_feature_names[21])
        x_train_rad3.columns = f3
        x_test_rad3.columns = f3
        rad3_feature_names = f3

        ##########################################################
        print("\nRadiomics1:")
        # NOTE: Commenting out subset model until feature analysis complete
        # radiomics1_subset = ['original_shape_Maximum2DDiameterColumn', 'original_shape_Maximum2DDiameterRow','original_shape_Maximum3DDiameter', 'original_shape_Sphericity', 'original_shape_VoxelVolume', 'original_shape_Maximum2DDiameterSlice']
        radiomics1_subset = []
        perform_all_test('radiomics1', x_train_rad1, x_test_rad1, y_train, y_test, df_train, df_test, rad1_feature_names, split_idx, radiomics1_subset)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print("\nRadiomics2:")
        # NOTE: Commenting out subset model until feature analysis complete
        # radiomics2_subset = ['original_firstorder_Energy', 'original_firstorder_Kurtosis','original_firstorder_Maximum', 'original_firstorder_Median', 'original_firstorder_90Percentile', 'original_firstorder_RootMeanSquared']
        radiomics2_subset = []
        perform_all_test('radiomics2', x_train_rad2, x_test_rad2, y_train, y_test, df_train, df_test, rad2_feature_names, split_idx, radiomics2_subset)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print("\nRadiomics3:")
        # NOTE: Commenting out subset model until feature analysis complete
        #radiomics3_subset = ['original_ngtdm_Busyness', 'original_firstorder_RootMeanSquared']
        radiomics3_subset = []
        perform_all_test('radiomics3', x_train_rad3, x_test_rad3, y_train, y_test, df_train, df_test, rad3_feature_names, split_idx, [])

        ################################################################
        # Generate and use combination of features. 
        
        # NOTE: Commenting out combination model until feature analysis complete
#         print("\nCombination features:")


#         x_train_comb = x_train[clin_subset].copy()
#         x_train_comb = pd.merge(x_train_comb, x_train_rad1[radiomics1_subset], left_index=True, right_index=True, how='inner')
#         x_test_comb = x_test[clin_subset].copy()
#         x_test_comb = pd.merge(x_test_comb, x_test_rad1[radiomics1_subset], left_index=True, right_index=True, how='inner')

#         clin_feat_name_idx = [i for i, x in enumerate(list(x_train)) if x in clin_subset]
#         comb_feature_names = [clin_feature_names[i] for i in clin_feat_name_idx]
#         comb_feature_names += radiomics1_subset

#         perform_all_test('combination', x_train_comb, x_test_comb, y_train, y_test, df_train, df_test, comb_feature_names, split_idx, [])

main()