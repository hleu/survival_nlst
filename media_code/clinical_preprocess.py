"""Preprocess clinical data and save to a csv file """
import os
import numpy as np
import pandas as pd
import random
from argparse import ArgumentParser

def get_pids_split(pid_file):
	"""Get pids split based on saved pid file.
    pid_file: string. Location of pid file
    Returns: train_pos, val_pos, test_pos, train_neg, val_neg, test_neg: 
    	lists of pids for corresponding split
    """
    # Open pid file and read contents
	f = open(pid_file, 'r') 
	lines = f.readlines() 
	lines = [line.strip('\n') for line in lines]

	# Map lines to list of pids for each categories
	train_pos = list(map(int, lines[1].split(','))) 
	val_pos = list(map(int, lines[2].split(',')))  
	test_pos = list(map(int, lines[3].split(',')))  
	train_neg = list(map(int, lines[4].split(','))) 
	val_neg = list(map(int, lines[5].split(',')))  
	test_neg = list(map(int, lines[6].split(',')))  
	return train_pos, val_pos, test_pos, train_neg, val_neg, test_neg

def create_pids_split(pid_file, pos_csv, neg_csv, train_percent=.6, test_percent=.2):
	"""Get pids split based on saved pid file.
    pid_file: string. Location of pid file
    Returns: train_pos, val_pos, test_pos, train_neg, val_neg, test_neg: 
    	lists of pids for corresponding split
    """
	# get postive and negative pids
	pos_df = pd.read_csv(pos_csv, header=0,    sep=',', quotechar='"', error_bad_lines=False)
	neg_df = pd.read_csv(neg_csv, header=0,    sep=',', quotechar='"', error_bad_lines=False)
	pospids = list(pos_df.pid.unique().astype(int))
	negpids = list(neg_df.pid.unique().astype(int))

	# shuffle lists
	random.shuffle(pospids)
	random.shuffle(negpids)

	# create lists of pids
	train_pos = pospids[:int(len(pospids)*train_percent)] 
	val_pos = pospids[int(len(pospids)*train_percent):int(len(pospids)*(train_percent+test_percent))]
	test_pos = pospids[int(len(pospids)*(train_percent+test_percent)):]
	train_pos = list(map(str, train_pos)) 
	val_pos = list(map(str, val_pos))  
	test_pos = list(map(str, test_pos))  

	train_neg = negpids[:int(len(negpids)*train_percent)]
	val_neg = negpids[int(len(negpids)*train_percent):int(len(negpids)*(train_percent+test_percent))]
	test_neg = negpids[int(len(negpids)*(train_percent+test_percent)):]
	train_neg = list(map(str, train_neg)) 
	val_neg = list(map(str, val_neg))  
	test_neg = list(map(str, test_neg))  

	# save list of pids
	with open(pid_file, 'w') as f:
		f.write('train_pos, val_pos, test_pos, train_neg, val_neg, test_neg\n')
		for l in (train_pos, val_pos, test_pos, train_neg, val_neg, test_neg):
			f.write(', '.join(l) + '\n')

	# return split
	return train_pos, val_pos, test_pos, train_neg, val_neg, test_neg


def preprocess_clinical_df(pids, df):
	"""Get clinical dataframe from pids list.
    pids: list of pids
    Returns: 
    df:  modified clinical dataframe
    """
	# Modify dataframe with only pids we are interested in
	df = df[df['pid'].isin(pids)]
	# BMI: Compute BMI from weight and height
	df.loc[:, 'BMI'] =  df['weight']/df['height']/df['height']*703
	# All disease: Get all Personal Disease History information and fill nan with 0
	df.loc[:,'diagasbe':'diagstro'] = \
		df.loc[:,'diagasbe':'diagstro'].fillna(0).astype(bool)
	# Personal Cancer History combincation: 
	# Combine all Personal Cancer History into one column 
	df.loc[:, 'personal_cancer'] = \
		df.loc[:,'cancblad':'canctran'].fillna(0).sum(axis=1).astype(bool)
	# Family History of Lung Cancer combincation: 
	# Combine all Family Cancer History into one column 
	df.loc[:, 'fam_cancer'] = \
		df.loc[:,'famfather':'famchild'].fillna(0).sum(axis=1).astype(bool)

	# Compute survival information
	# Events (e) - Boolean
	cancer = [] # Patient has cancer
	death = [] # Patient died before end of study
	death_cancer = [] # Patient confirmed died of cancer before end of study
	# Duration (t) - Numerical
	days_to_cancer = [] # Duration from start of study to cancer diagnosis/censoring (days)
	days_to_death = [] # Duration from start of study to death/censoring (days)
	# 1st positive scan - Numerical
	days_1stpos = [] # Duration from start of study to first scan (days)
	yr_1stpos = [] # Duration from start of study to first scan (year)

	# Iterate over rows with iterrows()
	for index, row in df.iterrows():
		# Cancer
		# Decide cancer based on 'can_scr' 
		# Days based on 'candx_days' (cancer) and fup_days (non_cancer)
		if (np.isnan(row['candx_days'])):
			cancer.append(0)
			days_to_cancer.append(row['fup_days'])
		else:                
			cancer.append(1)
			days_to_cancer.append(row['candx_days'])
		# Death
		if (row['finaldeathlc']==0 or row['finaldeathlc']==1):
			# Death due to lung cancer or work-up of suspected lung cancer
			if (row['finaldeathlc']==1):
				death.append(1)
				death_cancer.append(1)
				days_to_death.append(row['death_days'])
			# Death not due to lung cancer
			else:
				death.append(1)
				death_cancer.append(0)
				days_to_death.append(row['death_days'])
		else:
			# No confirmed death before end of study 
				death.append(0)
				death_cancer.append(0)
				days_to_death.append(row['fup_days'])

		# 1st positive scan
		if (row.scr_iso0 in [4,5,6] or row.scr_res0 in [4,5,6]): # year 1 positive
			yr_1stpos.append(0)
			days_1stpos.append(row.scr_days0)
		elif (row.scr_iso1 in [4,5,6] or row.scr_res1 in [4,5,6]): # year 2 positive
			yr_1stpos.append(1)
			days_1stpos.append(row.scr_days1)
		elif (row.scr_iso2 in [4,5,6] or row.scr_res2 in [4,5,6]): # year 3 positive
			yr_1stpos.append(2)
			days_1stpos.append(row.scr_days2)
		else: # no positive scan
			yr_1stpos.append(np.nan)
			days_1stpos.append(np.nan)

	# Add computed survival information
	df.loc[:, 'cancer'] = cancer
	df.loc[:, 'days_to_cancer'] = days_to_cancer
	df.loc[:, 'death'] = death
	df.loc[:, 'death_cancer'] = death_cancer
	df.loc[:, 'days_to_death'] = days_to_death
	df.loc[:, 'days_1stpos'] = days_1stpos
	df.loc[:, 'yr_1stpos'] = yr_1stpos
	df.loc[:, 'days_1stpos_death'] = df['days_to_death']- df['days_1stpos']
    
	# Get only columns we need for clinical information
	df = df [list(df.loc[:,'cancer':'days_1stpos_death']) + list(df.loc[:,'scr_days0':'scr_days2']) \
		+ list(df.loc[:,'de_type':'scr_iso2']) + ['pid'] \
		+ ['age', 'educat', 'ethnic', 'gender', 'race', 'BMI', 'cigsmok', 'pkyr', 'smokeage', 'smokeday', 'smokelive', 'smokework', 'smokeyr'] \
		+ list(df.loc[:,'diagasbe':'diagstro']) \
		+ ['personal_cancer', 'fam_cancer']]
    
	return df

if __name__ == '__main__':
	# parse arguments
	parser = ArgumentParser()
	parser.add_argument('--root', type=str, help='path to the NLST folder', \
		default='/mnt/nlst/', required=False)
	parser.add_argument('--clinical_csv', type=str, help='path to the prsn CSV file', \
		default='package-nlst-512.2019-08-13/CT_and_Path_w_CT/nlst_15kpct_prsn_062119.csv', required=False)
	parser.add_argument('--pids_file', type=str, help='path to the pid file', \
		default='pids.txt', required=False)
	parser.add_argument('--positive_csv', type=str, help='path to the postive nodule CSV file', \
		default='pospick.csv', required=False)
	parser.add_argument('--negative_csv', type=str, help='path to the negative nodule CSV file', \
		default='dr_menchaca_report_negative.csv', required=False)

	# get nlst folder path
	args = parser.parse_args()
	root = args.root

	# get clinical_csv file path
	clinical_csv = root + args.clinical_csv
	assert os.path.exists(clinical_csv)

	# get pid file path, if exists, get split, else create split based on nodules info csv
	pids_file = root+args.pids_file
	if (os.path.exists(pids_file)):
		train_pos, val_pos, test_pos, train_neg, val_neg, test_neg = get_pids_split(pids_file)
	else:
		pos_csv = root + args.positive_csv
		assert os.path.exists(pos_csv)
		neg_csv = root + args.negative_csv
		assert os.path.exists(neg_csv)
		train_pos, val_pos, test_pos, train_neg, val_neg, test_neg = create_pids_split(pids_file, pos_csv, neg_csv)

	# save clinical file
	pids = train_pos + val_pos + test_pos + train_neg + val_neg + test_neg
	clinical_df = preprocess_clinical_df(pids, pd.read_csv(clinical_csv))
	clinical_df.to_csv(os.path.join(root, 'clinical_preprocessed.csv'))

