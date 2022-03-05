import numpy as np
import pandas as pd
import random
import scipy.ndimage

import os
import os.path as osp
from glob import glob

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn_pandas import DataFrameMapper
    
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torchtuples as tt

# import time

############################################################
#  Preprocess Clinical Data
############################################################
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

def get_clinical_arr(df, pids, surv_type='1stslice_death'):
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

	df = df[df['pid'].isin(pids)] # leave only pids we are interested in
	# Add survival information depending on survival analysis type
	if (surv_type == '1stslice_death'):
		cols_leave.extend(['days_1stpos_death', 'death'])
	
	# Preprocess clinical data: 
	# OneHotEncoder for Categorical
	# Median SimpleImputer and StandardScaler for Numerical
	# Leave Boolean column
	onehot = [([col], OneHotEncoder(categories = [categories_dict[col]])) for col in categories] 
	standardize = [([col], [SimpleImputer(missing_values=np.nan, strategy='mean'), MinMaxScaler()]) for col in numericals]
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
    
	return df, clinical, duration, event, numericals+categories_ret+cols_leave

############################################################
#  Preprocess Imaging Data
############################################################
def load_nodule(npfolder, pid, yr, ab_num, mean_nods = 443.16226803816096, std_nods = 574.5850006260475):
	"""Load nodule from npy file based on information from input abnormality row, 
	cut to size (64, 128, 128), then normalize based on pre-computed mean and std of all nodules.
	Nodules from npy files has been cut roughly (90x90x90 mm^3) from nodule csv file
	resampled to new_spacing (slice thickness = 1.4, pixel spacing = .7, .7) to size (64, 128, 128),

    pid, yr, ab_num: string. Nodule identifiers: patient ID, study year, abnormality ID 
    mean_nods: pre-computed nodules mean
    std_nods: pre-computed nodules standard deviation
	Returns:
	nodule: 3D torch tensor of nodule pixel array, type float, size (slices=64, h=128, w=128)
		nodules image from npy file
    """
	path = npfolder + str(pid) + '_T' +str(yr)+'_'+str(ab_num) + '.npy'
	nodule = np.load(path)
	nodule = nodule[:64, :128, :128]
	nodule = (nodule - mean_nods)/std_nods
	return nodule


############################################################
#  NLST Dataset
############################################################
def collate_fn(batch):
	"""Stacks the entries of a nested tuple 
	if batch made of tuples else stack tensors and return
	"""
	if (len(batch[0])==2):
		return tt.tuplefy(batch).stack()
	else:
		return torch.stack(batch)

class NLSTData(Dataset):
	def __init__(self, pids, mode, nodule_csv = '', clinical_csv = '', img_npy_folder = '', 
		clinical = True, image = True, nodule = 'one', surv_type='1stslice_death'):
		"""Load a subset of the NLST Dataset.
		pids: list of input pids
		mode: mode  - 'train' or 'test'
		nodule_csv: string. Nodule file (ct-ab)
		clinical_csv: string. Clinical file (prsn)
		clinical: bool. Indicate return clinical information or not  
		image: bool. Indicate return image or not  
		nodule: string. Type of return nodule - 
			'one': return only one nodule per patient, 
			'onetimepoint': return one nodule per patient per time point,
			'all': return all nodule per patient
		surv_type: string. Type of surival analysis - 
			'1stslice_death': from 1st scan to death
        """
		self.mode = mode       
		self.data_len = len(pids)
		self.include_clinical = clinical
		self.include_img = image
		self.include_nodule = nodule
		self.img_folder = img_npy_folder
        
		# Preprocess clinical information
		self.clinical_df = pd.read_csv(clinical_csv, index_col=0)
		self.clinical_df, self.clinical_arr, self.time_arr, self.event_arr, self.clinical_arr_cat = \
        get_clinical_arr(self.clinical_df, pids)
		self.clinical_shape = np.shape(self.clinical_arr)
		self.time, self.event = \
        tt.tuplefy(self.time_arr, self.event_arr).to_tensor()

		# Get nodules information
		if (self.include_nodule):
			self.nodule = pd.read_csv(nodule_csv)
			self.nodule = self.nodule[self.nodule['pid'].isin(pids)] # leave only pids we are interested in
			self.nodule = self.nodule[self.nodule['cancer']==1] # leave only cancer ones
			self.nodule = self.nodule[self.nodule['fs']==1] # leave only first scan

		# Create index-pid dict
		self.idx_pid = {}
		i = 0
		for pid in pids:
			self.idx_pid[i] = pid
			i+=1
            
	def set_type(self, mode):
		# This function return clinical dataframe
		if (mode == 'clinical'):
			self.include_clinical = True
			self.include_img = False
		if (mode == 'nodule'):
			self.include_clinical = False
			self.include_img = True
		if (mode == 'comb'):
			self.include_clinical = True
			self.include_img = True
		return 
    
	def set_mode(self, mode):
		# This function set mode train or test
		self.mode = mode
		return 
    
	def get_clinical_df(self):
		# This function return clinical dataframe
		return self.clinical_df 
    
	def get_clinical_cat(self):
		# This function return clinical categories
		return self.clinical_arr_cat
    
	def get_clinical_shape(self):
		# This function return clinical categories
		return self.clinical_shape

	def get_surv_clinical(self):
		# This function return surival clinical numpy array 
		return self.clinical_arr

	def get_surv_nodules(self):
		# This function return surival nodule numpy array 
		nodules_ret = []
		for pid in self.idx_pid.values():
			curr_nodules = self.nodule[self.nodule["pid"] == pid]
			for index, row in curr_nodules.iterrows():
				nodules = load_nodule(self.img_folder, int(row['pid']), int(row['study_yr']), int(row['sct_ab_num']))
			nodules = np.asarray(nodules, dtype = np.float32)
			nodules_ret.append(nodules)
		return np.asarray(nodules_ret, dtype = np.float32)
    
	def get_surv_target(self):
		# This function return surival target as tuple of int numpy array (time, event) 
		return self.time_arr, self.event_arr
    
	def get_clinical(self, index):
		# This function return clinical array as 2D torch tensor from numpy array size (num patient, features=42)
		return torch.from_numpy(self.clinical_arr[index])
    
	def get_image(self, index, nodule = 'one'):
		"""Get image information based on index.
        index: int. Index of patient 
		Returns:
		nodules: 3D torch tensor of nodule pixel array, type float, size (slices=64, h=128, w=128)
		TODO: Add variation to get image (load the whole image/lung segmented image, 
			or several nodule based on nodule input )
			Ex: One nodule, return: nodule pixel array (torch), image pixel array (torch); 
			Multiple nodule, return: list(list(nodule pixel array (torch))_by_yr), list(image pixel array (torch))?
        """
		nodules = np.array([])
		pid = self.idx_pid[index]
		curr_nodules = self.nodule[self.nodule["pid"] == pid]
		for index, row in curr_nodules.iterrows():
			nodules = load_nodule(self.img_folder, int(row['pid']), int(row['study_yr']), int(row['sct_ab_num']))
		nodules = np.asarray(nodules, dtype = np.float32)
		return torch.from_numpy(nodules)
    
	def __getitem__(self, index):
		"""Get patient information based on index.
        index: int. Index of patient 
		Returns:
		General case: test split - ((clinical_info, nodules)), train or val split - 
			((clinical_info, nodules), (time, event)) : tuple of two tuples 
			clinical_info: torch tensor size clinical features = 42, type float
			nodules: torch tensor size (slices = 64, x, y = 128), type float
			time: torch tensor size 1, type int
			event: torch tensor size 1, type int
		Clinical only: test split - (clinical_info), other splits - (clinical_info, (time, event))
		Nodule only: test split - (nodules), other splits - (nodules, (time, event))
        """
		# Clinical info
# 		print(index)
# 		if type(index) is not int:
# 			raise ValueError(f"Need `index` to be `int")#. Got {type(index)})

		if (self.include_clinical):
			clinical = self.get_clinical(index)
			if (self.include_img == False):
				if (self.mode == 'test'):
					return clinical
				return clinical, (self.time[index], self.event[index])
            
		if (self.include_nodule or self.include_img):
			nodules = self.get_image(index)
			if (self.include_clinical == False):
				if (self.mode == 'test'):
					return nodules
				return nodules, (self.time[index], self.event[index])

		if (self.mode == 'test'):
			return (clinical, nodules)

		return (clinical, nodules), (self.time[index], self.event[index])

	def __len__(self):
		return self.data_len

    
############################################################
#  NLST Dataset BatchSampler
############################################################
def chunklist(indices, chunk_size):
	# function divide list indices into list of chunks of size chunk_size
	return [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]

def chunklist(indices, chunk_size):
	# function divide list indices into list of chunks of size chunk_size
	return [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]

class BatchNLSTSampler(Sampler):
	def __init__(self, dataset, batch_size):
		"""Sampler makes sure each batch has at least one event==1.
		dataset: input dataset
		batch_size: input batch_size
		"""
		event = dataset.get_surv_target()[1] # get event from dataset
		self.inds_1 = [i for i, e in enumerate(event) if e == 1] # index for event==1
		self.inds_0 = [i for i, e in enumerate(event) if e == 0] # index for event==0
		self.batch_size = batch_size
		self.num_batches = int(np.ceil(len(event)/self.batch_size))

	def __iter__(self):
		# Shuffle index for event==1
		random.shuffle(self.inds_1)
		# Separate event==1 to list of size total number of batch and the rest
		inds_1tmp = self.inds_1[:self.num_batches]
		# Combine the rest of event==1 with event==0, shuffle it
		inds_comb = self.inds_1[self.num_batches:] + self.inds_0
		random.shuffle(inds_comb)
        
		# Separate event==1 to chunk of 1 and combine event to batch_size-1
		inds_1tmp = chunklist(inds_1tmp, 1)
		inds_comb = chunklist(inds_comb, self.batch_size-1)
        
		# For each batch, combine 1 event==1 with batch_size-1 event_combine
		# to make sure each batch has at least one event==1, shuffle that
		combined = []
		for i in range (self.num_batches):
			if (i < len(inds_comb)):
				tmp = inds_1tmp[i]+inds_comb[i]
			else:
				tmp = inds_1tmp[i]
			random.shuffle(tmp)
			combined.append(tmp)
            
		# Shuffle return iter combined batches and return
		random.shuffle(combined)
		return iter(combined)
    
	def __len__(self):
		return self.num_batches

    
class BatchNLSTSampler2(Sampler):
	def __init__(self, dataset, batch_size):
		"""Sampler makes sure each batch has at least one event==1.
		dataset: input dataset
		batch_size: input batch_size
		"""
		self.event = dataset.get_surv_target()[1] # get event from dataset
		self.num_samples = len(self.event)
		self.batch_size = batch_size
		self.num_batches = int(np.ceil(self.num_samples/self.batch_size))
		
	def shuffle_epoch(self):
		valid_epoch = False
        
		while valid_epoch is not True:
			perms = np.arange(self.num_samples)
			random.shuffle(perms)
			tmp = True                
			for b in range(self.num_batches):
				batch_indices = perms[b*self.batch_size:(b+1)*self.batch_size]
				pos_batch = sum([self.event[idx] for idx in batch_indices])
				if pos_batch == 0:
					tmp = False
					break
			if (tmp):
				valid_epoch = True
		return perms

	def __iter__(self):
		indices = self.shuffle_epoch()
        
		combined = []
		for b in range (self.num_batches):
			combined.append(indices[b*self.batch_size:(b+1)*self.batch_size].tolist())
            
		# Shuffle return iter combined batches and return
		random.shuffle(combined)
		return iter(combined)
    
	def __len__(self):
		return self.num_batches
