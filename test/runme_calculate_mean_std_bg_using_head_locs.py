import os
fdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.insert(0,fdir)

from depth_rgb_to_bboxes import utils_depth as udp

import random

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


if __name__=='__main__':
	# Detected heads folders' folder
	heads_main_folder = '/data/HectorSanchez/database/PeopleCounter/READY/head_location/camara4/'
	ims_main_folder = '/data/HectorSanchez/database/PeopleCounter/READY/images/camara4/'
	mean_std_main_out_folder = '/data/HectorSanchez/database/PeopleCounter/READY/mean_std/'

	folders = sorted(os.listdir(heads_main_folder))

	head_files = {}
	bg_files = {}
	for i,folder in enumerate(folders):
		print('>>>> {} of {}'.format(i+1,len(folders)))
		# For heads location files
		folder_path = os.path.join(heads_main_folder,folder)
		files = sorted(os.listdir(folder_path))
		files = [x.replace('.txt','.png') for x in files]
		head_files[folder] = files

		# For images
		folder_path = os.path.join(ims_main_folder,folder)
		files = sorted(os.listdir(folder_path))
		files = [x for x in files if '.png' in x]
		# Select images with no people
		files = [x for x in files if x not in head_files[folder]]
		bg_files[folder] = files

		# Create Mean Std
		out_dir = os.path.join(mean_std_main_out_folder,'camara4',folder)
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)

		fpath = os.path.join(folder_path,bg_files[folder][0])
		dim = udp.load_dim(fpath)
		#print(dim.shape)
		#input('WF')
		all_dims = np.zeros((dim.shape[0],dim.shape[1],0),dtype=np.float64)

		pbar = tqdm(total=100)
		#for file in bg_files[folder]:
		for i in range(100):
			file = random.choice(bg_files[folder])
			fpath = os.path.join(folder_path,file)
			try:
				dim = udp.load_dim(fpath)
				dim = np.expand_dims(dim,axis=-1)
				all_dims = np.concatenate((all_dims,dim),axis=-1)
			except Exception as e:
				print(e)
			pbar.update(1)
		pbar.close()

		bg_mean = np.mean(all_dims,axis=-1) # Mean Background
		bg_std = np.std(all_dims,axis=-1) # Std dev Background

		np.save(os.path.join(out_dir,'bg_mean'),bg_mean)
		np.save(os.path.join(out_dir,'bg_std'),bg_std)