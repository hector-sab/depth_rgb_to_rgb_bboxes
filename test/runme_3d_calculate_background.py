"""
Calculates the Mean and Std dev of the background by
analising which depth images are indeed background
with the SSIM strategy.
"""
import os
fdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.insert(0,fdir)

from depth_rgb_to_bboxes import utils_depth as udp

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__=='__main__':
	#fdir = '../ims/'
	fdir = '/data/HectorSanchez/database/PeopleCounter/camara1/00000000/'
	print(fdir)
	bg_extractor = udp.FindBackgroundDepth()

	bg_extractor.calculate_sim(fdir)
	bg_extractor.determine_background(.748)
	#bg_extractor.ssim_chart()
	print('--->',len(bg_extractor.background_fnames))

	if True:
		# Visualize background depth images
		for file in bg_extractor.background_fnames:
			z = udp.load_dim(fdir+file)
			
			fig,ax = plt.subplots()
			ax.imshow(z)
			plt.show()
			plt.close(fig)

	# Calculate Mean and Std dev
	tmp_dim = udp.load_dim(fdir+bg_extractor.background_fnames[0])
	all_dims = np.zeros((tmp_dim.shape[0],tmp_dim.shape[1],0),dtype=np.float64)

	# Stack all the background depth images 
	pbar = tqdm(total=len(bg_extractor.background_fnames))
	for file in bg_extractor.background_fnames:
			dim = udp.load_dim(fdir+file)
			dim = np.expand_dims(dim,axis=-1)
			all_dims = np.concatenate((all_dims,dim),axis=-1)
			pbar.update(1)
	pbar.close()

	# Calculate Mean and Std dev
	bg_mean = np.mean(all_dims,axis=-1) # Mean Background
	bg_std = np.std(all_dims,axis=-1) # Std dev Background
	
	if False:
		# Define saving directory
		npy_out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		npy_out_dir = os.path.join(npy_out_dir,'mean_std','camera1')

		if not os.path.exists(npy_out_dir):
			os.makedirs(npy_out_dir)

		# Save mean and std dev
		np.save(os.path.join(npy_out_dir,'bg_camera1_mean.npy'),bg_mean)
		np.save(os.path.join(npy_out_dir,'bg_camera1_std.npy'),bg_std)
	
	if True:
		# Display mean and std dev
		fig1,ax1 = plt.subplots()
		ax1.set_title('Mean')
		ax1.imshow(bg_mean)
		fig2,ax2 = plt.subplots()
		ax2.set_title('Std')
		ax2.imshow(bg_std)
		plt.show()
		plt.close(fig1)
		plt.close(fig2)

