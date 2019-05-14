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

import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__=='__main__':
	# For Camera 1. Homography Adjustment
	#M = np.array([[8.21107477e-01,3.38614263e-03,2.27771764e+01],
	#	[-1.05755095e-02,8.43191419e-01,4.44988096e+01],
	#	[-1.65387419e-04,-1.17132272e-04,1.00000000e+00]])

	# For Camera 3. Homography Adjustment
	M = np.array([[9.49374573e-01,-2.30083863e-03,2.08312357e+01],
		[8.02288948e-03,9.23875695e-01,3.02569001e+01],
		[2.90156601e-05,-2.01306910e-05,1.00000000e+00]])

	main_dir = '/data/HectorSanchez/database/PeopleCounter/camara3/'
	main_out_dir = '/data2/HectorSanchez/database/PeopleCounter/FINAL_RUN/camara3/mean_std/'
	folders = sorted(os.listdir(main_dir))

	bg_extractor = udp.FindBackgroundDepth()

	ssim_ths = []


	folders = [x for i,x in enumerate(folders) if i==0]
	for i,folder in enumerate(folders):
		print('{}/{} Folder: {}'.format(i,len(folders),folder))

		fdir = os.path.join(main_dir,folder)
		print('--- Calculating SSIM:')
		bg_extractor.calculate_sim(fdir)
		bg_extractor.ssim_chart()
		th = float(input('Select a threshold: '))
		ssim_ths.append(th)

	for i,folder in enumerate(folders):
		print('{}/{} Folder: {}'.format(i,len(folders),folder))

		fdir = os.path.join(main_dir,folder)
		print('--- Calculating SSIM:')
		bg_extractor.calculate_sim(fdir)
		#bg_extractor.ssim_chart()
		th = ssim_ths[i]
		#th = float(input('Select a threshold: '))
		#th = 0.78
		bg_extractor.determine_background(th)
		print('    ====> BG frames',len(bg_extractor.background_fnames))

		if False:
			# Visualize background depth images
			for file in bg_extractor.background_fnames:
				z = udp.load_dim(fdir+file)
				
				fig,ax = plt.subplots()
				ax.imshow(z)
				plt.show()
				plt.close(fig)

		# Calculate Mean and Std dev
		if len(bg_extractor.background_fnames)==0: continue

		tmp_dim_path = os.path.join(fdir,bg_extractor.background_fnames[0])
		tmp_dim = udp.load_dim(tmp_dim_path)
		#all_dims = np.zeros((tmp_dim.shape[0],tmp_dim.shape[1],0),dtype=np.float64)
		all_dims = []

		# Stack all the background depth images 
		print('Calculating Mean and Std:')
		pbar = tqdm(total=len(bg_extractor.background_fnames))
		for file in bg_extractor.background_fnames:
				path = os.path.join(fdir,file)
				dim = udp.load_dim(path,M=M)
				#dim = np.expand_dims(dim,axis=-1)
				#all_dims = np.concatenate((all_dims,dim),axis=-1)
				all_dims.append(dim)
				pbar.update(1)
		pbar.close()
		all_dims = np.array(all_dims)
		print('        ',all_dims.shape)

		# Calculate Mean and Std dev
		#if np.sum(all_dims<0)>0:
		#	print('BAD NEWS:',np.sum(all_dims<0))

		#for dim in all_dims:
		#	if np.sum(dim<0)>0:
		#		print('YUP',np.sum(dim<0))
		#	dim[dim<0] = 0
		#	if np.sum(dim<0)>0:
		#		print('YUP2',np.sum(dim<0))
		#	fig,ax = plt.subplots()
		#	ax.imshow(dim)
		#	plt.show()
		#	plt.close(fig)
		#all_dims[all_dims<0] = 0

		bg_mean = np.mean(all_dims,axis=0) # Mean Background
		bg_std = np.std(all_dims,axis=0) # Std dev Background
		
		if True:
			# Define saving directory
			npy_out_dir = os.path.join(main_out_dir,folder)

			if not os.path.exists(npy_out_dir):
				os.makedirs(npy_out_dir)

			# Save mean and std dev
			np.save(os.path.join(npy_out_dir,'bg_mean.npy'),bg_mean)
			np.save(os.path.join(npy_out_dir,'bg_std.npy'),bg_std)

			#with open(os.path.join(npy_out_dir,'ssim_list.p'), 'wb') as fp:
			#	pickle.dump(bg_extractor.ssim, fp)

		
		if False:
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

