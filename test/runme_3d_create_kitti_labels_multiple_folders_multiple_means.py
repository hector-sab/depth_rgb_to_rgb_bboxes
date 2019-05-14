"""
Translates and rotates the depth projection into the cam1 RGB sensor position.
"""
import os
fdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.insert(0,fdir)

from depth_rgb_to_bboxes import utils_depth as udp

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def save_bbox2file(path,bboxes):
	# [x_left,y_top,x_right,y_bottom]
	line = 'person 0.0 0 0.0 {} {} {} {} 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0'
	if len(bboxes)>0:
		with open(path,'w') as f:
			for i,bbox in enumerate(bboxes):
				f.write(line.format(bbox[0],bbox[1],bbox[2],bbox[3]))
				if i<len(bboxes)-1:
					f.write('\n')

def create_lbs(fnames,bboxes):
	#print('Creating labels...')
	#pbar = tqdm(range(len(fnames)))
	for i in range(len(fnames)):
		fname = fnames[i]
		im_bboxes = bboxes[i]
		
		save_bbox2file(fname[:-3]+'txt',im_bboxes)


rot = [[0.99966,-0.00676,0.02532],
	   [0.00729,0.99975,-0.02100],
	   [-0.02517,0.02118,0.99946]]
rot = np.array(rot)

trans = [-0.01953,-0.00103,-0.00416]
trans = np.array(trans)

fl=[583.87,582.29] # Focal Lenght
pp=[228.75,329.44] # Principal Point

# For Camera 1. Homography Adjustment
M = np.array([[8.21107477e-01,3.38614263e-03,2.27771764e+01],
	[-1.05755095e-02,8.43191419e-01,4.44988096e+01],
	[-1.65387419e-04,-1.17132272e-04,1.00000000e+00]])

# TODO: Folder 14 has a shape (721,480). WTF?
if __name__=='__main__':
	mean_std_main_dir = '/data2/HectorSanchez/database/PeopleCounter/FINAL_RUN/camara1/mean_std/'
	lbs_main_out_dir = '/data2/HectorSanchez/database/PeopleCounter/FINAL_RUN/camara1/labels_v2/'
	dpt_dir = '/data/HectorSanchez/database/PeopleCounter/camara1/'

	folders = sorted(os.listdir(mean_std_main_dir))

	for i,folder in enumerate(folders):
		if i<90:
			continue
		#"""
		if i>120:
			break
		#"""
		print('{}/{} Folder: {}'.format(i,len(folders)-1,folder))

		lbs_out_dir = os.path.join(lbs_main_out_dir,folder)
		if not os.path.exists(lbs_out_dir):
			os.makedirs(lbs_out_dir)

		# Set the location of the mean and std files
		mpath = os.path.join(mean_std_main_dir,folders[13],'bg_mean.npy')
		spath = os.path.join(mean_std_main_dir,folders[13],'bg_std.npy')

		#mpath = os.path.join(mean_std_main_dir,folder,'bg_mean.npy')
		#spath = os.path.join(mean_std_main_dir,folder,'bg_std.npy')


		# Prepare object that removes the background
		bg_remover = udp.BackgroundRemover()
		bg_remover.set_background(mpath,spath)

		# Get all the depth images
		fdir = os.path.join(dpt_dir,folder)
		files = sorted(os.listdir(fdir))
		files = [x for i,x in enumerate(files) if '.png' in x and i>100]

		# This is the only thing needed
		bb_generator = udp.BBoxGenerator()
		bb_generator.set_background(mpath,spath)
		print(bb_generator.bg_remover.bg_mean.shape)
		
		for file in tqdm(files):
			#print('--->',file)
			fim = file.replace('depth','color')
			fim = fim.replace('png','jpg')
			rgb_im_path = os.path.join(fdir,fim)
			#im = cv2.imread(rgb_im_path)
			
			dpt_im_path = os.path.join(fdir,file)
			dim = udp.load_dim(dpt_im_path,M=M)
			
			if dim.shape[0]>480 or dim.shape[1]>640:
				continue

			bboxes = bb_generator.determine_bboxes(dim,dx=0,dy=0,fx=1,fy=1)

			lbs_path = os.path.join(lbs_out_dir,fim.replace('jpg','txt'))
			create_lbs([lbs_path],[bboxes])