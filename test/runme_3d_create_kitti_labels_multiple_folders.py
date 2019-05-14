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

# For Camera 3. Homography Adjustment
M = np.array([[9.49374573e-01,-2.30083863e-03,2.08312357e+01],
	[8.02288948e-03,9.23875695e-01,3.02569001e+01],
	[2.90156601e-05,-2.01306910e-05,1.00000000e+00]])

if __name__=='__main__':
	# Set the location of the mean and std files
	mean_std_dir = '/data2/HectorSanchez/database/PeopleCounter/FINAL_RUN/camara3/mean_std/00000000/'
	mpath = os.path.join(mean_std_dir,'bg_mean.npy')
	spath = os.path.join(mean_std_dir,'bg_std.npy')

	bb_generator = udp.BBoxGenerator()
	bb_generator.set_background(mpath,spath)
	#bb_generator.set_translation_matrix(trans)
	#bb_generator.set_rotation_matrix(rot)
	#bb_generator.set_focal_lenght(np.array(fl))
	#bb_generator.set_principal_point(np.array(pp))

	# Set images folders
	folders_dir = '/data/HectorSanchez/database/PeopleCounter/camara3/'
	folders = sorted(os.listdir(folders_dir))

	# Set main output dir
	main_out_dir = '/data2/HectorSanchez/database/PeopleCounter/FINAL_RUN/camara3/labels/'

	for i,folder in enumerate(folders):
		if i<67:
			"""
			folder 61, 66 have some issues with depth images
			"""
			continue
		print('---> Folder {} of {}: {}'.format(i+1,len(folders),folder))
		#if i<26:
		#	continue
		folder_dir = os.path.join(folders_dir,folder)
		files = sorted(os.listdir(folder_dir))
		files = [x for x in files if '.png' in x]

		out_dir = os.path.join(main_out_dir,folder)
		print('out_dir:',out_dir)
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)

		pbar = tqdm(total=len(files))
		for file in files:
			fim = file.replace('depth','color')
			fim = fim.replace('png','jpg')
			dim = udp.load_dim(os.path.join(folder_dir,file),M=M)
			bboxes = bb_generator.determine_bboxes(dim)#,dx=-10,dy=30,fx=1,fy=1)
			create_lbs([os.path.join(out_dir,fim)],[bboxes])
			pbar.update(1)
		pbar.close()