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
import open3d as o3d
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

def create_lbs(fnames,bboxes,out_dir):
	print('Creating labels...')
	pbar = tqdm(range(len(fnames)))
	for i in pbar:
		fname = fnames[i]
		im_bboxes = bboxes[i]
		
		save_bbox2file(out_dir+fname[:-3]+'txt',im_bboxes)


rot = [[0.99966,-0.00676,0.02532],
	   [0.00729,0.99975,-0.02100],
	   [-0.02517,0.02118,0.99946]]
rot = np.array(rot)

trans = [-0.01953,-0.00103,-0.00416]
trans = np.array(trans)

fl=[583.87,582.29] # Focal Lenght
pp=[228.75,329.44] # Principal Point


if __name__=='__main__':
	# Set the location of the mean and std files
	mean_std_dir = os.path.join(fdir,'mean_std/camera1/')
	mpath = os.path.join(mean_std_dir,'bg_camera1_mean.npy')
	spath = os.path.join(mean_std_dir,'bg_camera1_std.npy')

	# Get all the depth images
	fdir = '../ims/'
	files = sorted(os.listdir(fdir))
	files = [x for i,x in enumerate(files) if '.png' in x and i>45]

	# This is the only thing needed
	bb_generator = udp.BBoxGenerator()
	bb_generator.set_background(mpath,spath)
	bb_generator.set_translation_matrix(trans)
	bb_generator.set_rotation_matrix(rot)
	bb_generator.set_focal_lenght(np.array(fl))
	bb_generator.set_principal_point(np.array(pp))
	
	out_dir = '../ims_lbs/'
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
		
	for file in files:
		print('--->',file)
		fim = file.replace('depth','color')
		fim = fim.replace('png','jpg')

		dim = udp.load_dim(fdir+file)
		bboxes = bb_generator.determine_bboxes(dim,dx=-10,dy=30,fx=1,fy=1)

		create_lbs([fim],[bboxes],out_dir)