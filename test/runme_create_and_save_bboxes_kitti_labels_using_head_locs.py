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

def create_lbs(fnames,bboxes,out_dir):
	#print('Creating labels...')
	#pbar = tqdm(range(len(fnames)))
	for i in range(len(fnames)):
		fname = fnames[i]
		im_bboxes = bboxes[i]
		
		save_bbox2file(os.path.join(out_dir,fname[:-3]+'txt'),im_bboxes)

if __name__=='__main__':
	heads_main_folder = '/data/HectorSanchez/database/PeopleCounter/READY/head_location/camara4/'
	ims_main_folder = '/data/HectorSanchez/database/PeopleCounter/READY/images/camara4/'
	mean_std_folder = '/data/HectorSanchez/database/PeopleCounter/READY/mean_std/camara4/'
	bboxes_main_out_dir = '/data/HectorSanchez/database/PeopleCounter/READY/bboxes/camara4/'
	folders = sorted(os.listdir(heads_main_folder))

	head_files = {}

	for i,folder in enumerate(folders):
		print('>>>> {} of {}'.format(i+1,len(folders)))

		out_dir = os.path.join(bboxes_main_out_dir,folder)
		print('Output dir:', out_dir)

		mean_path = os.path.join(mean_std_folder,folder,'bg_mean.npy')
		std_path = os.path.join(mean_std_folder,folder,'bg_std.npy')

		bb_generator = udp.BBoxGenerator()
		bb_generator.set_background(mean_path,std_path)

		# For heads location files
		files = sorted(os.listdir(os.path.join(heads_main_folder,folder)))
		files = [x.replace('.txt','.png') for x in files]
		head_files[folder] = files

		folder_path = os.path.join(ims_main_folder,folder)

		pbar = tqdm(total=len(head_files[folder]))
		for file in head_files[folder]:
			fpath = os.path.join(folder_path,file)
			dim = udp.load_dim(fpath)

			bboxes = bb_generator.determine_bboxes(dim)

			###
			# Inspect visualiy that the bboxes are correct
			if False:
				print(file)
				plt.imshow(dim)
				plt.show()
				plt.imshow(bb_generator.mask)
				plt.show()
				for bbox in bboxes:
					dim = cv2.rectangle(dim,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,255,0),3)
					print(bbox)
				plt.imshow(dim)
				plt.show()
			###

			if not os.path.exists(out_dir):
				os.makedirs(out_dir)

			create_lbs([file],[bboxes],out_dir)
			pbar.update(1)
		pbar.close()