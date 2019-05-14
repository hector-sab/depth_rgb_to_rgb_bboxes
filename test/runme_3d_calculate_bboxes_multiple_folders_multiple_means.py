"""
Translates and rotates the depth projection into the cam1 RGB sensor position.
"""
import os
fdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.insert(0,fdir)

from depth_rgb_to_bboxes import utils_depth as udp

import cv2
from tqdm import tqdm
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


rot = [[0.99966,-0.00676,0.02532],
	   [0.00729,0.99975,-0.02100],
	   [-0.02517,0.02118,0.99946]]
rot = np.array(rot)

trans = [-0.01953,-0.00103,-0.00416]
trans = np.array(trans)

fl=[583.87,582.29] # Focal Lenght
pp=[228.75,329.44] # Principal Point


# For Camera 1. Homography Adjustment
#M = np.array([[8.21107477e-01,3.38614263e-03,2.27771764e+01],
#	[-1.05755095e-02,8.43191419e-01,4.44988096e+01],
#	[-1.65387419e-04,-1.17132272e-04,1.00000000e+00]])

# For Camera 3. Homography Adjustment
M = np.array([[9.49374573e-01,-2.30083863e-03,2.08312357e+01],
	[8.02288948e-03,9.23875695e-01,3.02569001e+01],
	[2.90156601e-05,-2.01306910e-05,1.00000000e+00]])


if __name__=='__main__':
	mean_std_main_dir = '/data2/HectorSanchez/database/PeopleCounter/FINAL_RUN/camara3/mean_std/00000000/'
	lbs_out_dir = '/data2/HectorSanchez/database/PeopleCounter/FINAL_RUN/camara3/labels/'
	dpt_dir = '/data/HectorSanchez/database/PeopleCounter/camara3/'

	folders = sorted(os.listdir(dpt_dir))

	for i,folder in enumerate(folders):
		print('{}/{} Folder: {}'.format(i,len(folders)-1,folder))

		# Set the location of the mean and std files
		mpath = os.path.join(mean_std_main_dir,folders[0],'bg_mean.npy')
		spath = os.path.join(mean_std_main_dir,folders[0],'bg_std.npy')


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
		
		for file in tqdm(files):
			#print('--->',file)
			fim = file.replace('depth','color')
			fim = fim.replace('png','jpg')
			rgb_im_path = os.path.join(fdir,fim)
			im = cv2.imread(rgb_im_path)

			
			dpt_im_path = os.path.join(fdir,file)
			dim = udp.load_dim(dpt_im_path,M=M)
			x,y = udp.projection_correction(dim)

			bboxes_test = bb_generator.determine_bboxes(dim)#,dx=0,dy=0,fx=1,fy=1)
			
			# Show mask in im
			im = im + np.expand_dims(bb_generator.mask,axis=-1)*100

			for bbox in bboxes_test:
				im = cv2.rectangle(im,pt1=(bbox[0],bbox[1]),pt2=(bbox[2],bbox[3]),color=(0,0,255),
				thickness=3)



			fig1,ax1 = plt.subplots()
			ax1.imshow(im[...,::-1])
			#fig2,ax2 = plt.subplots()
			#ax2.imshow(mask)
			#plt.show()
			plt.pause(0.2)
			plt.close(fig1)
			#plt.close(fig2)


			if len(y)==0:
				# No points found
				continue
			
			if False:
				# Plot with open3d
				xyz = np.zeros((np.size(x),3))
				xyz[:,0] = np.reshape(x,-1)
				xyz[:,1] = np.reshape(y,-1)
				xyz[:,2] = np.reshape(dim,-1)
				pcd = o3d.PointCloud()
				pcd.points = o3d.Vector3dVector(xyz)

				o3d.draw_geometries([pcd])
			
			if False:
				fig = plt.figure()
				ax = fig.gca(projection='3d')
				ax.set_aspect('equal')
				#surf = ax.plot_surface(x,y,dim,cmap='gray')
				#surf = ax.plot_trisurf(x,y,dim,cmap='gray')
				ax.scatter(x,y,dim)

				# Invert z axis
				plt.gca().invert_zaxis()
				# Maximaze window
				#plt.get_current_fig_manager().window.showMaximized()
				plt.show()
				plt.close(fig)