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
import matplotlib.pyplot as plt


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

	# Prepare object that removes the background
	bg_remover = udp.BackgroundRemover()
	bg_remover.set_background(mpath,spath)

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
	
	for i,file in enumerate(files):
		print('--->',file)
		
		fim = file.replace('depth','color')
		fim = fim.replace('png','jpg')
		im = cv2.imread(fdir+fim)

		
		dim = udp.load_dim(fdir+file)
		x,y = udp.projection_correction(dim)

		bboxes_test = bb_generator.determine_bboxes(dim,dx=-10,dy=30,fx=1,fy=1)
		
		# Show mask in im
		im = im + np.expand_dims(bb_generator.mask,axis=-1)*100

		for bbox in bboxes_test:
			im = cv2.rectangle(im,pt1=(bbox[0],bbox[1]),pt2=(bbox[2],bbox[3]),color=(0,0,255),
			thickness=3)



		fig1,ax1 = plt.subplots()
		ax1.imshow(im[...,::-1])
		ax1.axis('off')
		fig1.savefig("/home/hectorsab/Downloads/seminario2/bbox{:04d}.png".format(i), bbox_inches='tight')
		fig2,ax2 = plt.subplots()
		ax2.imshow(bb_generator.mask)
		ax2.axis('off')
		fig2.savefig("/home/hectorsab/Downloads/seminario3/mask{:04d}.png".format(i), bbox_inches='tight')
		
		#plt.show()
		#plt.pause(0.2)
		plt.close(fig1)
		plt.close(fig2)


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
			plt.get_current_fig_manager().window.showMaximized()
			plt.show()
			plt.close(fig)