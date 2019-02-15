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

	# For camera 1
	x_min = -1078.3400329779568
	y_min = -1442.6339578413751
	x_max = 1682.8418437836763
	y_max = 1385.4591844350853
	
	for file in files:
		print('--->',file)
		
		fim = file.replace('depth','color')
		fim = fim.replace('png','jpg')
		im = cv2.imread(fdir+fim)
		
		dim = udp.load_dim(fdir+file)
		x,y = udp.projection_correction(dim)

		#### Rotate, Translate, and reproyect to 2D
		# Translate
		x += trans[0]
		y += trans[1]
		dim += trans[2]

		xyz = np.zeros((np.size(x),3))
		xyz[:,0] = np.reshape(x,-1)
		xyz[:,1] = np.reshape(y,-1)
		xyz[:,2] = np.reshape(dim,-1)
		
		# Rotation
		xyz = np.dot(xyz,rot)

		# Reproyect
		us = np.zeros((307200))#[]
		vs = np.zeros((307200))#[]

		## Calculate new X,Y position in the image
		us = (xyz[:,0]/xyz[:,2])*fl[0] + pp[0]
		vs = (xyz[:,1]/xyz[:,2])*fl[1] + pp[1]
		
		us = us.astype(np.int32)
		vs = vs.astype(np.int32)

		x_min = np.amin(us)
		x_max = np.amax(us)
		y_min = np.amin(vs)
		y_max = np.amax(vs)
		
		#new_im = np.zeros((y_max-y_min,x_max-x_min))
		dim = np.zeros((480,640))
		us -= x_min
		vs -= y_min

		us_mask = np.logical_and(us>=0,us<640)
		vs_mask = np.logical_and(vs>=0,vs<480)

		mask = np.logical_and(us_mask,vs_mask)

		us = us[mask]
		vs = vs[mask]
		zs = xyz[:,2][mask.reshape(-1)]
		dim[vs,us] = zs
		####

		mask = bg_remover.remove_background(dim)
		mask_bool = mask.astype(np.bool)
		
		blobs = udp.find_blobs(mask)
		blobs = udp.filter_blobs(blobs,min=5000)
		bboxes = udp.blobs_to_bboxes(blobs,dx=-10,dy=30,fx=1,fy=1)

		# Show mask in im
		im = im + np.expand_dims(mask,axis=-1)*100

		for bbox in bboxes:
			im = cv2.rectangle(im,pt1=(bbox[0],bbox[1]),pt2=(bbox[2],bbox[3]),color=(0,0,255),
			thickness=3)

		fig1,ax1 = plt.subplots()
		ax1.imshow(im[...,::-1])
		#fig2,ax2 = plt.subplots()
		#ax2.imshow(mask)
		plt.show()
		#plt.pause(0.2)
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
			plt.get_current_fig_manager().window.showMaximized()
			plt.show()
			plt.close(fig)