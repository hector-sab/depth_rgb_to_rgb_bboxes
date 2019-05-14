import os
fdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.insert(0,fdir)

from depth_rgb_to_bboxes import utils_depth as udp

import cv2
import numpy as np
#import open3d as o3d
import matplotlib.pyplot as plt

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
	files = [x for i,x in enumerate(files) if '.png' in x and i>35]

	# For camera 1
	x_min = -1078.3400329779568
	y_min = -1442.6339578413751
	x_max = 1682.8418437836763
	y_max = 1385.4591844350853
	
	#fig1,ax1 = plt.subplots()
	#fig2,ax2 = plt.subplots()
	for file in files:
		print('--->',file)
		
		fim = file.replace('depth','color')
		fim = fim.replace('png','jpg')
		im = cv2.imread(fdir+fim)
		
		dim = udp.load_dim(fdir+file)
		x,y = udp.projection_correction(dim)
		
		# Tries to center for camera 1
		# += 1078.3400329779568
		#y += 1442.6339578413751

		mask = bg_remover.remove_background(dim)
		#plt.imshow(mask)
		#plt.show()
		#plt.pause(0.2)

		mask_bool = mask.astype(np.bool)
		
		#dim = dim[mask_bool]
		#x = x[mask_bool]
		#y = y[mask_bool]
		
		

		blobs = udp.find_blobs(mask)
		blobs = udp.filter_blobs(blobs,min=5000)
		#bboxes = udp.blobs_to_bboxes(blobs,dx=30,dy=40,fx=0.9,fy=1)
		bboxes = udp.blobs_to_bboxes(blobs,dx=0,dy=0,fx=1,fy=1)

		# Show mask in im
		#im = im + np.expand_dims(mask,axis=-1)*100
		print('---bboxes>',len(bboxes))
		mask = np.dstack((mask,mask,mask))
		mask *= 255
		print(mask.shape)

		if True:
			for bbox in bboxes:
				#im = cv2.rectangle(im,pt1=(bbox[0],bbox[1]),pt2=(bbox[2],bbox[3]),color=(0,0,255),
				#thickness=3)
				mask = cv2.rectangle(mask,pt1=(bbox[0],bbox[1]),pt2=(bbox[2],bbox[3]),color=(255,0,0),
				thickness=3)

		#fig1,ax1 = plt.subplots()
		#ax1.imshow(im[...,::-1])
		fig2,ax2 = plt.subplots()
		ax2.imshow(mask)
		#plt.show()
		plt.axis('off')
		plt.savefig('/home/hectorsab/Documents/Tesis/people_bboxes/'+file,bbox_inches='tight')
		plt.pause(0.1)
		#plt.close(fig1)
		plt.close(fig2)


		if len(y)==0:
			# No points found
			continue
		
		# Plot with open3d
		if False:
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