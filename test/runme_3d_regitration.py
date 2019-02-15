"""
NOT WORKING. In this script the only thing that is performed
is to add a background pointcloud to another.
"""
import os
fdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.insert(0,fdir)

from depth_rgb_to_bboxes import utils_depth as viz

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


if __name__=='__main__':
	# List all files on the directory
	dir_ = '../ims/'
	fpaths = sorted(os.listdir(dir_))

	FLAG_FIRST_TIME = False
	for path in fpaths:
		if '.png' in path:
			print(path)
			if not FLAG_FIRST_TIME:
				z = viz.load_dim(dir_+path)
				x,y = viz.projection_correction(z)

				x = x[1:-1,1:-1]
				y = y[1:-1,1:-1]
				z = z[1:-1,1:-1]

				background = np.zeros((np.size(x),3))
				background[:,0] = np.reshape(x,-1)
				background[:,1] = np.reshape(y,-1)
				background[:,2] = np.reshape(z,-1)
				continue

			# Load depth image
			z = viz.load_dim(dir_+path)

			# Generates corrected projection
			x,y = viz.projection_correction(z)

			# Removes first row and col
			x = x[1:-1,1:-1]
			y = y[1:-1,1:-1]
			z = z[1:-1,1:-1]
			
			

			#print('--> ',np.sum(z))
			xyz = np.zeros((np.size(x),3))
			xyz[:,0] = np.reshape(x,-1)
			xyz[:,1] = np.reshape(y,-1)
			xyz[:,2] = np.reshape(z,-1)
			
			#xyz = np.vstack((xyz,background))
			#print(xyz.shape)
			
			#http://www.open3d.org/docs/tutorial/Basic/working_with_numpy.html
			pcd = o3d.PointCloud()
			pcd.points = o3d.Vector3dVector(xyz)

			#o3d.draw_geometries([pcd])
			#x,y = np.meshgrid(np.arange(z.shape[1]),np.arange(z.shape[0]))
			plt.imshow(z)
			plt.show()
