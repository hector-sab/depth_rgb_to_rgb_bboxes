import os
fdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.insert(0,fdir)

from depth_rgb_to_bboxes import utils_depth as viz

if __name__=='__main__':
	import open3d as o3d
	import numpy as np

	# List all files on the directory
	dir_ = '/data/HectorSanchez/database/PeopleCounter/camara3/00000000/'
	fpaths = sorted(os.listdir(dir_))

	FLAG_FIRST_TIME = False
	for path in fpaths:
		if '.png' in path:
			#print(path)
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
				FLAG_FIRST_TIME = True
				continue
				
			if True:
				# Uses matplotlib 
				#viz.plot2d_heat(dir_+path)
				viz.plot3d(dir_+path)
			else:
				# Uses Open3d
				z = viz.load_dim(dir_+path)
				x,y = viz.projection_correction(z)

				x = x[1:-1,1:-1]
				y = y[1:-1,1:-1]
				z = z[1:-1,1:-1]
				
				###
				#min_x = np.argmin(background[:,0])
				#x += min_x

				#min_y = np.argmin(background[:,1])
				#y += min_y

				#min_x = np.argmin(background[:,0])
				#z += min_x
				###
				###
				#x = x + background[:,0]
				#x = x + background[:,0]
				#x = x + background[:,0]
				###

				print('--> ',np.sum(z))
				xyz = np.zeros((np.size(x),3))
				xyz[:,0] = np.reshape(x,-1)
				xyz[:,1] = np.reshape(y,-1)
				xyz[:,2] = np.reshape(z,-1)
				
				#http://www.open3d.org/docs/tutorial/Basic/working_with_numpy.html
				pcd = o3d.PointCloud()
				pcd.points = o3d.Vector3dVector(xyz)

				o3d.draw_geometries([pcd])