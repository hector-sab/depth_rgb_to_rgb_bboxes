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
	dir_ = '../ims/'
	fpaths = sorted(os.listdir(dir_))

	for path in fpaths:
		if '.png' in path:
			z = udp.load_dim(dir_+path)
			x,y = udp.projection_correction(z)

			# Translate
			x += trans[0]
			y += trans[1]
			z += trans[2]

			xyz = np.zeros((np.size(x),3))
			xyz[:,0] = np.reshape(x,-1)
			xyz[:,1] = np.reshape(y,-1)
			xyz[:,2] = np.reshape(z,-1)
			
			# Rotation
			xyz = np.dot(xyz,rot)


			#pcd = o3d.PointCloud()
			#pcd.points = o3d.Vector3dVector(xyz)
			#o3d.draw_geometries([pcd])

			us = np.zeros((307200))#[]
			vs = np.zeros((307200))#[]

			us = (xyz[:,0]/xyz[:,2])*fl[0] + pp[0]
			vs = (xyz[:,1]/xyz[:,2])*fl[1] + pp[1]
			
			us = us.astype(np.int32)
			vs = vs.astype(np.int32)

			x_min = np.amin(us)
			x_max = np.amax(us)
			y_min = np.amin(vs)
			y_max = np.amax(vs)
			
			#new_im = np.zeros((y_max-y_min,x_max-x_min))
			new_im = np.zeros((480,640))

			us -= x_min
			vs -= y_min

			us_mask = np.logical_and(us>=0,us<640)
			vs_mask = np.logical_and(vs>=0,vs<480)

			mask = np.logical_and(us_mask,vs_mask)

			us = us[mask]
			vs = vs[mask]
			zs = xyz[:,2][mask.reshape(-1)]
			new_im[vs,us] = zs

			if False:
				for i in range(len(us)):
					#print('UVs ims',i)
					u = us[i] - x_min
					v = vs[i] - y_min
					#print(v,u,xyz[i,2],end=' ')
					if v>=480 or v<0:
						#print()
						continue
					if  u>=640 or u<0:
						#print()
						continue
					#print('***')
					new_im[v,u] = xyz[i,2]


			fig1,ax1 = plt.subplots()
			ax1.imshow(new_im)

			fim = path.replace('depth','color')
			fim = fim.replace('png','jpg')
			im = cv2.imread(dir_+fim)
			fig2,ax2 = plt.subplots()
			ax2.imshow(im)

			plt.show()

