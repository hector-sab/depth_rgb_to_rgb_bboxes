"""
Maps depth image space to rgb image space using camera parameters.
Source:
http://burrus.name/index.php/Research/KinectCalibration

Useful viz:
https://jasonchu1313.github.io/2017/10/01/kinect-calibration/
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_im(path):
    if 'jpg' in path:
        im = cv2.imread(path)
        im = im[...,::-1]
    elif 'png' in path:
        im = cv2.imread(path,-1)
        im = im.astype(np.float32)
        # Mask to identify noise
        mask = im>=2047
        # Preprocessing to remove noise
        im = cv2.inpaint(im,mask.astype(np.uint8),3,cv2.INPAINT_TELEA)
        # Parameters for camera 1
        param1 = 3.12
        param2 = -0.002868
        # Convert from disparity to Depth
        im = 1000/(param2*im + param1)
    return(im)

main_dir = '/home/hectorsab/data/databases/PeopleCounter/camara1/00000000/'
im_rgb_name = 'color1391619277_448535.jpg'
im_dpt_name = im_rgb_name.replace('color','depth').replace('.jpg','.png')

im_rgb_path = os.path.join(main_dir,im_rgb_name)
im_dpt_path = os.path.join(main_dir,im_dpt_name)

im_rgb = load_im(im_rgb_path)
im_dpt = load_im(im_dpt_path)



#####
cxd,cyd =  329.44,228.75
fxd,fyd = 582.29,583.87
Rd = np.array([[0.99966,-0.00676,0.02532],
			   [0.00729,0.99975,-0.02100],
			   [-0.02517,0.02118,0.99946]])
Td = np.array([-0.01953,-0.00103,-0.00416])
Td = Td.reshape(-1,1)

xd,yd = np.meshgrid(np.arange(0,640),np.arange(0,480))

x = (xd - cxd)*im_dpt/fxd
y = (yd - cyd)*im_dpt/fyd

x = x.reshape(1,-1)
y = y.reshape(1,-1)
z = im_dpt.reshape(1,-1)

P = np.vstack((x,y,z))

P_ = np.dot(Rd,P) + Td

#>>> Calculate metric coord im
xmet = P_[0,:].astype(np.int32)
xmet += np.abs(np.amin(xmet))
ymet = P_[1,:].astype(np.int32)
ymet += np.abs(np.amin(ymet))
zmet = P_[2,:]
im_dpt_met = np.zeros((np.amax(ymet)+1,np.amax(xmet)+1))

im_dpt_met[ymet,xmet] = zmet

#>>> Calculate RGB Coord of Depth

cxrgb,cyrgb = 306.61,270.72
fxrgb,fyrgb = 513.79,515.41

xrgb = (P_[0,:]*fxrgb/P_[2,:]) + cxrgb
yrgb = (P_[1,:]*fyrgb/P_[2,:]) + cyrgb

xrgb = xrgb.astype(np.int32)
yrgb = yrgb.astype(np.int32)
#>>> Reconstruct image
im_dpt_adj = np.zeros((int(np.amax(yrgb))+1,int(np.amax(xrgb))+1))

im_dpt_adj[yrgb,xrgb] = P_[2,:]

#>>> Store the reprojected image in a 480x640 container
im_dpt_adj2 = np.zeros((480,640))
ymax = im_dpt_adj.shape[0]
xmax = im_dpt_adj.shape[1]

if ymax>480: ymax = 480

if xmax>640: xmax = 640

im_dpt_adj2[:ymax,:xmax] = im_dpt_adj[:ymax,:xmax]

#plt.imshow(im_dpt_adj2)
#plt.show()
#####


im_dpt = im_dpt_adj2*255/np.amax(im_dpt_adj2)
im_dpt = np.stack((im_dpt,im_dpt,im_dpt),axis=-1)

print(np.amax(im_dpt_adj2),im_dpt.shape)
print(np.amax(im_rgb))

im_dpt2 = Image.fromarray(im_dpt.astype('uint8'))
im_rgb2 = Image.fromarray(im_rgb.astype('uint8'))

im_mask = Image.blend(im_rgb2,im_dpt2,0.6
	8)
plt.imshow(im_mask)
plt.show()
