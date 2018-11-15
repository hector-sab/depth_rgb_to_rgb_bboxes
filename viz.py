import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_dim(path):
    im = cv2.imread(path,-1)
    im = im.astype(np.float32)
    im[im>=2047] = 0

    # Convert from disparity to Depth
    dim = 1000/(-0.0024*im + 3.15)
    return(dim)

def plot3d(path,scale=1):
    #z = cv2.imread(path)
    #if len(z.shape)>2:
    #    z = z[...,1]
    
    z = load_dim(path)
    x,y = np.meshgrid(np.arange(z.shape[1]),np.arange(z.shape[0]))
    
    #print(x.shape,y.shape,z.shape)
    #x = x.reshape(-1,1)
    #y = y.reshape(-1,1)
    #z = im.reshape(-1,1)
    z = scale*z
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    
    if True:
        surf = ax.plot_surface(x,y,z,cmap='gray')
    else:
        ax.scatter(x,y,z)
    plt.show()

def plot2d_heat(dir_,fname):
    dpath = dir_+fname
    ipath = dir_+'color'+fname[5:-4]+'.jpg'
    #dim = cv2.imread(dpath)
    #dim = scale*dim
    #if len(dim.shape)>2:
    #    dim = dim[...,0]
    dim = load_dim(dpath)
    #print(dim.shape)
     
    #im = cv2.imread(ipath)
    
    plt.figure(0)
    plt.imshow(dim,cmap='hot')
    #plt.figure(1)
    #plt.imshow(im)
    plt.show()

if __name__=='__main__':
    # List all files on the directory
    dir_ = 'ims/'
    fpaths = sorted(os.listdir(dir_))
    #print('Files Found:')
    for path in fpaths:
        if '.png' in path:
            print(path)
            im = cv2.imread(dir_+path)
            
            #for i in range(3):
            #    plt.imshow(im[...,i],cmap='hot')
            #    plt.show()
            #plot2d_heat(dir_,path)
            plot3d(dir_+path)
