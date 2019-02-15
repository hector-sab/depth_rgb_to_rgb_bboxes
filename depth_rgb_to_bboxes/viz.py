import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_dim(path):
    """
    Loads depth image
    """
    im = cv2.imread(path,-1)
    im = im.astype(np.float32)

    mask = im>=2047

    # Preprocessing to remove noise
    im = cv2.inpaint(im,mask.astype(np.uint8),3,cv2.INPAINT_TELEA)
    #im = im[1:-1,1:-1]

    #im[im>=2047] = 0

    # Convert from disparity to Depth
    # Parameters for camera 1
    param1 = 3.12
    param2 = -0.002868
    dim = 1000/(param2*im + param1)
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

    ##### Tries to fix projection 
    if True:
        # Camera 1 parameters
        fl = [583.87,582.29] # Focal Length [Horizontal,Verticla]
        pp = [228.75,329.44] # Principal Point. Center of the camera. [Horizontal,Verticla]
        x = z*(x-pp[1])/fl[1]
        y = z*(y-pp[0])/fl[0]
        #x = x.astype(np.int32)
        #y = y.astype(np.int32)
    #####

    # Scale
    z = scale*z

    # For displaying it correctly
    #argmax = np.argmax(z)
    #z = argmax - z

    # Remove first row col of pixels to remove noise
    x = x[1:-1,1:-1]
    y = y[1:-1,1:-1]
    z = z[1:-1,1:-1]
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    
    if True:
        surf = ax.plot_surface(x,y,z,cmap='gray')
    else:
        ax.scatter(x,y,z)

    # Invert z axis
    plt.gca().invert_zaxis()
    # Maximaze window
    plt.get_current_fig_manager().window.showMaximized()
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
