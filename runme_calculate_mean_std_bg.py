# Calculates the mean and standard deviation of the background
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import utils as ut

if __name__=='__main__':
    path = 'ims/'
    sims = ut.FindPeople()
    sims.runme(path,alpha=0.98)
    
    # Files without people
    fnames = sims.files
    fnames = ['depth'+x[5:-3]+'png' for x in fnames if x not in sims.fname_wppl]

    tmp_im = ut.load_dim2dpt(path+fnames[0])

    all_dims = np.zeros((tmp_im.shape[0],tmp_im.shape[1],0),dtype=np.float64)
    
    print('Calculating Mean and Std Dev....')
    pbar = tqdm(range(len(fnames)))
    for i in pbar:
        fname = fnames[i]
        #print('{}/{} - {}'.format(i,len(fnames),fname))
        dim = ut.load_dim2dpt(path+fname)
        dim = np.expand_dims(dim,axis=-1)
        #print('---->',dim.shape,all_dims.shape)
        all_dims = np.concatenate((all_dims,dim),axis=-1)
        #print('--->',all_dims.shape)
        #plt.imshow(im)#[...,::-1])
        #plt.show()

    bk_mean = np.mean(all_dims,axis=-1) # Mean Background
    bk_std = np.std(all_dims,axis=-1) # Std dev Background

    np.save('mean_bg.npy',bk_mean)
    np.save('std_bg.npy',bk_std) 
