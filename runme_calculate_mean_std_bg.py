# Calculates the mean and standard deviation of the background
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import utils as ut

if __name__=='__main__':
    if False:
        path = 'ims/'
        sims = ut.FindPeople()
        sims.runme(path,alpha=0.98)
        
        # Files without people
        fnames = sims.files
        fnames = ['depth'+x[5:-3]+'png' for x in fnames if x not in sims.fname_wppl]
    else:
        # Manual selection instead of automatic....
        # Folder 00000123
        path = dir_ = '/data/HectorSanchez/database/PeopleCounter/camara1/00000126/'
        fnames = ['1391699147_355912','1391699147_472388','1391699147_588756','1391699147_704799',
                  '1391699147_821060','1391699166_072044','1391699217_315239','1391699217_431247',
                  '1391699226_472556','1391699226_585844','1391699226_702384','1391699226_822246',
                  '1391699284_621990','1391699284_738154','1391699284_854357','1391699284_970441']
        fnames = ['depth'+x+'.png' for x in fnames]

    tmp_im = ut.load_dim2dpt(path+fnames[0])

    all_dims = np.zeros((tmp_im.shape[0],tmp_im.shape[1],0),dtype=np.float64)
    
    print('Calculating Mean and Std Dev....')
    pbar = tqdm(range(len(fnames)))
    for i in pbar:
        fname = fnames[i]
        #print(fname)
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

    np.save('mean_bg_00000126.npy',bk_mean)
    np.save('std_bg_00000126.npy',bk_std) 
