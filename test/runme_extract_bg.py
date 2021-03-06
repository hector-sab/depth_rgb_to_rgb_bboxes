# Calculates the mean and standard deviation of the background
import os
fdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.insert(0,fdir)

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from depth_rgb_to_bboxes import utils as ut

if __name__=='__main__':
    dir_ = './'
    path = dir_+'ims/'
    generator = ut.BboxGenerator()
    generator.set_bg(mpath='mean_bg.npy',spath='std_bg.npy')
    bboxes,fnames = generator.get_bboxes(path=path,return_fnames=True)

    for i in range(len(fnames)):
        fname = fnames[i]
        im = cv2.imread(path+fname)
        
        # Draw bboxes
        ## Select the minimun y_top from where to start
        ## drawing bboxes
        boxes = bboxes[i]
        boxes = ut.select_bboxes_inside_area(boxes)
        #bbcent = (boxes[:,1]+boxes[:,3])/2
        #bbmask = bbcent>100
        #boxes = boxes[bbmask,:]
        for bbox in boxes:
            cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),4)

        plt.imshow(im[...,::-1])
        plt.show()
