# This script oly shows the bboxes that would be
# saved. Does not sabe anything at all
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import utils as ut

if __name__=='__main__':
    dir_ = '/data/HectorSanchez/database/PeopleCounter/camara1/'
    path = dir_+'00000000/'
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