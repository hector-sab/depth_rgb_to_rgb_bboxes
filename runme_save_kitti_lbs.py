import os
from tqdm import tqdm

import utils as ut

if __name__=='__main__':
    dir_ = 'ims/'
    out_dir = 'lbs/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    generator = ut.BboxGenerator()
    generator.set_bg(mpath='mean_bg.npy',spath='std_bg.npy')
    bboxes,fnames = generator.get_bboxes(path=dir_,return_fnames=True)

    # [x_left,y_top,x_right,y_bottom]
    line = 'person 0 0 0 {} {} {} {} 0 0 0 0 0 0 0 0'

    print('Creating labels...')
    pbar = tqdm(range(len(fnames)))
    for i in pbar:
        fname = fnames[i]
        im_bboxes = bboxes[i]
        
        if len(im_bboxes)>0:
            with open(out_dir+fname[:-3]+'txt','w') as f:
                for j,bbox in enumerate(im_bboxes):
                    f.write(line.format(bbox[0],bbox[1],bbox[2],bbox[3]))
                    if j<len(im_bboxes)-1:
                        f.write('\n')