import os
from tqdm import tqdm

import utils as ut

def save_bbox2file(path,bboxes):
    # [x_left,y_top,x_right,y_bottom]
    line = 'person 0 0 0 {} {} {} {} 0 0 0 0 0 0 0 0'
    if len(bboxes)>0:
        with open(path,'w') as f:
            for i,bbox in enumerate(bboxes):
                f.write(line.format(bbox[0],bbox[1],bbox[2],bbox[3]))
                if i<len(bboxes)-1:
                    f.write('\n')

def create_lbs(fnames,bboxes):
    print('Creating labels...')
    pbar = tqdm(range(len(fnames)))
    for i in pbar:
        fname = fnames[i]
        im_bboxes = bboxes[i]
        
        save_bbox2file(out_dir+fname[:-3]+'txt',im_bboxes)

if __name__=='__main__':
    dir_ = '/data/HectorSanchez/database/PeopleCounter/camara1/00000006/'
    out_dir = '/data/HectorSanchez/database/PeopleCounter/camara1_lbs/00000006/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    generator = ut.BboxGenerator()
    generator.set_bg(mpath='mean_bg.npy',spath='std_bg.npy')
    bboxes,fnames = generator.get_bboxes(path=dir_,return_fnames=True)

    create_lbs(fnames,bboxes)

    #print('Creating labels...')
    #pbar = tqdm(range(len(fnames)))
    #for i in pbar:
    #    fname = fnames[i]
    #    im_bboxes = bboxes[i]
    #    
    #    save_bbox2file(out_dir+fname[:-3]+'txt',im_bboxes)