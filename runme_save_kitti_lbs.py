import os
from tqdm import tqdm

import utils as ut

def save_bbox2file(path,bboxes):
    # [x_left,y_top,x_right,y_bottom]
    line = 'person 0.0 0 0.0 {} {} {} {} 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0'
    if len(bboxes)>0:
        with open(path,'w') as f:
            for i,bbox in enumerate(bboxes):
                f.write(line.format(bbox[0],bbox[1],bbox[2],bbox[3]))
                if i<len(bboxes)-1:
                    f.write('\n')

def create_lbs(fnames,bboxes,out_dir):
    print('Creating labels...')
    pbar = tqdm(range(len(fnames)))
    for i in pbar:
        fname = fnames[i]
        im_bboxes = bboxes[i]
        
        save_bbox2file(out_dir+fname[:-3]+'txt',im_bboxes)

if __name__=='__main__':
    # For single folder
    if False:
        dir_ = '/data/HectorSanchez/database/PeopleCounter/camara1/00000024/'
        out_dir = '/data/HectorSanchez/database/PeopleCounter/camara1_lbs/00000024/'

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        generator = ut.BboxGenerator()
        generator.set_bg(mpath='mean_bg.npy',spath='std_bg.npy')
        bboxes,fnames = generator.get_bboxes(path=dir_,return_fnames=True)

        create_lbs(fnames,bboxes,out_dir)
    else:
        # For multiple folders
        main_dir = '/data/HectorSanchez/database/PeopleCounter/camara1/'
        folders = sorted(os.listdir(main_dir))

        out_main_dir = '/data/HectorSanchez/database/PeopleCounter/camara1_lbs/'

        generator = ut.BboxGenerator()
        generator.set_bg(mpath='mean_bg.npy',spath='std_bg.npy')

        for i in range(82,len(folders)):
            folder = folders[i]
            print(folder)
            path_dir = main_dir+folder+'/'
            path_out_dir = out_main_dir+folder+'/'

            if not os.path.exists(path_out_dir):
                os.makedirs(path_out_dir)

            bboxes,fnames = generator.get_bboxes(path=path_dir,return_fnames=True)
            create_lbs(fnames,bboxes,path_out_dir)

