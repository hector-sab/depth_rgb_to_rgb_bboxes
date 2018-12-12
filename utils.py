import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import skimage as sk
from skimage.measure import compare_ssim as ssim

def load_dim2dpt(path,mask=False):
    # Load Image as uint16 data
    im = cv2.imread(path,-1)
    # Convert it to float32
    im = im.astype(np.float64)
    # Remove all the points that are not part of the
    # disparity. (Values equal or bigger than 2074)
    inv_mask = im>=2047 # Invalid mask
    im[inv_mask] = 0
    
    # To depth
    dim = 1000/(-0.0024*im +3.15)
    
    # Inpainting
    #dim = dim.astype(np.uint8)
    #inv_mask = inv_mask.astype(np.uint8)
    #print(dim.dtype,inv_mask.dtype)
    #dim = cv2.inpaint(dim,inv_mask,7,cv2.INPAINT_TELEA)
    #dim = dim.astype(np.float64)
    if not mask:
        return(dim)
    else:
        return(dim,inv_mask)

class FindPeople:
    def __init__(self):
        self.fname_wppl = [] # Ims with people in int
        self.sims = [] # Similarities

    def runme(self,folder,alpha=0.94):
        # Main program. Finds all the images with people
        self.folder = folder
        self.files = sorted(os.listdir(self.folder))
        self.files = [x for x in self.files if '.jpg' in x]
        
        print('Detecting People on Images...')
        pbar = tqdm(range(len(self.files)))
        for i in pbar:
            im = cv2.imread(self.folder+self.files[i])
            try:
                im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print('--->',self.folder+self.files[i])
                input('Error')

            im = cv2.resize(im,None,fx=0.5,fy=0.5)

            if not i:
                im_p = np.copy(im)
                continue

            s = ssim(im,im_p)

            if s<alpha:
                self.fname_wppl.append(self.files[i])

            self.sims.append(s)
            im_p = np.copy(im)

    def chart(self):
        # Print the structural similarity index in a chart
        ind = np.arange(len(self.sims),dtype=np.int32)
        plt.plot(ind,self.sims,'o-')
        plt.show()

    def reset(self):
        self.fname_wppl = []
        self.sims = []

def find_blobs(mask):
    # Find all the blobs in a mask
    # mask (np.array): 2D array 
    blob_lbs = sk.measure.label(mask,background=0)
    return(blob_lbs)

def filter_blobs(blobs,min=None,max=None):
    # Filter the blobs by size of area
    # Args:
    #    blobs (np.array): 2D array which contains a mask of blobs 
    #        that are labeled labeled
    #    min (int): Minimum area desired in order to mantain the blob
    #    max (int): Maximum area desired in ordert to mantain the blob

    # Counts how many elements each blob has
    elems,count = np.unique(blobs,return_counts=True)
    # Determine which blobs have the minimum size
    if min is not None:
        mask = count>=min
        for i,valid in enumerate(mask):
            if i!=0 and not valid:
                # If they don't have the minimum size
                # Set them to zero
                blobs = (blobs!=i)*blobs
    # Determine which blobs have the maximum size
    if max is not None:
        mask = count<=max
        for i,valid in enumerate(mask):
            if i!=0 and not valid:
                blobs = (blobs!=i)*blobs

    return(blobs)

def lbBlob2bboxes(blobs):
    # Converts from labeled blob mask to bboxes
    # Returns an array of bboxes found
    # Format: shape -> [None,2]
    #         structure -> [x_left,y_top,x_right,y_bottom]
    # Args:
    #    blobs (np.array): 2D array containing the labeled blobs
    
    bboxes = []
    elems = np.unique(blobs)
    for elem in elems:
        if elem!=0:
            y_ind,x_ind = np.where(blobs==elem)
 
            dy = 20
            dx = -20
            y_top = np.min(y_ind) + dy
            y_bottom = np.max(y_ind) + dy
            x_left = np.min(x_ind) + dx
            x_right = np.max(x_ind) + dx
            bboxes.append([x_left,y_top,x_right,y_bottom])
    
    if len(bboxes)>0:
        bboxes = np.array(bboxes)
    else:
        bboxes = np.zeros((0,4))

    return(bboxes)

def select_bboxes_inside_area(bboxes,y_top=100):
    bbcent = (bboxes[:,1]+bboxes[:,3])/2
    bbmask = bbcent>y_top
    boxes = bboxes[bbmask,:]
    return(boxes)

class BboxGenerator:
    # Generate bounding boxes from depth images
    # It uses their RGB counterparts to find which
    #  images have people by using SSIM.
    def __init__(self):
        self.bg_mean = None
        self.bg_std = None
        #self.vmax = None
        self.up_mask = None
        self.lo_mask = None

        self.sims = FindPeople()

    def set_bg(self,mpath='mean_bg.npy',spath='std_bg.npy'):
        # mpath: Mean path of background
        # spath: Std dev path of background
        self.bg_mean = np.load(mpath)
        self.bg_std = np.load(spath)
        
        # Limit the std dev.
        #### TODO: Possible Improvement. Remove zeros from equation and count the stable
        ####       pixel with actual values
        vmax = np.max(self.bg_std)
        hist = (self.bg_std/vmax)*255
        ## Reduce the possible depths
        hist = hist.astype(np.uint8)
        pval = 7 # Pixel value to crop from
        mask = hist>pval
        self.bg_std[mask] = (pval/255)*vmax

        self.up_mask = self.bg_mean+self.bg_std*4
        self.lo_mask = self.bg_mean-self.bg_std*4
        
    def get_bboxes(self,path,return_fnames=False,reset=True):
        # Returns the bboxes in a list of arrays of shape [None,4]
        # Where 4 is # [x_left,y_top,x_right,y_bottom]
        if reset:
            self.sims.reset()
        self.sims.runme(path,alpha=0.98)

        # Files with people
        fnames = self.sims.fname_wppl # Images path
        dfnames = ['depth'+x[5:-3]+'png' for x in fnames] # Depths path

        all_bboxes = []
        
        print('Calculating Bounding Boxs...')
        pbar = tqdm(range(len(dfnames)))
        ind2del = [] # fnames to be deleted
        for i in pbar:
            dfname = dfnames[i]
            try:
                dim = load_dim2dpt(path+dfname)
            except Exception as e:
                # In case this image is not procesed, delet it
                print('---->Error with:',path+dfname)
                ind2del.append(i)
                continue

            mask = np.logical_and(dim<=self.up_mask,dim>=self.lo_mask)
            mask = np.logical_not(mask)
        
            # Clean noise....
            kernel = np.ones((5,5),np.uint8)
            mask = cv2.erode(mask.astype(np.uint8),kernel,iterations=2)
            mask = cv2.dilate(mask,kernel,iterations=2)

            ### S: TMP
            #plt.imshow(mask)
            #plt.show()
            ### E: TMP

            # Blob detection
            blob_lbs = find_blobs(mask)
            blob_lbs = filter_blobs(blob_lbs,min=3200)

            # Get bboxes
            bboxes = lbBlob2bboxes(blob_lbs)

            # Make sure the bbox that are displayed start located after
            # the noise sourse
            bboxes = select_bboxes_inside_area(bboxes)
            all_bboxes.append(bboxes)

        if len(ind2del)>0:
            for i in ind2del[::-1]:
                del fnames[i]

        if return_fnames:
            return(all_bboxes,fnames)
        else:
            return(all_bboxes)