import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
        bg_mean = np.load('mean_bg.npy')
        bg_std = np.load('std_bg.npy')
	
        plt.imshow(bg_mean)
        plt.show()

        plt.imshow(bg_std)
        plt.show()

        # For histogram
        vmax = np.max(bg_std)
        hist = (bg_std/vmax)*255
        hist = hist.astype(np.uint8)

        elems,count = np.unique(hist,return_counts=True)
        print('vmax: {} | # elems: {}'.format(vmax,len(elems)))

        ####
        tmp_elems = elems.reshape(-1,1)
        tmp_count = count.reshape(-1,1)
        tmp = np.hstack((tmp_elems,tmp_count))
        print(tmp)
        print(np.sum(count[5:]))
        ####

        plt.bar(elems,count)
        plt.show()
