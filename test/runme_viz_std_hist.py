import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
        #bg_mean = np.load('../mean_bg.npy')
        #bg_std = np.load('../std_bg.npy')
        bg_mean = np.load('../mean_std/camera1/bg_camera1_mean.npy')
        bg_std = np.load('../mean_std/camera1/bg_camera1_std.npy')
	
        plt.imshow(bg_mean)
        plt.show()

        ### 3D mean
        from mpl_toolkits.mplot3d import Axes3D
        X,Y = np.meshgrid(np.arange(bg_mean.shape[1]),
                np.arange(bg_mean.shape[0]))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X,Y,bg_mean,cmap='ocean')
        plt.show()
        ###

        plt.imshow(bg_std)
        plt.show()

        ### 3D std
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X,Y,bg_std,cmap='ocean')
        plt.show()
        ###

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
