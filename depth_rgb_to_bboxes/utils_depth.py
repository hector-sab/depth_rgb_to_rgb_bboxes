import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as sk
from skimage.measure import compare_ssim as ssim

def load_dim(path):
	"""
	Loads depth image

	Args:
	  path (str): Location of the disparity map file.
	
	Returns;
	  dim (np.ndarray): 3D scen contained in the disparity map file.

	TODO: Find the meaning of param1 and param2. Depth parameters. 
	  Make them available from function call.
	"""
	im = cv2.imread(path,-1)
	im = im.astype(np.float32)

	# Mask to identify noise
	mask = im>=2047

	# Preprocessing to remove noise
	im = cv2.inpaint(im,mask.astype(np.uint8),3,cv2.INPAINT_TELEA)

	# Parameters for camera 1
	param1 = 3.12
	param2 = -0.002868

	# Convert from disparity to Depth
	dim = 1000/(param2*im + param1)
	return(dim)

def projection_correction(dim,fl=[583.87,582.29],pp=[228.75,329.44]):
	"""
	Corrects the 3D proyection

	Args:
	  dim (np.ndarray): Depth image
	  fl (tuple|list|np.ndarray): Focal Lenght of the camera 
		  used to capture dim. [Horizontal,Vertical]
	  pp (tuple|list|np.ndarray): Principal Point of the camera 
		  used to capture dim. Center of the camera in the image plane. 
		  [Horizontal,Vertical]

	Returns:
	  x|y (np.ndarray): Depth image x,y indexes of dim with corrected projection

	Note: All default parameters are for camera 1
	"""
	# Mesh Grid for coord
	x,y = np.meshgrid(np.arange(dim.shape[1]),np.arange(dim.shape[0]))

	x = dim*(x-pp[1])/fl[1]
	y = dim*(y-pp[0])/fl[0]

	return(x,y)

def plot3d(path,scale=1):
	"""
	Plots a disparity map into a 3D view.

	Args:
	  path (str): Location of the disparity map file.
	  scale (int|float): Scales the size of the 3D object in Z.

	Returns:
	  fig (): Reference to the fig object.
	"""
	#z = cv2.imread(path)
	#if len(z.shape)>2:
	#    z = z[...,1]
	
	z = load_dim(path)
	x,y = projection_correction(z)

	# Scale
	z = scale*z

	# For displaying it correctly
	#argmax = np.argmax(z)
	#z = argmax - z

	# Tries to center everyting at zero
	print('min x:',np.amin(x))
	print('min y:',np.amin(y))
	print('max x:',np.amax(x))
	print('max y:',np.amax(y))
	# Parameters for camera 1
	x += 1078.3400329779568
	y += 1442.6339578413751

	# Remove first row col of pixels to remove noise
	x = x[1:-1,1:-1]
	y = y[1:-1,1:-1]
	z = z[1:-1,1:-1]
	
	# Create figure 
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.set_aspect('equal')
	
	if True:
		surf = ax.plot_surface(x,y,z,cmap='gray')
	else:
		ax.scatter(x,y,z)

	# Invert z axis
	plt.gca().invert_zaxis()
	# Maximaze window
	plt.get_current_fig_manager().window.showMaximized()
	plt.show()
	plt.close(fig)
	return(fig)

def plot2d_heat(path):
	"""
	Plots a disparity map as a heat map
	
	Args:
	  path (str): Location of the disparity map file.

	Returns:
	  fig (): Reference to the fig object.
	"""
	dim = load_dim(path)
	
	fig,ax = plt.subplots()
	ax.imshow(dim,cmap='hot')
	plt.show()
	plt.close(fig)
	return(fig)

class FindBackgroundDepth:
	def __init__(self):
		self.background_fnames = []
		self.sims = []

	def calculate_sim(self,directory):
		"""
		Calculate the structural similarities between images.
		"""
		self.sims = []
		files = sorted(os.listdir(directory))
		self.files = [x for x in files if '.png' in x]

		dim_prev = None
		pbar = tqdm(range(len(self.files)))
		for i in pbar:
			file = self.files[i]
			fpath = os.path.join(directory,file)
			
			dim = load_dim(fpath)
			dim = cv2.resize(dim,None,fx=0.5,fy=0.5)

			if dim_prev is None:
				dim_prev = np.copy(dim)
				continue
			
			sim = ssim(dim,dim_prev)
			self.sims.append(sim)
			dim_prev = np.copy(dim)

	def determine_background(self,alpha=0.90):
		"""
		Determine what files are background based on their 
		similarities.
		"""
		# Resets the list of background files
		self.background_fnames = []

		for i,sim in enumerate(self.sims):
			if sim>=alpha:
				if self.files[i] not in self.background_fnames:
					self.background_fnames.append(self.files[i])
				if self.files[i+1] not in self.background_fnames:
					self.background_fnames.append(self.files[i+1])

	def ssim_chart(self):
		"""
		Display the SSIMs calculated in a chart
		"""
		ind = np.arange(len(self.sims),dtype=np.int32)
		plt.plot(ind,self.sims,'o-')
		plt.show()

	def reset_all(self):
		self.background_fnames = []
		self.sims = []

class BackgroundRemover:
	def __init__(self):
		self.bg_mean = None
		self.bg_std = None

		self.mask_up = None
		self.mask_lo = None

	def set_background(self,mpath,spath):
		"""
		Sets the Mean and Std Dev of the background
		Args:
		  mpath (str): Mean path
		  spath (str): Std dev path
		"""
		self.bg_mean = np.load(mpath)
		self.bg_std = np.load(spath)

		self.mask_up = self.bg_mean + self.bg_std*20
		self.mask_lo = self.bg_mean - self.bg_std*20

	def remove_background(self,dim):
		"""
		Returns a mask containing the 3D space with no background
		"""
		mask = np.logical_and(dim<=self.mask_up,dim>=self.mask_lo)
		mask = np.logical_not(mask)

		kernel = np.ones((5,5),np.uint8)
		mask = cv2.erode(mask.astype(np.uint8),kernel,iterations=2)
		mask = cv2.dilate(mask,kernel,iterations=2)
		return(mask)

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

def blobs_to_bboxes(blobs,xyz=None,dx=0,dy=0,fx=1,fy=1):
	# Converts from labeled blob mask to bboxes
	# Returns an array of bboxes found
	# Format: shape -> [None,2]
	#         structure -> [x_left,y_top,x_right,y_bottom]
	# Args:
	#    blobs (np.array): 2D array containing the labeled blobs
	
	bboxes = []
	elems = np.unique(blobs) # How may blobs exist
	for elem in elems:
		# Zero indicates the background
		if elem!=0:
			if xyz is not None:
				# Obtains the bboxes with the corrected projection 
				# of the points
				mask = blobs==elem
				mask = mask.astype(np.bool)

				# Obtain X and Y coord for the blob
				x = xyz[:,0][mask.reshape(-1)]
				y = xyz[:,1][mask.reshape(-1)]

				# obtain min and max
				x_min = np.amin(x)
				x_max = np.amax(x)
				y_min = np.amin(y)
				y_max = np.amax(y)

				bboxes.append([x_min,y_min,x_max,y_max])
			else:
				y_ind,x_ind = np.where(blobs==elem)
				#dy = 40
				#dx = 30
				#fx = 0.9
				#fy = 1
				y_min = np.min(y_ind) + dy
				y_max = (np.max(y_ind) + dy)*fy
				x_min = np.min(x_ind) + dx
				x_max = (np.max(x_ind) + dx)*fx
				bboxes.append([x_min,y_min,x_max,y_max])
	
	if len(bboxes)>0:
		bboxes = np.array(bboxes)
	else:
		bboxes = np.zeros((0,4))

	bboxes = bboxes.astype(np.int32)
	return(bboxes)

class BBoxGenerator:
	def __init__(self):
		self.bg_remover = BackgroundRemover()
		self.translation_m = np.array([0,0,0])
		self.rotation_m = np.array([[1,0,0],
		                            [0,1,0],
		                            [0,0,1]])
		self.fl = None
		self.pp = None

	def set_background(self,mpath,spath):
		self.bg_remover.set_background(mpath,spath)

	def set_translation_matrix(self,matrix):
		self.translation_m = matrix

	def set_rotation_matrix(self,matrix):
		self.rotation_m = matrix

	def set_focal_lenght(self,fl):
		self.fl = fl

	def set_principal_point(self,pp):
		self.pp = pp

	def determine_bboxes(self,dim,dx=0,dy=0,fx=1,fy=1):
		"""
		returns bboxes
		"""
		if self.fl is not None and self.pp is not None:
			x,y = projection_correction(dim,self.fl,self.pp)

			# Translate
			x += self.translation_m[0]
			y += self.translation_m[1]
			dim += self.translation_m[2]

			xyz = np.zeros((np.size(x),3))
			self.xyz = xyz
			xyz[:,0] = np.reshape(x,-1)
			xyz[:,1] = np.reshape(y,-1)
			xyz[:,2] = np.reshape(dim,-1)

			# Rotation
			xyz = np.dot(xyz,self.rotation_m)

			# Reproyect
			us = np.zeros((307200))
			vs = np.zeros((307200))

			## Calculate new X,Y position in the image
			us = (xyz[:,0]/xyz[:,2])*self.fl[0] + self.pp[0]
			vs = (xyz[:,1]/xyz[:,2])*self.fl[1] + self.pp[1]
			
			us = us.astype(np.int32)
			vs = vs.astype(np.int32)

			x_min = np.amin(us)
			x_max = np.amax(us)
			y_min = np.amin(vs)
			y_max = np.amax(vs)

			#dim = np.zeros((y_max-y_min,x_max-x_min))
			dim = np.zeros((480,640))
			us -= x_min
			vs -= y_min

			us_mask = np.logical_and(us>=0,us<640)
			vs_mask = np.logical_and(vs>=0,vs<480)

			mask = np.logical_and(us_mask,vs_mask)

			us = us[mask]
			vs = vs[mask]
			zs = xyz[:,2][mask.reshape(-1)]
			dim[vs,us] = zs

		# Obtain mask with no background
		mask = self.bg_remover.remove_background(dim)
		

		###
		# Filter objects in the floor
		h_limit = 120
		above_mask = np.copy(mask)
		above_mask[h_limit:] = dim[h_limit:]<2700
		mask = np.logical_and(mask,above_mask)
		mask = mask.astype(np.uint8)
		###
		#print(mask.dtype)

		self.mask = mask

		# Find Blobs
		blobs = find_blobs(mask)
		blobs = filter_blobs(blobs,min=5000)

		# Get bboxes
		bboxes = blobs_to_bboxes(blobs,dx=dx,dy=dy,fx=fx,fy=fy)		

		return(bboxes)

if __name__=='__main__':
	# List all files on the directory
	dir_ = 'ims/'
	fpaths = sorted(os.listdir(dir_))
	#print('Files Found:')
	for path in fpaths:
		if '.png' in path:
			print(path)
			im = cv2.imread(dir_+path)
			
			#for i in range(3):
			#    plt.imshow(im[...,i],cmap='hot')
			#    plt.show()
			#plot2d_heat(dir_,path)
			plot3d(dir_+path)
