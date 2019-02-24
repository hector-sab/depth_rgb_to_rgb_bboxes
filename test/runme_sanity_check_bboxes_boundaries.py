# Checks that all the bboxes in the label files are within the
# allowed limits

import os

if __name__=='__main__':
	main_lbs_dir = '/data/HectorSanchez/database/PeopleCounter/camara1_lbs_v2/'
	folders = sorted(os.listdir(main_lbs_dir))

	min_x_min = 0
	min_y_min = 0
	max_x_max = 639 # Width of the images
	max_y_max = 479 # Height of the images

	str_line = '{} 0.0 0 0.0 {} {} {} {} 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0'
	total_files = 0
	for i in range(len(folders)):
		folder = folders[i]
		print(folder)
		path_dir = main_lbs_dir+folder+'/'
		fnames = sorted(os.listdir(path_dir))

		total_files += len(fnames)
		for j in range(len(fnames)):
			fname = fnames[j]

			flines = []
			with open(path_dir+fname,'r') as f:
				file = f.readlines()
				for oline in file:
					if len(oline)==0:
						continue
					oline = oline.strip('\n')
					line = oline.split(' ')

					label = line[0]
					x_min = int(float(line[4]))
					y_min = int(float(line[5]))
					x_max = int(float(line[6]))
					y_max = int(float(line[7]))

					# Make sure that the boundaries are respected
					if x_min<min_x_min:
						x_min = min_x_min
					if y_min<min_y_min:
						y_min = min_y_min
					if x_max>max_x_max:
						x_max = max_x_max
					if y_max>max_y_max:
						y_max = max_y_max
					flines.append(str_line.format(label,x_min,y_min,x_max,y_max))

			with open(path_dir+fname,'w') as f:
				for i,line in enumerate(flines):
					f.write(line)
					if i<len(flines)-1:
						f.write('\n')
	print('Total # of files processed:',total_files)