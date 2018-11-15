The objective of all this project is to retrieve the bboxes of people from a set of RGB and Disparity images.

Assumptions: The background is static, thus, never changes; or if it changes it does gradually and verry slowly.

## Workflow
- Calcula the mean and standar deviation of the values that compose the background in the disparity maps.
- Extract the bounding boxes from the disparity map.

## Scripts
- Calculation of Mean and STD: ```runme_calculate_mean_std_bg.py```
- Calculate bboxes and visualize results with the RGB images (not saving them): ```runme_extract_bg.py```
- Calculate and save the bboxes in a folder with KITTI format: ```runme_save_kitti_lbs.py```


There's also the option to visualize the mean and std dev maps we got in step #1 with ```runme_viz_std_hist.py```

## Data

Right now, the code works with having a ```ims/``` folder which contains ```color#######_####.jpg``` and ```depth#######_####.png```.
