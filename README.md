The objective of all this project is to retrieve the bboxes op people from a set of images that have both RGB and Disparity.

Assumptions: The background is static, thus, never changes; or if it changes it does gradually and verry slowly.

## Workflow
- Calcula the mean and standar deviation of the values that compose the background in the disparity maps.
- Extract the bounding boxes from the disparity map.

## Scripts
- Calculation of Mean and STD: ```runme_calculate_mean_std_bg.py```
- Calculate bboxes and visualize results with the RGB images (not saving them): ```runme_extract_bg.py```


There's also the option to visualize the mean and std dev maps we got in step #1 with ```runme_viz_std_hist.py```