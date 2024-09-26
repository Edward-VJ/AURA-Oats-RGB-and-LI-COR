# Top-leaf-segmentation-and-phenotyping-with-RGB-and-LI-Cor-data-for-Oat-plants
Code for the paper "Correlating top leaf RGB analysis with LI-COR sensor data to detect water stress in Oats within a high throughput phenotyping system"

This code is open source, and can be used a platform to replicate our results on similar data, though personalization for specific datasets will be required.
The workflow for our system is as follows:

1: train.ipynb: train YOLO model on our data
2: batch_dept_creation.py: - create depth map predictions using Depth Anything
3: threshold depth as binary mask
4: OAT_SAM.ipynb: segment oats into seperate parts, apply YOLO labels to the segmentations, extract top leaf
5: Extract RGB Metrics.ipynb: retrieve rgb indices, and calculate stats for them
6: Correlation Calculations.ipynb calculate pearson correlation, write results as csvs and scatterplots.

If you have any questions feel free to contact the maintainer over email.
