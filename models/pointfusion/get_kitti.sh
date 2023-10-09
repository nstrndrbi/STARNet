#!/bin/bash

# Get kitti 3d object detection dataset
mkdir kitti_3d
cd kitti_3d
mkdir src
cd srcS
echo "=== Dowloading Images"
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
echo "=== Dowloading Point Clouds"
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip
echo "=== Dowloading Calibration files"
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip
echo "=== Dowloading Labels"
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip 
echo "=== Done"