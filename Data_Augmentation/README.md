# Data Augmentation script for YOLO data set making

## How to use

- Create a 'image' directory and a 'image_aug' directory, put all the images(in .jpg) and labels(in .txt) into the 'image' directory

- You can adjust the parameters in the python script. For example, you can change the output file name, or have more combinations of augmentation by calling the augmentation functions

- The augmented images and corresponding label files will be created in the 'image_aug' directory

## TO DO

- Finish the general_affine_transformation and perspective_transformation
- Currently I'm using the labelling script by AlexeyAB, I'll try to make one myself
