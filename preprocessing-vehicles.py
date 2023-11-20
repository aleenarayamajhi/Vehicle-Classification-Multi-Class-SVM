# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 01:26:16 2023

@author: ar52624
"""

import os
import cv2

# Define the list of class labels
classes = ['car', 'airplane', 'ship', 'bike']

# Specify the directory path where images are stored for each class
base_dir = 'C:/Users/ar52624/Desktop/Fall 2023/Pattern Recognition/car-bike-airplane/my-dataset/2000/'

# Define the folder to save processed images
processed_dir = 'C:/Users/ar52624/Desktop/Fall 2023/Pattern Recognition/car-bike-airplane/my-dataset/2000/processedimages/'

# Iterate over each class
for class_name in classes:
    class_dir = os.path.join(base_dir, class_name)
    processed_class_dir = os.path.join(processed_dir, class_name)
    
    # Create the processed class directory if it doesn't exist
    os.makedirs(processed_class_dir, exist_ok=True)

    # List all the files in the class directory
    image_files = os.listdir(class_dir)

    # Iterate over the files and process each one
    for image_file in image_files:
        # Construct the full file paths
        image_path = os.path.join(class_dir, image_file)
        processed_image_path = os.path.join(processed_class_dir, image_file)

        # Check if the file is a regular file (not a directory)
        if os.path.isfile(image_path):
            # Open and read the image
            image = cv2.imread(image_path)
            
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Resize the image to the desired size (e.g., 100x100)
            resized_image = cv2.resize(gray_image, (128, 128))

            # Save the processed image
            cv2.imwrite(processed_image_path, resized_image)

print('Image processing completed.')
