# Vehicle-Classification-Multi-Class-SVM
In this project, I have classified four types of vehicles: airplane, bike, car, and ship using SVM classifier as a part of my assignment for Pattern Recognition course at University of Georgia. The details of the project are explained in a table below.
![Comparision Table](https://github.com/aleenarayamajhi/Vehicle-Classification-Multi-Class-SVM/assets/126793934/b7a182e1-cdff-4915-b62e-44ea3beb2be8)
The dataset for this project were taken from [GitHub](https://github.com/ghanimmustafa/SIFT_BoW_SVM_Object_Classification) and [Kaggle](https://www.kaggle.com/datasets/abtabm/multiclassimagedatasetairplanecar). The sample of images for each class are presented below:
![image](https://github.com/aleenarayamajhi/Vehicle-Classification-Multi-Class-SVM/assets/126793934/c7674391-5d08-4d42-a072-9f2c9d450647) Some addiitonal datasets were compiled to make 2000 images in each class. Then, image preprocessing was performed to convert images into grayscale to reduce noise and improve computational efficiency. Also, the images were reduced to 128 x 128 PX. For image preprocessing, theres's a code file provided in this repo. And, feature extraction was done using HOG and LBP combine after normalization with 8126 features total. The code for feature extraction is provided in this repo. SVM was used as a classifier. The accuracies were also compared with KNN and Decision Tree. The flowchart of this project can be explained using the following figure. 
![flow-chart-github](https://github.com/aleenarayamajhi/Vehicle-Classification-Multi-Class-SVM/assets/126793934/97302337-94ba-4b0a-b77d-c51985bde9b3)






