# Import necessary libraries
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from skimage.feature import local_binary_pattern
import skimage.feature as skif

# Specify the directory paths for different classes
classes = ["car", "airplane", "bike", "ship"]

# Initialize lists to store features and labels
feat = []
label = []

for class_idx, class_name in enumerate(classes):
    images_dir = f"C:\\Users\\ar52624\\Desktop\\Fall 2023\\Pattern Recognition\\car-bike-airplane\\my-dataset\\2000\\processedimages\\{class_name}"

    # List all the files in the directory
    imagecount = os.listdir(images_dir)

    # Iterate over the files and read each one
    for value in imagecount:
        # Construct the full file path
        image_path = os.path.join(images_dir, value)

        # Check if the file is a regular file (not a directory)
        if os.path.isfile(image_path):
            # Open and read the file
            dataset = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Define HOG parameters
            orientations = 9  # Number of gradient orientations
            pixels_per_cell = (8, 8)  # Size of a cell in pixels
            cells_per_block = (2, 2)  # Number of cells in each block

            # Compute HOG features
            hog_features, hog_image = skif.hog(
                dataset,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                visualize=True,
                block_norm='L2-Hys'
            )
            mhd = np.mean(hog_features)
            shd = np.std(hog_features)
            n_hog_features = (hog_features - mhd) / shd

            radius = 4
            n_points = 6 * radius
            method = 'uniform'
            lbp_image = local_binary_pattern(dataset, n_points, radius, method=method)
            lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

            # Normalize the histogram
            lbp_hist = lbp_hist.astype("float")
            lbp_hist /= (lbp_hist.sum() + 1e-8)
            mld = np.mean(lbp_hist)
            sld = np.std(lbp_hist)
            n_lbp_hist = (lbp_hist - mld) / sld

            features = np.concatenate((n_hog_features, n_lbp_hist))

            # Append features to the list and assign labels
            feat.append(features)
            label.append(class_idx)

# Initialize the SVM classifier
clf = svm.SVC(C=0.1, kernel='poly', degree=3, coef0=2)

# Perform cross-validation
cv_scores = cross_val_score(clf, feat, label, cv=5)

# Print the cross-validation accuracies
print("Cross-validation Accuracies:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feat, label, test_size=0.2, random_state=42)

# Fit the SVM model to the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Generate the confusion matrix
confusion = confusion_matrix(y_test, y_pred)

# Print the confusion matrix and classification report
print("Confusion Matrix:")
print(confusion)

class_names = ["car", "airplane", "bike", "ship"]
fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(confusion, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Print the classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)
#############################################################################
#############################################################################
####### KNN ###############
from sklearn.neighbors import KNeighborsClassifier
# Initialize the KNN classifier with the desired number of neighbors (e.g., n_neighbors=5)
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Make predictions using the KNN classifier
knn_predictions = knn_classifier.predict(X_test)

# Calculate accuracy using the KNN classifier
knn_accuracy = accuracy_score(y_test, knn_predictions)
print(f'KNN Test Accuracy: {knn_accuracy * 100:.2f}%')

# Generate the confusion matrix using the KNN classifier
knn_confusion = confusion_matrix(y_test, knn_predictions)

# Print the confusion matrix and classification report for KNN
print("KNN Confusion Matrix:")
print(knn_confusion)

# Print the classification report for KNN
knn_report = classification_report(y_test, knn_predictions)
print("KNN Classification Report:")
print(knn_report) 

#############################################################################
#############################################################################
####### Decision Trees ###############
# Import the DecisionTreeClassifier class
from sklearn.tree import DecisionTreeClassifier

# Initialize the Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier()

# Fit the Decision Tree model to the training data
decision_tree_classifier.fit(X_train, y_train)

# Make predictions using the Decision Tree classifier
decision_tree_predictions = decision_tree_classifier.predict(X_test)

# Calculate accuracy using the Decision Tree classifier
decision_tree_accuracy = accuracy_score(y_test, decision_tree_predictions)
print(f'Decision Tree Test Accuracy: {decision_tree_accuracy * 100:.2f}%')

# Generate the confusion matrix using the Decision Tree classifier
decision_tree_confusion = confusion_matrix(y_test, decision_tree_predictions)

# Print the confusion matrix and classification report for the Decision Tree
print("Decision Tree Confusion Matrix:")
print(decision_tree_confusion)

# Print the classification report for the Decision Tree
decision_tree_report = classification_report(y_test, decision_tree_predictions)
print("Decision Tree Classification Report:")
print(decision_tree_report)

