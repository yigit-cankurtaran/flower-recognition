import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# define constants
DATA_DIR = 'flowers'
IMG_SIZE = 64
NUM_CLASSES = 5

# define function to load images


def load_data(data_dir=DATA_DIR):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(class_names.index(class_name))

    images = np.array(images)
    labels = np.array(labels)
    return images, labels, class_names


# Load data
images, labels, class_names = load_data(DATA_DIR)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42)

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0
