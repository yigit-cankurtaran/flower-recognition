import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

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

# Create the convolutional base
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(
    IMG_SIZE, IMG_SIZE, 3)))  # 32 filters, 3x3 kernel
model.add(MaxPooling2D((2, 2)))  # 2x2 pooling
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

# Compile and train the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# print the validation accuracy
print('Validation accuracy:', model.evaluate(X_test, y_test)[1])

# Save the model
model.save('flowers_model.h5')
