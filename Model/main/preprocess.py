import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_images_from_folders(folder_path, image_size):

    """
    This function loads the images from the folder and returns a numpy array of the images and their corresponding labels
    """

    # Defining the lists storing the data and the correponding labels 
    X = []
    y = []

    # Defining label or class names 
    label_names = os.listdir(folder_path)

    for label, label_name in enumerate(label_names):
        label_folder = os.path.join(folder_path, label_name)

        # Checks if the label folder exists
        if not os.path.isdir(label_folder):
            continue

        # Iterates through all images in the label folder
        for filename in os.listdir(label_folder):
            image_path = os.path.join(label_folder, filename)
            try:
                with Image.open(image_path) as img:
                    # Converts image to RGB (i.e. pixel values are between 0 and 255) and resizes it to the specified size
                    image = img.convert("RGB").resize(image_size)
                    X.append(np.array(image))
                    y.append(label)
            except:
                continue

    return np.array(X), np.array(y), label_names

def make_subsets(X, y):

    """
    This function splits the data into training, validation and testing subsets with 80% of the data for training, 10% for validation and 10% for testing.
    """

    # First, split the data into training and temp (i.e. testing and validation)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)

    # Then split the temp set into validation and testing
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=0.5, random_state=42, stratify=y_temp)

    return X_train, X_val, X_test, y_train, y_val, y_test

def create_generators(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):

    """
    This function creates generators for the training, validation and testing data.
    """

    # Data augmentation on training data
    train_datagen = ImageDataGenerator(
        rescale=1./255,               # Rescaling pixel values to [0, 1]
        rotation_range=40,            # Augmentations (rotation)
        width_shift_range=0.2,        # Augmentation (shift)
        height_shift_range=0.2,       # Augmentation (shift)
        shear_range=0.2,              # Augmentation (shear)
        zoom_range=0.2,               # Augmentation (zoom)
        horizontal_flip=True,         # Augmentation (flip)
        brightness_range= [0.5, 1.5]  # Augmentation (brightness)
    )

    train_generator = train_datagen.flow(
        X_train,                      # Numpy array of training images
        y_train,                      # Numpy array of integer labels  
        batch_size=batch_size,        # Batch size of 32
    )

    # Data augmentation on validation data
    val_datagen = ImageDataGenerator(rescale=1./255)  

    val_generator = val_datagen.flow(
        X_val, 
        y_val,                      
        batch_size=batch_size,     
    )

    # Data augmentation on test data
    test_datagen = ImageDataGenerator(rescale=1./255)  

    test_generator = test_datagen.flow(
        X_test, 
        y_test,         
        batch_size=batch_size, 
        shuffle = False             
    )

    return train_generator, val_generator, test_generator