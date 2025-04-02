import os
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image

def load_one_folder(folder_path, label, image_size):
    images = []
    for filename in os.listdir(folder_path):
        with Image.open(os.path.join(folder_path, filename)) as img:
            image = img.convert("RGB").resize(image_size)
            images.append(np.array(image, dtype=np.uint8))
    return images, [label] * len(images)

def load_all_folders(folder_path, label_names, image_size):
    X = []
    y = []
    for label in range(len(label_names)):
        X_temp, y_temp = load_one_folder(os.path.join(folder_path, label_names[label]), label, image_size)
        X += X_temp
        y += y_temp
    return np.array(X), np.array(y)


def make_subsets():
    folder_path = "../Data/garbage-dataset/"
    label_names = [filename for filename in os.listdir(folder_path)]

    X, y = load_all_folders(folder_path, label_names, (400, 400))
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    return X_train, y_train, X_test, y_test, label_names
    # return X_train, y_train, X_test, y_test, label_names