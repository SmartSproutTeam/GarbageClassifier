import os, pathlib, shutil
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
import pandas as pd
from tensorflow.keras.applications import DenseNet201

import time

from evaluation_metrics import plot_history, calculate_metrics, plot_confusion_matrix

def make_subsets(original_path, new_base_path, class_names, total_images, image_size, batch_size):
    """
    This function creates a new base directory and splits images into train, validation and test folders (60:20:20 split). 
    Within each of the train and test folders, are copies of images corresponding to the 3 classes.

    """

    original_dir = pathlib.Path(original_path)
    new_base_dir = pathlib.Path(new_base_path)

    # Skip if new base directory already exists
    if new_base_dir.exists():
        print(f"Directory already exists. Skipping dataset creation.")
    else: 
        print("Creating new dataset directory...")
    
        # Calculating the number of images per class; remove this out if it's even number of images for each class
        images_per_class = total_images // len(class_names)
        train_count = int(images_per_class * 0.6) # 60% of images for training
        val_count = int(images_per_class * 0.2) # 20% of images for validation
        test_count = images_per_class - train_count - val_count # 20% of images for testing

        for category in class_names:
            category_dir = original_dir / category

            # Get all image files in the category folder
            image_files = sorted([
                file for file in category_dir.iterdir()
                if file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
            ])

            # Limit to only the first `images_per_class` files; can remove this if even number of images available
            image_files = image_files[:images_per_class]

            # First split: train vs temp (val + test)
            train_files, temp_files = train_test_split(
                image_files, test_size=(val_count + test_count), random_state=42, shuffle=True
            )

            # Second split: val vs test
            val_files, test_files = train_test_split(
                temp_files, test_size=test_count, random_state=42, shuffle=True
            )

            # Copy files to train and test folders
            for subset_name, files in [("train", train_files), ("validation", val_files), ("test", test_files)]:
                dest_dir = new_base_dir / subset_name / category
                os.makedirs(dest_dir, exist_ok=True)
                for file_path in files:
                    shutil.copyfile(file_path, dest_dir / file_path.name)
    
    # Loading datasets from created folders
    print("\nLoading datasets...")

    train_dataset = image_dataset_from_directory(
        new_base_dir / "train",
        image_size= image_size,
        batch_size=batch_size,  
        label_mode="int"  
    )

    validation_dataset = image_dataset_from_directory(
        new_base_dir / "validation",
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int"
    )

    test_dataset = image_dataset_from_directory(
        new_base_dir / "test",
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle = False
    )

    return train_dataset, test_dataset, validation_dataset

def define_base_model():
    """
    This function defines the base model i.e DenseNet201
    """

    conv_base = DenseNet201(
        weights="imagenet",
        include_top=False,
    )

    # This empties the list of trainable weights i.e. freezing weights of convolutional base  
    # Convolutional layers in pre-trained models have learnt features e.g. edges, shapes etc. 
    # Therefore we extract these features from the convolutional base, to be used in tailored model 
    conv_base.trainable = False

    # Printing model summary 
    # conv_base.summary()

    return conv_base

def build_model():
    """
    This function builds a convolutional neural network model.

    """

    # Data augmentation 
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.111),     # ~40 degrees (0.111 * 360 ≈ 40°)
        layers.RandomZoom(0.2),           # zoom up to 20%
        layers.RandomFlip("horizontal"),  # horizontal flip
        layers.RandomWidth(0.2),          # width shift up to 20%
        layers.RandomHeight(0.2),         # height shift up to 20%
    ])

    # Defining the inputs as shape (224, 224, 3), data augmentation and rescaling
    inputs = keras.Input(shape=(224, 224, 3))  
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x) # Rescaling values between [0,1]

    # Passing through the convolutional base
    conv_base = define_base_model()
    x = conv_base(x)

    # Reducing the entire feature map into single vector; reduces overfitting and has fewer parameter than Flatten
    x = layers.GlobalAveragePooling2D()(x)
    
    # Defining Dense and Dropout layers
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # Defining the outputs 
    outputs = layers.Dense(3, activation = "softmax")(x) # defined for 3 output classes

    model = keras.Model(inputs, outputs)

    # Compiling the model
    model.compile(optimizer="rmsprop", #==================================================== can adjust to adam or sgd
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model

def train_model(model, train_dataset, validation_dataset, best_model_file):
    """
    This function trains the model.
    """

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=best_model_file,
            monitor="val_loss",
            save_best_only=True
        )
    ]

    history = model.fit(
        train_dataset,
        epochs=20, 
        validation_data=validation_dataset,
        callbacks=callbacks  
    )
    
    print(f"Saving model to:{best_model_file}")
    
    return history

def test_model(test_dataset, best_model_file):  
    """
    This function tests the model.
    """
    
    # Testing the model
    test_model = keras.models.load_model(best_model_file)
    test_loss, test_acc = test_model.evaluate(test_dataset)
    print(f"\nTest accuracy: {test_acc:.3f}")
    
    return test_acc, test_loss


if __name__ == "__main__":

    start_time = time.time()

    # Initialising folder path and class names
    original_path = 'C:\\Users\\laure\\OneDrive\\Desktop\\AI Studio\\Z. First Sprint - test MLOps pipeline, organisation\\Classification Model'
    new_base_path= 'C:\\Users\\laure\\OneDrive\\Desktop\\AI Studio\\Z. First Sprint - test MLOps pipeline, organisation\\Classification Model\\dataset_dir'
    folder_names = ["biological", "plastic", "trash"]
    class_names = ["Plastic", "Organic", "Non-Plastic & Non-Organic"]

    # Defining image size and batch size
    image_size = (224, 224)
    batch_size = 32
    total_images = 900 # Defined as 900 for testing purposes due to uneven number of images for each class; remove this out if even number of images for each class

    # # Developing training and testing subsets; remove total images if even number of images for each class
    train_dataset, test_dataset, validation_dataset = make_subsets(original_path, new_base_path, folder_names, total_images, image_size, batch_size)

    # Defining the best model
    best_model_file = "best_model.keras"

    # Training the model
    model = build_model()
    history = train_model(model, train_dataset, validation_dataset, best_model_file)

    # Testing the model; this could be removed for final version 
    test_acc, test_loss = test_model(test_dataset, best_model_file)

    # Saving history 
    history_df = pd.DataFrame(history.history)
    history_df.to_csv("training_history.csv", index=False)

    # Loading history and best model 
    history_df = pd.read_csv("training_history.csv")
    best_model = keras.models.load_model(best_model_file)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal training time: {elapsed_time:.2f} seconds")

    # Evaluating metrics and plotting confusion matrix
    y_true, y_pred = calculate_metrics(test_dataset, best_model, class_names)
    plot_confusion_matrix(y_true, y_pred, class_names)

    # Plotting history 
    plot_history(history_df)







