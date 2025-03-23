import os, pathlib, shutil
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import pandas as pd

from evaluation_metrics import plot_history

def make_subsets(original_path, new_base_path, class_names, total_images, image_size, batch_size):
    """
    This function creates a new base directory and splits images into train, validation and test folders (60:20:20 split). 
    Within each of the train and test folders, are copies of images corresponding to the 3 classes.

    Input:
        original_path: path to the original dataset
        new_base_path: path to the new base directory
        class_names: list of class names

    Output:
        train_dataset: the training dataset
        test_dataset: the testing dataset
        validation_dataset: the validation dataset
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
        label_mode="int"  # or "categorical" depending on model
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
        label_mode="int"
    )

    return train_dataset, test_dataset, validation_dataset

def build_model():
    """
    This function builds a convolutional neural network model.

    Input:
        None

    Output:
        model: the compiled model
    """

    # Defining the inputs as shape (224, 224, 3) 
    inputs = keras.Input(shape=(224, 224, 3))

    # Rescaling inputs to [0,1] range 
    x = layers.Rescaling(1./255)(inputs)

    # Implementing convolutional layers and pooling 
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)

    # Flattening the output 
    x = layers.Flatten()(x)

    # Defining the outputs 
    outputs = layers.Dense(3, activation = "softmax")(x) # defined for 3 output classes

    model = keras.Model(inputs = inputs, outputs = outputs)

    # Compiling the model
    model.compile(optimizer="rmsprop", 
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # Printing the model summary
    model.summary()

    return model

def train_model(model, train_dataset, validation_dataset, best_model_file):
    """
    This function trains the model.

    Input:
        model: the model to train
        train_dataset: the training dataset
        best_model_file: the path to the best model
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
        epochs=30, #  ========================================================================== Adjust this for testing 
        validation_data=validation_dataset,
        callbacks=callbacks  
    )
    
    print(f"Saving model to:{best_model_file}")
    
    return history
    
def test_model(test_dataset, best_model_file):  
    """
    This function tests the model.

    Input:
        test_dataset: the test dataset
        best_model_file: the file path to the best model
    """
    
    # Testing the model
    test_model = keras.models.load_model(best_model_file)
    test_loss, test_acc = test_model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc:.3f}")
    
    return test_acc, test_loss


if __name__ == "__main__":

    # Initialising folder path and class names
    original_path = 'C:\\Users\\laure\\OneDrive\\Desktop\\AI Studio\\Z. First Sprint - test MLOps pipeline, organisation\\Classification Model'
    new_base_path= 'C:\\Users\\laure\\OneDrive\\Desktop\\AI Studio\\Z. First Sprint - test MLOps pipeline, organisation\\Classification Model\\dataset_dir'
    class_names = ["biological", "plastic", "trash"]

    # Defining image size and batch size
    image_size = (224, 224)
    batch_size = 32
    total_images = 900 # Defined as 900 for testing purposes due to uneven number of images for each class; remove this out if even number of images for each class

    # Developing training and testing subsets; remove total images if even number of images for each class
    train_dataset, test_dataset, validation_dataset = make_subsets(original_path, new_base_path, class_names, total_images, image_size, batch_size)

    # for data_batch, labels_batch in train_dataset:
    #     print("\ndata batch shape:", data_batch.shape)
    #     print("labels batch shape:", labels_batch.shape)
    #     break

    # Defining the best model
    best_model_file = "best_model.keras"

    # Develop and train the model
    model = build_model()
    history = train_model(model, train_dataset, validation_dataset, best_model_file)

    # Testing the model
    test_acc, test_loss = test_model(test_dataset, best_model_file)

    # Saving history 
    history_df = pd.DataFrame(history.history)
    history_df.to_csv("training_history.csv", index=False)

    # Loading history and best model 
    history_df = pd.read_csv("training_history.csv")
    best_model = keras.models.load_model(best_model_file)

    # Plotting history 
    plot_history(history_df)




