import os
import random
import shutil
import numpy as np
import uuid
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

def is_valid_image(path):
    """
    Checks if an image is valid and readable using PIL.
    Returns True if valid, False if corrupted or unreadable.
    """
    try:
        with Image.open(path) as img:
            img.verify()  # Only checks integrity
        return True
    except Exception:
        return False

def downsample_images(input_dir, output_dir, target_count, seed):
    """
    Downsamples or copies only valid images from input_dir to output_dir.
    Ensures no class has more than target_count images.
    """
    os.makedirs(output_dir, exist_ok=True)

    label_names = os.listdir(input_dir)

    # Looping through each class 
    for label_name in label_names:
        label_folder = os.path.join(input_dir, label_name)

        if not os.path.isdir(label_folder):
            continue

        output_label_folder = os.path.join(output_dir, label_name)
        os.makedirs(output_label_folder, exist_ok=True)

        # Skip if already has target number of valid images
        existing_output_files = [
            f for f in os.listdir(output_label_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if len(existing_output_files) == target_count:
            print(f"[{label_name}] Already has {target_count} images — skipping.")
            continue

        # Collect valid image files only
        all_files = [
            f for f in os.listdir(label_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        valid_files = []
        for f in all_files:
            full_path = os.path.join(label_folder, f)
            if is_valid_image(full_path):
                valid_files.append(f)
            else:
                print(f"[{label_name}] Skipped corrupted image: {f}")

        if len(valid_files) == 0:
            print(f"[{label_name}] No valid images found — skipping.")
            continue

        # Copying valid images or downsampling 
        if len(valid_files) <= target_count:
            print(f"[{label_name}] Number of invalid and valid images: {len(all_files)}.")
            print(f"[{label_name}] Copying all {len(valid_files)} valid images.")
            for f in valid_files:
                src_path = os.path.join(label_folder, f)
                dst_path = os.path.join(output_label_folder, f)
                shutil.copy(src_path, dst_path)
        else:
            print(f"[{label_name}] Number of invalid and valid images: {len(all_files)}.")
            print(f"[{label_name}] Downsampling from {len(valid_files)} to {target_count} valid images.")
            random.seed(seed)
            selected_files = random.sample(valid_files, target_count)
            for f in selected_files:
                src_path = os.path.join(label_folder, f)
                dst_path = os.path.join(output_label_folder, f)
                shutil.copy(src_path, dst_path)

def augment_images(output_dir, target_count, seed):
    """
    Augments each class folder inside the output directory until it reaches the specified target count.
    Augmentation is only applied to original (non-augmented) images.
    """

    # Defining augmentation transformations
    augment_params = {
        'rotation_range': 40,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'shear_range': 0.2,
        'zoom_range': 0.2,
        'horizontal_flip': True,
    }

    # Initialising the augmentation generator
    datagen = ImageDataGenerator(**augment_params)

    # Looping through each class folder inside output_dir
    for class_name in os.listdir(output_dir):
        class_output_path = os.path.join(output_dir, class_name)

        # Skip if not a directory
        if not os.path.isdir(class_output_path):
            continue

        # Get a list of only original images, and skip augmented or 'aug' files
        original_images = [
            f for f in os.listdir(class_output_path)
            if f.lower().endswith(('jpg', 'jpeg', 'png')) and not f.startswith('aug')
        ]

        # Defining current number of images 
        current_count = len([
            f for f in os.listdir(class_output_path)
            if f.lower().endswith(('jpg', 'jpeg', 'png'))
        ])

        # Identifying how many images are needed i.e. number of images to augment
        generate_count = target_count - current_count

        # Skip this class if it's already balanced
        if generate_count <= 0:
            print(f"[{class_name}] No augmentation needed.")
            continue

        print(f"[{class_name}] Augmenting {generate_count} images...")

        # Shuffling base images once
        random.seed(seed)

        # Generating shuffled list of base images of same lengths but different order
        base_images = random.sample(original_images, len(original_images)) 
        augment_index = 0 # index pointer that tracks current position in base images list 
        i = 0 # i counts the number of augmented images generated and saved

        # Generating images until target_count is met
        while i < generate_count:
            
            # Reshuffling if all base images have been used
            if augment_index >= len(base_images):
                base_images = random.sample(original_images, len(original_images))
                augment_index = 0

            # Defining image name and image path 
            img_name = base_images[augment_index]
            img_path = os.path.join(class_output_path, img_name)

            try:
                # Changing to numpy array to use for ImageDataGenerator
                img = load_img(img_path)
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)

                # Performing augmentation on one image at a time 
                for batch in datagen.flow(x, batch_size=1):

                    # Taking just the first image 
                    augmented_img = array_to_img(batch[0])

                    # Create unique filename 
                    original_base = os.path.splitext(img_name)[0]
                    unique_id = uuid.uuid4().hex[:8]  # shorter uuid
                    new_filename = f"{class_name}_aug_{original_base}_{unique_id}.jpeg"

                    # Saving image 
                    save_path = os.path.join(class_output_path, new_filename)
                    augmented_img.save(save_path)

                    # Incrementing indexes 
                    i += 1
                    augment_index +=1 

                    # Print the actual count of saved files so far
                    current_total = len([
                        f for f in os.listdir(class_output_path)
                        if f.lower().strip().endswith(('.jpg', '.jpeg', '.png'))
                    ])
                    print(f"[{class_name}] Generated {i}/{generate_count} → Saved {current_total}")

                    # Breaking because datagen.flow() is infinite; we break immediately after one image is saved
                    break  
            except Exception as e:
                print(f"[{class_name}] Skipped {img_name} due to error: {e}")
                continue

def load_images_from_folders(folder_path, image_size):
    """
    Loads images from subfolders and returns X (image array), y (class labels), and class names.
    """

    X = []
    y = []

    # List of class folders
    label_names = sorted(os.listdir(folder_path))  # Sorted for consistent label ordering

    for label, label_name in enumerate(label_names):
        label_folder = os.path.join(folder_path, label_name)

        if not os.path.isdir(label_folder):
            continue

        for img_name in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_name)

            try:
                # Loads and resizes images into specified image size 
                # Converts images to RGB (i.e. pixel values are b/w 0 to 255), and stores images and labels into arrays 
                img = load_img(img_path, target_size=image_size, color_mode='rgb') 
                img_array = img_to_array(img)
                X.append(img_array)
                y.append(label)
            except Exception as e:
                print(f"Skipped {img_path} due to error: {e}")
                continue

    # Print statements for checking 
    print("\n")
    print(f"Length of X is {len(X)}")
    print(f"Length of y is {len(y)}")

    return np.array(X), np.array(y), label_names

def make_subsets(X, y, seed):

    """
    This function splits the data into training, validation and testing subsets with 80% of the data for training, 10% for validation and 10% for testing.
    """

    # First, split the data into training and temp (i.e. testing and validation)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.8, random_state=seed, stratify=y)

    # Then split the temp set into validation and testing
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=0.5, random_state=seed, stratify=y_temp)

    # Print statements for checking 
    print("\n")
    print("X_train shape:", X_train.shape)
    print("X_val shape:  ", X_val.shape)
    print("X_test shape: ", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape:  ", y_val.shape)
    print("y_test shape: ", y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test

def create_generators(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):

    """
    This function creates generators for the training, validation and testing data.
    """

    # Data augmentation on training data
    train_datagen = ImageDataGenerator(
        rescale=1./255,               # Rescaling pixel values to [0, 1]
        brightness_range= [0.5, 1.5]  # Augmentation (brightness)
    )

    train_generator = train_datagen.flow(
        X_train,                      # Numpy array of training images
        y_train,                      # Numpy array of integer labels  
        batch_size=batch_size,        # Batch size of 32
    )

    # Data augmentation on validation data
    val_datagen = ImageDataGenerator(rescale=1./255)  # Rescaling pixel values to [0, 1]

    val_generator = val_datagen.flow(
        X_val, 
        y_val,                      
        batch_size=batch_size,     
    )

    # Data augmentation on test data
    test_datagen = ImageDataGenerator(rescale=1./255)  # Rescaling pixel values to [0, 1]

    test_generator = test_datagen.flow(
        X_test, 
        y_test,         
        batch_size=batch_size, 
        shuffle = False             
    )

    return train_generator, val_generator, test_generator



