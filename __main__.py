import pandas as pd
from tensorflow import keras
from preprocess import downsample_images, augment_images, load_images_from_folders, make_subsets, create_generators
from evaluate import plot_history, plot_confusion_matrix, calculate_metrics
from model import build_model, train_model

def main():
    """
    Main entry point for running the entire pipeline.
    """

    # Initialising dataset path and balanced dataset path 
    dataset_path = 'D:\\10. AI Studio Dataset\\test' # ============================== EDIT THIS FOR GOOGLE ENTERPRISE
    balanced_dataset_path = 'D:\\10. AI Studio Dataset\\test_balanced' # ============ EDIT THIS FOR GOOGLE ENTERPRISE

    # Defining image size, batch size, seed and target count for each class 
    image_size = (224, 224)
    batch_size = 32
    seed = 42
    target_count = 2000

    # Downsampling each class to <= target_count and saving into balanced_dataset_path
    downsample_images(dataset_path, balanced_dataset_path, target_count, seed)

    # Augmenting images to ensure all image classes meet target count and saving into balanced_dataset_path
    augment_images(balanced_dataset_path, target_count, seed)

    # Loading images from balanced_dataset_path
    X, y, label_names = load_images_from_folders(balanced_dataset_path, image_size)

    # Initialising integer labels i.e. 0, 1, 2 for waste categories
    int_labels = [i for i in range(len(label_names))]

    # Creating training, validation and testing subsets
    X_train, X_val, X_test, y_train, y_val, y_test = make_subsets(X, y, seed)

    # Creating generators
    train_generator, val_generator, test_generator = create_generators(X_train, y_train, X_val, y_val, X_test, y_test, batch_size)

    # Defining the best model
    best_model_file = "best_model.keras"

    # Training the model
    model = build_model(image_size, len(label_names))
    history = train_model(model, train_generator, val_generator, best_model_file)

    # Saving history 
    history_df = pd.DataFrame(history.history)
    history_df.to_csv("training_history.csv", index=False)

    # Loading history and best model 
    history_df = pd.read_csv("training_history.csv")
    model = keras.models.load_model(best_model_file)
    
    # Evaluating metrics and plotting confusion matrix
    y_pred = calculate_metrics(test_generator, y_test, model, label_names)
    plot_confusion_matrix(y_test, y_pred, label_names, int_labels)

    # Plotting history 
    plot_history(history_df)

if __name__ == "__main__":

    main()
