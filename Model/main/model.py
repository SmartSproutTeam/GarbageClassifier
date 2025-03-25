from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet201

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

def build_model(label_number):
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

    inputs = keras.Input(shape=(400, 400, 3))  
    x = data_augmentation(inputs)

    # Passing through the convolutional base
    conv_base = define_base_model()
    x = conv_base(x)

    # Reducing the entire feature map into single vector; reduces overfitting and has fewer parameter than Flatten
    x = layers.GlobalAveragePooling2D()(x)
    
    # Defining Dense and Dropout layers
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # Defining the outputs 
    outputs = layers.Dense(label_number, activation = "softmax")(x) # defined for 3 output classes

    model = keras.Model(inputs, outputs)

    # Compiling the model
    model.compile(optimizer="rmsprop", #==================================================== can adjust to adam or sgd
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model

def train_model(model, train_X, train_y):
    history = model.fit(
        train_X,
        train_y,
        epochs=20, 
        validation_split=0.2,
    )
    
    return history

def test_model(model, test_X, test_y):  
    # test_model = keras.models.load_model(best_model_file)
    test_loss, test_acc = model.evaluate(test_X, test_y)
    print(f"\nTest accuracy: {test_acc:.3f}")
    
    return test_acc, test_loss
