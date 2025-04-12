import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from clearml import Task, Logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.model import build_model, train_model
from main.preprocess import create_generators


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name="GarbageClassifier", task_name="Train model")
logger = Logger.current_logger()

# Arguments
args = {
    'dataset_task_id': '', # replace the value only when you need debug locally
}
task.connect(args)

# only create the task, we will actually execute it later
task.execute_remotely() # After passing local testing, you should uncomment this command to initial task to ClearML

print('Retrieving dataset')
dataset_task = Task.get_task(task_id=args['dataset_task_id'])
X_train = dataset_task.artifacts['X_train'].get()
X_test = dataset_task.artifacts['X_test'].get()
X_val = dataset_task.artifacts['X_val'].get()
y_train = dataset_task.artifacts['y_train'].get()
y_test = dataset_task.artifacts['y_test'].get()
y_val = dataset_task.artifacts['y_val'].get()
label_names = dataset_task.artifacts['label_names'].get()
image_size = dataset_task.artifacts['image_size'].get()
print('Dataset loaded')


# Creating generators
batch_size = 32
train_generator, val_generator, test_generator = create_generators(X_train, y_train, X_val, y_val, X_test, y_test, batch_size)

# Defining the best model
best_model_file = "best_model.keras"

print('Generators loaded')

# Training the model
model = build_model(image_size, len(label_names))
history = train_model(model, train_generator, val_generator, best_model_file)

training_accuracy = history.history['accuracy'][-1]
validation_accuracy = history.history['val_accuracy'][-1]

print(f"Training Accuracy: {training_accuracy}")
print(f"Validation Accuracy: {validation_accuracy}")

task.upload_artifact('model', model)