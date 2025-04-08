import pickle
from clearml import Task

from main.preprocess import load_images_from_folders, make_subsets


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name="examples", task_name="Charlie Pipeline step 2 process dataset")

# program arguments
# Use either dataset_task_id to point to a tasks artifact or
# use a direct url with dataset_url
args = {
    'dataset_task_id': '', #update id if it needs running locally
    'dataset_url': '',
    'random_state': 42,
    'test_size': 0.2,
}

# store arguments, later we will be able to change them from outside the code
task.connect(args)
print('Arguments: {}'.format(args))

# only create the task, we will actually execute it later
task.execute_remotely()

# get dataset from task's artifact
if args['dataset_task_id']:
    dataset_upload_task = Task.get_task(task_id=args['dataset_task_id'])
    print(f"Found dataset in task: {args['dataset_task_id']}")
    print("Available artifacts:", dataset_upload_task.artifacts.keys())

    # Assuming the artifact name was 'my_dataset' and is a folder path
    dataset_folder_path = dataset_upload_task.artifacts['my_dataset'].get_local_copy()
else:
    raise ValueError("Missing dataset_task_id!")

X, y, labels = load_images_from_folders(dataset_folder_path, image_size=args['image_size'])
X_train, X_val, X_test, y_train, y_val, y_test = make_subsets(X, y)

# === Upload Processed Data ===
task.upload_artifact('X_train', X_train)
task.upload_artifact('X_val', X_val)
task.upload_artifact('X_test', X_test)
task.upload_artifact('y_train', y_train)
task.upload_artifact('y_val', y_val)
task.upload_artifact('y_test', y_test)
task.upload_artifact('label_names', labels)

print("Artifacts uploaded in background.")
print("Done.")
