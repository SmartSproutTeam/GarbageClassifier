from clearml import Task
import os

# Create a dataset experiment
task = Task.init(project_name="examples", task_name="Charlie Pipeline step 1 dataset artifact")

# Only create the task, we will actually execute it later
task.execute_remotely()

# Path to the zipped dataset file
zipped_dataset = "./archive.zip"

print(f"Current working directory: {os.getcwd()}")

# Resolve and verify the file path
absolute_path = os.path.abspath(zipped_dataset)
print(f"Resolved absolute path: {absolute_path}")
if not os.path.exists(absolute_path):
    raise FileNotFoundError(f"File '{absolute_path}' does not exist!")

# Upload the zipped file as an artifact
task.upload_artifact(name="dataset", artifact_object=zipped_dataset)

print(f"Uploaded zipped dataset '{zipped_dataset}' as an artifact.")

# We are done
print("Done")