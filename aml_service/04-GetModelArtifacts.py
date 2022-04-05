import json, os
from azureml.core import Workspace, Datastore, Dataset

# workspace authentication
from azureml.core.authentication import AzureCliAuthentication
cli_auth = AzureCliAuthentication()

# set workspace
ws = Workspace.from_config(path="aml_config/aml_config.json", auth=cli_auth)

# load mlops config
with open("aml_config/config.json") as f:
    config = json.load(f)

datastore = Datastore(ws, config["datastore_name"])
dataset = Dataset.get_by_name(ws, config["dataset_name"]) 
source_directory = config["source_directory"]
test_path = os.path.join(source_directory, config["test_folder"])

# retrieve the latest trained model (Custom Vision), label and test dataset (images)
dataset.download(target_path=test_path, overwrite=True)

print(f'[INFO] Downloaded latest model artifacts and test dataset to {test_path}')