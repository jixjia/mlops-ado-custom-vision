import os
import json
import sys
from azureml.core import Workspace, Experiment, Datastore, Dataset, ScriptRunConfig, Environment
from azureml.core.runconfig import RunConfiguration, CondaDependencies
from azureml.core.conda_dependencies import CondaDependencies
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData, StepSequence
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import PublishedPipeline
from azureml.pipeline.core.graph import PipelineParameter
from azureml.core.compute import ComputeTarget
from azureml.core.model import Model
from azureml.core.resource_configuration import ResourceConfiguration

# workspace authentication
from azureml.core.authentication import AzureCliAuthentication
cli_auth = AzureCliAuthentication()

# set workspace
ws = Workspace.from_config(path="aml_config/aml_config.json", auth=cli_auth)

# load mlops config
with open("aml_config/config.json") as f:
    config = json.load(f)

model_name = config["model_name"]
source_directory = config["source_directory"]
test_folder = config["test_folder"]
model_dir = os.path.join(source_directory, test_folder, 'models')

try:
    # register the latest Custom Vision model for version tracking (auto increment)
    model = Model.register(workspace = ws,
                        model_name = model_name,
                        model_path = model_dir,
                        model_framework = Model.Framework.TENSORFLOW,
                        tags = {'source': "custom_vision", 'type': "object_detection"},
                        description = "Custom Vision object detection model for detecting Japanese vehicle license plates",
                        resource_configuration = ResourceConfiguration(cpu=1, memory_in_gb=2),
                        )

    print(f'[INFO] Successfully registered the latest model {model.name} (version: {model.version})')

except Exception as e:
    print(e.args)
    sys.exit(0)