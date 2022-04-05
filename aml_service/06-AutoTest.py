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

# workspace authentication
from azureml.core.authentication import AzureCliAuthentication
cli_auth = AzureCliAuthentication()

# set workspace
ws = Workspace.from_config(path="aml_config/aml_config.json", auth=cli_auth)

# load mlops config
with open("aml_config/config.json") as f:
    config = json.load(f)

aml_pipeline_name = "build-pipeline"
experiment_name = config["experiment_name"]
aml_cluster_name = config["compute_cluster_name"]
source_directory = config["source_directory"]
datastore = Datastore(ws, config["datastore_name"])
dataset = Dataset.get_by_name(ws, config["dataset_name"]) 
compute_target = ws.compute_targets[aml_cluster_name]
env = Environment.get(ws, config["env_name"])
exp = Experiment(ws, experiment_name)

# mount test dataset to compute target 
input_data = dataset.as_mount()

# auto-test job config
src = ScriptRunConfig(source_directory=os.path.join(source_directory, config["test_folder"]),
                      script='auto-test.py',
                      arguments=['--data_folder', input_data],
                      compute_target=compute_target,
                      environment=env)

# submit job and start tracking
run = exp.submit(config=src)

# monitor run
run.wait_for_completion()