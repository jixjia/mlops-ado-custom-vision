import os
import json
import datetime
import uuid
import sys
from azureml.core import Workspace, Environment
from azureml.core.webservice import Webservice
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.webservice import AciWebservice

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
source_directory = config["source_directory"]
deploy_folder = config["deploy_folder"]
model_name = config["model_name"]
model_description = config["model_description"]
env = Environment.get(ws, config["env_name"])

# fetch latest registered model
registered_model = Model(ws, model_name)

# container host
aci_config = AciWebservice.deploy_configuration(cpu_cores = 1, 
                                               memory_gb = 2, 
                                               tags = {'model':model_name, 'framework':config["env_name"]}, 
                                               description = model_description)

# inference config
inference_config = InferenceConfig(entry_script = "score.py", 
                                   source_directory = os.path.join(source_directory, deploy_folder), 
                                   environment = env,
                                   enable_gpu = False)

# container host name (using model version as suffix)
service_name = f'mlops-custom-vision-v{registered_model.version}'

try:
    service = Model.deploy(workspace = ws, 
                        name = service_name, 
                        models = [registered_model], 
                        inference_config = inference_config, 
                        deployment_config = aci_config,
                        overwrite = True)

    # provision & deploy
    service.wait_for_deployment(show_output=True)

    print(f"Deployed ACI Webservice: {service.name} \nWebservice Uri: {service.scoring_uri}")

    # write endpoint detail to local for unit testing in next step
    aci_webservice = {}
    aci_webservice["aci_service_name"] = service.name
    aci_webservice["aci_service_edndpoint"] = service.scoring_uri
    webservice_path = os.path.join(source_directory, deploy_folder, "aci_webservice.json")

    with open(webservice_path, "w") as f:
        json.dump(aci_webservice, f)

except Exception as e:
    print(e.args)
    sys.exit(0)