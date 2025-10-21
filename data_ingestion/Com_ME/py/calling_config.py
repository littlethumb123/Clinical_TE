import os
import json

#-----------------------------------------------------------------------------
# THIS FILE CONTAINS ALL CONFIG LEVEL DETAILS AND DAG_TAGS-- PATH OF CONFIG AND READING CONFIG IS BEING DONE IN THIS PYTHON FILE
#-----------------------------------------------------------------------------

DAG_PATH = os.environ.get('DAGS_FOLDER')

config_path = 'hcm-cm-de-hcb/transformer-embeddings-v2/config/'
config_path = os.path.join(DAG_PATH, config_path)

with open(config_path + 'config.json') as f:
    config = json.load(f)

DAG_TAGS = ["tenant:hcm-cm-de", f"owner:{config['config']['USER']}@aetna.com"]