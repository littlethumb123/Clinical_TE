import os, sys, re
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from airflow.utils.dates import days_ago
from airflow.models import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import BranchPythonOperator,PythonOperator,PythonVirtualenvOperator
# from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook
# from airflow.utils.email import send_email
# from airflow.exceptions import AirflowException
# from airflow.providers.google.cloud.operators.gcs import GCSDeleteObjectsOperator
# from airflow.providers.google.cloud.transfers.bigquery_to_gcs import BigQueryToGCSOperator
# import pandas as pd
# from airflow.operators.bash_operator import BashOperator

import calling_config as ct
import email_context as ec
from datetime import datetime, timedelta
# import pendulum
# from dateutil.relativedelta import relativedelta
# from google.cloud import bigquery
# from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook
# from google.auth import impersonated_credentials
# from airflow.exceptions import AirflowException


# constant vars
TENANT = "clin-analytics-hcb"
REPO_NAME = "enhanced-rap-cp"
PROJECT_ID = os.environ.get("GCP_PROJECT")
DAG_PATH = os.environ.get('DAGS_FOLDER')
ENV = PROJECT_ID.split('-')[-1]

connect_sa = f"clin-analytics-hcb-connect@{PROJECT_ID}.iam.gserviceaccount.com"
resource_sa = f"gchcb-clin-analytics-ontpd@{PROJECT_ID}.iam.gserviceaccount.com"
decrypt_sa = "gchcb-clin-analytics-d-ontpd@anbc-hcb-dev.iam.gserviceaccount.com"
CMEK_KEY = f"projects/cvs-key-vault-nonprod/locations/us-east4/keyRings/gkr-nonprod-us-east4/cryptoKeys/gk-{PROJECT_ID}-us-east4"

if (f"{ENV}" == "test"):
    resource_sa = "gchcb-clin-analytics-onppq@anbc-hcb-test.iam.gserviceaccount.com"
    connect_sa = "clin-analytics-hcb-connect@anbc-hcb-test.iam.gserviceaccount.com"
    decrypt_sa = "gchcb-clin-analytics-d-onppq@anbc-hcb-test.iam.gserviceaccount.com"
    CMEK_KEY = f"projects/cvs-key-vault-nonprod/locations/us-east4/keyRings/gkr-nonprod-us-east4/cryptoKeys/gk-{PROJECT_ID}-us-east4"
if (f"{ENV}" == "prod"):
    resource_sa = "gchcb-clin-analytics-onppp@anbc-hcb-prod.iam.gserviceaccount.com"
    connect_sa = "clin-analytics-hcb-connect@anbc-hcb-prod.iam.gserviceaccount.com"
    decrypt_sa = "gchcb-clin-analytics-d-onppp@anbc-hcb-prod.iam.gserviceaccount.com"
    CMEK_KEY = f"projects/cvs-key-vault-prod/locations/us-east4/keyRings/gkr-prod-us-east4/cryptoKeys/gk-{PROJECT_ID}-us-east4"

REGION = ct.config['config']['REGION']
USER = ct.config['config']['USER']
DAG_ID = f"{TENANT}-{REPO_NAME}-master"
## DATASETS
DATASET = ct.config['config']['DATASET'].format(ENV=ENV)
DEC_DATASET = ct.config['config']['DEC_DATASET'].format(ENV=ENV)
SHARE_BQDB = ct.config['config']['SHARE_BQDB'].format(ENV=ENV)
FINAL_DATASET = (DATASET if ENV=="prod" else DEC_DATASET)
## SQL PARAMS
sql_folder_path = f"{TENANT}/{REPO_NAME}/sql/"
owner_name = f"{USER}_aetna_com"
databases = ct.config['config']
databases["DATASET"] = DATASET
databases["DEC_DATASET"] = DEC_DATASET
databases["FINAL_DATASET"]=(databases["DATASET"] if ENV=="prod" else databases["DEC_DATASET"])
params = databases
params['OWNER'] = owner_name
params['project_name'] = PROJECT_ID
current_dt = datetime.now()
current_dt = current_dt.strftime('%Y-%m-%d')
params['current_dt']=current_dt

## email function
def success_call(context):
    ec.email_function(context, DAG_ID,"Success",ct.config['config']['success_email_to'])
def fail_call(context):
    ec.email_function(context, DAG_ID,"Failed",ct.config['config']['fail_email_to'])
 
## DAG PARAMS
default_args = {
    'project_id': PROJECT_ID,
    'retries': 0,
    "email_on_failure": False,
    "depends_on_past": False,
    'email_on_retry': True,
    'retry_delay': timedelta(minutes=3),
    "on_success_callback": success_call,
    "on_failure_callback": fail_call
}
dag_defaults = {}
default_args.update(dag_defaults)


 
# tags for DAG
with DAG(
        DAG_ID,
        start_date = days_ago(7),
        ## dev run daily
        schedule_interval= None, ##("0 12 * * *" if ENV=="dev" else None),
        access_control={'clin-analytics-hcb': {'can_read'}},
        max_active_runs = 1,
        tags=ct.DAG_TAGS,
        default_args=default_args,
        catchup=False,
        template_searchpath=os.path.join(DAG_PATH, sql_folder_path),
        # this ensure that Airflow checks the files in Composer DAG bucket for SQL files
        user_defined_macros=params,
        params={"isinit": 0,
                "te_base":"",
                "prefix":ct.config['config']['prefix'],
                "costcenter": ct.config['config']['COSTCENTER'],
                "owner": owner_name,
                "DEC_DATASET":DEC_DATASET,
                "FINAL_DATASET": FINAL_DATASET,
                "historydays": 30,
                "recipient": []
               }
    
) as dag:  # Must specify your tenant name and owner email as tags
    
    
    trigger_transformer_input = TriggerDagRunOperator(
        task_id = "trigger_transformer_input",
        trigger_dag_id = f'{TENANT}-{REPO_NAME}-input',
        wait_for_completion=True,
        conf = { "te_base": '{{ dag_run.conf["te_base"] }}',
                 "isinit": '{{ dag_run.conf["isinit"] if dag_run.conf.get("isinit") else 0 }}',
                    "costcenter": '{{ dag_run.conf["costcenter"] }}',
                    "owner": '{{ dag_run.conf["owner"] }}',
                    "DEC_DATASET":'{{ dag_run.conf["DEC_DATASET"] }}',
                    "FINAL_DATASET": '{{ dag_run.conf["FINAL_DATASET"] }}',
                    "prefix": '{{ dag_run.conf["prefix"] }}',
                    "historydays": '{{ dag_run.conf["historydays"] }}',
                    "recipient": []
                },
        dag = dag
    )		

    
    trigger_get_embeddings = TriggerDagRunOperator(
        task_id = "trigger_get_embeddings",
        trigger_dag_id = f"{TENANT}-{REPO_NAME}-get-embeddings",
        wait_for_completion=True,
        conf = {"input_table": f"{FINAL_DATASET}"+"."+"{{ dag_run.conf['prefix'] }}"+"_o3_score_ending",
                "prefix": '{{ dag_run.conf["prefix"] }}',
                "isinit": '{{ dag_run.conf["isinit"] if dag_run.conf.get("isinit") else 0 }}'
                },
        dag = dag
    )
    
    

trigger_transformer_input >> trigger_get_embeddings
