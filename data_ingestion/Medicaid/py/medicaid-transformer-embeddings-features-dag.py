import os, sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from airflow.utils.dates import days_ago
from airflow import models
from airflow.models import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonVirtualenvOperator,PythonOperator,BranchPythonOperator
from airflow.utils.email import send_email
from airflow.operators.dagrun_operator import TriggerDagRunOperator

import json
import logging

from datetime import datetime, timedelta, date

from google.cloud import bigquery
from google.auth import impersonated_credentials
from airflow.operators.bash_operator import BashOperator

def merge(*args):
    j = {}
    for i in args:
        j.update(i)
    return j

#########################################################
### get_config.py

PROJECT_ID   = os.environ.get("GCP_PROJECT")
DAG_PATH     = os.environ.get('DAGS_FOLDER')
ENV          = ''.join([x for x in ['dev', 'test', 'prod'] if x in PROJECT_ID])
C_PROJECT_ID = f'anbc-{ENV}-hcm-cm-de'
REPO_NAME    = "medicaid-transformer-embeddings"
TENANT       = "hcm-cm-de-hcb"

md_te_env_config_path = f'{TENANT}/{REPO_NAME}/config/'
sql_folder_path       = f'{TENANT}/{REPO_NAME}/sql/'

md_te_config_path     = os.path.join(DAG_PATH, md_te_env_config_path)

with open(md_te_config_path + f'{ENV}-{REPO_NAME}-config.json') as f:
     config = json.load(f)

print(config)

REGION       = config['config']['REGION']
USER         = config['config']['USER']
USER_EMAIL   = f"{USER}@aetna.com"
spark_jars   = config['config']['bq_spark_jar']
CLUSTER_NAME = config['config']['DATAPROC_VARS']['CLUSTER_NAME']
MPP_GPUS     = config['config']['DATAPROC_VARS']['MPP_GPUS']
group_name   = config['config']['DATAPROC_VARS']['GROUP_NAME']
CODE_BUCKET  = config['config']['CODE_BUCKET']
DATA_BUCKET  = config['config']['DATA_BUCKET']

### End of get_config.py
#########################################################
DAG_ID = f"hcm-cm-de-medicaid-transformer-embeddings-features"

CUR_DIR     = os.getcwd()

# config assign
#################################
###     Bigquery Job Params    ##
#################################
BQ_ENV_VARS  = config['config']['BQ_ENV_VARS']
DATASET      = config['config']['BQ_ENV_VARS']['DATASET']
SCHEMA       = config['config']['BQ_ENV_VARS']['CM_MD_SCHEMA']
DEC_SCHEMA   = config['config']['BQ_ENV_VARS']['DEC_SCHEMA']
SHARE_SCHEMA = config['config']['BQ_ENV_VARS']['SHARE_SCHEMA']
TBL_PREFIX   = config['config']['BQ_ENV_VARS']['PREFIX']
OWNER        = USER
COST_CENTER  = config['config']['BQ_ENV_VARS']['COST_CENTER']
BQDATABASE   = config['config']['BQ_ENV_VARS']['CM_MD_SCHEMA']
PREFIX       = config['config']['BQ_ENV_VARS']['PREFIX']
bq_dataset   = config['config']['BQ_ENV_VARS']['DATASET']
owner_name   = config['config']['BQ_ENV_VARS']['OWNER']
UNIQUE_ID    = config['config']['UNIQUE_ID']
LABELS_CALL  = config['config']['labels']['CREATE_TABLE_LABEL']
LABELS       = LABELS_CALL.format(OWNER=owner_name, COSTCENTER=COST_CENTER, UNIQUE_ID=UNIQUE_ID)
BQ_ENV_VARS['LABELS'] = LABELS

DAG_TAGS = [f"tenant:hcm-cm-de-hcb", "owner:navaneethakrishnanp@aetna.com", "model:medicaid-transformer-embeddings"] 

# Service accounts #
if ENV=="dev":
   env_id = 'ontpd'
elif ENV=="test":
   env_id = 'onppq'
elif ENV=="prod":
   env_id = 'onppp'

resource_sa = f"gchcb-hcm-cm-de-{env_id}@anbc-{ENV}-hcm-cm-de.iam.gserviceaccount.com"
decrypt_sa  = f"gchcb-hcm-cm-de-dec-{env_id}@anbc-{ENV}-hcm-cm-de.iam.gserviceaccount.com"
connect_sa  = f"hcm-cm-de-hcb-connect@anbc-{ENV}-hcm-cm-de.iam.gserviceaccount.com"

# Email Notification and Spanner Logging
sys.path.insert(0, f"{os.environ.get('DAGS_FOLDER')}/hcm-cm-de-hcb/common-spanner-util/")
from utils.configs import (import_config,
                           template_config,
                           dag_status_config,
                           spanner_config)
from utils.spanner_connector import SpannerConnector
from utils.utils import Logging
from utils import utils
from airflow.exceptions import AirflowException
import pendulum

def publish_email(context, status, color):
    to_email = ['cm_complex_care_solutions_de@cvshealth.com']
    cc_email = ['navaneethakrishnanp@aetna.com']
    PROJECT_ID = os.environ.get("GCP_PROJECT")
    ENV        = ''.join([x for x in ['dev', 'test', 'prod'] if x in PROJECT_ID])
    dag_run_id = context.get('dag_run')
    dag_id = context["task_instance"].dag_id
    task_id = context["task_instance"].task_id
    logs_url = context.get('task_instance').log_url
    date_time = str(pendulum.now("US/Eastern"))
    subject = "Composer Dag Status | Env: " + ENV + " | Model: MD Transformer Embeddings | Task id: " + task_id + " | Status : " + status
    email_body = """<tr>
    <p>Hi ,</p> 
    <p>Below are the details about the run : {dag_run_id}</p>
    <p>Dag_id   : {dag_id},</p> 
    <p>Task_id  : {task_id},</p> 
    <p>Status   : <b style="color:{color};">{status}</b>,</p>
    <p>Date_Time: {date_time},</p>
    <p>Log_URL  : {logs_url},</p>
    <p>              </p>
    <p>Thanks,\n     </p>
    <p>IO-OPS       </p>
    </tr>""".format(dag_run_id=dag_run_id, dag_id=dag_id, task_id=task_id, logs_url=logs_url, status=status,
                    color=color, date_time=date_time)
    try:
        send_email(to=to_email, cc=cc_email, subject=subject, html_content=email_body)
    except Exception as err:
        raise AirflowException("Error while sending mail: ", err)

def devops_spanner_update(context, status):
    spanner_connector = SpannerConnector()
    log = Logging()
    log.global_constant()

    date_time = str("\'" + str(pendulum.now("US/Eastern").to_datetime_string()) + "\'")
    run_id = pendulum.parse(str(pendulum.now("US/Eastern"))).int_timestamp
    dag_id = int(context["dag"].dag_id.split('-')[-1])
    airflow_dag_name = str("\'" + context["dag"].dag_id + "\'")
    airflow_task_name = str("\'" + context["task_instance"].task_id + "\'")
    start_time_utc = pendulum.parse(context["ts"], tz='UTC')
    start_time = str("\'" + str(pendulum.timezone('US/Eastern').convert(start_time_utc).to_datetime_string()) + "\'")

    if status == 'Failed':
        log = str("\'" + context.get('task_instance').log_url + "\'")
    else:
        log = str("\'" + 'NULL' + "\'")
    status = str("\'" + status + "\'")

    devops_dict = {
        'query': 'INSERT INTO clinical_analytics_dag_summary(run_id, dag_id, airflow_task_name, airflow_dag_name, start_time, update_timestamp, status, error_message) VALUES ({run_id}, {dag_id}, {airflow_task_name}, {airflow_dag_name}, {start_time}, {date_time}, {status}, {log})'}

    query = "".join(devops_dict['query'].format(
        run_id=run_id,
        dag_id=dag_id,
        airflow_task_name=airflow_task_name,
        airflow_dag_name=airflow_dag_name,
        start_time=start_time,
        date_time=date_time,
        status=status,
        log=log
    ))

    spanner_connector.execute_update_query(query)

def success_call(context):
    status = "Success"
    #devops_spanner_update(context, status)
    publish_email(context, status, "green")

def fail_call(context):
    status = "Failed"
    #devops_spanner_update(context, status)
    publish_email(context, status, "tomato")

def BQInsertJobFromFile(filename, service_account):
    jinja_query = "{% include '" + filename + "'%}"
    return BigQueryInsertJobOperator(
        task_id=os.path.basename(filename.split(".")[0]),
        impersonation_chain=service_account,
        configuration={"query": {"query": jinja_query, "useLegacySql": False}},
    )    

default_args = {
    'start_date': days_ago(1),
    'project_id': C_PROJECT_ID,
    'retries': 0,
    "email_on_failure": False,
    "depends_on_past": False,
    "email_on_retry": False,
    "on_success_callback": success_call,
    "on_failure_callback": fail_call,
}

# tags for DAG
with models.DAG(
        DAG_ID,
        schedule_interval=None,
        default_args=default_args,
        catchup=False,
        template_searchpath=os.path.join(os.environ.get('DAGS_FOLDER'), sql_folder_path),
        # this ensure that Airflow checks the files in Composer DAG bucket for SQL files
        user_defined_macros=BQ_ENV_VARS,
        tags=DAG_TAGS,
) as dag:  # Must specify your tenant name and owner email as tags

    pre_req             = BQInsertJobFromFile('000_Pre_Requisite.sql'                , resource_sa)
    ext_index_data      = BQInsertJobFromFile('001_Extract_Membership_Index.sql'     , resource_sa)
    embeddings_features = BQInsertJobFromFile('002_Generate_Embeddings_Features.sql' , resource_sa)

    trigger_te_dag = TriggerDagRunOperator(
        task_id="trigger_transformer_embeddings_dag",
        trigger_dag_id='hcm-cm-de-medicaid-transformer-embeddings-model',
        wait_for_completion=True,
        dag=dag
    ) 

    start_task = DummyOperator(task_id='start_task')
    end_task   = DummyOperator(task_id='end_task')

    start_task >> pre_req >> ext_index_data >> embeddings_features >> trigger_te_dag >> end_task