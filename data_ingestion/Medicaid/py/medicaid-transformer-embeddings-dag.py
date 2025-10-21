import os,sys
import json
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from airflow.utils import dates
from airflow.utils.dates import days_ago
from airflow import models
from airflow.models import DAG
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.email import send_email
from airflow.operators.dagrun_operator import TriggerDagRunOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.providers.google.cloud.operators.dataproc import (
    DataprocCreateClusterOperator,
    DataprocSubmitJobOperator,
    DataprocDeleteClusterOperator
)
from airflow.operators.bash import BashOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.providers.google.cloud.operators.bigquery import (
    BigQueryDeleteTableOperator,
    BigQueryExecuteQueryOperator,
)
from airflow.providers.google.cloud.transfers.bigquery_to_gcs import BigQueryToGCSOperator
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

md_te_config_path = os.path.join(DAG_PATH, md_te_env_config_path)

with open(md_te_config_path + f'{ENV}-{REPO_NAME}-config.json') as f:
     config = json.load(f)

print(config)

REGION       = config['config']['REGION']
TENANT       = config['config']['TENANT']
USER         = config['config']['USER']
spark_jars   = config['config']['bq_spark_jar']
CLUSTER_NAME = config['config']['DATAPROC_VARS']['CLUSTER_NAME']
MPP_GPUS     = config['config']['DATAPROC_VARS']['MPP_GPUS']
group_name   = config['config']['DATAPROC_VARS']['GROUP_NAME']
CODE_BUCKET  = config['config']['CODE_BUCKET']
DATA_BUCKET  = config['config']['DATA_BUCKET']

### End of get_config.py
#########################################################
DAG_ID = f"hcm-cm-de-medicaid-transformer-embeddings-model"

DAGS_FOLDER = os.environ["DAGS_FOLDER"]
CUR_DIR = os.getcwd()

# config assign
#################################
###     Bigquery Job Params    ##
#################################
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

DAG_TAGS = [f"tenant:hcm-cm-de-hcb", "owner:navaneethakrishnanp@aetna.com", "model:medicaid-transformer-embeddings"] 

GCP_ENV = PROJECT_ID.replace("anbc-", "")
SUBNET = f"https://www.googleapis.com/compute/v1/projects/insurance-vpc/regions/us-east4/subnetworks/sn-aa-use4-anbc-{ENV}-share"

# Service accounts #
resource_sa     = "gchcb-hcm-cm-de-ontpd@anbc-dev-hcm-cm-de.iam.gserviceaccount.com"
decrypt_sa      = "gchcb-hcm-cm-de-dec-ontpd@anbc-dev-hcm-cm-de.iam.gserviceaccount.com"
connect_sa      = "hcm-cm-de-hcb-connect@anbc-dev-hcm-cm-de.iam.gserviceaccount.com"

if ENV=="test":
    resource_sa = "gchcb-hcm-cm-de-onppq@anbc-test-hcm-cm-de.iam.gserviceaccount.com"
    decrypt_sa  = "gchcb-hcm-cm-de-dec-onppq@anbc-test-hcm-cm-de.iam.gserviceaccount.com"
    connect_sa  = "hcm-cm-de-hcb-connect@anbc-test-hcm-cm-de.iam.gserviceaccount.com"
elif ENV=="prod":
    resource_sa = "gchcb-hcm-cm-de-onppp@anbc-prod-hcm-cm-de.iam.gserviceaccount.com"
    decrypt_sa  = "gchcb-hcm-cm-de-dec-onppp@anbc-prod-hcm-cm-de.iam.gserviceaccount.com"
    connect_sa  = "hcm-cm-de-hcb-connect@anbc-prod-hcm-cm-de.iam.gserviceaccount.com"

# CMEK Key
if ENV=='prod':
    CMEK_KEY = f"projects/cvs-key-vault-prod/locations/us-east4/keyRings/gkr-prod-us-east4/cryptoKeys/gk-anbc-{ENV}-hcm-cm-de-us-east4"
    zscaler_tag ="zsccproduse4"
else:
    CMEK_KEY = f"projects/cvs-key-vault-nonprod/locations/us-east4/keyRings/gkr-nonprod-us-east4/cryptoKeys/gk-anbc-{ENV}-hcm-cm-de-us-east4"
    zscaler_tag = "zsccnpuse4"

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

#################################
###     Dataproc Job Params    ##
#################################
TAB_PREFIX = TBL_PREFIX
TRANSFORMER_INPUT = f"{SCHEMA}.{PREFIX}_O3_SCORE_ENDING"
py_script_path = f'gs://{CODE_BUCKET}/md-transformer-embeddings-model/'
py_script = f"{py_script_path}medicaid-asdb-claims-embedding_transformer_embeddings.py"
#################################

tgt_model_path = 'md-transformer-embeddings-model'
tgt_model_name = 'bestModel_singlegpu'

CLUSTER_TAG = 'mpp_gpu'
dataproc_resource_sa=decrypt_sa
dataproc_db=f"{SCHEMA}"
if ENV=='prod':
    dataproc_resource_sa=resource_sa
    dataproc_db=f"{SCHEMA}"
PYSPARK_FULLSCALE_JOB = {
    "placement": {
        "cluster_name": f'{CLUSTER_NAME}'
    },
    "pyspark_job": {
        "main_python_file_uri": py_script,
        "properties": {
            "spark.submit.deployMode": "client",
            "spark.jars": spark_jars,
        },
        "args": [f"--resource_sa={dataproc_resource_sa}", f"--bqdb={dataproc_db}", f"--tabprefix={TAB_PREFIX}", f"--envtype={ENV}", \
                 f"--owner={OWNER}", f"--costcenter={COST_CENTER}", f"--tenant={TENANT}", f"--tgt_bucket_name={CODE_BUCKET}", \
                 f"--tgt_model_path={tgt_model_path}", f"--tgt_model_name={tgt_model_name}", \
                 f"--clustertag={CLUSTER_TAG}", f"--mppgpus={MPP_GPUS}",\
                 f"--smpl_tbl={TRANSFORMER_INPUT}"],
                 "jar_file_uris": [spark_jars]
    }
}

###############################
#       cluster config        # 
###############################
CLUSTER_CONFIG = {
    "master_config": {
        "num_instances": 1,
        "machine_type_uri": "g2-standard-96",
        "disk_config": {"boot_disk_size_gb": 1024},
        "accelerators": [{
            "accelerator_type_uri": "nvidia-l4",
            "accelerator_count": 8,
        }],
    },
    "worker_config": {
        "num_instances": 2,
        "machine_type_uri": "g2-standard-16",
        "disk_config": {"boot_disk_size_gb": 1024},
        "accelerators": [{
            "accelerator_type_uri": "nvidia-l4",
            "accelerator_count": 1,
        }],
    },
    "secondary_worker_config": {
        "num_instances": 0,
        "machine_type_uri": "g2-standard-16",
        "disk_config": {"boot_disk_size_gb": 1024},
        "is_preemptible": False,
        "accelerators": [{
            "accelerator_type_uri": "nvidia-l4",
            "accelerator_count": 1,
        }],
    },
    "software_config": {
        "image_version": "2.2.6-ubuntu22",
        "properties": {
            "dataproc:efm.spark.shuffle": "primary-worker",
            "dataproc:efm.mapreduce.shuffle": "hcfs",
        },
    },
    "gce_cluster_config": {
        "zone_uri": "us-east4-c",
        "subnetwork_uri": SUBNET,  # specify subnet name
        "internal_ip_only": True,
        "tags": [f"dataproc-allow-internal-{ENV}-share", f"{zscaler_tag}"],
        "service_account": resource_sa,  # specify your tenant's resource SA.
        "service_account_scopes": [
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/bigquery"
        ],
        "metadata": {
            "PIP_PACKAGES": "torch==2.2.* xgboost scikit-learn scipy catboost==1.2.* joblib packaging spglm gcsfs google-cloud-logging",
            "rapids-runtime":"SPARK",
            "install-gpu-agent": "true",
        },
        "shielded_instance_config": {
            "enable_integrity_monitoring": False,
            "enable_vtpm": False,
            "enable_secure_boot": False,
        },
    },
    "endpoint_config": {"enable_http_port_access": True},
    "lifecycle_config": {
        "idle_delete_ttl":{"seconds": 3600 * 1}, # maximum 1 hour
        "auto_delete_ttl":{"seconds": 3600 * 8}, # maximum 18 hours
    },
    "encryption_config": {
        "gce_pd_kms_key_name": CMEK_KEY
    },
    "initialization_actions": [
        {
            "executable_file": f"gs://goog-dataproc-initialization-actions-us-east4/spark-rapids/spark-rapids.sh",
            "execution_timeout": {"seconds": 2700}
        },
        {
            "executable_file": "gs://goog-dataproc-initialization-actions-us-east4/python/pip-install.sh",
            "execution_timeout": {"seconds": 2700}
        },
    ]
}

#auto_scaling_policy = (
#    f"projects/{PROJECT_ID}/regions/us-east4/autoscalingPolicies/io-{GCP_ENV}"
#)

default_args_value = {
    "retries": 0,
    "project_id": C_PROJECT_ID,
    "email_on_failure": False,
    "depends_on_past": False,
    "email_on_retry": False,
    "on_success_callback": success_call,
    "on_failure_callback": fail_call,
    "retry_delay": timedelta(minutes=3)
}

SQL_CHECK_DATAPROC_IS_UP_JOB = {
    "placement": {
        "cluster_name": f'{CLUSTER_NAME}'
    },
    "spark_sql_job": {
        "query_list": {
            "queries": ["SELECT 'The Dataproc Cluster is up and running'"]
        },
        "properties": {
            "spark.submit.deployMode": "client"
        },
    }
}

with models.DAG(
        DAG_ID,
        tags=DAG_TAGS,
        default_args=default_args_value,
        start_date=days_ago(1),
        schedule_interval=None,
        #user_defined_macros=params,
        template_searchpath=os.path.join(os.environ.get('DAGS_FOLDER'), sql_folder_path),
        description="MedicaidTransformerEmbeddingsDAG"
) as dag:


    dataproc_create_mpp_gpu_cluster = DataprocCreateClusterOperator(
        task_id="dataproc_create_mpp_gpu_cluster",
        impersonation_chain=connect_sa,  # specify your tenant's connect SA, fully qualified name.
        cluster_name=CLUSTER_NAME,  # include your tenant name
        region=REGION,
        cluster_config=CLUSTER_CONFIG,
        retries=5,
        retry_delay=timedelta(minutes=5),
        use_if_exists=True,
        labels={
            "tenant": TENANT,
            "created-by": USER,
            "cost-center": COST_CENTER,
        },  
    )

    assign_permissions = BashOperator(
        task_id="assign_permissions",
        bash_command=f"bash {DAGS_FOLDER}/dataproc-set-iam.sh {CLUSTER_NAME} us-east4 group:{group_name}@cvshealth.com {C_PROJECT_ID}",
    )

    check_if_mpp_gpu_cluster_is_running_job = DataprocSubmitJobOperator(
        task_id="check_if_mpp_gpu_cluster_is_running_job",
        project_id=C_PROJECT_ID,
        impersonation_chain=connect_sa,
        region = 'us-east4',
        job=SQL_CHECK_DATAPROC_IS_UP_JOB,
    )

    md_transformer_gpu_scoring = DataprocSubmitJobOperator(
        task_id='md_transformer_gpu_scoring',
        project_id=C_PROJECT_ID,
        impersonation_chain=connect_sa,
        region = 'us-east4',
        job=PYSPARK_FULLSCALE_JOB,
    )
    
    delete_mpp_gpu_dp_cluster = DataprocDeleteClusterOperator(
        task_id="delete_mpp_gpu_dp_cluster",
        on_success_callback=success_call,
        project_id=C_PROJECT_ID,
        impersonation_chain=connect_sa,
        region = 'us-east4',
        cluster_name=f'{CLUSTER_NAME}',
        retries=1
    )
    
    start_task = DummyOperator(task_id='start')
    end_task   = DummyOperator(task_id='end')

    """
    trigger_te_features_dag = TriggerDagRunOperator(
        task_id="trigger_transformer_embeddings_features_dag",
        trigger_dag_id='hcm-cm-de-cm-mdcd-transformer-embeddings-features',
        dag=dag
    ) 
    """

start_task >> dataproc_create_mpp_gpu_cluster >> assign_permissions >> check_if_mpp_gpu_cluster_is_running_job >> md_transformer_gpu_scoring >> delete_mpp_gpu_dp_cluster >> end_task