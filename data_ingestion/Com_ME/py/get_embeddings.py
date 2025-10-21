import os, sys, re
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from airflow.utils.dates import days_ago
from airflow.models import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.providers.google.cloud.operators.dataproc import (
    DataprocCreateClusterOperator,
    DataprocSubmitJobOperator,
    DataprocDeleteClusterOperator
)
from airflow.operators.bash_operator import BashOperator
import calling_config as ct
import email_context as ec
from datetime import datetime, timedelta

# constant vars
TENANT = "clin-analytics-hcb"
REPO_NAME = "transformer-embeddings"
PROJECT_ID = os.environ.get("GCP_PROJECT")
DAG_PATH = os.environ.get('DAGS_FOLDER')
ENV = PROJECT_ID.split('-')[-1]

connect_sa = f"clin-analytics-hcb-connect@{PROJECT_ID}.iam.gserviceaccount.com"
resource_sa = f"gchcb-clin-analytics-ontpd@{PROJECT_ID}.iam.gserviceaccount.com"
dataproc_resource_sa="gchcb-clin-analytics-d-ontpd@anbc-hcb-test.iam.gserviceaccount.com"
decrypt_sa = "gchcb-clin-analytics-d-ontpd@anbc-hcb-dev.iam.gserviceaccount.com"
CMEK_KEY = f"projects/cvs-key-vault-nonprod/locations/us-east4/keyRings/gkr-nonprod-us-east4/cryptoKeys/gk-{PROJECT_ID}-us-east4"
group_name = "gchcb-clin-analytics-ontpd"
if (f"{ENV}" == "test"):
    resource_sa = "gchcb-clin-analytics-onppq@anbc-hcb-test.iam.gserviceaccount.com"
    connect_sa = "clin-analytics-hcb-connect@anbc-hcb-test.iam.gserviceaccount.com"
    dataproc_resource_sa="gchcb-clin-analytics-d-onppq@anbc-hcb-test.iam.gserviceaccount.com"
    decrypt_sa = "gchcb-clin-analytics-d-onppq@anbc-hcb-test.iam.gserviceaccount.com"
    CMEK_KEY = f"projects/cvs-key-vault-nonprod/locations/us-east4/keyRings/gkr-nonprod-us-east4/cryptoKeys/gk-{PROJECT_ID}-us-east4"
    group_name = "gchcb-clin-analytics-ontpqs"
if (f"{ENV}" == "prod"):
    resource_sa = "gchcb-clin-analytics-onppp@anbc-hcb-prod.iam.gserviceaccount.com"
    dataproc_resource_sa="gchcb-clin-analytics-onppp@anbc-hcb-test.iam.gserviceaccount.com"
    connect_sa = "clin-analytics-hcb-connect@anbc-hcb-prod.iam.gserviceaccount.com"
    decrypt_sa = "gchcb-clin-analytics-d-onppp@anbc-hcb-prod.iam.gserviceaccount.com"
    CMEK_KEY = f"projects/cvs-key-vault-prod/locations/us-east4/keyRings/gkr-prod-us-east4/cryptoKeys/gk-{PROJECT_ID}-us-east4"
    group_name = "gchcb-clin-analytics-ontpp"
    
REGION = ct.config['config']['REGION']
USER = ct.config['config']['USER']
DAG_ID = f"{TENANT}-{REPO_NAME}-get"
COSTCENTER=ct.config['config']['COSTCENTER']
## DATASETS
DATASET = ct.config['config']['DATASET'].format(ENV=ENV)
DEC_DATASET = ct.config['config']['DEC_DATASET'].format(ENV=ENV)
# SHARE_BQDB = ct.config['config']['SHARE_BQDB'].format(ENV=ENV)
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
#################################
###     Dataproc Job Params    ##
#################################
GCP_ENV = PROJECT_ID.replace("anbc-", "")
SUBNET = f"io-hcb-{ENV}-vm"
CODE_BUCKET = ct.config['config']['CODE_BUCKET'].format(ENV=ENV)
DATA_BUCKET = ct.config['config']['DATA_BUCKET'].format(ENV=ENV)
spark_jars = ct.config['config']['bq_spark_jar']
CLUSTER_NAME = ct.config['config']['cluster_name']
MPP_GPUS = ct.config['config']['MPP_GPUS']
# TAB_PREFIX = ct.config['config']['prefix']
py_script_path = f'gs://{CODE_BUCKET}/{REPO_NAME}'
py_script = f"{py_script_path}/transformer_dataproc.py"
# tgt_bucket_name = (f"clin-analytics-code-hcb-{ENV}" if ENV=="dev" else "clin-analytics-shared-hcb-prod")
tgt_model_path = ct.config['config']['model_folder']
tgt_model_name = 'bestModel_singlegpu'
CLUSTER_TAG = 'mpp_gpu'

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
        "args": [f"--resource_sa={dataproc_resource_sa}", f"--bqdb={FINAL_DATASET}", f"--tabprefix={{ params.prefix }}", f"--envtype={ENV}", \
                 f"--owner={USER}", f"--costcenter={COSTCENTER}", f"--tenant={TENANT}", f"--tgt_bucket_name={DATA_BUCKET}", \
                 f"--tgt_model_path={tgt_model_path}", f"--tgt_model_name={tgt_model_name}", \
                 f"--clustertag={CLUSTER_TAG}", f"--mppgpus={MPP_GPUS}",\
                 f"--smpl_tbl={{ params.input_table }}"],
                 "jar_file_uris": [spark_jars]
    }
}

###############################
    # cluster config 
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
    # "worker_config": {
    #     "num_instances": 2,
    #     "machine_type_uri": "g2-standard-16",
    #     "disk_config": {"boot_disk_size_gb": 512},
    #     "accelerators": [{
    #         "accelerator_type_uri": "nvidia-l4",
    #         "accelerator_count": 1,
    #     }],
    # },
    # "secondary_worker_config": {
    #     "num_instances": 0,
    #     "machine_type_uri": "g2-standard-16",
    #     "disk_config": {"boot_disk_size_gb": 1024},
    #     "is_preemptible": False,
    #     "accelerators": [{
    #         "accelerator_type_uri": "nvidia-l4",
    #         "accelerator_count": 1,
    #     }],
    # },
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
        "tags": ["io-dataproc"],
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
    "config_bucket": f"clin-analytics-data-hcb-{ENV}",  # specify your tenant's data bucket, example: REPLACE_ME-data-lab
    "autoscaling_config": {
        "policy_uri": f"projects/{PROJECT_ID}/regions/us-east4/autoscalingPolicies/io-{GCP_ENV}"
    },  # specify autoscaling policy uri
    "endpoint_config": {"enable_http_port_access": True},
    "lifecycle_config": {
        "idle_delete_ttl":{"seconds": 3600 * 1}, # maximum 1 hour
        "auto_delete_ttl":{"seconds": 3600 * 8}, # maximum 8 hours
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

auto_scaling_policy = (
    f"projects/{PROJECT_ID}/regions/us-east4/autoscalingPolicies/io-{GCP_ENV}"
)
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
#     "on_success_callback": success_call,
    "on_failure_callback": fail_call
}
dag_defaults = {}
default_args.update(dag_defaults)
  
# tags for DAG
with DAG(
        DAG_ID,
        start_date = days_ago(7),
        ## dev run daily
        schedule_interval=None,
        access_control={'clin-analytics-hcb': {'can_read'}},
        max_active_runs = 1,
        tags=ct.DAG_TAGS,
        default_args=default_args,
        catchup=False,
        template_searchpath=os.path.join(DAG_PATH, sql_folder_path),
        # this ensure that Airflow checks the files in Composer DAG bucket for SQL files
        user_defined_macros=params,
        params={"input_table":"",
                "emb_init":0,
                "prefix":"",
                "INPUT_DATASET":FINAL_DATASET,
                "FINAL_DATASET":FINAL_DATASET
                               }
    
) as dag:  # Must specify your tenant name and owner email as tags
    
    dataproc_create_cluster = DataprocCreateClusterOperator(
        task_id="dataproc_create_cluster",
        impersonation_chain=connect_sa,  # specify your tenant's connect SA, fully qualified name.
        cluster_name=CLUSTER_NAME,  # include your tenant name
        region=REGION,
        cluster_config=CLUSTER_CONFIG,
        retries=5,
        retry_delay=timedelta(minutes=5),
        #cluster_config=CLUSTER_CONFIG,
        use_if_exists=True,
        labels={
            "tenant": TENANT,
            "created-by": USER,
            "cost-center": COSTCENTER,
        },  
    )

    assign_permissions = BashOperator(
        task_id="assign_permissions",
        bash_command=f"bash {DAG_PATH}/dataproc-set-iam.sh {CLUSTER_NAME} us-east4 group:{group_name}@cvshealth.com",
    )

    # check_if_cluster_is_running_job = DataprocSubmitJobOperator(
    #     task_id="check_if_cluster_is_running_job",
    #     project_id=PROJECT_ID,
    #     impersonation_chain=connect_sa,
    #     region = 'us-east4',
    #     job=SQL_CHECK_DATAPROC_IS_UP_JOB,
    # )


    transformer_gpu_scoring = DataprocSubmitJobOperator(
        task_id='transformer_gpu_scoring',
        project_id=PROJECT_ID,
        impersonation_chain=connect_sa,
        region = 'us-east4',
        job=PYSPARK_FULLSCALE_JOB,
        params={"input_table": "{{ dag_run.conf['input_table'] }}",
                "prefix": "{{ dag_run.conf['prefix'] }}"
                }
    )
    
    delete_cluster = DataprocDeleteClusterOperator(
        task_id="delete_cluster",
        project_id=PROJECT_ID,
        impersonation_chain=connect_sa,
        region = 'us-east4',
        cluster_name=f'{CLUSTER_NAME}',
        retries=1
    )

    
    history = BigQueryInsertJobOperator(
            task_id='emb_history',
            impersonation_chain=dataproc_resource_sa,  # SA Airflow uses to impersonate while interacting with BQ
            configuration={
                'labels': {
                    'owner': owner_name,
                    'dag-name': DAG_ID,  # dag name
                    'task-name': 'emb_history',  # task name same as task_id
                    'sequence': '1'  # sequence to define the order
                },
                'query': {
                    'query': "{% include 'emb_history.sql'%}",
                    'useLegacySql': False,  # uses standard SQL as dialect.
                    "queryParameters": [
                     {"parameterType": {"type": "INTEGER"},
                      "parameterValue": {"value": '{{ dag_run.conf["emb_init"] if dag_run.conf.get("emb_init") else 0 }}'},
                      "name": "emb_init"
                    }
                ]
                }
            }
        )
    

dataproc_create_cluster >> assign_permissions >> transformer_gpu_scoring >> delete_cluster
transformer_gpu_scoring >> history