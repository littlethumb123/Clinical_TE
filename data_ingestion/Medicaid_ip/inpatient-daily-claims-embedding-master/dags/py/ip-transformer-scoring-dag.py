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
TENANT = "hcm-cm-de"
USER = "zhongc"
PROJECT_ID = os.environ.get("GCP_PROJECT")
ENV = PROJECT_ID.split('-')[-1]
COMPUTE_PROJECT_ID = f'anbc-{ENV}-hcm-cm-de'
DAG_PATH = os.environ.get('DAGS_FOLDER')
REPO_NAME = "inpatient-daily-claims-embedding"

cp_ip_env_config_path = f'{TENANT}-hcb/{REPO_NAME}/config/'
sql_relative_script_path = f'{TENANT}-hcb/{REPO_NAME}/sql/'


cp_ip_config_path = os.path.join(DAG_PATH,  cp_ip_env_config_path)
#airflow_config_path = os.path.join(DAG_PATH,  airflow_env_config_path)
sql_script_path = os.path.join(DAG_PATH,  sql_relative_script_path)


with open(cp_ip_config_path + f'ip_config.json') as f:
    config = json.load(f)

print(config)

REGION = config['config']['REGION']
USER_EMAIL = f"{USER}"+f"@aetna.com"
#DAG_POSTFIX_SCORING = config['config']['DAG_POSTFIX_SCORING']
spark_jars = config['config']['bq_spark_jar']
CLUSTER_NAME = config['config']['mpp_gpu_cluster_name']


CODE_BUCKET = config['config']['CODE_BUCKET'].format(ENV=ENV)
DATA_BUCKET = config['config']['DATA_BUCKET'].format(ENV=ENV)

### End of get_config.py
#########################################################
DAG_ID = f"hcm-cm-de-{REPO_NAME}"

DAGS_FOLDER = os.environ["DAGS_FOLDER"]
CUR_DIR = os.getcwd()

# config assign
#################################
###     Bigquery Job Params    ##
#################################

DATASET = config['config']['bq_dataset']['DATASET'].format(ENV=ENV)
DEC_DATASET = config['config']['bq_dataset']['DEC_DATASET'].format(ENV=ENV)
SHARE_DATASET = config['config']['bq_dataset']['SHARE_DATASET'].format(ENV=ENV)
TBL_PREFIX = config['config']['bq_variable']['TABLE_PREFIX']
OWNER = 'zhongc'
COST_CENTER = config['config']['bq_dataset']['COSTCENTER']
TARGET_DB = f"{PROJECT_ID}.{DATASET}"
PREFIX = config['config']['bq_dataset']['PREFIX']
bq_dataset = config['config']['bq_dataset']
bq_dataset = {x:bq_dataset[x].format(ENV=ENV) if isinstance(bq_dataset[x], str) else bq_dataset[x] for x in bq_dataset}

params = bq_dataset
RUN_DATE = date.today()
V_RUN_DT = RUN_DATE.strftime("%Y-%m-%d")
params["current_dt"]=V_RUN_DT
params['COSTCENTER'] = COST_CENTER
params['TARGET_DB'] = TARGET_DB
params['DEC_TARGET_DB'] = f"{PROJECT_ID}.{DEC_DATASET}"
params['PREFIX']=PREFIX
params['FINAL_DB']=(TARGET_DB if ENV=="prod" else f"{PROJECT_ID}.{DEC_DATASET}")

DAG_TAGS = [f"tenant: hcm-cm-de", f"owner: {OWNER}@aetna.com"] # TODO: parametrize

GCP_ENV = PROJECT_ID.replace("anbc-", "")
email_list = list([f"{USER_EMAIL}"])

connect_sa = f"hcm-cm-de-hcb-connect@anbc-{ENV}-hcm-cm-de.iam.gserviceaccount.com"
resource_sa = "gchcb-hcm-cm-de-ontpd@anbc-dev-hcm-cm-de.iam.gserviceaccount.com"
decrypt_sa = "gchcb-hcm-cm-de-dec-ontpd@anbc-dev-hcm-cm-de.iam.gserviceaccount.com"
dataproc_resource_sa = "gchcb-hcm-cm-de-dec-ontpd@anbc-dev-hcm-cm-de.iam.gserviceaccount.com"
CMEK_KEY = f"projects/cvs-key-vault-nonprod/locations/us-east4/keyRings/gkr-nonprod-us-east4/cryptoKeys/gk-{COMPUTE_PROJECT_ID}-us-east4"
group_name = "gchcb-hcm-cm-de-ontpd"
if (f"{ENV}" == "test"):
    resource_sa = "gchcb-hcm-cm-de-onppq@anbc-test-hcm-cm-de.iam.gserviceaccount.com"
    decrypt_sa = "gchcb-hcm-cm-de-dec-onppq@anbc-test-hcm-cm-de.iam.gserviceaccount.com"
    dataproc_resource_sa = "gchcb-hcm-cm-de-dec-onppq@anbc-test-hcm-cm-de.iam.gserviceaccount.com"
    group_name = "gchcb-hcm-cm-de-ontpq"
if (f"{ENV}" == "prod"):
    resource_sa = "gchcb-hcm-cm-de-onppp@anbc-prod-hcm-cm-de.iam.gserviceaccount.com"
    decrypt_sa = "gchcb-hcm-cm-de-dec-onppp@anbc-prod-hcm-cm-de.iam.gserviceaccount.com"
    dataproc_resource_sa = "gchcb-hcm-cm-de-onppp@anbc-prod-hcm-cm-de.iam.gserviceaccount.com"
    CMEK_KEY = f"projects/cvs-key-vault-prod/locations/us-east4/keyRings/gkr-prod-us-east4/cryptoKeys/gk-{COMPUTE_PROJECT_ID}-us-east4"
    group_name = "gchcb-hcm-cm-de-ontpp"


###############################
    # cluster config 
###############################
TAB_PREFIX = config['config']['bq_dataset']['EMB_PREFIX']
TRANSFORMER_INPUT = f"{params['FINAL_DB']}.{PREFIX}_o3_score_ending_tmp"
network_tag=(config['config']['prod_network_tag'] if ENV=="prod" else config['config'][f'network_tag'])
MPP_GPUS=8
CLUSTER_CONFIG = {
    "master_config": {
        "num_instances": 1,
        "machine_type_uri": "g2-standard-96",
        "disk_config": {"boot_disk_size_gb": 2048},
        "accelerators": [{
            "accelerator_type_uri": "nvidia-l4",
            "accelerator_count": 8,
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
        "subnetwork_uri": f"https://www.googleapis.com/compute/v1/projects/insurance-vpc/regions/us-east4/subnetworks/sn-aa-use4-anbc-{ENV}-share",  # specify subnet name
        "internal_ip_only": True,
        "tags": [f"dataproc-allow-internal-{ENV}-share",f"{network_tag}"],
        "service_account": dataproc_resource_sa,  # specify your tenant's resource SA.
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

#################################
###     Dataproc Job Params    ##
#################################
py_script_path = f'gs://{CODE_BUCKET}/{REPO_NAME}/'
py_script = f"{py_script_path}transformer_embeddings_5_ip.py"
#################################
tgt_model_path = f'{REPO_NAME}'
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
        "args": [f"--resource_sa={dataproc_resource_sa}", f"--bqdb={params['FINAL_DB']}", f"--tabprefix={TAB_PREFIX}", f"--envtype={ENV}", \
                 f"--owner={OWNER}", f"--costcenter={COST_CENTER}", f"--tenant={TENANT}", f"--tgt_bucket_name={CODE_BUCKET}", \
                 f"--tgt_model_path={tgt_model_path}", f"--tgt_model_name={tgt_model_name}", \
                 f"--clustertag={CLUSTER_TAG}", f"--mppgpus={MPP_GPUS}",\
                 f"--smpl_tbl={TRANSFORMER_INPUT}"],
                 "jar_file_uris": [spark_jars]
    }
}

def custom_failure_notification(context):
    dag_run = context.get('dag_run')
    task_instances = dag_run.get_task_instances()
    msg = "DAG job run has failed. Please, check a log."
    subject = str(GCP_ENV).upper()  +" : DAG " + str(task_instances) + " - Failed."
    send_email(to=email_list,subject=subject, html_content=msg)
    # this function check

def custom_success_notification(context):
    dag_run = context.get('dag_run')
    task_instances = dag_run.get_task_instances()
    msg = f"{DAG_ID} DAG job run has completed."
    subject = str(GCP_ENV).upper()  +" : DAG " + str(task_instances) + " - Succeeded."
    send_email(to=email_list,subject=subject, html_content=msg)
    # this function check

default_args_value = {
    "retries": 0,
    "project_id": COMPUTE_PROJECT_ID,
    "email": ["zhongc@aetna.com"],
    "email_on_failure": True,
    "email_on_retry": True,
    "on_failure_callback": custom_failure_notification,
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
        start_date=days_ago(7),
        schedule_interval=("30 4 * * 5" if ENV=='prod' else None),
        user_defined_macros=params,
        template_searchpath=os.path.join(os.environ.get('DAGS_FOLDER'), sql_script_path),
        description="CPIPTransformerScoringFullPopulationDAG"
) as dag:

    def greeting():
        import logging
        logging.info('Hello World!')

    get_members = BigQueryInsertJobOperator(
        task_id='get_members',
        impersonation_chain=resource_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': OWNER,
                'dag-name': DAG_ID,  # dag name
                'task-name': 'member_table',  # task name same as task_id
                'sequence': '1'  # sequence to define the order
            },
            'query': {
                'query': "{% include '01_member.sql'%}",
                'useLegacySql': False  # uses standard SQL as dialect.
            }
        }
    )
    get_edw_clms = BigQueryInsertJobOperator(
        task_id='get_edw_clms',
        impersonation_chain=resource_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': OWNER,
                'dag-name': DAG_ID,  # dag name
                'task-name': 'get_edw_clms',  # task name same as task_id
                'sequence': '1'  # sequence to define the order
            },
            'query': {
                'query': "{% include '02_a_edw_claims.sql'%}",
                'useLegacySql': False  # uses standard SQL as dialect.
            }
        }
    )

    insights_clms_dtl = BigQueryInsertJobOperator(
        task_id='insights_clms_dtl',
        impersonation_chain=resource_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': OWNER,
                'dag-name': DAG_ID,  # dag name
                'task-name': 'insights_clms_dtl',  # task name same as task_id
                'sequence': '1'  # sequence to define the order
            },
            'query': {
                'query': "{% include '02_b_insights_clms_dtl.sql'%}",
                'useLegacySql': False  # uses standard SQL as dialect.
            }
        }
    )

    insights_clms_hdr = BigQueryInsertJobOperator(
        task_id='insights_clms_hdr',
        impersonation_chain=resource_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': OWNER,
                'dag-name': DAG_ID,  # dag name
                'task-name': 'insights_clms_hdr',  # task name same as task_id
                'sequence': '1'  # sequence to define the order
            },
            'query': {
                'query': "{% include '02_b_insights_clms_hdr.sql'%}",
                'useLegacySql': False  # uses standard SQL as dialect.
            }
        }
    )

    insights_clms = BigQueryInsertJobOperator(
        task_id='insights_clms',
        impersonation_chain=resource_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': OWNER,
                'dag-name': DAG_ID,  # dag name
                'task-name': '02_b_insights_clms',  # task name same as task_id
                'sequence': '1'  # sequence to define the order
            },
            'query': {
                'query': "{% include '02_b_insights_clms.sql'%}",
                'useLegacySql': False  # uses standard SQL as dialect.
            }
        }
    )

    rx_cur_archive_dec = BigQueryInsertJobOperator(
        task_id='rx_cur_archive_dec',
        impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': OWNER,
                'dag-name': DAG_ID,  # dag name
                'task-name': 'rx_cur_archive_dec',  # task name same as task_id
                'sequence': '1'  # sequence to define the order
            },
            'query': {
                'query': "{% include '02_c_rx_cur_archive_dec.sql'%}",
                'useLegacySql': False  # uses standard SQL as dialect.
            }
        }
    )

    edw_archive_clms_dec = BigQueryInsertJobOperator(
        task_id='edw_archive_clms_dec',
        impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': OWNER,
                'dag-name': DAG_ID,  # dag name
                'task-name': 'edw_archive_clms_dec',  # task name same as task_id
                'sequence': '1'  # sequence to define the order
            },
            'query': {
                'query': "{% include '02_d_edw_archive_clms_dec.sql'%}",
                'useLegacySql': False  # uses standard SQL as dialect.
            }
        }
    )
    
    combine_edw_insights_dec = BigQueryInsertJobOperator(
        task_id='combine_edw_insights_dec',
        impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': OWNER,
                'dag-name': DAG_ID,  # dag name
                'task-name': 'combine_edw_insights_dec',  # task name same as task_id
                'sequence': '1'  # sequence to define the order
            },
            'query': {
                'query': "{% include '03_combine_edw_insights_dec.sql'%}",
                'useLegacySql': False  # uses standard SQL as dialect.
            }
        }
    )
   
    combine_archive_dec = BigQueryInsertJobOperator(
        task_id='combine_archive_dec',
        impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': OWNER,
                'dag-name': DAG_ID,  # dag name
                'task-name': 'combine_archive_dec',  # task name same as task_id
                'sequence': '1'  # sequence to define the order
            },
            'query': {
                'query': "{% include '04_combine_archive_dec.sql'%}",
                'useLegacySql': False  # uses standard SQL as dialect.
            }
        }
    )

    add_icd_dec = BigQueryInsertJobOperator(
        task_id='add_icd_dec',
        impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': OWNER,
                'dag-name': DAG_ID,  # dag name
                'task-name': 'add_icd_dec',  # task name same as task_id
                'sequence': '1'  # sequence to define the order
            },
            'query': {
                'query': "{% include '05_a_add_icd_dec.sql'%}",
                'useLegacySql': False  # uses standard SQL as dialect.
            }
        }
    )

    add_icd_archive_dec = BigQueryInsertJobOperator(
        task_id='add_icd_archive_dec',
        impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': OWNER,
                'dag-name': DAG_ID,  # dag name
                'task-name': 'add_icd_archive_dec',  # task name same as task_id
                'sequence': '1'  # sequence to define the order
            },
            'query': {
                'query': "{% include '05_b_add_icd_archive_dec.sql'%}",
                'useLegacySql': False  # uses standard SQL as dialect.
            }
        }
    )

    icd_archive_combine_dec = BigQueryInsertJobOperator(
        task_id='icd_archive_combine_dec',
        impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': OWNER,
                'dag-name': DAG_ID,  # dag name
                'task-name': 'icd_archive_combine_dec',  # task name same as task_id
                'sequence': '1'  # sequence to define the order
            },
            'query': {
                'query': "{% include '06_icd_archive_combine_dec.sql'%}",
                'useLegacySql': False  # uses standard SQL as dialect.
            }
        }
    )

    prep_transformer_a_dec = BigQueryInsertJobOperator(
        task_id='prep_transformer_a_dec',
        impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': OWNER,
                'dag-name': DAG_ID,  # dag name
                'task-name': 'prep_transformer_a_dec',  # task name same as task_id
                'sequence': '1'  # sequence to define the order
            },
            'query': {
                'query': "{% include '07_prep_transformer_a_dec.sql'%}",
                'useLegacySql': False  # uses standard SQL as dialect.
            }
        }
    )
    dataproc_create_mpp_gpu_cluster = DataprocCreateClusterOperator(
        task_id="dataproc_create_mpp_gpu_cluster",
        impersonation_chain=connect_sa,  # specify your tenant's connect SA, fully qualified name.
        cluster_name=CLUSTER_NAME,  # include your tenant name
        region=REGION,
        cluster_config=CLUSTER_CONFIG,
        retries=3,
        use_if_exists=True,
        retry_delay=timedelta(minutes=10),
        #cluster_config=CLUSTER_CONFIG,
        labels={
            "tenant": TENANT,
            "created-by": USER,
            "cost-center": COST_CENTER,
        },  
    )

    assign_permissions = BashOperator(
        task_id="assign_permissions",
        bash_command=f"bash {DAGS_FOLDER}/dataproc-set-iam.sh {CLUSTER_NAME} us-east4 group:{group_name}@cvshealth.com {COMPUTE_PROJECT_ID}",
    )

    check_if_mpp_gpu_cluster_is_running_job = DataprocSubmitJobOperator(
        task_id="check_if_mpp_gpu_cluster_is_running_job",
        project_id=COMPUTE_PROJECT_ID,
        impersonation_chain=connect_sa,
        region = 'us-east4',
        job=SQL_CHECK_DATAPROC_IS_UP_JOB,
    )


    cp_ip_transformer_gpu_scoring_full_population = DataprocSubmitJobOperator(
        task_id='cp_ip_transformer_gpu_scoring_full_population',
        project_id=COMPUTE_PROJECT_ID,
        impersonation_chain=connect_sa,
        region = 'us-east4',
        job=PYSPARK_FULLSCALE_JOB,
    )
    
    delete_mpp_gpu_dp_cluster = DataprocDeleteClusterOperator(
        task_id="delete_mpp_gpu_dp_cluster",
        on_success_callback=custom_success_notification,
        project_id=COMPUTE_PROJECT_ID,
        impersonation_chain=connect_sa,
        region = 'us-east4',
        cluster_name=f'{CLUSTER_NAME}',
        trigger_rule="all_done",
        retries=1
    )

    emb_history = BigQueryInsertJobOperator(
        task_id='embs_history',
        impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': OWNER,
                'dag-name': DAG_ID,  # dag name
                'task-name': 'embs_history',  # task name same as task_id
                'sequence': '1'  # sequence to define the order
            },
            'query': {
                'query': "{% include 'embs_history.sql'%}",
                'useLegacySql': False  # uses standard SQL as dialect.
            }
        }
    )

    trigger_me_model_dag = TriggerDagRunOperator(
    task_id="trigger_mojo_dag",
    trigger_dag_id='hcm-cm-de-weekly-inpatient-me',
    dag=dag
    )



get_members >> [insights_clms_dtl ,insights_clms_hdr] >> insights_clms >> combine_edw_insights_dec
get_members >> get_edw_clms >> combine_edw_insights_dec
get_members >> rx_cur_archive_dec >> combine_edw_insights_dec
get_members >> edw_archive_clms_dec >> combine_edw_insights_dec
combine_edw_insights_dec >> combine_archive_dec >> add_icd_dec >> icd_archive_combine_dec
combine_edw_insights_dec >> combine_archive_dec >> add_icd_archive_dec >> icd_archive_combine_dec
icd_archive_combine_dec >> prep_transformer_a_dec >> dataproc_create_mpp_gpu_cluster
dataproc_create_mpp_gpu_cluster >> assign_permissions >> check_if_mpp_gpu_cluster_is_running_job >> cp_ip_transformer_gpu_scoring_full_population >> [emb_history, delete_mpp_gpu_dp_cluster, trigger_me_model_dag]