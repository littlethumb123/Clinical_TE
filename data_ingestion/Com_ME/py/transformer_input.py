import os, sys, re
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from airflow.utils.dates import days_ago
from airflow.models import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from  airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator,PythonVirtualenvOperator

import calling_config as ct
import email_context as ec
from datetime import datetime, timedelta


# constant vars
TENANT = "hcm-cm-de"
REPO_NAME = "transformer-embeddings-v2"
PROJECT_ID = os.environ.get("GCP_PROJECT")
ENV = PROJECT_ID.split('-')[-1]
COMPUTE_PROJECT_ID = f'anbc-{ENV}-hcm-cm-de'
DAG_PATH = os.environ.get('DAGS_FOLDER')

connect_sa = f"hcm-cm-de-hcb-connect@anbc-{ENV}-hcm-cm-de.iam.gserviceaccount.com"
resource_sa = "gchcb-hcm-cm-de-ontpd@anbc-dev-hcm-cm-de.iam.gserviceaccount.com"
decrypt_sa = "gchcb-hcm-cm-de-dec-ontpd@anbc-dev-hcm-cm-de.iam.gserviceaccount.com"
CMEK_KEY = f"projects/cvs-key-vault-nonprod/locations/us-east4/keyRings/gkr-nonprod-us-east4/cryptoKeys/gk-{COMPUTE_PROJECT_ID}-us-east4"
group_name = "gchcb-hcm-cm-de-ontpd"
if (f"{ENV}" == "test"):
    resource_sa = "gchcb-hcm-cm-de-onppq@anbc-test-hcm-cm-de.iam.gserviceaccount.com"
    decrypt_sa = "gchcb-hcm-cm-de-dec-onppq@anbc-test-hcm-cm-de.iam.gserviceaccount.com"
    group_name = "gchcb-hcm-cm-de-ontpq"
if (f"{ENV}" == "prod"):
    resource_sa = "gchcb-hcm-cm-de-onppp@anbc-prod-hcm-cm-de.iam.gserviceaccount.com"
    decrypt_sa = "gchcb-hcm-cm-de-dec-onppp@anbc-prod-hcm-cm-de.iam.gserviceaccount.com"
    CMEK_KEY = f"projects/cvs-key-vault-prod/locations/us-east4/keyRings/gkr-prod-us-east4/cryptoKeys/gk-{COMPUTE_PROJECT_ID}-us-east4"
    group_name = "gchcb-hcm-cm-de-ontpp"
REGION = ct.config['config']['REGION']
USER = ct.config['config']['USER']
DAG_ID = f"{TENANT}-{REPO_NAME}-input"
## DATASETS
DATASET = PROJECT_ID+"."+ct.config['config']['DATASET'].format(ENV=ENV)
DEC_DATASET = PROJECT_ID+"."+ct.config['config']['DEC_DATASET'].format(ENV=ENV)
SHARE_BQDB = ct.config['config']['bq']['SHARE_BQDB'].format(ENV=ENV)
FINAL_DATASET = (DATASET if ENV=="prod" else DEC_DATASET)

## SQL PARAMS
sql_folder_path = f"{TENANT}-hcb/{REPO_NAME}/sql/"
owner_name = f"{USER}_aetna_com"
databases = ct.config['config']['bq']
# databases["DATASET"] = DATASET
# databases["DEC_DATASET"] = DEC_DATASET
databases["SHARE_BQDB"] = SHARE_BQDB
# databases["FINAL_DATASET"]=(databases["DATASET"] if ENV=="prod" else databases["DEC_DATASET"])
params = databases
params['OWNER'] = owner_name
params['project_name'] = PROJECT_ID
params['DATASET'] = DATASET
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
    'project_id': COMPUTE_PROJECT_ID,
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
        schedule_interval=None,
        access_control={'hcm-cm-de-hcb': {'can_read'}},
        max_active_runs = 1,
        tags=ct.DAG_TAGS,
        default_args=default_args,
        catchup=False,
        template_searchpath=os.path.join(DAG_PATH, sql_folder_path),
        # this ensure that Airflow checks the files in Composer DAG bucket for SQL files
        user_defined_macros=params,
        params={"te_base": "",
                "emb_init":0,
                "costcenter": f"{ct.config['config']['COSTCENTER']}",
                "owner": owner_name,
                "DEC_DATASET": DEC_DATASET,
                "FINAL_DATASET": FINAL_DATASET,
                "prefix": "",
                "historydays":0,
                "recipient":[]
               }
    
) as dag:  # Must specify your tenant name and owner email as tags
    
    te_attach_member_id = BigQueryInsertJobOperator(
            task_id='0_te_attach_member_id',
            impersonation_chain=resource_sa,  # SA Airflow uses to impersonate while interacting with BQ
            configuration={
                'labels': {
                    'owner': owner_name,
                    'dag-name': DAG_ID,  # dag name
                    'task-name': '0_te_attach_member_id',  # task name same as task_id
                    'sequence': '1'  # sequence to define the order
                },
                'query': {
                    'query': "{% include '0_te_attach_member_id.sql'%}",
                    'useLegacySql': False
                },
                
            },
            params={"owner": "{{ dag_run.conf['owner'] }}",
                     "costcenter": "{{ dag_run.conf['costcenter'] }}"
                    }
        )
    
    dly_clm_dec = BigQueryInsertJobOperator(
            task_id='1_a_dly_clm_dec',
            impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
            configuration={
                'labels': {
                    'owner': owner_name,
                    'dag-name': DAG_ID,  # dag name
                    'task-name': '1_a_dly_clm_dec',  # task name same as task_id
                    'sequence': '1'  # sequence to define the order
                },
                'query': {
                    'query': "{% include '1_a_dly_clm_dec.sql'%}",
                    'useLegacySql': False
                },
                
            },
            params={"owner": "{{ dag_run.conf['owner'] }}",
                     "costcenter": "{{ dag_run.conf['costcenter'] }}"
                    }
        )

    mnthly_clm_dec = BigQueryInsertJobOperator(
        task_id='1_b_mnthly_clm_dec',
        impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': owner_name,
                'dag-name': DAG_ID,  # dag name
                'task-name': '1_b_mnthly_clm_dec',  # task name same as task_id
                'sequence': '2'  # sequence to define the order
            },
            'query': {
                'query': "{% include '1_b_mnthly_clm_dec.sql'%}",
                'useLegacySql': False
            },
            
        },
            params={"owner": "{{ dag_run.conf['owner'] }}",
                     "costcenter": "{{ dag_run.conf['costcenter'] }}"
                    }
    )

    archv_clm_dec = BigQueryInsertJobOperator(
        task_id='1_c_archv_clm_dec',
        impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': owner_name,
                'dag-name': DAG_ID,  # dag name
                'task-name': '1_c_archv_clm_dec',  # task name same as task_id
                'sequence': '2'  # sequence to define the order
            },
            'query': {
                'query': "{% include '1_c_archv_clm_dec.sql'%}",
                'useLegacySql': False # uses standard SQL as dialect. 
            },
              
        },
            params={"owner": "{{ dag_run.conf['owner'] }}",
                     "costcenter": "{{ dag_run.conf['costcenter'] }}"
                    }
    )
    
    monthly_rx_dec = BigQueryInsertJobOperator(
        task_id='1_d_monthly_rx_dec',
        impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': owner_name,
                'dag-name': DAG_ID,  # dag name
                'task-name': '1_d_monthly_rx_dec',  # task name same as task_id
                'sequence': '2'  # sequence to define the order
            },
            'query': {
                'query': "{% include '1_d_monthly_rx_dec.sql'%}",
                'useLegacySql': False,  # uses standard SQL as dialect.
            },
            
        },
            params={"owner": "{{ dag_run.conf['owner'] }}",
                     "costcenter": "{{ dag_run.conf['costcenter'] }}"
                    }
    )

    archv_rx_dec = BigQueryInsertJobOperator(
        task_id='1_e_archv_rx_dec',
        impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': owner_name,
                'dag-name': DAG_ID,  # dag name
                'task-name': '1_e_archv_rx_dec',  # task name same as task_id
                'sequence': '2'  # sequence to define the order
            },
            'query': {
                'query': "{% include '1_e_archv_rx_dec.sql'%}",
                'useLegacySql': False
            },
            
        },
            params={"owner": "{{ dag_run.conf['owner'] }}",
                     "costcenter": "{{ dag_run.conf['costcenter'] }}"
                    }
    )

    combine_clm_dec = BigQueryInsertJobOperator(
        task_id='2_a_combine_clm_dec',
        impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': owner_name,
                'dag-name': DAG_ID,  # dag name
                'task-name': '2_a_combine_clm_dec',  # task name same as task_id
                'sequence': '2'  # sequence to define the order
            },
            'query': {
                'query': "{% include '2_a_combine_clm_dec.sql'%}",
                'useLegacySql': False
            },
            
        },
            params={"owner": "{{ dag_run.conf['owner'] }}",
                     "costcenter": "{{ dag_run.conf['costcenter'] }}"
                    }
    )

    rx_combine_dec = BigQueryInsertJobOperator(
        task_id='2_b_rx_combine_dec',
        impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': owner_name,
                'dag-name': DAG_ID,  # dag name
                'task-name': '2_b_rx_combine_dec',  # task name same as task_id
                'sequence': '2'  # sequence to define the order
            },
            'query': {
                'query': "{% include '2_b_rx_combine_dec.sql'%}",
                'useLegacySql': False
            },
            
        },
            params={"owner": "{{ dag_run.conf['owner'] }}",
                     "costcenter": "{{ dag_run.conf['costcenter'] }}"
                    }
    )
    
    
    dly_clm_ln_dec = BigQueryInsertJobOperator(
        task_id='3_a_dly_clm_ln_dec',
        impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': owner_name,
                'dag-name': DAG_ID,  # dag name
                'task-name': '3_a_dly_clm_ln_dec',  # task name same as task_id
                'sequence': '10'  # sequence to define the order
            },
            'query': {
                'query': "{% include '3_a_dly_clm_ln_dec.sql'%}",
                'useLegacySql': False
            },
            
        },
            params={"owner": "{{ dag_run.conf['owner'] }}",
                     "costcenter": "{{ dag_run.conf['costcenter'] }}"
                    }
    )
    
    mnthly_clm_ln_dec = BigQueryInsertJobOperator(
        task_id='3_b_mnthly_clm_ln_dec',
        impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': owner_name,
                'dag-name': DAG_ID,  # dag name
                'task-name': '3_b_mnthly_clm_ln_dec',  # task name same as task_id
                'sequence': '2'  # sequence to define the order
            },
            'query': {
                'query': "{% include '3_b_mnthly_clm_ln_dec.sql'%}",
                'useLegacySql': False
            },
            
        },
            params={"owner": "{{ dag_run.conf['owner'] }}",
                     "costcenter": "{{ dag_run.conf['costcenter'] }}"
                    }
    )

    archv_clm_ln_bk1_dec = BigQueryInsertJobOperator(
        task_id='3_c_archv_clm_ln_bk1_dec',
        impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': owner_name,
                'dag-name': DAG_ID,  # dag name
                'task-name': '3_c_archv_clm_ln_bk1_dec',  # task name same as task_id
                'sequence': '2'  # sequence to define the order
            },
            'query': {
                'query': "{% include '3_c_archv_clm_ln_bk1_dec.sql'%}",
                'useLegacySql': False
            },
            
        },
            params={"owner": "{{ dag_run.conf['owner'] }}",
                     "costcenter": "{{ dag_run.conf['costcenter'] }}"
                    }
    )

    archv_clm_ln_bk2_dec = BigQueryInsertJobOperator(
        task_id='3_d_archv_clm_ln_bk2_dec',
        impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': owner_name,
                'dag-name': DAG_ID,  # dag name
                'task-name': '3_d_archv_clm_ln_bk2_dec',  # task name same as task_id
                'sequence': '2'  # sequence to define the order
            },
            'query': {
                'query': "{% include '3_d_archv_clm_ln_bk2_dec.sql'%}",
                'useLegacySql': False
            },
            
        },
            params={"owner": "{{ dag_run.conf['owner'] }}",
                     "costcenter": "{{ dag_run.conf['costcenter'] }}"
                    }
    )
    
    combine_cln_ln_dec = BigQueryInsertJobOperator(
        task_id='4_a_combine_cln_ln_dec',
        impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
        configuration={
            'labels': {
                'owner': owner_name,
                'dag-name': DAG_ID,  # dag name
                'task-name': '4_a_combine_cln_ln_dec',  # task name same as task_id
                'sequence': '2'  # sequence to define the order
            },
            'query': {
                'query': "{% include '4_a_combine_cln_ln_dec.sql'%}",
                'useLegacySql': False
            },
           
        },
            params={"owner": "{{ dag_run.conf['owner'] }}",
                     "costcenter": "{{ dag_run.conf['costcenter'] }}"
                    }
    )

    prep_transformer_dec = BigQueryInsertJobOperator(
        task_id='5_a_prep_transformer_dec',
        impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
        on_success_callback=success_call,
        configuration={
            'labels': {
                'owner': owner_name,
                'dag-name': DAG_ID,  # dag name
                'task-name': '5_a_prep_transformer_dec',  # task name same as task_id
                'sequence': '2'  # sequence to define the order
            },
            'query': {
                'query': "{% include '5_a_prep_transformer_dec.sql'%}",
                'useLegacySql': False
            },
            
        },
            params={"owner": "{{ dag_run.conf['owner'] }}",
                     "costcenter": "{{ dag_run.conf['costcenter'] }}"
                    }
    )

    history_input = BigQueryInsertJobOperator(
        task_id='6_a_history_input',
        impersonation_chain=decrypt_sa,  # SA Airflow uses to impersonate while interacting with BQ
        on_success_callback=success_call,
        configuration={
            'labels': {
                'owner': owner_name,
                'dag-name': DAG_ID,  # dag name
                'task-name': '6_a_history_input',  # task name same as task_id
                'sequence': '2'  # sequence to define the order
            },
            'query': {
                'query': "{% include '6_a_history_input.sql'%}",
                'useLegacySql': False,  # uses 5_a_prep_transformer_dec SQL as dialect.
                "queryParameters": [
                        {"parameterType": {"type": "INTEGER"},
                        "parameterValue": {"value": '{{ dag_run.conf["historydays"] if dag_run.conf.get("historydays") else 0 }}'},
                        "name": "historydays"
                        },
                        {"parameterType": {"type": "INTEGER"},
                        "parameterValue": {"value": '{{ dag_run.conf["emb_init"] if dag_run.conf.get("emb_init") else 0 }}'},
                        "name": "emb_init"
                        }
                    ]
            },
           
        },
            params={"owner": "{{ dag_run.conf['owner'] }}",
                     "costcenter": "{{ dag_run.conf['costcenter'] }}"
                    }
    )
    # start = DummyOperator(task_id='start')
    stage_2 = DummyOperator(task_id='stage_2')
    stage_3 = DummyOperator(task_id='stage_3')

te_attach_member_id >> [dly_clm_dec, mnthly_clm_dec,archv_clm_dec, monthly_rx_dec,archv_rx_dec] >> stage_2
stage_2 >> [combine_clm_dec,rx_combine_dec] >> stage_3
stage_3 >> [dly_clm_ln_dec, mnthly_clm_ln_dec, archv_clm_ln_bk1_dec, archv_clm_ln_bk2_dec] >> combine_cln_ln_dec >> prep_transformer_dec >> history_input