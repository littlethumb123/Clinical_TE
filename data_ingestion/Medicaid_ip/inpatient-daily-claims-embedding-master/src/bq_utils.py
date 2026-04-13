import google.auth
from google.auth import impersonated_credentials
from google.cloud import bigquery
from google.cloud import storage
from pathlib import Path
import traceback

import warnings
warnings.filterwarnings('ignore')
import datetime
import numpy as np
import pandas as pd



#################################
### class bq_utils
#################################

class bq_utils:     ### TODO make referencing accp_utils/bq_utils.py work in Dataproc DAGs
    def get_bigquery_client(resource_sa = "gchcb-clin-analytics-ontpd@anbc-hcb-dev.iam.gserviceaccount.com"):
        credentials, project= google.auth.default()
        envname = f'{project}'.split('-')[-1]
        #target_credentials = impersonated_credentials.Credentials(credentials, target_principal=resource_sa, target_scopes = ["https://www.googleapis.com/auth/cloud-platform"])
        target_credentials = credentials
        client = bigquery.Client(credentials=target_credentials)
        return client

    def sql_exec(sql_query, resource_sa = "gchcb-clin-analytics-ontpd@anbc-hcb-dev.iam.gserviceaccount.com", bqclient = None, run_in_batch=False):
        if (bqclient == None):
            bq_client= bq_utils.get_bigquery_client(resource_sa)
        else:
            bq_client = bqclient
        if (run_in_batch):
            job_config = bigquery.QueryJobConfig(
                # Run at batch priority, which won't count toward concurrent rate limit.
                priority=bigquery.QueryPriority.BATCH
            )
            query_job = bq_client.query(sql_query,job_config=job_config)
        else:
            query_job = bq_client.query(sql_query)
        return query_job.result()

    def query_db(sql_query, resource_sa = "gchcb-clin-analytics-ontpd@anbc-hcb-dev.iam.gserviceaccount.com", bqclient = None, run_in_batch=False):
        if (bqclient == None):
            bq_client= bq_utils.get_bigquery_client(resource_sa)
        else:
            bq_client = bqclient
        if (run_in_batch):
            job_config = bigquery.QueryJobConfig(
                # Run at batch priority, which won't count toward concurrent rate limit.
                priority=bigquery.QueryPriority.BATCH
            )
            query_job = bq_client.query(sql_query,job_config=job_config)
        else:
            query_job = bq_client.query(sql_query)
        result_dataframe = query_job.to_dataframe()
        return result_dataframe

    def df_to_gbq(pandas_df, bq_tablename,  chunksize, resource_sa):
        credentials, project= google.auth.default()
        envname = f'{project}'.split('-')[-1]
        #target_credentials = impersonated_credentials.Credentials(credentials, target_principal=resource_sa, target_scopes = ["https://www.googleapis.com/auth/cloud-platform"])
        target_credentials = credentials
        pandas_df.to_gbq(
            destination_table=bq_tablename,
            chunksize=chunksize,
            project_id=project,
            if_exists="replace",
            credentials=target_credentials
        )

    def read_gbq(bq_tablename, resource_sa):
        credentials, project= google.auth.default()
        envname = f'{project}'.split('-')[-1]
        #target_credentials = impersonated_credentials.Credentials(credentials, target_principal=resource_sa, target_scopes = ["https://www.googleapis.com/auth/cloud-platform"])
        target_credentials = credentials
        df_pd = pd.read_gbq(
            query_or_table=bq_tablename,
            dialect="standard",
            project_id=project,
            credentials=target_credentials,
            use_bqstorage_api=True
        )
        return df_pd

