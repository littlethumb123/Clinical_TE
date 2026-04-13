import google.auth
from google.auth import impersonated_credentials
from google.cloud import bigquery
from google.cloud import storage
from pathlib import Path
from io import BytesIO
import pickle
import joblib
from datetime import date, datetime, timedelta
import traceback


import warnings
warnings.filterwarnings('ignore')
# Imports Python standard library logging
import logging

logging.basicConfig(level=logging.INFO)
pylogger = logging.getLogger(__name__)

def apperr(msg):
    pylogger.error(f'ERROR: {msg}')
    #

def applog(msg):
    pylogger.info(f'info: {msg}')
    #

#################################
### class gs_utils
#################################

class gs_utils:
    def get_storage_client(resource_sa = "gchcb-clin-analytics-ontpd@anbc-hcb-dev.iam.gserviceaccount.com"):
        credentials, project= google.auth.default()
        #target_credentials = impersonated_credentials.Credentials(credentials, target_principal=resource_sa, target_scopes = ["https://www.googleapis.com/auth/cloud-platform"])
        client= storage.Client(credentials=credentials)
        return client

    def get_code_bucket(envname, tenant, resource_sa = "gchcb-clin-analytics-ontpd@anbc-hcb-dev.iam.gserviceaccount.com"):
        storage_client = gs_utils.get_storage_client(resource_sa)
        tenant_str = f"{tenant}-{envname}"
        bucket_name = tenant_str.replace("-hcb-", "-code-hcb-")
        bucket = storage_client.get_bucket(bucket_name)
        return bucket, bucket_name

    def get_data_bucket(envname, tenant, resource_sa = "gchcb-clin-analytics-ontpd@anbc-hcb-dev.iam.gserviceaccount.com"):
        storage_client = gs_utils.get_storage_client(resource_sa)
        tenant_str = f"{tenant}-{envname}"
        bucket_name = tenant_str.replace("-hcb-", "-data-hcb-")
        bucket = storage_client.get_bucket(bucket_name)
        return bucket, bucket_name

    def get_named_bucket(envname, tenant, bucket_name, resource_sa = "gchcb-clin-analytics-ontpd@anbc-hcb-dev.iam.gserviceaccount.com"):
        storage_client = gs_utils.get_storage_client(resource_sa)
        bucket = storage_client.get_bucket(bucket_name)
        return bucket

    def read_file_from_bucket(bucket, relative_path, file_name):
        blob = bucket.blob(f'{relative_path}/{file_name}')
        blob.download_to_filename(file_name)
        with open(file_name,"r") as f:
            file_text = f.read()
        return file_text

    def download_file_from_bucket(bucket_type, relative_path, file_name, local_file_name, envname, tenant, resource_sa, bucket_name):
        bucket = None
        if (bucket_type == 'code'):
            bucket, bname = gs_utils.get_code_bucket(envname, tenant, resource_sa)
        elif (bucket_type == 'data'):
            bucket, bname = gs_utils.get_data_bucket(envname, tenant, resource_sa)
        else:
            bucket = gs_utils.get_named_bucket(envname, tenant, bucket_name, resource_sa)
        #
        if f'{file_name}'.__contains__('/'):
            blob = bucket.blob(f'{file_name}')
        else:
            blob = bucket.blob(f'{relative_path}/{file_name}')
        blob.download_to_filename(local_file_name)

    def read_file_from_std_bucket(bucket_type, relative_path, file_name, mode, envname, tenant, resource_sa, bucket_name):
        bucket = None
        if (bucket_type == 'code'):
            bucket, bname = gs_utils.get_code_bucket(envname, tenant, resource_sa)
        elif (bucket_type == 'data'):
            bucket, bname = gs_utils.get_data_bucket(envname, tenant, resource_sa)
        else:
            bucket = gs_utils.get_named_bucket(envname, tenant, bucket_name, resource_sa)
        #
        blob = bucket.blob(f'{relative_path}/{file_name}')
        blob.download_to_filename(file_name)
        with open(file_name, mode) as f:
            file_bytes = f.read()
        return file_bytes

    def joblib_data_dumper(df_head, file_name, bucket_name, envname, tenant, resource_sa):
        bucket = gs_utils.get_named_bucket(envname, tenant, bucket_name, resource_sa)
        blob = bucket.blob(file_name)
        with blob.open("wb", ignore_flush=True) as f:
            joblib.dump(df_head, f)

    def joblib_data_loader(df_head, filename, bucket_name, envname, tenant, resource_sa):
        bucket = gs_utils.get_named_bucket(envname, tenant, bucket_name, resource_sa)
        blob = bucket.blob(filename)
        data = BytesIO()
        blob.download_to_file(data)
        data=joblib.load(data)
        return data

#################################
