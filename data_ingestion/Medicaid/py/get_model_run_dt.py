def fetch_run_dt(resource_sa, db_schema):
    # #### Import Packages
    import subprocess
    import numpy as np
    import pandas as pd
    pd.set_option('display.max_columns', 150)
    import sys
    import os
    import re
    import logging
    import warnings
    import time
    from datetime import datetime
    import pytz
    import google.auth
    from google.auth import impersonated_credentials
    from google.cloud import bigquery
    from google.cloud import storage
        
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    print("Resource SA: ", resource_sa)
    print("DB Schema  : ", db_schema)

    credentials, project = google.auth.default()
    target_credentials = impersonated_credentials.Credentials(
        source_credentials=credentials,
        target_principal=resource_sa,
        target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
        lifetime=3500
    )

    bq_client = bigquery.Client(credentials = target_credentials)
    
    start_time = time.time()
    ## Check if the source v_rap_api_score_bkup table is updated for the current date
    query = """
            SELECT run_dt FROM {}.MD_MODEL_RUN_DATE_CONFIG WHERE model_id = 'Medicaid_Transformer_Embeddings'
            """.format(db_schema)
    
    print("SQL Query: ",query)
    get_model_run_dt = bq_client.query(query)
    get_model_run_dt.result()
    
    model_run_dt = get_model_run_dt.to_dataframe()
    model_run_dt = model_run_dt.values[0]
    run_dt = ''.join(model_run_dt)
    
    print("Model Run Date: ", run_dt)

    return(run_dt)
         