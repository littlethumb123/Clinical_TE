import xgboost as xgb
import google.auth

from google.cloud import bigquery
from google.auth import impersonated_credentials

credentials, project= google.auth.default()
client = bigquery.Client(credentials=credentials)

### NOTE: change below to the params used in production ###
job_labels = {"costcenter": 13070, "owner": "palmere1_aetna_com"}

load_config = bigquery.LoadJobConfig(labels=job_labels)

### NOTE: change below to the table used in production ###
TARGET_TABLE ="cm_medicaid_hcb_dev.a534354_medicaid_IP_predict"

model = xgb.XGBClassifier()
model.load_model("MDCD_IP_2024_xgb_v_2_1_1.json")
#clin-analytics-data-hcb-dev/a534354/IP_model_xgboost/mdcd_ip_2024_final_model.cbm if we want to point to the bucket...

### NOTE: change below to the table used in production ###
sql = """SELECT * FROM `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_dag_check_features`""" 
df = client.query(sql).to_dataframe() 

df = df.rename(columns={'coa_population_group_TANF_CHIP': 'coa_population_group_TANF/CHIP', 'coa_population_group_Dual_Elig': 'coa_population_group_Dual Elig'})
df.set_index('asdb_member_key', inplace = True)
            
#y_pred_test_class = model.predict(df)    
df['y_pred_test'] = model.predict_proba(df)[:, 1]

output = df[['y_pred_test']]
output = output.reset_index()

load_job = client.load_table_from_dataframe(
    dataframe = output,
    destination = TARGET_TABLE,
    job_config = load_config,
)

print(load_job.result())