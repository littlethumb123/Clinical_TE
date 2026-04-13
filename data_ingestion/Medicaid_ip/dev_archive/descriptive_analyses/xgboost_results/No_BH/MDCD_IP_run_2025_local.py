########################################################################################
####### Project: Medicaid IP model 2024                                          #######
####### Original Author: Elle Palmer                                             #######
####### Date: 2024-08-29                                                         #######
####### Last modified by:                                       #######
####### On:                                                            #######
####### Population: All Medicaid                                                 #######
########################################################################################

#######################
### import packages ###
#######################
import xgboost as xgb
import google.auth
import pyspark
import time
import pandas as pd

from google.auth import impersonated_credentials
from google.cloud import bigquery
from google.cloud import storage

#rom pyspark import SparkContext, SparkConf
#rom pyspark.sql import SparkSession

#rom pyspark.sql import functions as F
#rom pyspark.sql.functions import to_date, col, udf, when, lit 
#rom pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DateType
#from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
#from pyspark.ml import Pipeline
#from pyspark.ml.classification import LogisticRegression

###############
### set up  ###
###############
#sc = SparkContext()
#conf = SparkConf()

#conf.set('google.cloud.auth.service.account.enable', 'true')
#conf.set("viewsEnabled", "true")

#spark = SparkSession.builder.appName("MD IP Pyspark Predict").getOrCreate()

TEMPORARY_GCS_BUCKET = 'clin-analytics-data-hcb-dev'

#spark.conf.set('temporaryGcsBucket', TEMPORARY_GCS_BUCKET)

client = bigquery.Client()
model = xgb.XGBClassifier()

bucket_name='clin-analytics-data-hcb-dev'
PROJECT_ID='anbc-hcb-dev'
DATASET_ID='cm_medicaid_hcb_dev'

print("Bucket Name: ",bucket_name)
print("PROJECT ID: " ,PROJECT_ID)
print("DATASET ID: " ,DATASET_ID)

SOURCE_TABLE = PROJECT_ID + '.' + DATASET_ID + '.a534354_medicaid_ip_features'
TARGET_TABLE = 'a534354_medicaid_IP_predict'

credentials, project = google.auth.default()
client = storage.Client(credentials=credentials)
bucket=client.get_bucket(bucket_name)
bq_client = bigquery.Client(credentials=credentials)

client = bigquery.Client()
#blob_name=bucket.blob('cm-mdcd-adhoc-files/cm-mdcd-avoidable-ed-model/MDCD_Avoidable_ED_Model_xgb_model.pkl')

model = xgb.XGBClassifier()
#model.load_model("clin-analytics-data-hcb-dev/a534354/IP_model_xgboost/mdcd_ip_2024_final_model.cbm")
model.load_model("mdcd_ip_2024_final_model.cbm")

############################
# ## import and prep data ###
# ###########################
# STEP 8: Create prediction table and merge it with index table.
def create_prediction_table(index, y_pred):
    print("STEP 8: Creating the prediction table...")

    index_pd = index.toPandas()
    ed = pd.DataFrame(y_pred, columns=['avoidable_ed_riskscore'])
    ed['avoidable_ed_riskscore'] = ed['avoidable_ed_riskscore'].round(decimals = 6)
    prediction = pd.concat([index_pd, ed], axis=1)
    prediction['run_date']     = pd.to_datetime('today').strftime("%Y-%m-%d")
    prediction['asdb_elig_dt'] = pd.to_datetime('today').strftime("%Y-%m-01")
   
    return prediction


# Create prediction table and merge it with index table.
def create_prediction_table(index, y_pred):
    print("STEP 4: Creating the prediction table...")

    index_pd = index.toPandas()
    ip = pd.DataFrame(y_pred, columns=['ip_riskscore'])
    ip['ip_riskscore'] = ip['ip_riskscore'].round(decimals = 6)
    prediction = pd.concat([index_pd, ip], axis=1)
    prediction['run_date']     = pd.to_datetime('today').strftime("%Y-%m-%d")
    prediction['asdb_elig_dt'] = pd.to_datetime('today').strftime("%Y-%m-01")
   
    return prediction

###################################################################
# ## fill missing embeddings with 0 values per standard protocol ###
# ##################################################################
def load_data_from_bq(SOURCE_TABLE):    
    print("STEP 1: Fetching Features from BQ...")

    sql_string = '''SELECT * from {}'''.format(SOURCE_TABLE)
    print('SQL String: ', sql_string)
    print('Source Table: ', SOURCE_TABLE)

    #data = spark.read.format('bigquery').option('table', SOURCE_TABLE).option('query', sql_string).load()
    #data.printSchema()
    sql = """SELECT * FROM `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_dag_check_features`"""
    data = client.query(sql).to_dataframe()
    print("Total Number of records in features table: ",data.count())

    return data

def index_seperator(data):
    print("STEP 2: Seperating index data...")
    
    data = data.rename(columns={'coa_population_group_TANF_CHIP': 'coa_population_group_TANF/CHIP', 'coa_population_group_Dual_Elig': 'coa_population_group_Dual Elig'})
    index_columns = ['asdb_member_key', ]
    index_data = data.loc[:, index_columns]
    feature_data = data.set_index('asdb_member_key', inplace = True)

    return index_data, feature_data

# Split data into chunks with size 500k to stay within memory limit
def split_features_df(data):
    print("STEP 4: Spliting result table into chunks...")

    n = len(data) // 500000
    if n > 1:
        df_split = np.array_split(data, n)
        return df_split, n
    else:
        return data, 0
    
# Save predictions to TARGET_TABLE
def write_to_bq(input_df,client,project_id,dataset_id,output_table,n):
    print("STEP 5: Saving the predictions...")

    tmp_table = project_id + '.' + dataset_id + '.' + output_table + '_tmp'
    
    if n > 1:
        for i in range(0, n):
            print(f'Saving chunk [{i+1}/{n}]...')
            if i == 0:
                ## drop table if exists
                query = """DROP TABLE IF EXISTS {}""".format(tmp_table)
                query_job = client.query(query)
                query_job.result()
    
                job = client.load_table_from_dataframe(input_df[i], tmp_table,)
                job.result()  # Waits for table load to complete.
                print("Loaded dataframe to {}".format(tmp_table))
            else:
                job = client.load_table_from_dataframe(input_df[i], tmp_table,)
                job.result()  # Waits for table load to complete.
                print("Loaded dataframe to {}".format(tmp_table))
    else:
         ## drop table if exists
         query = """DROP TABLE IF EXISTS {}""".format(tmp_table)
         query_job = client.query(query)
         query_job.result()

         job = client.load_table_from_dataframe(input_df, tmp_table,)
         job.result()  # Waits for table load to complete.
         print("Loaded dataframe to {}".format(tmp_table))
            
    ## adding owner - for some reason loadjob config above not adding owner
    query = """
    ALTER TABLE {}
    SET OPTIONS (
    labels=[("owner", "navaneethakrishnanp_aetna_com"), ("cost_center", "13070"), ("unique_id", "hcm-cm-gen-md-dev")]
    )
    """.format(tmp_table)
    query_job = client.query(query)
    query_job.result()

    ## rounding the score values to 6 decimal places    
    query = """
    UPDATE `{}` SET ip_riskscore = round(ip_riskscore,6) WHERE CAST(run_date AS DATE) = CURRENT_DATE()
    """.format(tmp_table)
    print("UPDATE SQL",query)
    query_job = client.query(query)
    query_job.result()
    
    ## drop table if exists
    query = """
    DROP TABLE IF EXISTS {}.{}.{}""".format(project_id,dataset_id,output_table)
    query_job = client.query(query)
    query_job.result()
    
    ## renaming table name    
    query = """
    ALTER TABLE `{}` rename to `{}`
    """.format(tmp_table,output_table)
    print("ALTER SQL",query)
    query_job = client.query(query)
    query_job.result()

# Combine all the steps in single function
def combiner(SOURCE_TABLE, TARGET_TABLE, model, y_pred):
    start_time = time.time()
    #data = load_data_from_bq(SOURCE_TABLE)
    #index, data = index_seperator(data)
    #data.show(5)
    #feature_pd = data.toPandas()
    #y_pred = make_prediction(model, data)
    #y_pred = model.predict_proba(data)[:, 1]
    
    data = create_prediction_table(index, y_pred)
    data, n = split_features_df(data)

    client = bigquery.Client(project=PROJECT_ID)
    write_to_bq(data,client,PROJECT_ID,DATASET_ID,TARGET_TABLE,n)
    
    duration = time.time() - start_time
    print(f'Prediction has been completed!')
    print(f'Model predictions has been made in {duration} seconds.')

#combiner(SOURCE_TABLE, TARGET_TABLE, FEATURES, model)

sql = """
SELECT
    *
FROM 
    `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_dag_check_features`
"""
data = client.query(sql).to_dataframe() 

index_columns = ['asdb_member_key', ]
index = data.loc[:, index_columns] 

data = data.rename(columns={'coa_population_group_TANF_CHIP': 'coa_population_group_TANF/CHIP', 'coa_population_group_Dual_Elig': 'coa_population_group_Dual Elig'})
data.set_index('asdb_member_key', inplace = True)
               
#y_pred_test_class = model.predict(df)    
y_pred = model.predict_proba(data)[:, 1]

credentials, project= google.auth.default()
client = bigquery.Client(credentials=credentials)

job_labels = {"costcenter": 13070, "owner": "palmere1_aetna_com"}
load_config = bigquery.LoadJobConfig(labels=job_labels)
load_job = client.load_table_from_dataframe(
    dataframe = df,
    destination = TARGET_TABLE,
    job_config = load_config,
)

print(load_job.result())
