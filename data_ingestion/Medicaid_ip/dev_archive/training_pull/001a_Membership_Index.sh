#!/bin/bash

bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_member_index`'

bq query \
--use_legacy_sql=false \
'
CREATE OR REPLACE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_member_index`
--PARTITION BY RANGE_BUCKET(asdb_plan_key, GENERATE_ARRAY(0,100,1))
--CLUSTER BY asdb_elig_dt, coa_population_group
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
WITH pre AS (
  SELECT
     asdb_member_key
    , asdb_plan_key
    , asdb_plan_nm
    , asdb_elig_dt AS index_dt
    , coa_population_category
    , coa_population_group
    , ROW_NUMBER() OVER(PARTITION BY asdb_member_key ORDER BY RAND()) AS pos
  FROM 
    `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_member`
)
SELECT  
    pre.*   
FROM 
    pre
WHERE 
    pos <= 1
'