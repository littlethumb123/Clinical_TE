#!/bin/bash
bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_med_claims_yr1_v2`'

bq query \
--use_legacy_sql=false \
'
CREATE OR REPLACE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_med_claims_yr1_v2`
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
SELECT
    *
FROM 
    `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_med_claims_yr1`
WHERE 
    asdb_paid_dt < index_dt
'
