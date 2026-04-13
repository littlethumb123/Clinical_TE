#!/bin/bash

bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_member_index`'

bq query \
--use_legacy_sql=false \
'
CREATE OR REPLACE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_member_index`
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
SELECT
    asdb_member_key
    , asdb_plan_key
    , CAST(asdb_elig_dt AS DATE) AS index_dt
    , ss_cohort
    , coa_population_category
    , CASE WHEN TRIM(coa_population_category) = "ABD Non Dual LTSS" OR
         TRIM(coa_population_category) = "LTSS Only" OR 
              TRIM(coa_population_category) = "Dual Elig LTSS" OR 
              TRIM(coa_population_category) = "Dual Int LTSS"
              THEN "LTSS"
         WHEN TRIM(coa_population_category) = "ABD Non Dual Non LTSS" OR
              TRIM(coa_population_category) = "DD"
              THEN "ABD"
         WHEN TRIM(coa_population_category) = "BH Int SMI" OR
              TRIM(coa_population_category) = "BH Only"
              THEN "BH"
         WHEN TRIM(coa_population_category) = "DSNP Medicare Only" OR
              TRIM(coa_population_category) = "Dual Elig NonLTSS" 
              THEN "Dual Elig"
         WHEN TRIM(coa_population_category) = "Dual Int DD" OR
              TRIM(coa_population_category) = "Dual Int NonLTSS"
              THEN "Dual Int"  -- Dual Elig and Dual Int could be combined depending on analysis need
         WHEN TRIM(coa_population_category) = "CHIP" OR
              TRIM(coa_population_category) = "TANF" 
              THEN "TANF/CHIP"
         WHEN TRIM(coa_population_category) = "Expansion" 
              THEN "Expansion"
         ELSE "Other"
         END AS coa_population_group
    , final_score
    , final_riskstrat
FROM
    `anbc-hcb-dev.cm_medicaid_hcb_dev.a091749_all_cohort` AS a
'