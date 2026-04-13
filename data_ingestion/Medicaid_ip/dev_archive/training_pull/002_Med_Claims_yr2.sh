#!/bin/bash
bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_med_claims_yr2`'

bq query \
--use_legacy_sql=false \
'
CREATE OR REPLACE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_med_claims_yr2`
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
SELECT
       st.asdb_member_key
       , st.asdb_plan_key
       , st.index_dt
       , clm.claimid
       , clm.asdb_coe_id
       , coe.asdb_coe_general_type
       , coe.asdb_coe_sub_cat
       , clm.asdb_svc_prov_key
       , CAST(clm.asdb_incurred_dt AS DATE) AS asdb_incurred_dt
       , CAST(clm.asdb_paid_dt AS DATE) AS asdb_paid_dt
       , clm.location
       , clm.revcode
       , clm.servcode
       , clm.billtype
       , clm.prindiag
       , clm.paid_amt
       , clm.emis_cat
FROM 
       (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `'$ST'`) AS st
INNER JOIN 
       (WITH latest_partitions AS 
           (SELECT
               asdb_member_key
               , asdb_plan_key
               , claimid
               , asdb_svc_prov_key
               , asdb_incurred_dt
               , asdb_paid_dt
               , location
               , revcode
               , servcode
               , billtype
               , prindiag
               , paid_amt
               , emis_cat
               , insert_dts AS date
               , final_claim
               , status_header
               , status_detail
               , asdb_coe_id
            FROM 
                `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_CLM_DATA_STAGE`
            WHERE 
                CAST(insert_dts AS DATE) > DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY)
           )
           SELECT * 
           FROM latest_partitions
           WHERE date = (SELECT MAX(date) FROM latest_partitions)
                AND final_claim = 1
                AND TRIM(UPPER(status_header)) = "PAID"
                AND TRIM(UPPER(status_detail)) NOT IN ("DENY", "DENIED")
           
        ) AS clm
              ON st.asdb_member_key = clm.asdb_member_key
              AND st.asdb_plan_key = clm.asdb_plan_key
LEFT JOIN 
       `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_TYPE_OF_SERVICE` AS coe
              ON clm.asdb_coe_id = coe.asdb_coe_id
WHERE 
       CAST(asdb_incurred_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 24 MONTH) AND DATE_SUB(DATE_SUB(st.index_dt, INTERVAL 1 DAY), INTERVAL 12 MONTH)
'
