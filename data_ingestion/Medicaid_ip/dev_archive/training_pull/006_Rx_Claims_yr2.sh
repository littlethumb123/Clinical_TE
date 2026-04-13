#!/bin/bash

#-- Pull Rx Claims
bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_rx_claims`'

bq query \
--use_legacy_sql=false \
'
CREATE OR REPLACE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_rx_claims`
--PARTITION BY DATE_TRUNC(asdb_incurred_dt, MONTH)
--CLUSTER BY asdb_plan_key, asdb_member_key, asdb_mos
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
SELECT DISTINCT
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
    , rx.asdb_pharmacy_key
    , rx.prescriptionnum
    , CAST(rx.asdb_incurred_dt AS DATE) AS disp_dt
    , rx.days_supply
    , rx.script_ct
    , ROUND(CASE WHEN rx.days_supply >= 0.1 AND rx.days_supply < 30 THEN 30
               ELSE rx.days_supply END/30) AS scripts
    , rx.ndcnum AS ndc_cd
    , rx.gpi AS adjudicated_gpi_cd  --generic product code
    , SUBSTR(rx.gpi,1,4) AS gpi4
    , SUBSTR(rx.gpi,1,2) AS gpi2
    , rx.billed_amt  --billed_amt should be used to calculate rx cost
    , rx.claim_adj_amt
    , rx.copay_amt
    , CASE WHEN rx.pharmacytype="R" THEN 1 
        ELSE 0 
        END AS retail_flag  --retail
    , CASE WHEN rx.pharmacytype="M" THEN 1 
        ELSE 0 
        END AS mail_order_flag  --mail order
    , CASE WHEN rx.drugtype = 3 THEN 1 
        ELSE 0 
        END AS generic_fill_flag
    , CASE WHEN rx.drugtype = 2 THEN 1 
        ELSE 0 
        END AS branded_generic_fill_flag
    , CASE WHEN rx.drugtype = 4 THEN 1 
        ELSE 0 
        END AS otc_fill_flag
    , CASE WHEN rx.drugtype = 1 THEN 1 
        ELSE 0 
        END AS ss_brand_fill_flag
    , CASE WHEN rx.drugtype = 5 THEN 1 
        ELSE 0 
        END AS ms_brand_fill_flag
    , CASE WHEN rx.formularyflag="F" or rx.drugtype = 3 THEN 1 
        ELSE 0 
        END AS formulary_fill_flag
    , CASE WHEN c.maint_drug_cd="X" THEN 1 
        ELSE 0 
        END AS maint_drug_flag  --maintenance drug flag
    , CASE WHEN d.ndc IS NOT NULL THEN 1 
        ELSE 0 
        END AS specialty_rx_flag --specialty pharmacy drug flag
FROM 
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `'$ST'`) AS st
INNER JOIN 
    `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_RX_DATA_STAGE` AS rx

-- view already references SNAP table (edp_hcb_mdcd_core_src.SNAP_T_ASDB_ASDB_RX_DATA_STAGE_20230613T041729)

        ON st.asdb_member_key = rx.asdb_member_key
        AND st.asdb_plan_key = rx.asdb_plan_key
LEFT JOIN 
    `edp-prod-hcbstorage.edp_hcb_anbor_enrsrcv.EDW_DRUG` AS c

-- view references table (edp_hcb_anbor_enrsrc.T_EDW_DRUG)
--  but does not reference the available SNAP table (edp_hcb_anbor_enrsrc.SNAP_T_EDW_DRUG_20230712T203517)

        ON TRIM(rx.ndcnum) = TRIM(c.ndc_cd)
LEFT JOIN 
    `edp-prod-hcbstorage.edp_hcb_core_srcv.SPECDRUG_PWW_UNIVERSAL_SPEC_LIST` AS d

-- view references SNAP table (edp_hcb_core_src.SNAP_T_SPECDRUG_PWW_UNIVERSAL_SPEC_LIST_20221220T183429)
-- but the data are 7mo old compared to source table (edp_hcb_core_src.T_SPECDRUG_PWW_UNIVERSAL_SPEC_LIST)

        ON TRIM(rx.ndcnum) = TRIM(d.ndc)
WHERE 
    rx.ClaimType = "P"  --paid claims only
    AND CAST(rx.asdb_incurred_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 24 MONTH) AND DATE_SUB(DATE_SUB(st.index_dt, INTERVAL 1 DAY), INTERVAL 12 MONTH)
'



#-- Summarize pharmacy claims
bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_rx_tmp`'

bq query \
--use_legacy_sql=false \
'
CREATE OR REPLACE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_rx_tmp`
--PARTITION BY DATE_TRUNC(asdb_incurred_dt, MONTH)
--CLUSTER BY asdb_plan_key, asdb_member_key, asdb_mos
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
SELECT 
    asdb_member_key
    , asdb_plan_key
    , index_dt
    , MIN(disp_dt) AS first_disp_dt
    , MAX(disp_dt) AS last_disp_dt
    , COUNT(*) AS rx_claim_cnt
    , SUM(days_supply) AS days_supply_SUM
    , COUNT(DISTINCT ndc_cd) AS ndc_cnt
    , COUNT(DISTINCT adjudicated_gpi_cd) AS gpi_cnt
    , COUNT(DISTINCT gpi4) AS gpi4_cnt
    , COUNT(DISTINCT gpi2) AS gpi2_cnt
    , SUM(billed_amt) AS billed_amt_sum
    , SUM(claim_adj_amt) AS claim_adj_amt_sum
    , SUM(copay_amt) AS copay_amt_sum
    , SUM(retail_flag) AS retail_fills
    , SUM(mail_order_flag) AS mail_order_fills
    , SUM(generic_fill_flag) AS generic_fills
    , SUM(branded_generic_fill_flag) AS branded_generic_fills
    , SUM(otc_fill_flag) AS otc_fills
    , SUM(ss_brand_fill_flag) AS ss_brand_fills
    , SUM(ms_brand_fill_flag) AS ms_brand_fills
    , SUM(formulary_fill_flag) AS formulary_fills
    , SUM(maint_drug_flag) AS maint_drug_fills
    , SUM(CASE WHEN gpi2="27" THEN Scripts ELSE 0 END)        AS antidiabetic_scripts
    , SUM(CASE WHEN gpi2="27" THEN days_supply ELSE 0 END) AS antidiabetic_days_supply
    , SUM(CASE WHEN gpi2="33" THEN Scripts ELSE 0 END)        AS beta_blocker_scripts
    , SUM(CASE WHEN gpi2="33" THEN days_supply ELSE 0 END) AS beta_blocker_days_supply
    , SUM(CASE WHEN gpi2="36" THEN Scripts ELSE 0 END)        AS antihypertensive_scripts
    , SUM(CASE WHEN gpi2="36" THEN days_supply ELSE 0 END) AS antihypertensive_days_supply
    , SUM(CASE WHEN gpi2="39" THEN Scripts ELSE 0 END)        AS lipid_lowering_scripts
    , SUM(CASE WHEN gpi2="39" THEN days_supply ELSE 0 END) AS lipid_lowering_days_supply
    , SUM(CASE WHEN gpi2="34" THEN Scripts ELSE 0 END)        AS calcium_channel_blk_scripts
    , SUM(CASE WHEN gpi2="34" THEN days_supply ELSE 0 END) AS calcium_channel_blk_days_supply
    , SUM(CASE WHEN gpi2="37" THEN Scripts ELSE 0 END)        AS diuretic_scripts
    , SUM(CASE WHEN gpi2="37" THEN days_supply ELSE 0 END) AS diuretic_days_supply
    , SUM(CASE WHEN gpi2="32" THEN Scripts ELSE 0 END)        AS antianginal_agent_scripts
    , SUM(CASE WHEN gpi2="32" THEN days_supply ELSE 0 END) AS antianginal_agent_days_supply
    , SUM(CASE WHEN gpi2="58" THEN Scripts ELSE 0 END)        AS antidepressant_scripts
    , SUM(CASE WHEN gpi2="58" THEN days_supply ELSE 0 END) AS antidepressant_days_supply
    , SUM(CASE WHEN gpi2="59" THEN Scripts ELSE 0 END)        AS antipsychotic_scripts
    , SUM(CASE WHEN gpi2="59" THEN days_supply ELSE 0 END) AS antipsychotic_days_supply
    , SUM(CASE WHEN gpi2="57" THEN Scripts ELSE 0 END)        AS antianxiety_scripts
    , SUM(CASE WHEN gpi2="57" THEN days_supply ELSE 0 END) AS antianxiety_days_supply
    , SUM(CASE WHEN gpi2="72" THEN Scripts ELSE 0 END)        AS anticonvulsant_scripts
    , SUM(CASE WHEN gpi2="72" THEN days_supply ELSE 0 END) AS anticonvulsant_days_supply
    , SUM(CASE WHEN gpi4="4440" THEN Scripts ELSE 0 END)      AS inhaled_steroid_scripts
    , SUM(CASE WHEN gpi4="4440" THEN days_supply ELSE 0 END) AS inhaled_steroid_days_supply
FROM 
    `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_rx_claims`
GROUP BY 
    asdb_member_key
    , asdb_plan_key
    , index_dt
'

#-- Combine with cohort table
bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_rx_yr2`'

bq query \
--use_legacy_sql=false \
'
CREATE OR REPLACE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_rx_yr2`
--PARTITION BY DATE_TRUNC(asdb_incurred_dt, MONTH)
--CLUSTER BY asdb_plan_key, asdb_member_key, asdb_mos
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
SELECT 
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
    , rx.first_disp_dt
    , rx.last_disp_dt
    , COALESCE(rx.rx_claim_cnt, 0) AS rx_claim_cnt
    , COALESCE(rx.days_supply_sum, 0) AS days_supply_sum
    , COALESCE(rx.ndc_cnt, 0) AS ndc_cnt
    , COALESCE(rx.gpi_cnt, 0) AS gpi_cnt
    , COALESCE(rx.gpi4_cnt, 0) AS gpi4_cnt
    , COALESCE(rx.gpi2_cnt, 0) AS gpi2_cnt
    , COALESCE(rx.retail_fills, 0) AS retail_fills
    , COALESCE(rx.mail_order_fills, 0) AS mail_order_fills
    , COALESCE(rx.generic_fills, 0) AS generic_fills
    , COALESCE(rx.branded_generic_fills, 0) AS branded_generic_fills
    , COALESCE(rx.otc_fills, 0) AS otc_fills
    , COALESCE(rx.ss_brand_fills, 0) AS ss_brand_fills
    , COALESCE(rx.ms_brand_fills, 0) AS ms_brand_fills
    , COALESCE(rx.formulary_fills, 0) AS formulary_fills
    , COALESCE(rx.maint_drug_fills, 0) AS maint_drug_fills
    , COALESCE(rx.antidiabetic_scripts, 0) AS antidiabetic_scripts
    , COALESCE(rx.antidiabetic_days_supply, 0) AS antidiabetic_days_supply
    , COALESCE(rx.beta_blocker_scripts, 0) AS beta_blocker_scripts
    , COALESCE(rx.beta_blocker_days_supply, 0) AS beta_blocker_days_supply
    , COALESCE(rx.antihypertensive_scripts, 0) AS antihypertensive_scripts
    , COALESCE(rx.antihypertensive_days_supply, 0) AS antihypertensive_days_supply
    , COALESCE(rx.lipid_lowering_scripts, 0) AS lipid_lowering_scripts
    , COALESCE(rx.lipid_lowering_days_supply, 0) AS lipid_lowering_days_supply
    , COALESCE(rx.calcium_channel_blk_scripts, 0) AS calcium_channel_blk_scripts
    , COALESCE(rx.calcium_channel_blk_days_supply, 0) AS calcium_channel_blk_days_supply
    , COALESCE(rx.diuretic_scripts, 0) AS diuretic_scripts
    , COALESCE(rx.diuretic_days_supply, 0) AS diuretic_days_supply
    , COALESCE(rx.antianginal_agent_scripts, 0) AS antianginal_agent_scripts
    , COALESCE(rx.antianginal_agent_days_supply, 0) AS antianginal_agent_days_supply
    , COALESCE(rx.antidepressant_scripts, 0) AS antidepressant_scripts
    , COALESCE(rx.antidepressant_days_supply, 0) AS antidepressant_days_supply
    , COALESCE(rx.antipsychotic_scripts, 0) AS antipsychotic_scripts
    , COALESCE(rx.antipsychotic_days_supply, 0) AS antipsychotic_days_supply
    , COALESCE(rx.antianxiety_scripts, 0) AS antianxiety_scripts
    , COALESCE(rx.antianxiety_days_supply, 0) AS antianxiety_days_supply
    , COALESCE(rx.anticonvulsant_scripts, 0) AS anticonvulsant_scripts
    , COALESCE(rx.anticonvulsant_days_supply, 0) AS anticonvulsant_days_supply
    , COALESCE(rx.inhaled_steroid_scripts, 0) AS inhaled_steroid_scripts
    , COALESCE(rx.inhaled_steroid_days_supply, 0) AS inhaled_steroid_days_supply
FROM 
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `'$ST'`) AS st
LEFT JOIN 
    `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_rx_tmp` rx
        ON st.asdb_member_key = rx.asdb_member_key
        AND st.asdb_plan_key = rx.asdb_plan_key
'

#--- Remove temporary tables
bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_rx_tmp`'
