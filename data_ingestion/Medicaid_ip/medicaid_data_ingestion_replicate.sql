-- =====================================================================================
-- Medicaid IP Model - Data Ingestion Pipeline Replication
-- =====================================================================================
-- This SQL file replicates the exact feature and outcome generation procedure from
-- the original shell scripts in training_pull/ and Outcome_pulls/
-- 
-- Original Authors: Elle Palmer
-- Consolidated by: AI Assistant
-- Last Modified: January 2026
--
-- IMPORTANT: This script preserves all original logic, table structures, and naming
-- conventions. Do not modify without understanding the full pipeline dependencies.
-- =====================================================================================

-- =====================================================================================
-- CONFIGURATION VARIABLES (Set these before running)
-- =====================================================================================
-- To use this script:
-- 1. Update the variable values below to match your environment
-- 2. Run the entire script in BigQuery as a single session (scripting mode)
-- 3. All table names use the configured values
-- =====================================================================================

DECLARE GCP_PROJECT STRING DEFAULT 'edp-prod-storage';
DECLARE GCP_DB STRING DEFAULT 'edp_ent_sdoheir_cns';
DECLARE PREFIX STRING DEFAULT 'a964286_medicaid_ip_final_dataset_4_te_experiment_2023';
DECLARE OWNER STRING DEFAULT 'zhaopeng_xing_aetna_com';
DECLARE COST_CENTER STRING DEFAULT '13070';
DECLARE ELIG_START_DT DATE DEFAULT DATE('2023-01-01');
DECLARE ELIG_END_DT DATE DEFAULT DATE('2023-12-30');
DECLARE SDOH_YEAR INT64 DEFAULT 2023;
DECLARE DEFAULT_EXP INT64 DEFAULT 180;  -- Table expiration in days

-- Derived fully qualified table prefix
DECLARE TABLE_PREFIX STRING;
SET TABLE_PREFIX = CONCAT(GCP_PROJECT, '.', GCP_DB, '.', PREFIX);

-- =====================================================================================
-- NOTE: BigQuery scripting does not support variable substitution in table names.
-- The table names below use hardcoded values that match the DECLARE defaults above.
-- If you change the DECLARE values, you must also update the table names throughout.
-- 
-- Current Configuration:
--   Project:  edp-prod-storage
--   Dataset:  edp_ent_sdoheir_cns  
--   Prefix:   a964286_medicaid_ip_final_dataset_4_te_experiment_2023
-- =====================================================================================

-- =====================================================================================
-- SECTION 0: BASE MEMBER TABLE CREATION (001_Membership.sh - prerequisite)
-- =====================================================================================
-- Creates the base member table filtered by eligibility date range.
-- This is the prerequisite table that 001a_Membership_Index.sh reads from.
-- 
-- ⚠️ DATE FILTER APPLIED HERE: Members are filtered to only include those with
--    eligibility dates between ELIG_START_DT and ELIG_END_DT (2023-01-01 to 2023-12-30)
-- =====================================================================================

-- =====================================================================================
-- Medicaid IP Model - Data Ingestion Pipeline Replication
-- =====================================================================================
-- This SQL file replicates the exact feature and outcome generation procedure from
-- the original shell scripts in training_pull/ and Outcome_pulls/
-- 
-- Original Authors: Elle Palmer
-- Last Modified: January 2026
--
-- IMPORTANT: This script preserves all original logic, table structures, and naming
-- conventions. Do not modify without understanding the full pipeline dependencies.
-- =====================================================================================

-- =====================================================================================
-- SECTION 0: BASE MEMBER TABLE CREATION (001_Membership.sh - prerequisite)
-- =====================================================================================
-- Creates the base member table filtered by eligibility date range.
-- This is the prerequisite table that 001a_Membership_Index.sh reads from.
-- 
-- ⚠️ DATE FILTER APPLIED HERE: Members are filtered to only include those with
--    eligibility dates between ELIG_START_DT and ELIG_END_DT (2023-01-01 to 2023-12-30)
-- =====================================================================================

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member`;

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
SELECT DISTINCT
    asdb_member_key
    , asdb_plan_key
    , SPLIT(asdb_plan_code_version, "_")[offset(0)] AS asdb_plan_nm
    , CAST(asdb_elig_dt AS DATE) AS asdb_elig_dt
    , coa_population_category
    -- ⚠️ DERIVED COLUMN: coa_population_group is computed from coa_population_category
    , CASE 
        WHEN TRIM(coa_population_category) = "ABD Non Dual LTSS" OR
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
             THEN "Dual Int"
        WHEN TRIM(coa_population_category) = "CHIP" OR
             TRIM(coa_population_category) = "TANF" 
             THEN "TANF/CHIP"
        WHEN TRIM(coa_population_category) = "Expansion" 
             THEN "Expansion"
        ELSE "Other"
      END AS coa_population_group
FROM 
    `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ELIG_DATA_MBR_PER_MTH`
WHERE 
    -- ⚠️ DATE FILTER: Only include members with eligibility in the specified date range
    CAST(asdb_elig_dt AS DATE) BETWEEN DATE('2023-01-01') AND DATE('2023-12-30')
;


-- =====================================================================================
-- SECTION 1: MEMBER INDEX CREATION (001a_Membership_Index.sh)
-- =====================================================================================
-- Creates the base member population with index dates for feature extraction

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`;

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
WITH pre AS (
  SELECT
     asdb_member_key
    , asdb_plan_key
    , asdb_elig_dt AS index_dt
    , coa_population_category
    , coa_population_group
    , ROW_NUMBER() OVER(PARTITION BY asdb_member_key ORDER BY RAND()) AS pos
  FROM 
    `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member`
)
SELECT  
    pre.*   
FROM 
    pre
WHERE 
    pos <= 1
;
-- =====================================================================================
-- SECTION 2A: MEDICAL CLAIMS EXTRACTION - YEAR 1 (002_Med_Claims_yr1.sh)
-- =====================================================================================
-- Pulls medical claims for 12 months prior to index date

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_med_claims_yr1`;

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_med_claims_yr1`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
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
       , clm.asdb_pcp_prov_key
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
       (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
INNER JOIN 
       (WITH latest_partitions AS 
           (SELECT
               asdb_member_key
               , asdb_plan_key
               , claimid
               , asdb_svc_prov_key
               , asdb_pcp_prov_key
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
         CAST(asdb_incurred_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 12 MONTH) AND DATE_SUB(st.index_dt, INTERVAL 1 DAY)
;

-- =====================================================================================
-- SECTION 2B: MEDICAL CLAIMS EXTRACTION - YEAR 2 (002_Med_Claims_yr2.sh)
-- =====================================================================================
-- Pulls medical claims for 13-24 months prior to index date

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_med_claims_yr2`;

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_med_claims_yr2`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
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
       (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
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
;
-- =====================================================================================
-- SECTION 3A: COST AND UTILIZATION - YEAR 1 (003_Cost_and_Utilization_yr1.sh)
-- =====================================================================================

-- ----- 3A.1: ED Cases Year 1 -----
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ed_cases_yr1`;

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ed_cases_yr1`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
SELECT 
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
    , mc.asdb_incurred_dt AS ed_vis_dt
    , mc.event_ct
    , mc.prindiag
    , mc.cost
    , mc.op_severitylvl
    , CASE WHEN TRIM(nyu.avoidable_ind) = "Y" THEN 1 
        ELSE 0 
        END AS avoidable_er_visits
    , CASE WHEN TRIM(nyu.er_type) = "UNNECESSARY" THEN 1 
        ELSE 0 
        END AS unnecessary_er_visits
    , CASE WHEN TRIM(nyu.er_type) = "PREVENTABLE" THEN 1 
        ELSE 0 
        END AS preventable_er_visits
FROM 
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
INNER JOIN 
    `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ASDB_ICE_OP` AS mc
        ON st.asdb_member_key=mc.asdb_member_key
        AND st.asdb_plan_key=mc.asdb_plan_key
LEFT JOIN 
    `edp-prod-storage.edp_ent_core_src.ICD10_X_ER_TYPE` AS nyu
        ON TRIM(mc.prindiag) = TRIM(nyu.dx_cd)
WHERE 
    CAST(mc.asdb_incurred_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 12 MONTH) AND DATE_SUB(st.index_dt, INTERVAL 1 DAY)
    AND CAST(mc.asdb_coe_id AS INT64) = 20100
    AND event_ct=1
;

-- ----- 3A.2: ED Summary Year 1 -----
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ed_yr1`;

CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ed_yr1`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
SELECT 
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
    , COALESCE(SUM(mc.event_ct), 0) AS sum_ed_visits
    , CASE WHEN COALESCE(SUM(mc.event_ct), 0) > 0 THEN 1 ELSE 0 END AS ed_flag
    , COALESCE(SUM(mc.cost), 0) AS sum_ed_cost
    , COALESCE(SUM(mc.avoidable_er_visits), 0) AS sum_avoidable
    , COALESCE(SUM(mc.unnecessary_er_visits), 0) AS sum_unnecessary
    , COALESCE(SUM(mc.preventable_er_visits), 0) AS sum_preventable
    , MAX(mc.op_severitylvl) AS max_ed_severitylvl
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "1-Low" THEN mc.event_ct ELSE 0 END) AS low_sev_ed_visits
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "2-Low/Med" THEN mc.event_ct ELSE 0 END) AS low_med_sev_ed_visits
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "3-Med" THEN mc.event_ct ELSE 0 END) AS med_sev_ed_visits
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "4-Med/High" THEN mc.event_ct ELSE 0 END) AS med_high_sev_ed_visits
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "5-High" THEN mc.event_ct ELSE 0 END) AS high_sev_ed_visits
    , CASE WHEN SUM(CASE WHEN TRIM(mc.op_severitylvl) = "1-Low" THEN mc.event_ct ELSE 0 END) > 0 THEN 1 
        ELSE 0 
        END AS low_sev_ed_flag
    , CASE WHEN SUM(CASE WHEN TRIM(mc.op_severitylvl) = "2-Low/Med" THEN mc.event_ct ELSE 0 END) > 0 THEN 1 
        ELSE 0 
        END AS low_med_sev_ed_flag
    , CASE WHEN SUM(CASE WHEN TRIM(mc.op_severitylvl) = "3-Med" THEN mc.event_ct ELSE 0 END) > 0 THEN 1 
        ELSE 0 
        END AS med_sev_ed_flag
    , CASE WHEN SUM(CASE WHEN TRIM(mc.op_severitylvl) = "4-Med/High" THEN mc.event_ct ELSE 0 END) > 0 THEN 1 
        ELSE 0 
        END AS med_high_sev_ed_flag
    , CASE WHEN SUM(CASE WHEN TRIM(mc.op_severitylvl) = "5-High" THEN mc.event_ct ELSE 0 END) > 0 THEN 1 
        ELSE 0 
        END AS high_sev_ed_flag
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "1-Low" THEN mc.cost ELSE 0 END) AS low_sev_ed_cost
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "2-Low/Med" THEN mc.cost ELSE 0 END) AS low_med_sev_ed_cost
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "3-Med" THEN mc.cost ELSE 0 END) AS med_sev_ed_cost
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "4-Med/High" THEN mc.cost ELSE 0 END) AS med_high_sev_ed_cost
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "5-High" THEN mc.cost ELSE 0 END) AS high_sev_ed_cost
FROM 
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
LEFT JOIN 
    `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ed_cases_yr1` AS mc
        ON st.asdb_member_key = mc.asdb_member_key
        AND st.asdb_plan_key = mc.asdb_plan_key
GROUP BY 
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
;

-- ----- 3A.3: IP Cases Year 1 -----
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ip_cases_yr1`;

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ip_cases_yr1`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
SELECT 
    mc.asdb_member_key
    , mc.asdb_plan_key
    , st.index_dt
    , mc.asdb_event_start_dt
    , mc.asdb_event_end_dt
    , mc.final_discharge_dt
    , mc.prindiag
    , CASE WHEN mc.asdb_coe_id IN (10200,10700,10800)
            THEN "Acute"
        WHEN mc.asdb_coe_id IN (10000,10100,10300)
            THEN "Maternity/Infant"
        ELSE "Non-Acute"
        END AS ip_type
    , DATE_DIFF(mc.final_discharge_dt, mc.asdb_event_start_dt, DAY) AS calc_los
    , mc.event_ct
    , mc.admit_los
    , mc.paid_los
    , mc.cost AS ip_paid_amt
FROM
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
INNER JOIN 
    `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ICE_IP`  AS mc
        ON st.asdb_member_key=mc.asdb_member_key
        AND st.asdb_plan_key=mc.asdb_plan_key
WHERE 
    CAST(mc.asdb_event_start_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 12 MONTH) AND DATE_SUB(st.index_dt, INTERVAL 1 DAY)
    AND mc.event_ct=1
;

-- ----- 3A.4: IP Summary Year 1 -----
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ip_yr1`;

CREATE TABLE  `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ip_yr1`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
WITH acute AS (
    SELECT
        asdb_member_key
        , asdb_plan_key
        , index_dt
        , CASE WHEN SUM(event_ct) > 0 THEN 1 
            ELSE 0 
            END AS acute_ip_flag
        , SUM(event_ct) AS sum_acute_ip_admits
        , SUM(calc_los) AS sum_acute_calc_los
        , SUM(admit_los) AS sum_acute_admit_los
        , SUM(paid_los) AS sum_acute_paid_los
        , SUM(ip_paid_amt) AS sum_acute_ip_cost
    FROM  
        `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ip_cases_yr1`
    WHERE 
        ip_type = "Acute"
    GROUP BY 
        asdb_member_key
        , asdb_plan_key
        , index_dt
),
nonacute AS (
    SELECT 
        asdb_member_key
        , asdb_plan_key
        , CASE WHEN SUM(event_ct) > 0 THEN 1 
            ELSE 0 
            END AS non_acute_ip_flag
        , SUM(event_ct) AS sum_non_acute_ip_admits
        , SUM(calc_los) AS sum_non_acute_calc_los
        , SUM(admit_los) AS sum_non_acute_admit_los
        , SUM(paid_los) AS sum_non_acute_paid_los
        , SUM(ip_paid_amt) AS sum_non_acute_ip_cost
    FROM 
        `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ip_cases_yr1`
    WHERE 
        ip_type="Non-Acute"
    GROUP BY 
        asdb_member_key
        , asdb_plan_key
),
maternity AS (
    SELECT asdb_member_key,
        asdb_plan_key,
        CASE WHEN SUM(event_ct)>0 THEN 1 ELSE 0 END AS maternity_ip_flag,
        SUM(event_ct) AS sum_maternity_ip_admits,
        SUM(calc_los) AS sum_maternity_calc_los,
        SUM(admit_los) AS sum_maternity_admit_los,
        SUM(paid_los) AS sum_maternity_paid_los,
        SUM(ip_paid_amt) AS sum_maternity_ip_cost
    FROM 
        `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ip_cases_yr1`
    WHERE 
        ip_type="Maternity/Infant"
    GROUP BY 
        asdb_member_key
        , asdb_plan_key
)
SELECT 
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
    , COALESCE(a.acute_ip_flag, 0) AS acute_ip_flag
    , COALESCE(a.sum_acute_ip_admits, 0) AS sum_acute_ip_admits
    , COALESCE(a.sum_acute_calc_los, 0) AS sum_acute_calc_los
    , COALESCE(a.sum_acute_admit_los, 0) AS sum_acute_admit_los
    , COALESCE(a.sum_acute_paid_los, 0) AS sum_acute_paid_los
    , COALESCE(a.sum_acute_ip_cost, 0) AS sum_acute_ip_cost
    , COALESCE(b.non_acute_ip_flag, 0) AS non_acute_ip_flag
    , COALESCE(b.sum_non_acute_ip_admits, 0) AS sum_non_acute_ip_admits
    , COALESCE(b.sum_non_acute_calc_los, 0) AS sum_non_acute_calc_los
    , COALESCE(b.sum_non_acute_admit_los, 0) AS sum_non_acute_admit_los
    , COALESCE(b.sum_non_acute_paid_los, 0) AS sum_non_acute_paid_los
    , COALESCE(b.sum_non_acute_ip_cost, 0) AS sum_non_acute_ip_cost
    , COALESCE(c.maternity_ip_flag, 0) AS maternity_ip_flag
    , COALESCE(c.sum_maternity_ip_admits, 0) AS sum_maternity_ip_admits
FROM 
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
LEFT JOIN 
    acute AS a
        ON st.asdb_member_key = a.asdb_member_key
        AND st.asdb_plan_key = a.asdb_plan_key
LEFT JOIN 
    nonacute AS b
        ON st.asdb_member_key = b.asdb_member_key
        AND st.asdb_plan_key = b.asdb_plan_key
LEFT JOIN
    maternity AS c
        ON st.asdb_member_key = c.asdb_member_key
        AND st.asdb_plan_key = c.asdb_plan_key
;

-- ----- 3A.5: OP Summary Year 1 -----
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_op_yr1`;

CREATE TABLE  `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_op_yr1`
PARTITION BY RANGE_BUCKET(asdb_plan_key, GENERATE_ARRAY(0,100,1))
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
WITH clm AS(
    SELECT 
        *
        , CASE WHEN ROW_NUMBER() OVER(PARTITION BY asdb_member_key, asdb_plan_key, asdb_svc_prov_key, asdb_incurred_dt) = 1 THEN 1 
            ELSE 0 
            END AS op_ct
    FROM 
        `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_med_claims_yr1`
    WHERE 
        TRIM(asdb_coe_general_type) != "Inpatient"
        AND TRIM(emis_cat) != "Institutional Services"
        AND TRIM(emis_cat) != "Emergency"
)
SELECT 
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
    , COALESCE(SUM(clm.paid_amt), 0) AS sum_op_cost
    , COALESCE(SUM(clm.op_ct), 0) AS sum_op_visits
    , MAX(CASE WHEN clm.op_ct=1 THEN 1 ELSE 0 END) AS op_flag
FROM 
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
LEFT JOIN 
    clm
        ON st.asdb_member_key = clm.asdb_member_key
        AND st.asdb_plan_key = clm.asdb_plan_key
GROUP BY 
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
;

-- ----- 3A.6: Medical Claims Flag Year 1 -----
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_med_claims_flag_yr1`;

CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_med_claims_flag_yr1`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
WITH clm AS (
    SELECT 
        *
        , CASE WHEN TRIM(asdb_coe_general_type)="Inpatient" OR TRIM(emis_cat)="Institutional Services" THEN "Inpatient"
            WHEN TRIM(emis_cat)="Emergency" THEN "Emergency"
            ELSE "Outpatient" 
            END AS plc_svc_ctg
    FROM 
        `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_med_claims_yr1`
)
SELECT 
    clm.*
    , fac.prov_specialty
    ---- Cost Metrics
    , CASE WHEN plc_svc_ctg="Inpatient" THEN paid_amt ELSE 0 END AS inpatient_cost
    , CASE WHEN plc_svc_ctg="Emergency" THEN paid_amt ELSE 0 END AS emergency_cost
    , CASE WHEN plc_svc_ctg="Outpatient" THEN paid_amt ELSE 0 END AS outpatient_cost
    , CASE WHEN TRIM(emis_cat)="Community-Based Services" THEN paid_amt ELSE 0 END AS emis_community_cost
    , CASE WHEN TRIM(emis_cat)="Emergency" THEN paid_amt ELSE 0 END AS emis_ed_cost
    , CASE WHEN TRIM(emis_cat)="Home Health" THEN paid_amt ELSE 0 END AS emis_hh_cost
    , CASE WHEN TRIM(emis_cat)="Home-Based Services" THEN paid_amt ELSE 0 END AS emis_home_cost
    , CASE WHEN TRIM(emis_cat)="Inpatient Facility" THEN paid_amt ELSE 0 END AS emis_ip_cost
    , CASE WHEN TRIM(emis_cat)="Institutional Services" THEN paid_amt ELSE 0 END AS emis_ins_cost
    , CASE WHEN TRIM(emis_cat)="Laboratory" THEN paid_amt ELSE 0 END AS emis_lab_cost
    , CASE WHEN TRIM(emis_cat)="Medical Pharmacy" THEN paid_amt ELSE 0 END AS emis_mrx_cost
    , CASE WHEN TRIM(emis_cat)="Mental Health" THEN paid_amt ELSE 0 END AS emis_mh_cost
    , CASE WHEN TRIM(emis_cat)="Misc. Medical" THEN paid_amt ELSE 0 END AS emis_misc_cost
    , CASE WHEN TRIM(emis_cat)="Primary Physician" THEN paid_amt ELSE 0 END AS emis_pcp_cost
    , CASE WHEN TRIM(emis_cat)="Radiology" THEN paid_amt ELSE 0 END AS emis_radio_cost
    , CASE WHEN TRIM(emis_cat)="Selected Ambulatory Facility" THEN paid_amt ELSE 0 END AS emis_ambul_cost
    , CASE WHEN TRIM(emis_cat)="Specialist Physician" THEN paid_amt ELSE 0 END AS emis_spec_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" THEN paid_amt ELSE 0 END AS coe_ltc_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Inpatient" AND TRIM(asdb_coe_sub_cat)="Hospital" THEN paid_amt ELSE 0 END AS coe_ip_hos_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Inpatient" AND TRIM(asdb_coe_sub_cat)="Non Hospital" THEN paid_amt ELSE 0 END AS coe_ip_non_hos_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Laboratory" AND TRIM(asdb_coe_sub_cat)="Professional" THEN paid_amt ELSE 0 END AS coe_lab_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" AND TRIM(asdb_coe_sub_cat)="Community Based Services" THEN paid_amt ELSE 0 END AS coe_ltc_community_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" AND TRIM(asdb_coe_sub_cat)="Home Based Services" THEN paid_amt ELSE 0 END AS coe_ltc_home_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" AND TRIM(asdb_coe_sub_cat)="Institution" THEN paid_amt ELSE 0 END AS coe_ltc_ins_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Other" AND TRIM(asdb_coe_sub_cat)="Professional" THEN paid_amt ELSE 0 END AS coe_other_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Outpatient" AND TRIM(asdb_coe_sub_cat)="Hospital" THEN paid_amt ELSE 0 END AS coe_op_hos_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Outpatient" AND TRIM(asdb_coe_sub_cat)="Non Hospital" THEN paid_amt ELSE 0 END AS coe_op_non_hos_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Anesthesia" THEN paid_amt ELSE 0 END AS coe_anesth_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Evaluation & Management" THEN paid_amt ELSE 0 END AS coe_eval_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Maternity" THEN paid_amt ELSE 0 END AS coe_maternity_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Medicine" THEN paid_amt ELSE 0 END AS coe_mrx_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Mental Health" THEN paid_amt ELSE 0 END AS coe_mh_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Physician" THEN paid_amt ELSE 0 END AS coe_phy_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Surgery" THEN paid_amt ELSE 0 END AS coe_surg_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Radiology" AND TRIM(asdb_coe_sub_cat)="Professional" THEN paid_amt ELSE 0 END AS coe_radio_cost
    , CASE WHEN TRIM(fac.prov_specialty)="Urgent Care" OR TRIM(location)="20" OR TRIM(servcode)="S9083" THEN paid_amt ELSE 0 END AS uc_cost
    ---- Utilization Metrics
    , CASE WHEN TRIM(emis_cat)="Community-Based Services" THEN 1 ELSE 0 END AS emis_community_clm
    , CASE WHEN TRIM(emis_cat)="Emergency" THEN 1 ELSE 0 END AS emis_ed_clm
    , CASE WHEN TRIM(emis_cat)="Home Health" THEN 1 ELSE 0 END AS emis_hh_clm
    , CASE WHEN TRIM(emis_cat)="Home-Based Services" THEN 1 ELSE 0 END AS emis_home_clm
    , CASE WHEN TRIM(emis_cat)="Inpatient Facility" THEN 1 ELSE 0 END AS emis_ip_clm
    , CASE WHEN TRIM(emis_cat)="Institutional Services" THEN 1 ELSE 0 END AS emis_ins_clm
    , CASE WHEN TRIM(emis_cat)="Laboratory" THEN 1 ELSE 0 END AS emis_lab_clm
    , CASE WHEN TRIM(emis_cat)="Medical Pharmacy" THEN 1 ELSE 0 END AS emis_mrx_clm
    , CASE WHEN TRIM(emis_cat)="Mental Health" THEN 1 ELSE 0 END AS emis_mh_clm
    , CASE WHEN TRIM(emis_cat)="Misc. Medical" THEN 1 ELSE 0 END AS emis_misc_clm
    , CASE WHEN TRIM(emis_cat)="Primary Physician" THEN 1 ELSE 0 END AS emis_pcp_clm
    , CASE WHEN TRIM(emis_cat)="Radiology" THEN 1 ELSE 0 END AS emis_radio_clm
    , CASE WHEN TRIM(emis_cat)="Selected Ambulatory Facility" THEN 1 ELSE 0 END AS emis_ambul_clm
    , CASE WHEN TRIM(emis_cat)="Specialist Physician" THEN 1 ELSE 0 END AS emis_spec_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" THEN 1 ELSE 0 END as ltc_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Inpatient" AND TRIM(asdb_coe_sub_cat)="Hospital" THEN 1 ELSE 0 END AS coe_ip_hos_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Inpatient" AND TRIM(asdb_coe_sub_cat)="Non Hospital" THEN 1 ELSE 0 END AS coe_ip_non_hos_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Laboratory" AND TRIM(asdb_coe_sub_cat)="Professional" THEN 1 ELSE 0 END AS coe_lab_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" AND TRIM(asdb_coe_sub_cat)="Community Based Services" THEN 1 ELSE 0 END AS coe_ltc_community_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" AND TRIM(asdb_coe_sub_cat)="Home Based Services" THEN 1 ELSE 0 END AS coe_ltc_home_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" AND TRIM(asdb_coe_sub_cat)="Institution" THEN 1 ELSE 0 END AS coe_ltc_ins_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Other" AND TRIM(asdb_coe_sub_cat)="Professional" THEN 1 ELSE 0 END AS coe_other_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Outpatient" AND TRIM(asdb_coe_sub_cat)="Hospital" THEN 1 ELSE 0 END AS coe_op_hos_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Outpatient" AND TRIM(asdb_coe_sub_cat)="Non Hospital" THEN 1 ELSE 0 END AS coe_op_non_hos_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Anesthesia" THEN 1 ELSE 0 END AS coe_anesth_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Evaluation & Management" THEN 1 ELSE 0 END AS coe_eval_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Maternity" THEN 1 ELSE 0 END AS coe_maternity_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Medicine" THEN 1 ELSE 0 END AS coe_mrx_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Mental Health" THEN 1 ELSE 0 END AS coe_mh_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Physician" THEN 1 ELSE 0 END AS coe_phy_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Surgery" THEN 1 ELSE 0 END AS coe_surg_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Radiology" AND TRIM(asdb_coe_sub_cat)="Professional" THEN 1 ELSE 0 END AS coe_radio_clm
    , CASE WHEN TRIM(fac.prov_specialty)="Urgent Care" OR TRIM(location)="20" OR TRIM(servcode)="S9083" THEN 1 ELSE 0 END AS uc_clm
    , CASE WHEN (TRIM(clm.revcode) in ("0760","0761","0762","0769") AND (TRIM(clm.billtype) like "13%" or TRIM(clm.billtype) like "85%") AND (TRIM(clm.servcode)) in ("99217","99218","99219","99202","G0378","G0379","")) THEN 1 ELSE 0 END AS obs_clm
FROM 
    clm
LEFT JOIN 
    `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_SVC_PROV` AS fac
        ON clm.asdb_svc_prov_key = fac.asdb_svc_prov_key 
        AND clm.asdb_plan_key = fac.asdb_plan_key
;

-- ----- 3A.7: Other Cost Utilization Summary Year 1 -----
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_other_cost_utilization_yr1`;

CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_other_cost_utilization_yr1`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
SELECT
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
    , COUNT(DISTINCT clm.claimid) AS claim_cnt
    , COUNT(*) AS claim_line_cnt
    , COALESCE(SUM(clm.paid_amt), 0) AS sum_paid_amt
    -- Cost
    , COALESCE(SUM(clm.inpatient_cost), 0) AS inpatient_cost
    , COALESCE(SUM(clm.emergency_cost), 0) AS emergency_cost
    , COALESCE(SUM(clm.outpatient_cost), 0) AS outpatient_cost
    , COALESCE(SUM(clm.emis_community_cost), 0) AS emis_community_cost
    , COALESCE(SUM(clm.emis_ed_cost), 0) AS emis_ed_cost
    , COALESCE(SUM(clm.emis_hh_cost), 0) AS emis_hh_cost
    , COALESCE(SUM(clm.emis_home_cost), 0) AS emis_home_cost
    , COALESCE(SUM(clm.emis_ip_cost), 0) AS emis_ip_cost
    , COALESCE(SUM(clm.emis_ins_cost), 0) AS emis_ins_cost
    , COALESCE(SUM(clm.emis_lab_cost), 0) AS emis_lab_cost
    , COALESCE(SUM(clm.emis_mrx_cost), 0) AS emis_mrx_cost
    , COALESCE(SUM(clm.emis_mh_cost), 0) AS emis_mh_cost
    , COALESCE(SUM(clm.emis_misc_cost), 0) AS emis_misc_cost
    , COALESCE(SUM(clm.emis_pcp_cost), 0) AS emis_pcp_cost
    , COALESCE(SUM(clm.emis_radio_cost), 0) AS emis_radio_cost
    , COALESCE(SUM(clm.emis_ambul_cost), 0) AS emis_ambul_cost
    , COALESCE(SUM(clm.emis_spec_cost), 0) AS emis_spec_cost
    , COALESCE(SUM(clm.coe_ltc_cost), 0) AS coe_ltc_cost
    , COALESCE(SUM(clm.coe_ip_hos_cost), 0) AS coe_ip_hos_cost
    , COALESCE(SUM(clm.coe_ip_non_hos_cost), 0) AS coe_ip_non_hos_cost
    , COALESCE(SUM(clm.coe_lab_cost), 0) AS coe_lab_cost
    , COALESCE(SUM(clm.coe_ltc_community_cost), 0) AS coe_ltc_community_cost
    , COALESCE(SUM(clm.coe_ltc_home_cost), 0) AS coe_ltc_home_cost
    , COALESCE(SUM(clm.coe_ltc_ins_cost), 0) AS coe_ltc_ins_cost
    , COALESCE(SUM(clm.coe_other_cost), 0) AS coe_other_cost
    , COALESCE(SUM(clm.coe_op_hos_cost), 0) AS coe_op_hos_cost
    , COALESCE(SUM(clm.coe_op_non_hos_cost), 0) AS coe_op_non_hos_cost
    , COALESCE(SUM(clm.coe_anesth_cost), 0) AS coe_anesth_cost
    , COALESCE(SUM(clm.coe_eval_cost), 0) AS coe_eval_cost
    , COALESCE(SUM(clm.coe_maternity_cost), 0) AS coe_maternity_cost
    , COALESCE(SUM(clm.coe_mrx_cost), 0) AS coe_mrx_cost
    , COALESCE(SUM(clm.coe_mh_cost), 0) AS coe_mh_cost
    , COALESCE(SUM(clm.coe_phy_cost), 0) AS coe_phy_cost
    , COALESCE(SUM(clm.coe_surg_cost), 0) AS coe_surg_cost
    , COALESCE(SUM(clm.coe_radio_cost), 0) AS coe_radio_cost
    , COALESCE(SUM(clm.uc_cost), 0) AS uc_cost
    -- Utilization
    , COALESCE(SUM(clm.emis_community_clm), 0) AS emis_community_clm
    , COALESCE(SUM(clm.emis_ed_clm), 0) AS emis_ed_clm
    , COALESCE(SUM(clm.emis_hh_clm), 0) AS emis_hh_clm
    , COALESCE(SUM(clm.emis_home_clm), 0) AS emis_home_clm
    , COALESCE(SUM(clm.emis_ip_clm), 0) AS emis_ip_clm
    , COALESCE(SUM(clm.emis_ins_clm), 0) AS emis_ins_clm
    , COALESCE(SUM(clm.emis_lab_clm), 0) AS emis_lab_clm
    , COALESCE(SUM(clm.emis_mrx_clm), 0) AS emis_mrx_clm
    , COALESCE(SUM(clm.emis_mh_clm), 0) AS emis_mh_clm
    , COALESCE(SUM(clm.emis_misc_clm), 0) AS emis_misc_clm
    , COALESCE(SUM(clm.emis_pcp_clm), 0) AS emis_pcp_clm
    , COALESCE(SUM(clm.emis_radio_clm), 0) AS emis_radio_clm
    , COALESCE(SUM(clm.emis_ambul_clm), 0) AS emis_ambul_clm
    , COALESCE(SUM(clm.emis_spec_clm), 0) AS emis_spec_clm
    , COALESCE(SUM(clm.ltc_clm), 0) AS ltc_clm
    , COALESCE(SUM(clm.coe_ip_hos_clm), 0) AS coe_ip_hos_clm
    , COALESCE(SUM(clm.coe_ip_non_hos_clm), 0) AS coe_ip_non_hos_clm
    , COALESCE(SUM(clm.coe_lab_clm), 0) AS coe_lab_clm
    , COALESCE(SUM(clm.coe_ltc_community_clm), 0) AS coe_ltc_community_clm
    , COALESCE(SUM(clm.coe_ltc_home_clm), 0) AS coe_ltc_home_clm
    , COALESCE(SUM(clm.coe_ltc_ins_clm), 0) AS coe_ltc_ins_clm
    , COALESCE(SUM(clm.coe_other_clm), 0) AS coe_other_clm
    , COALESCE(SUM(clm.coe_op_hos_clm), 0) AS coe_op_hos_clm
    , COALESCE(SUM(clm.coe_op_non_hos_clm), 0) AS coe_op_non_hos_clm
    , COALESCE(SUM(clm.coe_anesth_clm), 0) AS coe_anesth_clm
    , COALESCE(SUM(clm.coe_eval_clm), 0) AS coe_eval_clm
    , COALESCE(SUM(clm.coe_maternity_clm), 0) AS coe_maternity_clm
    , COALESCE(SUM(clm.coe_mrx_clm), 0) AS coe_mrx_clm
    , COALESCE(SUM(clm.coe_mh_clm), 0) AS coe_mh_clm
    , COALESCE(SUM(clm.coe_phy_clm), 0) AS coe_phy_clm
    , COALESCE(SUM(clm.coe_surg_clm), 0) AS coe_surg_clm
    , COALESCE(SUM(clm.coe_radio_clm), 0) AS coe_radio_clm
    , COALESCE(SUM(clm.uc_clm), 0) AS uc_clm
    , COALESCE(SUM(clm.obs_clm), 0) AS obs_clm
FROM 
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
LEFT JOIN 
    `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_med_claims_flag_yr1` AS clm
        ON st.asdb_member_key=clm.asdb_member_key
        AND st.asdb_plan_key=clm.asdb_plan_key
GROUP BY 
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
;

-- =====================================================================================
-- SECTION 3B: COST AND UTILIZATION - YEAR 2 (003_Cost_and_Utilization_yr2.sh)
-- =====================================================================================
-- Note: Year 2 uses date range: 24 months to 13 months before index_dt

-- ----- 3B.1: ED Cases Year 2 -----
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ed_cases_yr2`;

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ed_cases_yr2`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
SELECT 
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
    , mc.asdb_incurred_dt AS ed_vis_dt
    , mc.event_ct
    , mc.prindiag
    , mc.cost
    , mc.op_severitylvl
    , CASE WHEN TRIM(nyu.avoidable_ind) = "Y" THEN 1 ELSE 0 END AS avoidable_er_visits
    , CASE WHEN TRIM(nyu.er_type) = "UNNECESSARY" THEN 1 ELSE 0 END AS unnecessary_er_visits
    , CASE WHEN TRIM(nyu.er_type) = "PREVENTABLE" THEN 1 ELSE 0 END AS preventable_er_visits
FROM 
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
INNER JOIN 
    `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ASDB_ICE_OP` AS mc
        ON st.asdb_member_key=mc.asdb_member_key
        AND st.asdb_plan_key=mc.asdb_plan_key
LEFT JOIN 
    `edp-prod-storage.edp_ent_core_src.ICD10_X_ER_TYPE` AS nyu
        ON TRIM(mc.prindiag) = TRIM(nyu.dx_cd)
WHERE 
    CAST(mc.asdb_incurred_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 24 MONTH) AND DATE_SUB(DATE_SUB(st.index_dt, INTERVAL 1 DAY), INTERVAL 12 MONTH)
    AND CAST(mc.asdb_coe_id AS INT64) = 20100
    AND event_ct=1
;

-- ----- 3B.2: ED Summary Year 2 -----
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ed_yr2`;

CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ed_yr2`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
SELECT 
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
    , COALESCE(SUM(mc.event_ct), 0) AS sum_ed_visits
    , CASE WHEN COALESCE(SUM(mc.event_ct), 0) > 0 THEN 1 ELSE 0 END AS ed_flag
    , COALESCE(SUM(mc.cost), 0) AS sum_ed_cost
    , COALESCE(SUM(mc.avoidable_er_visits), 0) AS sum_avoidable
    , COALESCE(SUM(mc.unnecessary_er_visits), 0) AS sum_unnecessary
    , COALESCE(SUM(mc.preventable_er_visits), 0) AS sum_preventable
    , MAX(mc.op_severitylvl) AS max_ed_severitylvl
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "1-Low" THEN mc.event_ct ELSE 0 END) AS low_sev_ed_visits
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "2-Low/Med" THEN mc.event_ct ELSE 0 END) AS low_med_sev_ed_visits
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "3-Med" THEN mc.event_ct ELSE 0 END) AS med_sev_ed_visits
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "4-Med/High" THEN mc.event_ct ELSE 0 END) AS med_high_sev_ed_visits
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "5-High" THEN mc.event_ct ELSE 0 END) AS high_sev_ed_visits
    , CASE WHEN SUM(CASE WHEN TRIM(mc.op_severitylvl) = "1-Low" THEN mc.event_ct ELSE 0 END) > 0 THEN 1 ELSE 0 END AS low_sev_ed_flag
    , CASE WHEN SUM(CASE WHEN TRIM(mc.op_severitylvl) = "2-Low/Med" THEN mc.event_ct ELSE 0 END) > 0 THEN 1 ELSE 0 END AS low_med_sev_ed_flag
    , CASE WHEN SUM(CASE WHEN TRIM(mc.op_severitylvl) = "3-Med" THEN mc.event_ct ELSE 0 END) > 0 THEN 1 ELSE 0 END AS med_sev_ed_flag
    , CASE WHEN SUM(CASE WHEN TRIM(mc.op_severitylvl) = "4-Med/High" THEN mc.event_ct ELSE 0 END) > 0 THEN 1 ELSE 0 END AS med_high_sev_ed_flag
    , CASE WHEN SUM(CASE WHEN TRIM(mc.op_severitylvl) = "5-High" THEN mc.event_ct ELSE 0 END) > 0 THEN 1 ELSE 0 END AS high_sev_ed_flag
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "1-Low" THEN mc.cost ELSE 0 END) AS low_sev_ed_cost
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "2-Low/Med" THEN mc.cost ELSE 0 END) AS low_med_sev_ed_cost
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "3-Med" THEN mc.cost ELSE 0 END) AS med_sev_ed_cost
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "4-Med/High" THEN mc.cost ELSE 0 END) AS med_high_sev_ed_cost
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "5-High" THEN mc.cost ELSE 0 END) AS high_sev_ed_cost
FROM 
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
LEFT JOIN 
    `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ed_cases_yr2` AS mc
        ON st.asdb_member_key = mc.asdb_member_key
        AND st.asdb_plan_key = mc.asdb_plan_key
GROUP BY 
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
;

-- ----- 3B.3: IP Cases Year 2 -----
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ip_cases_yr2`;

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ip_cases_yr2`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
SELECT 
    mc.asdb_member_key
    , mc.asdb_plan_key
    , st.index_dt
    , mc.asdb_event_start_dt
    , mc.asdb_event_end_dt
    , mc.final_discharge_dt
    , mc.prindiag
    , CASE WHEN mc.asdb_coe_id IN (10200,10700,10800) THEN "Acute"
        WHEN mc.asdb_coe_id IN (10000,10100,10300) THEN "Maternity/Infant"
        ELSE "Non-Acute"
        END AS ip_type
    , DATE_DIFF(mc.final_discharge_dt, mc.asdb_event_start_dt, DAY) AS calc_los
    , mc.event_ct
    , mc.admit_los
    , mc.paid_los
    , mc.cost AS ip_paid_amt
FROM
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
INNER JOIN 
    `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ICE_IP` AS mc
        ON st.asdb_member_key=mc.asdb_member_key
        AND st.asdb_plan_key=mc.asdb_plan_key
WHERE 
    CAST(mc.asdb_event_start_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 24 MONTH) AND DATE_SUB(DATE_SUB(st.index_dt, INTERVAL 1 DAY), INTERVAL 12 MONTH)
    AND mc.event_ct=1
;

-- ----- 3B.4: IP Summary Year 2 -----
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ip_yr2`;

CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ip_yr2`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
WITH acute AS (
    SELECT asdb_member_key, asdb_plan_key, index_dt
        , CASE WHEN SUM(event_ct) > 0 THEN 1 ELSE 0 END AS acute_ip_flag
        , SUM(event_ct) AS sum_acute_ip_admits
        , SUM(calc_los) AS sum_acute_calc_los
        , SUM(admit_los) AS sum_acute_admit_los
        , SUM(paid_los) AS sum_acute_paid_los
        , SUM(ip_paid_amt) AS sum_acute_ip_cost
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ip_cases_yr2`
    WHERE ip_type = "Acute"
    GROUP BY asdb_member_key, asdb_plan_key, index_dt
),
nonacute AS (
    SELECT asdb_member_key, asdb_plan_key
        , CASE WHEN SUM(event_ct) > 0 THEN 1 ELSE 0 END AS non_acute_ip_flag
        , SUM(event_ct) AS sum_non_acute_ip_admits
        , SUM(calc_los) AS sum_non_acute_calc_los
        , SUM(admit_los) AS sum_non_acute_admit_los
        , SUM(paid_los) AS sum_non_acute_paid_los
        , SUM(ip_paid_amt) AS sum_non_acute_ip_cost
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ip_cases_yr2`
    WHERE ip_type="Non-Acute"
    GROUP BY asdb_member_key, asdb_plan_key
),
maternity AS (
    SELECT asdb_member_key, asdb_plan_key
        , CASE WHEN SUM(event_ct)>0 THEN 1 ELSE 0 END AS maternity_ip_flag
        , SUM(event_ct) AS sum_maternity_ip_admits
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ip_cases_yr2`
    WHERE ip_type="Maternity/Infant"
    GROUP BY asdb_member_key, asdb_plan_key
)
SELECT 
    st.asdb_member_key, st.asdb_plan_key, st.index_dt
    , COALESCE(a.acute_ip_flag, 0) AS acute_ip_flag
    , COALESCE(a.sum_acute_ip_admits, 0) AS sum_acute_ip_admits
    , COALESCE(a.sum_acute_calc_los, 0) AS sum_acute_calc_los
    , COALESCE(a.sum_acute_admit_los, 0) AS sum_acute_admit_los
    , COALESCE(a.sum_acute_paid_los, 0) AS sum_acute_paid_los
    , COALESCE(a.sum_acute_ip_cost, 0) AS sum_acute_ip_cost
    , COALESCE(b.non_acute_ip_flag, 0) AS non_acute_ip_flag
    , COALESCE(b.sum_non_acute_ip_admits, 0) AS sum_non_acute_ip_admits
    , COALESCE(b.sum_non_acute_calc_los, 0) AS sum_non_acute_calc_los
    , COALESCE(b.sum_non_acute_admit_los, 0) AS sum_non_acute_admit_los
    , COALESCE(b.sum_non_acute_paid_los, 0) AS sum_non_acute_paid_los
    , COALESCE(b.sum_non_acute_ip_cost, 0) AS sum_non_acute_ip_cost
    , COALESCE(c.maternity_ip_flag, 0) AS maternity_ip_flag
    , COALESCE(c.sum_maternity_ip_admits, 0) AS sum_maternity_ip_admits
FROM 
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
LEFT JOIN acute AS a ON st.asdb_member_key = a.asdb_member_key AND st.asdb_plan_key = a.asdb_plan_key
LEFT JOIN nonacute AS b ON st.asdb_member_key = b.asdb_member_key AND st.asdb_plan_key = b.asdb_plan_key
LEFT JOIN maternity AS c ON st.asdb_member_key = c.asdb_member_key AND st.asdb_plan_key = c.asdb_plan_key
;

-- ----- 3B.5: OP Summary Year 2 -----
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_op_yr2`;

CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_op_yr2`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
WITH clm AS(
    SELECT *
        , CASE WHEN ROW_NUMBER() OVER(PARTITION BY asdb_member_key, asdb_plan_key, asdb_svc_prov_key, asdb_incurred_dt) = 1 THEN 1 ELSE 0 END AS op_ct
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_med_claims_yr2`
    WHERE TRIM(asdb_coe_general_type) != "Inpatient"
        AND TRIM(emis_cat) != "Institutional Services"
        AND TRIM(emis_cat) != "Emergency"
)
SELECT 
    st.asdb_member_key, st.asdb_plan_key, st.index_dt
    , COALESCE(SUM(clm.paid_amt), 0) AS sum_op_cost
    , COALESCE(SUM(clm.op_ct), 0) AS sum_op_visits
    , MAX(CASE WHEN clm.op_ct=1 THEN 1 ELSE 0 END) AS op_flag
FROM 
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
LEFT JOIN clm ON st.asdb_member_key = clm.asdb_member_key AND st.asdb_plan_key = clm.asdb_plan_key
GROUP BY st.asdb_member_key, st.asdb_plan_key, st.index_dt
;

-- ----- 3B.6: Other Cost Utilization Summary Year 2 -----
-- Note: Uses med_claims_yr2, similar structure to yr1
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_med_claims_flag_yr2`;

CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_med_claims_flag_yr2`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
WITH clm AS (
    SELECT *
        , CASE WHEN TRIM(asdb_coe_general_type)="Inpatient" OR TRIM(emis_cat)="Institutional Services" THEN "Inpatient"
            WHEN TRIM(emis_cat)="Emergency" THEN "Emergency"
            ELSE "Outpatient" END AS plc_svc_ctg
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_med_claims_yr2`
)
SELECT 
    clm.*
    , fac.prov_specialty
    , CASE WHEN plc_svc_ctg="Inpatient" THEN paid_amt ELSE 0 END AS inpatient_cost
    , CASE WHEN plc_svc_ctg="Emergency" THEN paid_amt ELSE 0 END AS emergency_cost
    , CASE WHEN plc_svc_ctg="Outpatient" THEN paid_amt ELSE 0 END AS outpatient_cost
    , CASE WHEN TRIM(emis_cat)="Community-Based Services" THEN paid_amt ELSE 0 END AS emis_community_cost
    , CASE WHEN TRIM(emis_cat)="Emergency" THEN paid_amt ELSE 0 END AS emis_ed_cost
    , CASE WHEN TRIM(emis_cat)="Home Health" THEN paid_amt ELSE 0 END AS emis_hh_cost
    , CASE WHEN TRIM(emis_cat)="Home-Based Services" THEN paid_amt ELSE 0 END AS emis_home_cost
    , CASE WHEN TRIM(emis_cat)="Inpatient Facility" THEN paid_amt ELSE 0 END AS emis_ip_cost
    , CASE WHEN TRIM(emis_cat)="Institutional Services" THEN paid_amt ELSE 0 END AS emis_ins_cost
    , CASE WHEN TRIM(emis_cat)="Laboratory" THEN paid_amt ELSE 0 END AS emis_lab_cost
    , CASE WHEN TRIM(emis_cat)="Medical Pharmacy" THEN paid_amt ELSE 0 END AS emis_mrx_cost
    , CASE WHEN TRIM(emis_cat)="Mental Health" THEN paid_amt ELSE 0 END AS emis_mh_cost
    , CASE WHEN TRIM(emis_cat)="Misc. Medical" THEN paid_amt ELSE 0 END AS emis_misc_cost
    , CASE WHEN TRIM(emis_cat)="Primary Physician" THEN paid_amt ELSE 0 END AS emis_pcp_cost
    , CASE WHEN TRIM(emis_cat)="Radiology" THEN paid_amt ELSE 0 END AS emis_radio_cost
    , CASE WHEN TRIM(emis_cat)="Selected Ambulatory Facility" THEN paid_amt ELSE 0 END AS emis_ambul_cost
    , CASE WHEN TRIM(emis_cat)="Specialist Physician" THEN paid_amt ELSE 0 END AS emis_spec_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" THEN paid_amt ELSE 0 END AS coe_ltc_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Inpatient" AND TRIM(asdb_coe_sub_cat)="Hospital" THEN paid_amt ELSE 0 END AS coe_ip_hos_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Inpatient" AND TRIM(asdb_coe_sub_cat)="Non Hospital" THEN paid_amt ELSE 0 END AS coe_ip_non_hos_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Laboratory" AND TRIM(asdb_coe_sub_cat)="Professional" THEN paid_amt ELSE 0 END AS coe_lab_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" AND TRIM(asdb_coe_sub_cat)="Community Based Services" THEN paid_amt ELSE 0 END AS coe_ltc_community_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" AND TRIM(asdb_coe_sub_cat)="Home Based Services" THEN paid_amt ELSE 0 END AS coe_ltc_home_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" AND TRIM(asdb_coe_sub_cat)="Institution" THEN paid_amt ELSE 0 END AS coe_ltc_ins_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Other" AND TRIM(asdb_coe_sub_cat)="Professional" THEN paid_amt ELSE 0 END AS coe_other_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Outpatient" AND TRIM(asdb_coe_sub_cat)="Hospital" THEN paid_amt ELSE 0 END AS coe_op_hos_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Outpatient" AND TRIM(asdb_coe_sub_cat)="Non Hospital" THEN paid_amt ELSE 0 END AS coe_op_non_hos_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Anesthesia" THEN paid_amt ELSE 0 END AS coe_anesth_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Evaluation & Management" THEN paid_amt ELSE 0 END AS coe_eval_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Maternity" THEN paid_amt ELSE 0 END AS coe_maternity_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Medicine" THEN paid_amt ELSE 0 END AS coe_mrx_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Mental Health" THEN paid_amt ELSE 0 END AS coe_mh_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Physician" THEN paid_amt ELSE 0 END AS coe_phy_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Surgery" THEN paid_amt ELSE 0 END AS coe_surg_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Radiology" AND TRIM(asdb_coe_sub_cat)="Professional" THEN paid_amt ELSE 0 END AS coe_radio_cost
    , CASE WHEN TRIM(fac.prov_specialty)="Urgent Care" OR TRIM(location)="20" OR TRIM(servcode)="S9083" THEN paid_amt ELSE 0 END AS uc_cost
    , CASE WHEN TRIM(emis_cat)="Community-Based Services" THEN 1 ELSE 0 END AS emis_community_clm
    , CASE WHEN TRIM(emis_cat)="Emergency" THEN 1 ELSE 0 END AS emis_ed_clm
    , CASE WHEN TRIM(emis_cat)="Home Health" THEN 1 ELSE 0 END AS emis_hh_clm
    , CASE WHEN TRIM(emis_cat)="Home-Based Services" THEN 1 ELSE 0 END AS emis_home_clm
    , CASE WHEN TRIM(emis_cat)="Inpatient Facility" THEN 1 ELSE 0 END AS emis_ip_clm
    , CASE WHEN TRIM(emis_cat)="Institutional Services" THEN 1 ELSE 0 END AS emis_ins_clm
    , CASE WHEN TRIM(emis_cat)="Laboratory" THEN 1 ELSE 0 END AS emis_lab_clm
    , CASE WHEN TRIM(emis_cat)="Medical Pharmacy" THEN 1 ELSE 0 END AS emis_mrx_clm
    , CASE WHEN TRIM(emis_cat)="Mental Health" THEN 1 ELSE 0 END AS emis_mh_clm
    , CASE WHEN TRIM(emis_cat)="Misc. Medical" THEN 1 ELSE 0 END AS emis_misc_clm
    , CASE WHEN TRIM(emis_cat)="Primary Physician" THEN 1 ELSE 0 END AS emis_pcp_clm
    , CASE WHEN TRIM(emis_cat)="Radiology" THEN 1 ELSE 0 END AS emis_radio_clm
    , CASE WHEN TRIM(emis_cat)="Selected Ambulatory Facility" THEN 1 ELSE 0 END AS emis_ambul_clm
    , CASE WHEN TRIM(emis_cat)="Specialist Physician" THEN 1 ELSE 0 END AS emis_spec_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" THEN 1 ELSE 0 END as ltc_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Inpatient" AND TRIM(asdb_coe_sub_cat)="Hospital" THEN 1 ELSE 0 END AS coe_ip_hos_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Inpatient" AND TRIM(asdb_coe_sub_cat)="Non Hospital" THEN 1 ELSE 0 END AS coe_ip_non_hos_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Laboratory" AND TRIM(asdb_coe_sub_cat)="Professional" THEN 1 ELSE 0 END AS coe_lab_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" AND TRIM(asdb_coe_sub_cat)="Community Based Services" THEN 1 ELSE 0 END AS coe_ltc_community_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" AND TRIM(asdb_coe_sub_cat)="Home Based Services" THEN 1 ELSE 0 END AS coe_ltc_home_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" AND TRIM(asdb_coe_sub_cat)="Institution" THEN 1 ELSE 0 END AS coe_ltc_ins_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Other" AND TRIM(asdb_coe_sub_cat)="Professional" THEN 1 ELSE 0 END AS coe_other_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Outpatient" AND TRIM(asdb_coe_sub_cat)="Hospital" THEN 1 ELSE 0 END AS coe_op_hos_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Outpatient" AND TRIM(asdb_coe_sub_cat)="Non Hospital" THEN 1 ELSE 0 END AS coe_op_non_hos_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Anesthesia" THEN 1 ELSE 0 END AS coe_anesth_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Evaluation & Management" THEN 1 ELSE 0 END AS coe_eval_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Maternity" THEN 1 ELSE 0 END AS coe_maternity_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Medicine" THEN 1 ELSE 0 END AS coe_mrx_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Mental Health" THEN 1 ELSE 0 END AS coe_mh_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Physician" THEN 1 ELSE 0 END AS coe_phy_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Surgery" THEN 1 ELSE 0 END AS coe_surg_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Radiology" AND TRIM(asdb_coe_sub_cat)="Professional" THEN 1 ELSE 0 END AS coe_radio_clm
    , CASE WHEN TRIM(fac.prov_specialty)="Urgent Care" OR TRIM(location)="20" OR TRIM(servcode)="S9083" THEN 1 ELSE 0 END AS uc_clm
    , CASE WHEN (TRIM(clm.revcode) in ("0760","0761","0762","0769") AND (TRIM(clm.billtype) like "13%" or TRIM(clm.billtype) like "85%") AND (TRIM(clm.servcode)) in ("99217","99218","99219","99202","G0378","G0379","")) THEN 1 ELSE 0 END AS obs_clm
FROM clm
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_SVC_PROV` AS fac
    ON clm.asdb_svc_prov_key = fac.asdb_svc_prov_key AND clm.asdb_plan_key = fac.asdb_plan_key
;

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_other_cost_utilization_yr2`;

CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_other_cost_utilization_yr2`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
SELECT
    st.asdb_member_key, st.asdb_plan_key, st.index_dt
    , COUNT(DISTINCT clm.claimid) AS claim_cnt
    , COUNT(*) AS claim_line_cnt
    , COALESCE(SUM(clm.paid_amt), 0) AS sum_paid_amt
    , COALESCE(SUM(clm.inpatient_cost), 0) AS inpatient_cost
    , COALESCE(SUM(clm.emergency_cost), 0) AS emergency_cost
    , COALESCE(SUM(clm.outpatient_cost), 0) AS outpatient_cost
    , COALESCE(SUM(clm.emis_community_cost), 0) AS emis_community_cost
    , COALESCE(SUM(clm.emis_ed_cost), 0) AS emis_ed_cost
    , COALESCE(SUM(clm.emis_hh_cost), 0) AS emis_hh_cost
    , COALESCE(SUM(clm.emis_home_cost), 0) AS emis_home_cost
    , COALESCE(SUM(clm.emis_ip_cost), 0) AS emis_ip_cost
    , COALESCE(SUM(clm.emis_ins_cost), 0) AS emis_ins_cost
    , COALESCE(SUM(clm.emis_lab_cost), 0) AS emis_lab_cost
    , COALESCE(SUM(clm.emis_mrx_cost), 0) AS emis_mrx_cost
    , COALESCE(SUM(clm.emis_mh_cost), 0) AS emis_mh_cost
    , COALESCE(SUM(clm.emis_misc_cost), 0) AS emis_misc_cost
    , COALESCE(SUM(clm.emis_pcp_cost), 0) AS emis_pcp_cost
    , COALESCE(SUM(clm.emis_radio_cost), 0) AS emis_radio_cost
    , COALESCE(SUM(clm.emis_ambul_cost), 0) AS emis_ambul_cost
    , COALESCE(SUM(clm.emis_spec_cost), 0) AS emis_spec_cost
    , COALESCE(SUM(clm.coe_ltc_cost), 0) AS coe_ltc_cost
    , COALESCE(SUM(clm.coe_ip_hos_cost), 0) AS coe_ip_hos_cost
    , COALESCE(SUM(clm.coe_ip_non_hos_cost), 0) AS coe_ip_non_hos_cost
    , COALESCE(SUM(clm.coe_lab_cost), 0) AS coe_lab_cost
    , COALESCE(SUM(clm.coe_ltc_community_cost), 0) AS coe_ltc_community_cost
    , COALESCE(SUM(clm.coe_ltc_home_cost), 0) AS coe_ltc_home_cost
    , COALESCE(SUM(clm.coe_ltc_ins_cost), 0) AS coe_ltc_ins_cost
    , COALESCE(SUM(clm.coe_other_cost), 0) AS coe_other_cost
    , COALESCE(SUM(clm.coe_op_hos_cost), 0) AS coe_op_hos_cost
    , COALESCE(SUM(clm.coe_op_non_hos_cost), 0) AS coe_op_non_hos_cost
    , COALESCE(SUM(clm.coe_anesth_cost), 0) AS coe_anesth_cost
    , COALESCE(SUM(clm.coe_eval_cost), 0) AS coe_eval_cost
    , COALESCE(SUM(clm.coe_maternity_cost), 0) AS coe_maternity_cost
    , COALESCE(SUM(clm.coe_mrx_cost), 0) AS coe_mrx_cost
    , COALESCE(SUM(clm.coe_mh_cost), 0) AS coe_mh_cost
    , COALESCE(SUM(clm.coe_phy_cost), 0) AS coe_phy_cost
    , COALESCE(SUM(clm.coe_surg_cost), 0) AS coe_surg_cost
    , COALESCE(SUM(clm.coe_radio_cost), 0) AS coe_radio_cost
    , COALESCE(SUM(clm.uc_cost), 0) AS uc_cost
    , COALESCE(SUM(clm.emis_community_clm), 0) AS emis_community_clm
    , COALESCE(SUM(clm.emis_ed_clm), 0) AS emis_ed_clm
    , COALESCE(SUM(clm.emis_hh_clm), 0) AS emis_hh_clm
    , COALESCE(SUM(clm.emis_home_clm), 0) AS emis_home_clm
    , COALESCE(SUM(clm.emis_ip_clm), 0) AS emis_ip_clm
    , COALESCE(SUM(clm.emis_ins_clm), 0) AS emis_ins_clm
    , COALESCE(SUM(clm.emis_lab_clm), 0) AS emis_lab_clm
    , COALESCE(SUM(clm.emis_mrx_clm), 0) AS emis_mrx_clm
    , COALESCE(SUM(clm.emis_mh_clm), 0) AS emis_mh_clm
    , COALESCE(SUM(clm.emis_misc_clm), 0) AS emis_misc_clm
    , COALESCE(SUM(clm.emis_pcp_clm), 0) AS emis_pcp_clm
    , COALESCE(SUM(clm.emis_radio_clm), 0) AS emis_radio_clm
    , COALESCE(SUM(clm.emis_ambul_clm), 0) AS emis_ambul_clm
    , COALESCE(SUM(clm.emis_spec_clm), 0) AS emis_spec_clm
    , COALESCE(SUM(clm.ltc_clm), 0) AS ltc_clm
    , COALESCE(SUM(clm.coe_ip_hos_clm), 0) AS coe_ip_hos_clm
    , COALESCE(SUM(clm.coe_ip_non_hos_clm), 0) AS coe_ip_non_hos_clm
    , COALESCE(SUM(clm.coe_lab_clm), 0) AS coe_lab_clm
    , COALESCE(SUM(clm.coe_ltc_community_clm), 0) AS coe_ltc_community_clm
    , COALESCE(SUM(clm.coe_ltc_home_clm), 0) AS coe_ltc_home_clm
    , COALESCE(SUM(clm.coe_ltc_ins_clm), 0) AS coe_ltc_ins_clm
    , COALESCE(SUM(clm.coe_other_clm), 0) AS coe_other_clm
    , COALESCE(SUM(clm.coe_op_hos_clm), 0) AS coe_op_hos_clm
    , COALESCE(SUM(clm.coe_op_non_hos_clm), 0) AS coe_op_non_hos_clm
    , COALESCE(SUM(clm.coe_anesth_clm), 0) AS coe_anesth_clm
    , COALESCE(SUM(clm.coe_eval_clm), 0) AS coe_eval_clm
    , COALESCE(SUM(clm.coe_maternity_clm), 0) AS coe_maternity_clm
    , COALESCE(SUM(clm.coe_mrx_clm), 0) AS coe_mrx_clm
    , COALESCE(SUM(clm.coe_mh_clm), 0) AS coe_mh_clm
    , COALESCE(SUM(clm.coe_phy_clm), 0) AS coe_phy_clm
    , COALESCE(SUM(clm.coe_surg_clm), 0) AS coe_surg_clm
    , COALESCE(SUM(clm.coe_radio_clm), 0) AS coe_radio_clm
    , COALESCE(SUM(clm.uc_clm), 0) AS uc_clm
    , COALESCE(SUM(clm.obs_clm), 0) AS obs_clm
FROM (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_med_claims_flag_yr2` AS clm
    ON st.asdb_member_key=clm.asdb_member_key AND st.asdb_plan_key=clm.asdb_plan_key
GROUP BY st.asdb_member_key, st.asdb_plan_key, st.index_dt
;

-- =====================================================================================
-- SECTION 4: CONDITIONS (004_Conditions.sh)
-- =====================================================================================
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_conditions`;

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_conditions`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
SELECT
    *
    , (abdominal_pain+AID+ANX+OST+AST+AUT+CHO+burns+cad+Cancer+narc+CBD+CHF+CRF+VNA+CHD+
        COP+CYS+DEP+DIA+EDO+esrd+EPL+CRO+MOH+HEM+HepC+immune+intel_dsblty+meta_cancer+
        liver_dis+MSS+OBE+oud+liver_other+paralysis+PAR+hmd+PVD+autoimmune+DEM+SCA+
        sleep_apnea+spinal_inj+back+substance+ALC+bipolar+psychoses) AS major_chronic_cnt
FROM 
    (SELECT
        st.asdb_member_key
        , st.asdb_plan_key
        , st.index_dt
        , MIN(b.rpt_end_dt) AS first_rpt
        , MAX(b.rpt_end_dt) AS last_rpt
        , MAX(CASE WHEN b.cond_rank=52 THEN 1 ELSE 0 END) AS abdominal_pain
        , MAX(CASE WHEN b.cond_rank=34 THEN 1 ELSE 0 END) AS AID
        , MAX(CASE WHEN b.cond_rank=69 THEN 1 ELSE 0 END) AS IDA
        , MAX(CASE WHEN b.cond_rank=41 THEN 1 ELSE 0 END) AS ANX
        , MAX(CASE WHEN b.cond_rank=61 THEN 1 ELSE 0 END) AS OST
        , MAX(CASE WHEN b.cond_rank=33 THEN 1 ELSE 0 END) AS AST
        , MAX(CASE WHEN b.cond_rank=45 THEN 1 ELSE 0 END) AS AUT
        , MAX(CASE WHEN b.cond_rank=51 THEN 1 ELSE 0 END) AS CHO
        , MAX(CASE WHEN b.cond_rank=39 THEN 1 ELSE 0 END) AS burns
        , MAX(CASE WHEN b.cond_rank=16 THEN 1 ELSE 0 END) AS cad
        , MAX(CASE WHEN b.cond_rank=29 THEN 1 ELSE 0 END) AS Cancer
        , MAX(CASE WHEN b.cond_rank=55 THEN 1 ELSE 0 END) AS narc
        , MAX(CASE WHEN b.cond_rank=17 THEN 1 ELSE 0 END) AS CBD
        , MAX(CASE WHEN b.cond_rank=4 THEN 1 ELSE 0 END) AS CHF
        , MAX(CASE WHEN b.cond_rank=3 THEN 1 ELSE 0 END) AS CRF
        , MAX(CASE WHEN b.cond_rank=62 THEN 1 ELSE 0 END) AS VNA
        , MAX(CASE WHEN b.cond_rank=30 THEN 1 ELSE 0 END) AS CHD
        , MAX(CASE WHEN b.cond_rank=44 THEN 1 ELSE 0 END) AS COP
        , MAX(CASE WHEN b.cond_rank=12 THEN 1 ELSE 0 END) AS CYS
        , MAX(CASE WHEN b.cond_rank=37 THEN 1 ELSE 0 END) AS DEP
        , MAX(CASE WHEN b.cond_rank=24 THEN 1 ELSE 0 END) AS DIA
        , MAX(CASE WHEN b.cond_rank=35 THEN 1 ELSE 0 END) AS EDO
        , MAX(CASE WHEN b.cond_rank=1 THEN 1 ELSE 0 END) AS esrd
        , MAX(CASE WHEN b.cond_rank=20 THEN 1 ELSE 0 END) AS EPL
        , MAX(CASE WHEN b.cond_rank=19 OR b.cond_rank=9 THEN 1 ELSE 0 END) AS CRO
        , MAX(CASE WHEN b.cond_rank=27 THEN 1 ELSE 0 END) AS MOH
        , MAX(CASE WHEN b.cond_rank=2 THEN 1 ELSE 0 END) AS HEM
        , MAX(CASE WHEN b.cond_rank=74 THEN 1 ELSE 0 END) AS HepC
        , MAX(CASE WHEN b.cond_rank=46 THEN 1 ELSE 0 END) AS HYP
        , MAX(CASE WHEN b.cond_rank=54 THEN 1 ELSE 0 END) AS HYC
        , MAX(CASE WHEN b.cond_rank=10 THEN 1 ELSE 0 END) AS immune
        , MAX(CASE WHEN b.cond_rank=72 THEN 1 ELSE 0 END) AS intel_dsblty
        , MAX(CASE WHEN b.cond_rank=6 THEN 1 ELSE 0 END) AS meta_cancer
        , MAX(CASE WHEN b.cond_rank=21 THEN 1 ELSE 0 END) AS liver_dis
        , MAX(CASE WHEN b.cond_rank=26 THEN 1 ELSE 0 END) AS MSS
        , MAX(CASE WHEN b.cond_rank=73 THEN 1 ELSE 0 END) AS OBE
        , MAX(CASE WHEN b.cond_rank=99 THEN 1 ELSE 0 END) AS oud
        , MAX(CASE WHEN b.cond_rank=64 THEN 1 ELSE 0 END) AS liver_other
        , MAX(CASE WHEN b.cond_rank=11 THEN 1 ELSE 0 END) AS paralysis
        , MAX(CASE WHEN b.cond_rank=42 THEN 1 ELSE 0 END) AS PAR
        , MAX(CASE WHEN b.cond_rank=57 THEN 1 ELSE 0 END) AS PUD
        , MAX(CASE WHEN b.cond_rank=18 THEN 1 ELSE 0 END) AS hmd
        , MAX(CASE WHEN b.cond_rank=50 THEN 1 ELSE 0 END) AS PVD
        , MAX(CASE WHEN b.cond_rank=43 THEN 1 ELSE 0 END) AS autoimmune
        , MAX(CASE WHEN b.cond_rank=32 THEN 1 ELSE 0 END) AS DEM
        , MAX(CASE WHEN b.cond_rank=7 THEN 1 ELSE 0 END) AS SCA
        , MAX(CASE WHEN b.cond_rank=66 THEN 1 ELSE 0 END) AS sleep_apnea
        , MAX(CASE WHEN b.cond_rank=13 THEN 1 ELSE 0 END) AS spinal_inj
        , MAX(CASE WHEN b.cond_rank=31 THEN 1 ELSE 0 END) AS back
        -- BH conditions
        , MAX(CASE WHEN b.cond_rank=22 THEN 1 ELSE 0 END) AS substance
        , MAX(CASE WHEN b.cond_rank=14 THEN 1 ELSE 0 END) AS ALC
        , MAX(CASE WHEN b.cond_rank=36 THEN 1 ELSE 0 END) AS bipolar 
        , MAX(CASE WHEN b.cond_rank=25 THEN 1 ELSE 0 END) AS psychoses
    FROM 
        (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
    LEFT JOIN 
        `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_PPM_MEMBER_CONDITION_HISTORY` AS b 
            ON st.asdb_member_key = b.ppm_member_key
            AND st.asdb_plan_key = b.ppm_plan_key
            AND DATE_TRUNC(st.index_dt, MONTH) BETWEEN DATE_ADD(LAST_DAY(CAST(b.rpt_end_dt AS DATE), MONTH), INTERVAL 1 DAY) 
                AND DATE_ADD(LAST_DAY(CAST(b.rpt_end_dt AS DATE), MONTH), INTERVAL 12 MONTH)
    GROUP BY 
        st.asdb_member_key
        , st.asdb_plan_key
        , st.index_dt
) tb
;

-- =====================================================================================
-- SECTION 5A: RX CLAIMS - YEAR 1 (006_Rx_Claims_yr1.sh)
-- =====================================================================================
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_rx_claims_yr1`;

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_rx_claims_yr1`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
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
    , ROUND(CASE WHEN rx.days_supply >= 0.1 AND rx.days_supply < 30 THEN 30 ELSE rx.days_supply END/30) AS scripts
    , rx.ndcnum AS ndc_cd
    , rx.gpi AS adjudicated_gpi_cd
    , SUBSTR(rx.gpi,1,4) AS gpi4
    , SUBSTR(rx.gpi,1,2) AS gpi2
    , rx.billed_amt
    , rx.claim_adj_amt
    , rx.copay_amt
    , CASE WHEN rx.pharmacytype="R" THEN 1 ELSE 0 END AS retail_flag
    , CASE WHEN rx.pharmacytype="M" THEN 1 ELSE 0 END AS mail_order_flag
    , CASE WHEN rx.drugtype = 3 THEN 1 ELSE 0 END AS generic_fill_flag
    , CASE WHEN rx.drugtype = 2 THEN 1 ELSE 0 END AS branded_generic_fill_flag
    , CASE WHEN rx.drugtype = 4 THEN 1 ELSE 0 END AS otc_fill_flag
    , CASE WHEN rx.drugtype = 1 THEN 1 ELSE 0 END AS ss_brand_fill_flag
    , CASE WHEN rx.drugtype = 5 THEN 1 ELSE 0 END AS ms_brand_fill_flag
    , CASE WHEN rx.formularyflag="F" or rx.drugtype = 3 THEN 1 ELSE 0 END AS formulary_fill_flag
    , CASE WHEN c.maint_drug_cd="X" THEN 1 ELSE 0 END AS maint_drug_flag
    , CASE WHEN d.ndc IS NOT NULL THEN 1 ELSE 0 END AS specialty_rx_flag
FROM 
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
INNER JOIN `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_RX_DATA_STAGE` AS rx
    ON st.asdb_member_key = rx.asdb_member_key AND st.asdb_plan_key = rx.asdb_plan_key
LEFT JOIN `edp-prod-storage.edp_hcb_anbor_enrsrc_prod.EDW_DRUG` AS c
    ON TRIM(rx.ndcnum) = TRIM(c.ndc_cd)
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_srcv.specdrug_pww_universal_spec_list` AS d
    ON TRIM(rx.ndcnum) = TRIM(d.ndc)
WHERE rx.ClaimType = "P"
    AND CAST(rx.asdb_incurred_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 12 MONTH) AND DATE_SUB(st.index_dt, INTERVAL 1 DAY)
;

-- Summarize Rx Year 1
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_rx_yr1`;

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_rx_yr1`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
WITH rx_tmp AS (
    SELECT asdb_member_key, asdb_plan_key, index_dt
        , MIN(disp_dt) AS first_disp_dt
        , MAX(disp_dt) AS last_disp_dt
        , COUNT(*) AS rx_claim_cnt
        , SUM(days_supply) AS days_supply_SUM
        , COUNT(DISTINCT ndc_cd) AS ndc_cnt
        , COUNT(DISTINCT adjudicated_gpi_cd) AS gpi_cnt
        , COUNT(DISTINCT gpi4) AS gpi4_cnt
        , COUNT(DISTINCT gpi2) AS gpi2_cnt
        , SUM(retail_flag) AS retail_fills
        , SUM(mail_order_flag) AS mail_order_fills
        , SUM(generic_fill_flag) AS generic_fills
        , SUM(branded_generic_fill_flag) AS branded_generic_fills
        , SUM(otc_fill_flag) AS otc_fills
        , SUM(ss_brand_fill_flag) AS ss_brand_fills
        , SUM(ms_brand_fill_flag) AS ms_brand_fills
        , SUM(formulary_fill_flag) AS formulary_fills
        , SUM(maint_drug_flag) AS maint_drug_fills
        , SUM(CASE WHEN gpi2="27" THEN Scripts ELSE 0 END) AS antidiabetic_scripts
        , SUM(CASE WHEN gpi2="27" THEN days_supply ELSE 0 END) AS antidiabetic_days_supply
        , SUM(CASE WHEN gpi2="33" THEN Scripts ELSE 0 END) AS beta_blocker_scripts
        , SUM(CASE WHEN gpi2="33" THEN days_supply ELSE 0 END) AS beta_blocker_days_supply
        , SUM(CASE WHEN gpi2="36" THEN Scripts ELSE 0 END) AS antihypertensive_scripts
        , SUM(CASE WHEN gpi2="36" THEN days_supply ELSE 0 END) AS antihypertensive_days_supply
        , SUM(CASE WHEN gpi2="39" THEN Scripts ELSE 0 END) AS lipid_lowering_scripts
        , SUM(CASE WHEN gpi2="39" THEN days_supply ELSE 0 END) AS lipid_lowering_days_supply
        , SUM(CASE WHEN gpi2="34" THEN Scripts ELSE 0 END) AS calcium_channel_blk_scripts
        , SUM(CASE WHEN gpi2="34" THEN days_supply ELSE 0 END) AS calcium_channel_blk_days_supply
        , SUM(CASE WHEN gpi2="37" THEN Scripts ELSE 0 END) AS diuretic_scripts
        , SUM(CASE WHEN gpi2="37" THEN days_supply ELSE 0 END) AS diuretic_days_supply
        , SUM(CASE WHEN gpi2="32" THEN Scripts ELSE 0 END) AS antianginal_agent_scripts
        , SUM(CASE WHEN gpi2="32" THEN days_supply ELSE 0 END) AS antianginal_agent_days_supply
        , SUM(CASE WHEN gpi2="58" THEN Scripts ELSE 0 END) AS antidepressant_scripts
        , SUM(CASE WHEN gpi2="58" THEN days_supply ELSE 0 END) AS antidepressant_days_supply
        , SUM(CASE WHEN gpi2="59" THEN Scripts ELSE 0 END) AS antipsychotic_scripts
        , SUM(CASE WHEN gpi2="59" THEN days_supply ELSE 0 END) AS antipsychotic_days_supply
        , SUM(CASE WHEN gpi2="57" THEN Scripts ELSE 0 END) AS antianxiety_scripts
        , SUM(CASE WHEN gpi2="57" THEN days_supply ELSE 0 END) AS antianxiety_days_supply
        , SUM(CASE WHEN gpi2="72" THEN Scripts ELSE 0 END) AS anticonvulsant_scripts
        , SUM(CASE WHEN gpi2="72" THEN days_supply ELSE 0 END) AS anticonvulsant_days_supply
        , SUM(CASE WHEN gpi4="4440" THEN Scripts ELSE 0 END) AS inhaled_steroid_scripts
        , SUM(CASE WHEN gpi4="4440" THEN days_supply ELSE 0 END) AS inhaled_steroid_days_supply
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_rx_claims_yr1`
    GROUP BY asdb_member_key, asdb_plan_key, index_dt
)
SELECT 
    st.asdb_member_key, st.asdb_plan_key, st.index_dt
    , rx.first_disp_dt, rx.last_disp_dt
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
FROM (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
LEFT JOIN rx_tmp rx ON st.asdb_member_key = rx.asdb_member_key AND st.asdb_plan_key = rx.asdb_plan_key
;

-- =====================================================================================
-- SECTION 5B: RX CLAIMS - YEAR 2 (006_Rx_Claims_yr2.sh)
-- =====================================================================================
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_rx_claims_yr2`;

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_rx_claims_yr2`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
SELECT DISTINCT
    st.asdb_member_key, st.asdb_plan_key, st.index_dt
    , rx.asdb_pharmacy_key, rx.prescriptionnum
    , CAST(rx.asdb_incurred_dt AS DATE) AS disp_dt
    , rx.days_supply, rx.script_ct
    , ROUND(CASE WHEN rx.days_supply >= 0.1 AND rx.days_supply < 30 THEN 30 ELSE rx.days_supply END/30) AS scripts
    , rx.ndcnum AS ndc_cd
    , rx.gpi AS adjudicated_gpi_cd
    , SUBSTR(rx.gpi,1,4) AS gpi4
    , SUBSTR(rx.gpi,1,2) AS gpi2
    , rx.billed_amt, rx.claim_adj_amt, rx.copay_amt
    , CASE WHEN rx.pharmacytype="R" THEN 1 ELSE 0 END AS retail_flag
    , CASE WHEN rx.pharmacytype="M" THEN 1 ELSE 0 END AS mail_order_flag
    , CASE WHEN rx.drugtype = 3 THEN 1 ELSE 0 END AS generic_fill_flag
    , CASE WHEN rx.drugtype = 2 THEN 1 ELSE 0 END AS branded_generic_fill_flag
    , CASE WHEN rx.drugtype = 4 THEN 1 ELSE 0 END AS otc_fill_flag
    , CASE WHEN rx.drugtype = 1 THEN 1 ELSE 0 END AS ss_brand_fill_flag
    , CASE WHEN rx.drugtype = 5 THEN 1 ELSE 0 END AS ms_brand_fill_flag
    , CASE WHEN rx.formularyflag="F" or rx.drugtype = 3 THEN 1 ELSE 0 END AS formulary_fill_flag
    , CASE WHEN c.maint_drug_cd="X" THEN 1 ELSE 0 END AS maint_drug_flag
    , CASE WHEN d.ndc IS NOT NULL THEN 1 ELSE 0 END AS specialty_rx_flag
FROM (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
INNER JOIN `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_RX_DATA_STAGE` AS rx
    ON st.asdb_member_key = rx.asdb_member_key AND st.asdb_plan_key = rx.asdb_plan_key
LEFT JOIN `edp-prod-storage.edp_hcb_anbor_enrsrc_prod.EDW_DRUG` AS c ON TRIM(rx.ndcnum) = TRIM(c.ndc_cd)
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_srcv.specdrug_pww_universal_spec_list` AS d ON TRIM(rx.ndcnum) = TRIM(d.ndc)
WHERE rx.ClaimType = "P"
    AND CAST(rx.asdb_incurred_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 24 MONTH) AND DATE_SUB(DATE_SUB(st.index_dt, INTERVAL 1 DAY), INTERVAL 12 MONTH)
;

-- Summarize Rx Year 2
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_rx_yr2`;

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_rx_yr2`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
WITH rx_tmp AS (
    SELECT asdb_member_key, asdb_plan_key, index_dt
        , MIN(disp_dt) AS first_disp_dt, MAX(disp_dt) AS last_disp_dt
        , COUNT(*) AS rx_claim_cnt, SUM(days_supply) AS days_supply_SUM
        , COUNT(DISTINCT ndc_cd) AS ndc_cnt, COUNT(DISTINCT adjudicated_gpi_cd) AS gpi_cnt
        , COUNT(DISTINCT gpi4) AS gpi4_cnt, COUNT(DISTINCT gpi2) AS gpi2_cnt
        , SUM(retail_flag) AS retail_fills, SUM(mail_order_flag) AS mail_order_fills
        , SUM(generic_fill_flag) AS generic_fills, SUM(branded_generic_fill_flag) AS branded_generic_fills
        , SUM(otc_fill_flag) AS otc_fills, SUM(ss_brand_fill_flag) AS ss_brand_fills
        , SUM(ms_brand_fill_flag) AS ms_brand_fills, SUM(formulary_fill_flag) AS formulary_fills
        , SUM(maint_drug_flag) AS maint_drug_fills
        , SUM(CASE WHEN gpi2="27" THEN Scripts ELSE 0 END) AS antidiabetic_scripts
        , SUM(CASE WHEN gpi2="27" THEN days_supply ELSE 0 END) AS antidiabetic_days_supply
        , SUM(CASE WHEN gpi2="33" THEN Scripts ELSE 0 END) AS beta_blocker_scripts
        , SUM(CASE WHEN gpi2="33" THEN days_supply ELSE 0 END) AS beta_blocker_days_supply
        , SUM(CASE WHEN gpi2="36" THEN Scripts ELSE 0 END) AS antihypertensive_scripts
        , SUM(CASE WHEN gpi2="36" THEN days_supply ELSE 0 END) AS antihypertensive_days_supply
        , SUM(CASE WHEN gpi2="39" THEN Scripts ELSE 0 END) AS lipid_lowering_scripts
        , SUM(CASE WHEN gpi2="39" THEN days_supply ELSE 0 END) AS lipid_lowering_days_supply
        , SUM(CASE WHEN gpi2="34" THEN Scripts ELSE 0 END) AS calcium_channel_blk_scripts
        , SUM(CASE WHEN gpi2="34" THEN days_supply ELSE 0 END) AS calcium_channel_blk_days_supply
        , SUM(CASE WHEN gpi2="37" THEN Scripts ELSE 0 END) AS diuretic_scripts
        , SUM(CASE WHEN gpi2="37" THEN days_supply ELSE 0 END) AS diuretic_days_supply
        , SUM(CASE WHEN gpi2="32" THEN Scripts ELSE 0 END) AS antianginal_agent_scripts
        , SUM(CASE WHEN gpi2="32" THEN days_supply ELSE 0 END) AS antianginal_agent_days_supply
        , SUM(CASE WHEN gpi2="58" THEN Scripts ELSE 0 END) AS antidepressant_scripts
        , SUM(CASE WHEN gpi2="58" THEN days_supply ELSE 0 END) AS antidepressant_days_supply
        , SUM(CASE WHEN gpi2="59" THEN Scripts ELSE 0 END) AS antipsychotic_scripts
        , SUM(CASE WHEN gpi2="59" THEN days_supply ELSE 0 END) AS antipsychotic_days_supply
        , SUM(CASE WHEN gpi2="57" THEN Scripts ELSE 0 END) AS antianxiety_scripts
        , SUM(CASE WHEN gpi2="57" THEN days_supply ELSE 0 END) AS antianxiety_days_supply
        , SUM(CASE WHEN gpi2="72" THEN Scripts ELSE 0 END) AS anticonvulsant_scripts
        , SUM(CASE WHEN gpi2="72" THEN days_supply ELSE 0 END) AS anticonvulsant_days_supply
        , SUM(CASE WHEN gpi4="4440" THEN Scripts ELSE 0 END) AS inhaled_steroid_scripts
        , SUM(CASE WHEN gpi4="4440" THEN days_supply ELSE 0 END) AS inhaled_steroid_days_supply
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_rx_claims_yr2`
    GROUP BY asdb_member_key, asdb_plan_key, index_dt
)
SELECT 
    st.asdb_member_key, st.asdb_plan_key, st.index_dt
    , rx.first_disp_dt, rx.last_disp_dt
    , COALESCE(rx.rx_claim_cnt, 0) AS rx_claim_cnt
    , COALESCE(rx.days_supply_sum, 0) AS days_supply_sum
    , COALESCE(rx.ndc_cnt, 0) AS ndc_cnt, COALESCE(rx.gpi_cnt, 0) AS gpi_cnt
    , COALESCE(rx.gpi4_cnt, 0) AS gpi4_cnt, COALESCE(rx.gpi2_cnt, 0) AS gpi2_cnt
    , COALESCE(rx.retail_fills, 0) AS retail_fills, COALESCE(rx.mail_order_fills, 0) AS mail_order_fills
    , COALESCE(rx.generic_fills, 0) AS generic_fills, COALESCE(rx.branded_generic_fills, 0) AS branded_generic_fills
    , COALESCE(rx.otc_fills, 0) AS otc_fills, COALESCE(rx.ss_brand_fills, 0) AS ss_brand_fills
    , COALESCE(rx.ms_brand_fills, 0) AS ms_brand_fills, COALESCE(rx.formulary_fills, 0) AS formulary_fills
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
FROM (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
LEFT JOIN rx_tmp rx ON st.asdb_member_key = rx.asdb_member_key AND st.asdb_plan_key = rx.asdb_plan_key
;

-- =====================================================================================
-- SECTION 6: DEMOGRAPHICS (007_Demographics.sh)
-- =====================================================================================
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_demographics`;

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_demographics`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
WITH mth AS (
    SELECT mnth.asdb_member_key, st.index_dt, CAST(mnth.asdb_elig_dt AS DATE) AS asdb_elig_dt
        , ROW_NUMBER() OVER(PARTITION BY(mnth.asdb_member_key) ORDER BY mnth.asdb_elig_dt) AS mnths
    FROM (SELECT asdb_member_key, asdb_elig_dt FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ELIG_DATA_MBR_PER_MTH`) AS mnth
    LEFT JOIN (SELECT asdb_member_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
        ON st.asdb_member_key = mnth.asdb_member_key
    WHERE CAST(mnth.asdb_elig_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 12 MONTH) AND index_dt
),
mth2 AS (
    SELECT mnth.asdb_member_key, st.index_dt, CAST(mnth.asdb_elig_dt AS DATE) AS asdb_elig_dt
        , ROW_NUMBER() OVER(PARTITION BY(mnth.asdb_member_key) ORDER BY mnth.asdb_elig_dt) AS mnths
    FROM (SELECT asdb_member_key, asdb_elig_dt FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ELIG_DATA_MBR_PER_MTH`) AS mnth
    LEFT JOIN (SELECT asdb_member_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
        ON st.asdb_member_key = mnth.asdb_member_key
    WHERE CAST(mnth.asdb_elig_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 24 MONTH) AND DATE_SUB(st.index_dt, INTERVAL 13 MONTH)
),
post AS (
    SELECT mnth.asdb_member_key, st.index_dt, CAST(mnth.asdb_elig_dt AS DATE) AS asdb_elig_dt
        , ROW_NUMBER() OVER(PARTITION BY(mnth.asdb_member_key) ORDER BY mnth.asdb_elig_dt) AS mnths
    FROM (SELECT asdb_member_key, asdb_elig_dt FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ELIG_DATA_MBR_PER_MTH`) AS mnth
    LEFT JOIN (SELECT asdb_member_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
        ON st.asdb_member_key = mnth.asdb_member_key
    WHERE CAST(mnth.asdb_elig_dt AS DATE) BETWEEN DATE_ADD(st.index_dt, INTERVAL 1 MONTH) AND DATE_ADD(st.index_dt, INTERVAL 6 MONTH)
)
SELECT 
    st.asdb_member_key, st.asdb_plan_key, st.index_dt
    , FLOOR(DATE_DIFF(DATE(st.index_dt), DATE(mb.dob), YEAR)) AS agenbr
    , mb.gender, mb.ethnicity_code, mb.primarylanguage_desc
    , COALESCE(mth.tenure, 0) AS tenure_yr1
    , COALESCE(mth2.tenure, 0) AS tenure_yr2
    , COALESCE(post.tenure, 0) AS post_mnths
    , zcuu.urbsubr, zcuu.zip_weight_avg_medinc
FROM (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_MEMBER` AS mb ON st.asdb_member_key = mb.asdb_member_key
LEFT JOIN (SELECT asdb_member_key, MAX(mnths) AS tenure FROM mth GROUP BY asdb_member_key) AS mth ON st.asdb_member_key = mth.asdb_member_key
LEFT JOIN (SELECT asdb_member_key, MAX(mnths) AS tenure FROM mth2 GROUP BY asdb_member_key) AS mth2 ON st.asdb_member_key = mth2.asdb_member_key
LEFT JOIN (SELECT asdb_member_key, MAX(mnths) AS tenure FROM post GROUP BY asdb_member_key) AS post ON st.asdb_member_key = post.asdb_member_key
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_tra_ckd_phm_srcv.ZIP_CENSUS_USPS_URBRUR` AS zcuu ON TRIM(mb.member_zip) = TRIM(zcuu.zip_cd)
;

-- =====================================================================================
-- SECTION 7: GEO ID (008_GeoID.sh)
-- =====================================================================================
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_geoid`;

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_geoid`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
WITH maxdt AS (
    SELECT iodb_member_key, MAX(source_pstd_dts) AS source_pstd_dts
    FROM `anbc-hcb-prod.insights_share_hcb_prod.v_enriched_address_medicaid`
    GROUP BY iodb_member_key
)
SELECT DISTINCT 
    st.asdb_member_key, st.asdb_plan_key, st.index_dt, mb.iodb_member_key, id.ctfips, id.bgfips
FROM (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
INNER JOIN (SELECT iodb_member_key, asdb_member_key, asdb_plan_key FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_MEMBER`) AS mb
    ON st.asdb_member_key=mb.asdb_member_key AND st.asdb_plan_key=mb.asdb_plan_key
INNER JOIN (SELECT block_code AS bgfips, CONCAT(fips_state_county_code, census_tract) AS ctfips, iodb_member_key, source_pstd_dts, geo_accuracy_code
    FROM `anbc-hcb-prod.insights_share_hcb_prod.v_enriched_address_medicaid`) AS id ON mb.iodb_member_key=id.iodb_member_key
INNER JOIN maxdt ON id.iodb_member_key=maxdt.iodb_member_key AND id.source_pstd_dts=maxdt.source_pstd_dts
WHERE TRIM(id.geo_accuracy_code) IN ("1", "2", "5", "6")
;

-- =====================================================================================
-- SECTION 8: ACS DATA (009_ACS.sh)
-- =====================================================================================
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_acs`;

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_acs`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
SELECT st.asdb_member_key, st.asdb_plan_key, st.index_dt
    , COALESCE(MAX(b.social_risk_score), 0) AS social_risk_score
    , COALESCE(MAX(b.sdi_score), 0) AS sdi_score
    , COALESCE(MAX(b.svi_score), 0) AS svi_score
    , COALESCE(MAX(b.adi_score), 0) AS adi_score
FROM (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_geoid` AS id ON st.asdb_member_key=id.asdb_member_key AND st.asdb_plan_key=id.asdb_plan_key
LEFT JOIN (SELECT * FROM `edp-prod-storage.edp_ent_sdoheir_srcv.srs_acs_block_group_allscores_historical_data` WHERE effective_year = 2022) AS b ON id.ctfips = b.ctfips
GROUP BY st.asdb_member_key, st.asdb_plan_key, st.index_dt
;

-- =====================================================================================
-- SECTION 9: CSDI/SDOH INDICES (011_CSDI_risk.sh)
-- =====================================================================================
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_csdi`;

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_csdi`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
SELECT st.asdb_member_key, st.asdb_plan_key, st.index_dt
    , COALESCE(MAX(c.citizenship_index), 0) AS citizenship_index
    , COALESCE(MAX(c.education_index), 0) AS education_index
    , COALESCE(MAX(c.food_access), 0) AS food_access
    , COALESCE(MAX(c.health_access), 0) AS health_access
    , COALESCE(MAX(c.health_habits), 0) AS health_habits
    , COALESCE(MAX(c.housing_desert), 0) AS housing_desert
    , COALESCE(MAX(c.housing_ownership), 0) AS housing_ownership     
    , COALESCE(MAX(c.housing_quality), 0) AS housing_quality     
    , COALESCE(MAX(c.income_index), 0) AS income_index    
    , COALESCE(MAX(c.income_inequality), 0) AS income_inequality    
    , COALESCE(MAX(c.language_score), 0) AS language_score    
    , COALESCE(MAX(c.natural_disaster), 0) AS natural_disaster 
    , COALESCE(MAX(c.poverty_score), 0) AS poverty_score 
    , COALESCE(MAX(c.proactive_health), 0) AS proactive_health
    , COALESCE(MAX(c.racial_diversity), 0) AS racial_diversity
    , COALESCE(MAX(c.social_isolation), 0) AS social_isolation    
    , COALESCE(MAX(c.technology_access), 0) AS technology_access    
    , COALESCE(MAX(c.transport_access), 0) AS transport_access     
    , COALESCE(MAX(c.unemployment_index), 0) AS unemployment_index    
    , COALESCE(MAX(c.water_quality), 0) AS water_quality     
    , COALESCE(MAX(c.disability_score), 0) AS disability_score    
    , COALESCE(MAX(c.health_infra), 0) AS health_infra    
    , COALESCE(MAX(c.social_risk_score), 0) AS social_risk_score    
FROM (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt, bgfips FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_geoid`) AS st
LEFT JOIN (SELECT * FROM `edp-prod-storage.edp_ent_sdoheir_srcv.risk_index_block_group_historical_data` WHERE effective_year = 2023) AS c
    ON TRIM(st.bgfips)=TRIM(c.bgfips)
GROUP BY st.asdb_member_key, st.asdb_plan_key, st.index_dt
;

-- =====================================================================================
-- SECTION 10: PREVENTATIVE CARE (010_preventative.sh)
-- =====================================================================================
-- Note: This section uses the med_claims_flag_yr1 table created in Section 3A.6
-- The preventative table creates flags for various preventive care services

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_preventative`;

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_preventative`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
SELECT
    asdb_member_key, asdb_plan_key, claimid, index_dt, asdb_incurred_dt
    , CASE WHEN plc_svc_ctg = "Outpatient"
            AND (asdb_coe_id IN (63000, 63100, 63200, 63300, 63400, 63500, 63600, 63999)
                OR TRIM(prindiag) in ("V20.2","V20.31","V20.32","V70.0","V70.3","V70.5","V70.6","V70.8","V70.9","V72.31","V72.3",
                    "Z00.110","Z00.111","Z00.129", "Z00.8","Z01.411","Z01.419","Z01.42"))
            AND TRIM(emis_cat) = "Primary Physician"
        THEN 1 ELSE 0 END AS pcp_op_visit
    , CASE WHEN plc_svc_ctg = "Outpatient"
            AND (asdb_coe_id IN (63000, 63100, 63200, 63300, 63400, 63500, 63600, 63999)
                OR TRIM(prindiag) in ("V20.2","V20.31","V20.32","V70.0","V70.3","V70.5","V70.6","V70.8","V70.9","V72.31","V72.3",
                    "Z00.110","Z00.111","Z00.129", "Z00.8","Z01.411","Z01.419","Z01.42"))
            AND NOT (TRIM(emis_cat) = "Primary Physician" OR TRIM(lower(prov_specialty)) LIKE "%gynecol%")
        THEN 1 ELSE 0 END AS spec_op_visit
    , CASE WHEN plc_svc_ctg = "Outpatient"
            AND (asdb_coe_id IN (63000, 63100, 63200, 63300, 63400, 63500, 63600, 63999)
                OR TRIM(prindiag) in ("V20.2","V20.31","V20.32","V70.0","V70.3","V70.5","V70.6","V70.8","V70.9","V72.31","V72.3",
                    "Z00.110","Z00.111","Z00.129", "Z00.8","Z01.411","Z01.419","Z01.42"))
            AND (TRIM(lower(prov_specialty)) LIKE "%midwif%" OR TRIM(lower(prov_specialty)) LIKE "%gynecol%")
        THEN 1 ELSE 0 END AS obgyn_mw_op_visit
    , CASE WHEN TRIM(servcode) in ("E0185","E0188","E0189","E0194","E0197","E0198","E0199","E0250","E0251","E0255","E0256","E0260",
            "E0261","E0265","E0266","E0290","E0291","E0292","E0293","E0294","E0295","E0296","E0297","E0300","E0301","E0302","E0303",
            "E0304","E0424","E0431","E0433","E0434","E0439","E0441","E0442","E0443","E0444","E0450","E0460","E0461","E0462","E0463",
            "E0464","E0470","E0471","E0472","E0480","E0482","E0483","E0484","E0570","E0575","E0580","E0585","E0601","E0607","E0627",
            "E0628","E0629","E0636","E0650","E0651","E0652","E0655","E0656","E0657","E0660","E0665","E0666","E0667","E0668","E0669",
            "E0671","E0672","E0673","E0675","E0692","E0693","E0694","E0720","E0730","E0731","E0740","E0744","E0745","E0747","E0748",
            "E0749","E0760","E0762","E0764","E0765","E0782","E0783","E0784","E0786","E0840","E0849","E0850","E0855","E0856","E0958",
            "E0959","E0960","E0961","E0966","E0967","E0968","E0969","E0971","E0973","E0974","E0978","E0980","E0981","E0982","E0983",
            "E0984","E0985","E0986","E0990","E0992","E0994","E1014","E1015","E1020","E1028","E1029","E1030","E1031","E1035","E1036",
            "E1037","E1038","E1039","E1161","E1227","E1228","E1232","E1233","E1234","E1235","E1236","E1237","E1238","E1296","E1297",
            "E1298","E1310","E2502","E2506","E2508","E2510","E2227","K0001","K0002","K0003","K0004","K0005","K0006","K0007","K0009",
            "K0606","K0730") THEN 1 ELSE 0 END AS dme_significant
    , CASE WHEN TRIM(servcode) in ("80061","83715","83716","83721","83718","83700","83701","83704","3048F","3049F","3050F") THEN 1 ELSE 0 END AS cholest_screen_claim
    , CASE WHEN TRIM(servcode) in ("83036","83037","3044F","3045F","3046F") THEN 1 ELSE 0 END AS hba1c_test_claim
    , CASE WHEN TRIM(servcode) in ("C9287","J0178","Q2048","Q2049","Q2043","C9296","Q2050","J8510","J8520","J8521","J8530","J8560","J8600","J8610","J8700","J8705","J8999")
          OR (TRIM(servcode) LIKE "J9%" AND TRIM(servcode) not in ("J9202","J9395","J9217")) THEN 1 ELSE 0 END AS chemo_clm_flg
    , CASE WHEN TRIM(servcode) in ("G0442","G0443") THEN 1 ELSE 0 END AS CMS_Alcohol_Misuse_Screening_Counseling
    , CASE WHEN TRIM(servcode) in ("0554T","0555T","0556T","0557T","0558T","76977","77078","77080","77081","77085","G0130")
            AND TRIM(prindiag) in ("E21.0","E21.3","E23.0","E34.2","E89.40","E89.41","M80.08xA","M80.88xA","M84.58xA","M84.68xA","N95.8","N95.9","Q78.0","S34.3xxA","Z78.0","Z79.3","Z79.51","Z79.52","Z79.811","Z79.818","Z79.83","Z87.310","E24","E28.3","M48","M81","M85.8","Q96","S12","S14","S22","S24","S32.0","S32.1","S32.2","S34.1")
        THEN 1 ELSE 0 END AS CMS_Bone_Mass_Measurements
    , CASE WHEN TRIM(servcode) in ("82465","83718","84478") AND TRIM(prindiag) = "Z13.6" THEN 1 ELSE 0 END AS CMS_Cardiovascular_Disease_Screening
    , CASE WHEN TRIM(servcode) in ("81528","82270","G0104","G0105","G0106","G0120","G0121","G0328") AND TRIM(prindiag) IN ("Z86.004","Z12.11","Z12.12") THEN 1 ELSE 0 END AS CMS_Colorectal_Cancer_Screening
    , CASE WHEN TRIM(servcode) in ("99406","99407") AND TRIM(prindiag) in ("F17.210","F17.211","F17.213","F17.218","F17.219","F17.220","F17.221","F17.223","F17.228","F17.229","F17.290","F17.291","F17.293","F17.298","F17.299","T65.211A","T65.212A","T65.213A","T65.214A","T65.221A","T65.222A","T65.223A","T65.224A","T65.291A","T65.292A","T65.293A","T65.294A","Z87.891") THEN 1 ELSE 0 END AS CMS_tobacco_use_counseling
    , CASE WHEN TRIM(servcode) in ("G0444") THEN 1 ELSE 0 END AS CMS_depression_screening
    , CASE WHEN TRIM(servcode) in ("82947","82950","82951") AND TRIM(prindiag) ="Z13.1" THEN 1 ELSE 0 END AS CMS_Diabetes_Screening
    , CASE WHEN TRIM(servcode) in ("G0499") THEN 1 ELSE 0 END AS CMS_hep_b_virus_Screening
    , CASE WHEN TRIM(servcode) in ("90739","90740","90743","90744","90746","90747","G0010") AND TRIM(prindiag) ="Z23" THEN 1 ELSE 0 END AS CMS_Hep_B_Virus_Vax
    , CASE WHEN TRIM(servcode) in ("G0446") THEN 1 ELSE 0 END AS CMS_IBT_for_CVD
    , CASE WHEN TRIM(servcode) in ("G0447","G0473") AND TRIM(prindiag) in ("Z68.30","Z68.31","Z68.32","Z68.33","Z68.34","Z68.35","Z68.36","Z68.37","Z68.38","Z68.39","Z68.41","Z68.42","Z68.43","Z68.44","Z68.45") THEN 1 ELSE 0 END AS CMS_IBT_for_obesity
    , CASE WHEN TRIM(servcode) in ("90630","90653","90654","90655","90656","90657","90658","90660","90662","90672","90673","90674","90682","90685","90686","90687","90688","90689","90694","90756","Q2034","Q2035","Q2036","Q2037","Q2038","Q2039","G0008") AND TRIM(prindiag) = "Z23" THEN 1 ELSE 0 END AS CMS_Influenza_Virus_Vaccine
    , CASE WHEN TRIM(servcode) in ("G0296","G0297") AND TRIM(prindiag) in ("F17.210","F17.211","F17.213","F17.218","F17.219","Z87.891") THEN 1 ELSE 0 END AS CMS_lung_Cancer_Screening
    , CASE WHEN TRIM(servcode) in ("97802","97803","97804","G0270","G0271") THEN 1 ELSE 0 END AS CMS_nutrition_therapy
    , CASE WHEN TRIM(servcode) in ("90670","90732","G0009") AND TRIM(prindiag) = "Z23" THEN 1 ELSE 0 END AS CMS_Pneumococcal_Vaccine
    , CASE WHEN TRIM(servcode) in ("G0476") AND TRIM(prindiag) in ("Z11.51","Z01.411","Z01.419") THEN 1 ELSE 0 END AS CMS_Cervical_cancer_hpv
    , CASE WHEN TRIM(servcode) in ("86631","86632","87110","87270","87320","87490","87491","87810","87800","87590","87591","87850","87800","86592","86593","86780","87340","87341","G0445") AND TRIM(prindiag) in ("Z11.3","Z11.59","Z34.00","Z34.01","Z34.02","Z34.03","Z34.80","Z34.81","Z34.82","Z34.83","Z34.90","Z34.91","Z34.92","Z34.93","Z72.51","Z72.52","Z72.53","Z72.89","O09.90","O09.91","O09.92","O09.93") THEN 1 ELSE 0 END AS CMS_sti_screening
    , CASE WHEN TRIM(servcode) in ("77063","77067") AND TRIM(prindiag) in ("N63.15","N63.25","Z12.31") THEN 1 ELSE 0 END AS CMS_screening_Mammography
    , CASE WHEN TRIM(servcode) in ("G0123","G0124","G0141","G0143","G0144","G0145","G0147","G0148","P3000","P3001") AND TRIM(prindiag) in ("Z72.51","Z72.52","Z72.53","Z77.29","Z77.9","Z91.89","Z92.89","Z01.411","Z01.419","Z12.4","Z12.72","Z12.79","Z12.89") THEN 1 ELSE 0 END AS CMS_screening_pap
    , CASE WHEN TRIM(servcode) = "G0101" AND TRIM(prindiag) in ("Z72.51","Z72.52","Z72.53","Z77.29","Z77.9","Z91.89","Z92.89","Z77.22","Z77.29","Z72.89","Z92.89","Z01.411","Z01.419","Z12.4","Z12.72","Z12.79","Z12.89") THEN 1 ELSE 0 END AS CMS_Screening_Pelvic_Exams
    , CASE WHEN TRIM(servcode) in ("G0108","G0109") THEN 1 ELSE 0 END AS CMS_Diabetes_Self_Management_Training
    , CASE WHEN TRIM(servcode) = "G0102" AND TRIM(prindiag) = "Z12.5" THEN 1 ELSE 0 END AS CMS_prostate_cancer_rectal_examination
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_med_claims_flag_yr1`
;

-- Summarize preventative
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_preventative_summary`;

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_preventative_summary`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
SELECT asdb_member_key, asdb_plan_key, index_dt
    , MIN(asdb_incurred_dt) AS first_prv_dt
    , MAX(asdb_incurred_dt) AS last_prv_dt
    , SUM(pcp_op_visit) AS sum_pcp
    , SUM(spec_op_visit) AS sum_spec
    , SUM(obgyn_mw_op_visit) AS sum_ob
    , SUM(dme_significant) AS sum_dme
    , SUM(cholest_screen_claim) AS sum_chol_lab
    , SUM(hba1c_test_claim) AS sum_a1c_lab
    , SUM(chemo_clm_flg) AS sum_chemo
    , SUM(cms_alcohol_misuse_screening_counseling) AS cms_alc_scrn
    , SUM(cms_bone_mass_measurements) AS cms_bone_scrn
    , SUM(cms_cardiovascular_disease_screening) AS cms_cvd_scrn
    , SUM(cms_colorectal_cancer_screening) AS cms_col_scrn
    , SUM(cms_tobacco_use_counseling) AS cms_tobacco
    , SUM(cms_depression_screening) AS cms_dep_scrn
    , SUM(cms_diabetes_screening) AS cms_t2d_scrn
    , SUM(cms_hep_b_virus_screening) AS cms_hepb_scrn
    , SUM(cms_hep_b_virus_vax) AS cms_hepb_vax
    , SUM(cms_ibt_for_cvd) AS cms_ibt_cvd
    , SUM(cms_ibt_for_obesity) AS cms_ibt_obese
    , SUM(cms_influenza_virus_vaccine) AS cms_flu_vax
    , SUM(cms_lung_cancer_screening) AS cms_lung_cancer_scrn
    , SUM(cms_nutrition_therapy) AS cms_nutrition
    , SUM(cms_pneumococcal_vaccine) AS cms_pneum_vax
    , SUM(cms_cervical_cancer_hpv) AS cms_hpv_scrn
    , SUM(cms_sti_screening) AS cms_sti_scrn
    , SUM(cms_screening_mammography) AS cms_mam_scrn
    , SUM(cms_screening_pap) AS cms_pap
    , SUM(cms_screening_pelvic_exams) AS cms_pelvic
    , SUM(cms_diabetes_self_management_training) AS cms_t2d_train
    , SUM(cms_prostate_cancer_rectal_examination) AS cms_prost_cancer_scrn
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_preventative`
GROUP BY asdb_member_key, asdb_plan_key, index_dt
;

-- =====================================================================================
-- SECTION 11: NON-EMBEDDING FEATURE BEAST (013_non_embedding_feature_beast.sh)
-- =====================================================================================
-- This is the final feature table that joins all intermediate tables together
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_non_embedding_features`;

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_non_embedding_features`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
SELECT
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
    , st.coa_population_category
    , st.coa_population_group
    -- ED Features Year 1
    , ed1.sum_ed_visits AS sum_ed_visits_yr1
    , ed1.ed_flag AS ed_flag_yr1
    , ed1.sum_avoidable AS sum_avoidable_yr1
    , ed1.sum_unnecessary AS sum_unnecessary_yr1
    , ed1.sum_preventable AS sum_preventable_yr1
    , ed1.low_sev_ed_visits AS low_sev_ed_visits_yr1
    , ed1.low_med_sev_ed_visits AS low_med_sev_ed_visits_yr1
    , ed1.med_sev_ed_visits AS med_sev_ed_visits_yr1
    , ed1.med_high_sev_ed_visits AS med_high_sev_ed_visits_yr1
    , ed1.high_sev_ed_visits AS high_sev_ed_visits_yr1
    , ed1.low_sev_ed_flag AS low_sev_ed_flag_yr1
    , ed1.low_med_sev_ed_flag AS low_med_sev_ed_flag_yr1
    , ed1.med_sev_ed_flag AS med_sev_ed_flag_yr1
    , ed1.med_high_sev_ed_flag AS med_high_sev_ed_flag_yr1
    , ed1.high_sev_ed_flag AS high_sev_ed_flag_yr1
    -- ED Features Year 2
    , ed2.ed_flag AS ed_flag_yr2   
    , ed2.sum_ed_visits AS sum_ed_visits_yr2
    , ed2.sum_avoidable AS sum_avoidable_yr2
    , ed2.sum_unnecessary AS sum_unnecessary_yr2
    , ed2.sum_preventable AS sum_preventable_yr2
    , ed2.low_sev_ed_visits AS low_sev_ed_visits_yr2
    , ed2.low_med_sev_ed_visits AS low_med_sev_ed_visits_yr2
    , ed2.med_sev_ed_visits AS med_sev_ed_visits_yr2
    , ed2.med_high_sev_ed_visits AS med_high_sev_ed_visits_yr2
    , ed2.high_sev_ed_visits AS high_sev_ed_visits_yr2
    , ed2.low_sev_ed_flag AS low_sev_ed_flag_yr2
    , ed2.low_med_sev_ed_flag AS low_med_sev_ed_flag_yr2
    , ed2.med_sev_ed_flag AS med_sev_ed_flag_yr2
    , ed2.med_high_sev_ed_flag AS med_high_sev_ed_flag_yr2
    , ed2.high_sev_ed_flag AS high_sev_ed_flag_yr2
    -- IP Features
    , ip1.acute_ip_flag AS acute_ip_flag_yr1
    , ip1.sum_acute_ip_admits AS sum_acute_ip_admits_yr1
    , ip1.sum_acute_calc_los AS sum_acute_calc_los_yr1
    , ip2.acute_ip_flag AS acute_ip_flag_yr2
    , ip2.sum_acute_ip_admits AS sum_acute_ip_admits_yr2
    , ip2.sum_acute_calc_los AS sum_acute_calc_los_yr2
    -- OP Features
    , op1.sum_op_visits AS sum_op_visits_yr1
    , op2.sum_op_visits AS sum_op_visits_yr2
    -- Utilization Claim Counts Year 1
    , ut1.emis_community_clm AS emis_community_clm_yr1
    , ut1.emis_ed_clm AS emis_ed_clm_yr1
    , ut1.emis_hh_clm AS emis_hh_clm_yr1
    , ut1.emis_home_clm AS emis_home_clm_yr1
    , ut1.emis_ip_clm AS emis_ip_clm_yr1
    , ut1.emis_ins_clm AS emis_ins_clm_yr1
    , ut1.emis_lab_clm AS emis_lab_clm_yr1
    , ut1.emis_mrx_clm AS emis_mrx_clm_yr1
    , ut1.emis_mh_clm AS emis_mh_clm_yr1
    , ut1.emis_misc_clm AS emis_misc_clm_yr1
    , ut1.emis_pcp_clm AS emis_pcp_clm_yr1
    , ut1.emis_radio_clm AS emis_radio_clm_yr1
    , ut1.emis_ambul_clm AS emis_ambul_clm_yr1
    , ut1.emis_spec_clm AS emis_spec_clm_yr1
    , ut1.ltc_clm AS ltc_clm_yr1
    , ut1.coe_ip_hos_clm AS coe_ip_hos_clm_yr1
    , ut1.coe_ip_non_hos_clm AS coe_ip_non_hos_clm_yr1
    , ut1.coe_lab_clm AS coe_lab_clm_yr1
    , ut1.coe_ltc_community_clm AS coe_ltc_community_clm_yr1
    , ut1.coe_ltc_home_clm AS coe_ltc_home_clm_yr1
    , ut1.coe_ltc_ins_clm AS coe_ltc_ins_clm_yr1
    , ut1.coe_other_clm AS coe_other_clm_yr1
    , ut1.coe_op_hos_clm AS coe_op_hos_clm_yr1
    , ut1.coe_op_non_hos_clm AS coe_op_non_hos_clm_yr1
    , ut1.coe_anesth_clm AS coe_anesth_clm_yr1
    , ut1.coe_eval_clm AS coe_eval_clm_yr1
    , ut1.coe_maternity_clm AS coe_maternity_clm_yr1
    , ut1.coe_mrx_clm AS coe_mrx_clm_yr1
    , ut1.coe_mh_clm AS coe_mh_clm_yr1
    , ut1.coe_phy_clm AS coe_phy_clm_yr1
    , ut1.coe_surg_clm AS coe_surg_clm_yr1
    , ut1.coe_radio_clm AS coe_radio_clm_yr1
    , ut1.uc_clm AS uc_clm_yr1
    , ut1.obs_clm AS obs_clm_yr1   
    -- Utilization Claim Counts Year 2
    , ut2.emis_community_clm AS emis_community_clm_yr2
    , ut2.emis_ed_clm AS emis_ed_clm_yr2
    , ut2.emis_hh_clm AS emis_hh_clm_yr2
    , ut2.emis_home_clm AS emis_home_clm_yr2
    , ut2.emis_ip_clm AS emis_ip_clm_yr2
    , ut2.emis_ins_clm AS emis_ins_clm_yr2
    , ut2.emis_lab_clm AS emis_lab_clm_yr2
    , ut2.emis_mrx_clm AS emis_mrx_clm_yr2
    , ut2.emis_mh_clm AS emis_mh_clm_yr2
    , ut2.emis_misc_clm AS emis_misc_clm_yr2
    , ut2.emis_pcp_clm AS emis_pcp_clm_yr2
    , ut2.emis_radio_clm AS emis_radio_clm_yr2
    , ut2.emis_ambul_clm AS emis_ambul_clm_yr2
    , ut2.emis_spec_clm AS emis_spec_clm_yr2
    , ut2.ltc_clm AS ltc_clm_yr2
    , ut2.coe_ip_hos_clm AS coe_ip_hos_clm_yr2
    , ut2.coe_ip_non_hos_clm AS coe_ip_non_hos_clm_yr2
    , ut2.coe_lab_clm AS coe_lab_clm_yr2
    , ut2.coe_ltc_community_clm AS coe_ltc_community_clm_yr2
    , ut2.coe_ltc_home_clm AS coe_ltc_home_clm_yr2
    , ut2.coe_ltc_ins_clm AS coe_ltc_ins_clm_yr2
    , ut2.coe_other_clm AS coe_other_clm_yr2
    , ut2.coe_op_hos_clm AS coe_op_hos_clm_yr2
    , ut2.coe_op_non_hos_clm AS coe_op_non_hos_clm_yr2
    , ut2.coe_anesth_clm AS coe_anesth_clm_yr2
    , ut2.coe_eval_clm AS coe_eval_clm_yr2
    , ut2.coe_maternity_clm AS coe_maternity_clm_yr2
    , ut2.coe_mrx_clm AS coe_mrx_clm_yr2
    , ut2.coe_mh_clm AS coe_mh_clm_yr2
    , ut2.coe_phy_clm AS coe_phy_clm_yr2
    , ut2.coe_surg_clm AS coe_surg_clm_yr2
    , ut2.coe_radio_clm AS coe_radio_clm_yr2
    , ut2.uc_clm AS uc_clm_yr2
    , ut2.obs_clm AS obs_clm_yr2
    -- Conditions
    , cond.abdominal_pain, cond.AID, cond.IDA, cond.ANX, cond.OST, cond.AST, cond.AUT, cond.CHO
    , cond.burns, cond.cad, cond.Cancer, cond.narc, cond.CBD, cond.CHF, cond.CRF, cond.VNA
    , cond.CHD, cond.COP, cond.CYS, cond.DEP, cond.DIA, cond.EDO, cond.esrd, cond.EPL, cond.CRO
    , cond.MOH, cond.HEM, cond.HepC, cond.HYP, cond.HYC, cond.immune, cond.intel_dsblty
    , cond.meta_cancer, cond.liver_dis, cond.MSS, cond.OBE, cond.oud, cond.liver_other
    , cond.paralysis, cond.PAR, cond.PUD, cond.hmd, cond.PVD, cond.autoimmune, cond.DEM
    , cond.SCA, cond.sleep_apnea, cond.spinal_inj, cond.back, cond.substance, cond.ALC
    , cond.bipolar, cond.psychoses, cond.major_chronic_cnt 
    -- Rx Features Year 1
    , rx1.rx_claim_cnt AS rx_claim_cnt_yr1
    , rx1.days_supply_sum AS days_supply_sum_yr1
    , rx1.ndc_cnt AS ndc_cnt_yr1
    , rx1.gpi_cnt AS gpi_cnt_yr1
    , rx1.gpi4_cnt AS gpi4_cnt_yr1
    , rx1.gpi2_cnt AS gpi2_cnt_yr1
    , rx1.retail_fills AS retail_fills_yr1
    , rx1.mail_order_fills AS mail_order_fills_yr1
    , rx1.generic_fills AS generic_fills_yr1
    , rx1.branded_generic_fills AS branded_generic_fills_yr1
    , rx1.otc_fills AS otc_fills_yr1
    , rx1.ss_brand_fills AS ss_brand_fills_yr1
    , rx1.ms_brand_fills AS ms_brand_fills_yr1
    , rx1.formulary_fills AS formulary_fills_yr1
    , rx1.maint_drug_fills AS maint_drug_fills_yr1
    , rx1.antidiabetic_scripts AS antidiabetic_scripts_yr1
    , rx1.antidiabetic_days_supply AS antidiabetic_days_supply_yr1
    , rx1.beta_blocker_scripts AS beta_blocker_scripts_yr1
    , rx1.beta_blocker_days_supply AS beta_blocker_days_supply_yr1
    , rx1.antihypertensive_scripts AS antihypertensive_scripts_yr1
    , rx1.antihypertensive_days_supply AS antihypertensive_days_supply_yr1
    , rx1.lipid_lowering_scripts AS lipid_lowering_scripts_yr1
    , rx1.lipid_lowering_days_supply AS lipid_lowering_days_supply_yr1
    , rx1.calcium_channel_blk_scripts AS calcium_channel_blk_scripts_yr1
    , rx1.calcium_channel_blk_days_supply AS calcium_channel_blk_days_supply_yr1
    , rx1.diuretic_scripts AS diuretic_scripts_yr1
    , rx1.diuretic_days_supply AS diuretic_days_supply_yr1
    , rx1.antianginal_agent_scripts AS antianginal_agent_scripts_yr1
    , rx1.antianginal_agent_days_supply AS antianginal_agent_days_supply_yr1
    , rx1.antidepressant_scripts AS antidepressant_scripts_yr1
    , rx1.antidepressant_days_supply AS antidepressant_days_supply_yr1
    , rx1.antipsychotic_scripts AS antipsychotic_scripts_yr1
    , rx1.antipsychotic_days_supply AS antipsychotic_days_supply_yr1
    , rx1.antianxiety_scripts AS antianxiety_scripts_yr1
    , rx1.antianxiety_days_supply AS antianxiety_days_supply_yr1
    , rx1.anticonvulsant_scripts AS anticonvulsant_scripts_yr1
    , rx1.anticonvulsant_days_supply AS anticonvulsant_days_supply_yr1
    , rx1.inhaled_steroid_scripts AS inhaled_steroid_scripts_yr1
    , rx1.inhaled_steroid_days_supply AS inhaled_steroid_days_supply_yr1
    -- Rx Features Year 2
    , rx2.rx_claim_cnt AS rx_claim_cnt_yr2
    , rx2.days_supply_sum AS days_supply_sum_yr2
    , rx2.ndc_cnt AS ndc_cnt_yr2
    , rx2.gpi_cnt AS gpi_cnt_yr2
    , rx2.gpi4_cnt AS gpi4_cnt_yr2
    , rx2.gpi2_cnt AS gpi2_cnt_yr2
    , rx2.retail_fills AS retail_fills_yr2
    , rx2.mail_order_fills AS mail_order_fills_yr2
    , rx2.generic_fills AS generic_fills_yr2
    , rx2.branded_generic_fills AS branded_generic_fills_yr2
    , rx2.otc_fills AS otc_fills_yr2
    , rx2.ss_brand_fills AS ss_brand_fills_yr2
    , rx2.ms_brand_fills AS ms_brand_fills_yr2
    , rx2.formulary_fills AS formulary_fills_yr2
    , rx2.maint_drug_fills AS maint_drug_fills_yr2
    , rx2.antidiabetic_scripts AS antidiabetic_scripts_yr2
    , rx2.antidiabetic_days_supply AS antidiabetic_days_supply_yr2
    , rx2.beta_blocker_scripts AS beta_blocker_scripts_yr2
    , rx2.beta_blocker_days_supply AS beta_blocker_days_supply_yr2
    , rx2.antihypertensive_scripts AS antihypertensive_scripts_yr2
    , rx2.antihypertensive_days_supply AS antihypertensive_days_supply_yr2
    , rx2.lipid_lowering_scripts AS lipid_lowering_scripts_yr2
    , rx2.lipid_lowering_days_supply AS lipid_lowering_days_supply_yr2
    , rx2.calcium_channel_blk_scripts AS calcium_channel_blk_scripts_yr2
    , rx2.calcium_channel_blk_days_supply AS calcium_channel_blk_days_supply_yr2
    , rx2.diuretic_scripts AS diuretic_scripts_yr2
    , rx2.diuretic_days_supply AS diuretic_days_supply_yr2
    , rx2.antianginal_agent_scripts AS antianginal_agent_scripts_yr2
    , rx2.antianginal_agent_days_supply AS antianginal_agent_days_supply_yr2
    , rx2.antidepressant_scripts AS antidepressant_scripts_yr2
    , rx2.antidepressant_days_supply AS antidepressant_days_supply_yr2
    , rx2.antipsychotic_scripts AS antipsychotic_scripts_yr2
    , rx2.antipsychotic_days_supply AS antipsychotic_days_supply_yr2
    , rx2.antianxiety_scripts AS antianxiety_scripts_yr2
    , rx2.antianxiety_days_supply AS antianxiety_days_supply_yr2
    , rx2.anticonvulsant_scripts AS anticonvulsant_scripts_yr2
    , rx2.anticonvulsant_days_supply AS anticonvulsant_days_supply_yr2
    , rx2.inhaled_steroid_scripts AS inhaled_steroid_scripts_yr2
    , rx2.inhaled_steroid_days_supply AS inhaled_steroid_days_supply_yr2
    -- Demographics
    , demo.agenbr, demo.gender, demo.ethnicity_code, demo.primarylanguage_desc
    , demo.tenure_yr1, demo.tenure_yr2, demo.post_mnths
    , demo.urbsubr, demo.zip_weight_avg_medinc
    -- ACS/SDOH
    , acs.social_risk_score AS acs_social_risk_score
    , acs.sdi_score, acs.svi_score, acs.adi_score
    -- CSDI
    , csdi.citizenship_index, csdi.education_index, csdi.food_access, csdi.health_access
    , csdi.health_habits, csdi.housing_desert, csdi.housing_ownership, csdi.housing_quality
    , csdi.income_index, csdi.income_inequality, csdi.language_score, csdi.natural_disaster
    , csdi.poverty_score, csdi.proactive_health, csdi.racial_diversity, csdi.social_isolation
    , csdi.technology_access, csdi.transport_access, csdi.unemployment_index, csdi.water_quality
    , csdi.disability_score, csdi.health_infra, csdi.social_risk_score AS csdi_social_risk_score
    -- Preventative Care
    , prev.first_prv_dt, prev.last_prv_dt, prev.sum_pcp, prev.sum_spec, prev.sum_ob
    , prev.sum_dme, prev.sum_chol_lab, prev.sum_a1c_lab, prev.sum_chemo
    , prev.cms_alc_scrn, prev.cms_bone_scrn, prev.cms_cvd_scrn, prev.cms_col_scrn
    , prev.cms_tobacco, prev.cms_dep_scrn, prev.cms_t2d_scrn, prev.cms_hepb_scrn
    , prev.cms_hepb_vax, prev.cms_ibt_cvd, prev.cms_ibt_obese, prev.cms_flu_vax
    , prev.cms_lung_cancer_scrn, prev.cms_nutrition, prev.cms_pneum_vax, prev.cms_hpv_scrn
    , prev.cms_sti_scrn, prev.cms_mam_scrn, prev.cms_pap, prev.cms_pelvic
    , prev.cms_t2d_train, prev.cms_prost_cancer_scrn
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index` AS st
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ed_yr1` AS ed1 ON st.asdb_member_key = ed1.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ed_yr2` AS ed2 ON st.asdb_member_key = ed2.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ip_yr1` AS ip1 ON st.asdb_member_key = ip1.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ip_yr2` AS ip2 ON st.asdb_member_key = ip2.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_op_yr1` AS op1 ON st.asdb_member_key = op1.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_op_yr2` AS op2 ON st.asdb_member_key = op2.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_other_cost_utilization_yr1` AS ut1 ON st.asdb_member_key = ut1.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_other_cost_utilization_yr2` AS ut2 ON st.asdb_member_key = ut2.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_conditions` AS cond ON st.asdb_member_key = cond.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_rx_yr1` AS rx1 ON st.asdb_member_key = rx1.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_rx_yr2` AS rx2 ON st.asdb_member_key = rx2.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_demographics` AS demo ON st.asdb_member_key = demo.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_acs` AS acs ON st.asdb_member_key = acs.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_csdi` AS csdi ON st.asdb_member_key = csdi.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_preventative_summary` AS prev ON st.asdb_member_key = prev.asdb_member_key
;

-- =====================================================================================
-- SECTION 12: OUTCOME TABLE (IP_model_outcome.sql)
-- =====================================================================================
-- Creates the outcome: acute inpatient admission within 181 days after index_dt

-- Step 1: Get all IP cases in outcome window
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_outcome_ip_cases`;

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_outcome_ip_cases`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
SELECT 
    mc.asdb_member_key
    , mc.asdb_event_start_dt
    , mc.asdb_event_end_dt
    , mc.final_discharge_dt
    , mc.prindiag
    , CASE WHEN mc.asdb_coe_id IN (10200,10700,10800) THEN "Acute"
        ELSE "Non-Acute"
        END AS ip_type
    , DATE_DIFF(mc.final_discharge_dt, mc.asdb_event_start_dt, DAY) AS calc_los
    , mc.event_ct
    , mc.admit_los
    , mc.paid_los
    , mc.cost AS ip_paid_amt
FROM
    (SELECT DISTINCT asdb_member_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
INNER JOIN 
    `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ICE_IP` AS mc
        ON st.asdb_member_key=mc.asdb_member_key
WHERE 
    CAST(mc.asdb_event_start_dt AS DATE) BETWEEN DATE_ADD(index_dt, INTERVAL 1 DAY) AND DATE_ADD(index_dt, INTERVAL 181 DAY)
    AND mc.event_ct=1
;

-- Step 2: Create final outcome table with aggregated metrics
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_outcome_ip`;

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_outcome_ip`
OPTIONS (labels = [("owner", "zhaopeng_xing_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
WITH acute AS (
    SELECT
        asdb_member_key
        , CASE WHEN SUM(event_ct) > 0 THEN 1 
            ELSE 0 
            END AS acute_ip_flag
        , SUM(event_ct) AS sum_acute_ip_admits
        , SUM(calc_los) AS sum_acute_calc_los
        , SUM(admit_los) AS sum_acute_admit_los
        , SUM(paid_los) AS sum_acute_paid_los
        , SUM(ip_paid_amt) AS sum_acute_ip_cost
    FROM  
        `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_outcome_ip_cases`
    WHERE 
        ip_type = "Acute"
    GROUP BY 
        asdb_member_key
)
SELECT 
    st.asdb_member_key
    , st.index_dt
    , COALESCE(a.acute_ip_flag, 0) AS acute_ip_flag
    , COALESCE(a.sum_acute_ip_admits, 0) AS sum_acute_ip_admits
    , COALESCE(a.sum_acute_calc_los, 0) AS sum_acute_calc_los
    , COALESCE(a.sum_acute_admit_los, 0) AS sum_acute_admit_los
    , COALESCE(a.sum_acute_paid_los, 0) AS sum_acute_paid_los
    , COALESCE(a.sum_acute_ip_cost, 0) AS sum_acute_ip_cost
FROM 
    (SELECT DISTINCT asdb_member_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index`) AS st
LEFT JOIN 
    acute AS a
        ON st.asdb_member_key = a.asdb_member_key
;

-- =====================================================================================
-- END OF PIPELINE
-- =====================================================================================
-- 
-- Summary of Tables Created:
-- --------------------------
-- 0.  a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member                    - BASE: Members filtered by date (2023-01-01 to 2023-12-30) ⚠️ DATE FILTER APPLIED
-- 1.  a964286_medicaid_ip_final_dataset_4_te_experiment_2023_member_index              - Base member population with index dates
-- 2.  a964286_medicaid_ip_final_dataset_4_te_experiment_2023_med_claims_yr1            - Medical claims 12mo before index
-- 3.  a964286_medicaid_ip_final_dataset_4_te_experiment_2023_med_claims_yr2            - Medical claims 13-24mo before index  
-- 4.  a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ed_cases_yr1/yr2          - ED visit-level data
-- 5.  a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ed_yr1/yr2                - ED summary features
-- 6.  a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ip_cases_yr1/yr2          - IP admission-level data
-- 7.  a964286_medicaid_ip_final_dataset_4_te_experiment_2023_ip_yr1/yr2                - IP summary features
-- 8.  a964286_medicaid_ip_final_dataset_4_te_experiment_2023_op_yr1/yr2                - OP summary features
-- 9.  a964286_medicaid_ip_final_dataset_4_te_experiment_2023_med_claims_flag_yr1/yr2   - Claims with cost/utilization flags
-- 10. a964286_medicaid_ip_final_dataset_4_te_experiment_2023_other_cost_utilization_yr1/yr2 - Aggregated cost/utilization
-- 11. a964286_medicaid_ip_final_dataset_4_te_experiment_2023_conditions                - Chronic condition flags
-- 12. a964286_medicaid_ip_final_dataset_4_te_experiment_2023_rx_claims_yr1/yr2         - Rx claim-level data
-- 13. a964286_medicaid_ip_final_dataset_4_te_experiment_2023_rx_yr1/yr2                - Rx summary features
-- 14. a964286_medicaid_ip_final_dataset_4_te_experiment_2023_demographics              - Age, gender, tenure, etc.
-- 15. a964286_medicaid_ip_final_dataset_4_te_experiment_2023_geoid                     - Geographic identifiers
-- 16. a964286_medicaid_ip_final_dataset_4_te_experiment_2023_acs                       - ACS SDOH scores
-- 17. a964286_medicaid_ip_final_dataset_4_te_experiment_2023_csdi                      - CSDI risk indices
-- 18. a964286_medicaid_ip_final_dataset_4_te_experiment_2023_preventative              - Preventive care claim flags
-- 19. a964286_medicaid_ip_final_dataset_4_te_experiment_2023_preventative_summary      - Preventive care summary
-- 20. a964286_medicaid_ip_final_dataset_4_te_experiment_2023_non_embedding_features    - FINAL FEATURE TABLE (300+ features)
-- 21. a964286_medicaid_ip_final_dataset_4_te_experiment_2023_outcome_ip_cases          - Outcome IP cases
-- 22. a964286_medicaid_ip_final_dataset_4_te_experiment_2023_outcome_ip                - FINAL OUTCOME TABLE
--
-- Primary Outcome Variable: acute_ip_flag (1 = IP admission within 181 days)
-- =====================================================================================
