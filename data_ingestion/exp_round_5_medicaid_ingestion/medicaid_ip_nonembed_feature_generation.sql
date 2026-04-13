/*==============================================================================
  MEDICAID IP NON-EMBEDDING FEATURE GENERATION FOR TRANSFORMER EXPERIMENT
  
  Purpose: Create non-embedding features for comparing:
           (1) Embedding only vs (2) Embedding + Non-embedding features
           in predicting Medicaid Inpatient (IP) admission risk
  
  Base Cohort: edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid
               (2023 calendar year, ONE random index_dt per member)
  
  Outcome Table: edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_outcome_ip_4_te_experiment
  
  Table Naming Convention: a964286_Medicaid_nonembed_feature_4_te_experiment_*
  
  Feature Windows (matching original IP model):
  - yr1: index_dt - 12 months to index_dt - 1 day (recent history)
  - yr2: index_dt - 24 months to index_dt - 13 months (older history)
  
  Feature Categories (~300 features total):
  1. ED visits and severity (yr1/yr2)
  2. IP admissions - acute/non-acute (yr1/yr2)
  3. OP visits (yr1/yr2)
  4. Cost & Utilization by claim type (yr1/yr2)
  5. Chronic conditions (48 flags from PPM)
  6. Pharmacy/Rx claims (yr1/yr2)
  7. Demographics (age, gender, ethnicity, language, tenure)
  8. ACS social risk scores (SDI, SVI, ADI)
  9. CSDI risk indices (22 indices)
  10. Preventative care services (25+ flags)
  
  Original Reference: anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_non_embedding_features
                      (from training_pull/013_non_embedding_feature_beast.sh)
  
  Author: CSDI Team
  Date: 2024-12-18
  
  ⚠️ EXECUTION ORDER: Run steps sequentially (Step 0 → Step 17)
  ⚠️ RUNTIME: Full pipeline may take 30-60 minutes depending on data volume
  
==============================================================================*/


/*==============================================================================
  STEP 0: CREATE MEMBER INDEX WITH PLAN KEY AND POPULATION CATEGORY
  
  Purpose: The transformer cohort (a834793_Medicaid) only has individual_id and 
           index_dt. Many source tables require asdb_plan_key for joins.
           This step enriches the cohort with:
           - asdb_plan_key (from eligibility at index_dt)
           - coa_population_category (Medicaid enrollment category)
           - coa_population_group (grouped categories for analysis)
  
  Input:  a834793_Medicaid (transformer cohort)
  Output: a964286_Medicaid_nonembed_feature_4_te_experiment_member_index
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
SELECT
    -- Primary identifiers
    st.individual_id AS asdb_member_key
    , e.asdb_plan_key
    , st.index_dt
    
    -- Original population category from eligibility
    , e.coa_population_category
    
    -- Grouped population categories for stratified analysis
    -- Grouping logic from original IP model (001a_Membership_Index.sh)
    , CASE 
        WHEN TRIM(e.coa_population_category) IN ('ABD Non Dual LTSS', 'LTSS Only', 'Dual Elig LTSS', 'Dual Int LTSS')
            THEN 'LTSS'  -- Long-Term Services & Supports
        WHEN TRIM(e.coa_population_category) IN ('ABD Non Dual Non LTSS', 'ABD Non Dual LTSS')
            THEN 'ABD'   -- Aged, Blind, Disabled
        WHEN TRIM(e.coa_population_category) IN ('BH Int SMI', 'BH Only')
            THEN 'BH'    -- Behavioral Health
        WHEN TRIM(e.coa_population_category) IN ('DSNP Medicare Only', 'Dual Elig NonLTSS')
            THEN 'Dual Elig'  -- Dual Eligible (Medicare + Medicaid)
        WHEN TRIM(e.coa_population_category) IN ('Dual Int DD', 'Dual Int NonLTSS')
            THEN 'Dual Int'   -- Dual Integrated
        WHEN TRIM(e.coa_population_category) = 'CHIP'
            THEN 'CHIP'       -- Children's Health Insurance Program
        WHEN TRIM(e.coa_population_category) = 'TANF'
            THEN 'TANF'       -- Temporary Assistance for Needy Families
        WHEN TRIM(e.coa_population_category) = 'Expansion'
            THEN 'Expansion'  -- Medicaid Expansion
        WHEN TRIM(e.coa_population_category) = 'Foster'
            THEN 'Foster'     -- Foster Care
        ELSE 'Other'
      END AS coa_population_group
FROM 
    `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid` AS st
INNER JOIN
    `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ELIG_DATA_MBR_PER_MTH` AS e
        ON st.individual_id = e.asdb_member_key
        AND st.index_dt = CAST(e.asdb_elig_dt AS DATE)
;


/*==============================================================================
  STEP 1a: MEDICAL CLAIMS - YEAR 1 (12 months before index_dt)
  
  Purpose: Extract medical claims with service categories for feature derivation.
           This is the base table for ED, IP, OP, and utilization features.
  
  Key Fields:
  - asdb_coe_id: Category of Episode (determines service type)
  - asdb_coe_general_type: General service category (Inpatient, Outpatient, etc.)
  - emis_cat: EMIS category for detailed utilization tracking
  - paid_amt: Payment amount for cost calculations
  
  Date Filter: index_dt - 12 months to index_dt - 1 day
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_med_claims_yr1`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_med_claims_yr1`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
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
    `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index` AS st
INNER JOIN 
    -- Use latest partition of claims data
    (
        WITH latest_partitions AS (
            SELECT
                asdb_member_key, asdb_plan_key, claimid
                , asdb_svc_prov_key, asdb_pcp_prov_key
                , asdb_incurred_dt, asdb_paid_dt
                , location, revcode, servcode, billtype, prindiag
                , paid_amt, emis_cat
                , insert_dts AS date
                , final_claim, status_header, status_detail, asdb_coe_id
            FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_CLM_DATA_STAGE`
            WHERE CAST(insert_dts AS DATE) > DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY)
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
    -- Year 1 window: 12 months before index date
    CAST(clm.asdb_incurred_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 12 MONTH) 
                                           AND DATE_SUB(st.index_dt, INTERVAL 1 DAY)
;


/*==============================================================================
  STEP 1b: MEDICAL CLAIMS - YEAR 2 (24-13 months before index_dt)
  
  Purpose: Same as Year 1 but for older historical period
  Date Filter: index_dt - 24 months to index_dt - 13 months
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_med_claims_yr2`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_med_claims_yr2`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
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
    `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index` AS st
INNER JOIN 
    (
        WITH latest_partitions AS (
            SELECT
                asdb_member_key, asdb_plan_key, claimid
                , asdb_svc_prov_key, asdb_pcp_prov_key
                , asdb_incurred_dt, asdb_paid_dt
                , location, revcode, servcode, billtype, prindiag
                , paid_amt, emis_cat
                , insert_dts AS date
                , final_claim, status_header, status_detail, asdb_coe_id
            FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_CLM_DATA_STAGE`
            WHERE CAST(insert_dts AS DATE) > DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY)
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
    -- Year 2 window: 24-13 months before index date
    CAST(clm.asdb_incurred_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 24 MONTH) 
                                           AND DATE_SUB(st.index_dt, INTERVAL 13 MONTH)
;


/*==============================================================================
  STEP 2a: ED CASES - YEAR 1
  
  Purpose: Extract Emergency Department visits with severity and classification.
           Uses NYU ED algorithm for avoidable/unnecessary/preventable flags.
  
  Key Definitions:
  - asdb_coe_id = 20100: Emergency Department visits
  - op_severitylvl: 1-Low to 5-High severity
  - NYU classifications: Avoidable, Unnecessary, Preventable
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ed_cases_yr1`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ed_cases_yr1`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
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
    -- NYU ED Algorithm classifications
    , CASE WHEN TRIM(nyu.avoidable_ind) = "Y" THEN 1 ELSE 0 END AS avoidable_er_visits
    , CASE WHEN TRIM(nyu.er_type) = "UNNECESSARY" THEN 1 ELSE 0 END AS unnecessary_er_visits
    , CASE WHEN TRIM(nyu.er_type) = "PREVENTABLE" THEN 1 ELSE 0 END AS preventable_er_visits
FROM 
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt 
     FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`) AS st
INNER JOIN 
    `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ASDB_ICE_OP` AS mc
        ON st.asdb_member_key = mc.asdb_member_key
        AND st.asdb_plan_key = mc.asdb_plan_key
LEFT JOIN 
    -- NYU ED Algorithm lookup table (ICD10 to ER Type mapping)
    `anbc-hcb-prod.cm_medicaid_hcb_prod.ICD10_X_ER_TYPE` AS nyu
        ON TRIM(mc.prindiag) = TRIM(nyu.dx_cd)
WHERE 
    CAST(mc.asdb_incurred_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 12 MONTH) 
                                          AND DATE_SUB(st.index_dt, INTERVAL 1 DAY)
    AND CAST(mc.asdb_coe_id AS INT64) = 20100  -- ED visits only
    AND mc.event_ct = 1  -- Primary events only
;


/*==============================================================================
  STEP 2b: ED SUMMARY - YEAR 1
  
  Purpose: Aggregate ED cases into member-level summary features
  
  Output Features:
  - sum_ed_visits: Total ED visits
  - ed_flag: Binary (any ED visit)
  - sum_avoidable/unnecessary/preventable: NYU classification counts
  - Severity-specific visit counts and flags (Low to High)
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ed_yr1`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ed_yr1`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
SELECT 
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
    -- Visit counts
    , COALESCE(SUM(mc.event_ct), 0) AS sum_ed_visits
    , CASE WHEN COALESCE(SUM(mc.event_ct), 0) > 0 THEN 1 ELSE 0 END AS ed_flag
    , COALESCE(SUM(mc.cost), 0) AS sum_ed_cost
    -- NYU classification counts
    , COALESCE(SUM(mc.avoidable_er_visits), 0) AS sum_avoidable
    , COALESCE(SUM(mc.unnecessary_er_visits), 0) AS sum_unnecessary
    , COALESCE(SUM(mc.preventable_er_visits), 0) AS sum_preventable
    , MAX(mc.op_severitylvl) AS max_ed_severitylvl
    -- Severity-specific visit counts
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "1-Low" THEN mc.event_ct ELSE 0 END) AS low_sev_ed_visits
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "2-Low/Med" THEN mc.event_ct ELSE 0 END) AS low_med_sev_ed_visits
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "3-Med" THEN mc.event_ct ELSE 0 END) AS med_sev_ed_visits
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "4-Med/High" THEN mc.event_ct ELSE 0 END) AS med_high_sev_ed_visits
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "5-High" THEN mc.event_ct ELSE 0 END) AS high_sev_ed_visits
    -- Severity-specific flags
    , CASE WHEN SUM(CASE WHEN TRIM(mc.op_severitylvl) = "1-Low" THEN mc.event_ct ELSE 0 END) > 0 THEN 1 ELSE 0 END AS low_sev_ed_flag
    , CASE WHEN SUM(CASE WHEN TRIM(mc.op_severitylvl) = "2-Low/Med" THEN mc.event_ct ELSE 0 END) > 0 THEN 1 ELSE 0 END AS low_med_sev_ed_flag
    , CASE WHEN SUM(CASE WHEN TRIM(mc.op_severitylvl) = "3-Med" THEN mc.event_ct ELSE 0 END) > 0 THEN 1 ELSE 0 END AS med_sev_ed_flag
    , CASE WHEN SUM(CASE WHEN TRIM(mc.op_severitylvl) = "4-Med/High" THEN mc.event_ct ELSE 0 END) > 0 THEN 1 ELSE 0 END AS med_high_sev_ed_flag
    , CASE WHEN SUM(CASE WHEN TRIM(mc.op_severitylvl) = "5-High" THEN mc.event_ct ELSE 0 END) > 0 THEN 1 ELSE 0 END AS high_sev_ed_flag
    -- Severity-specific costs (ADDED - was missing)
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "1-Low" THEN mc.cost ELSE 0 END) AS low_sev_ed_cost
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "2-Low/Med" THEN mc.cost ELSE 0 END) AS low_med_sev_ed_cost
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "3-Med" THEN mc.cost ELSE 0 END) AS med_sev_ed_cost
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "4-Med/High" THEN mc.cost ELSE 0 END) AS med_high_sev_ed_cost
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "5-High" THEN mc.cost ELSE 0 END) AS high_sev_ed_cost
FROM 
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt 
     FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`) AS st
LEFT JOIN 
    `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ed_cases_yr1` AS mc
        ON st.asdb_member_key = mc.asdb_member_key
        AND st.asdb_plan_key = mc.asdb_plan_key
GROUP BY 
    st.asdb_member_key, st.asdb_plan_key, st.index_dt
;


/*==============================================================================
  STEP 2c: ED CASES - YEAR 2
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ed_cases_yr2`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ed_cases_yr2`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
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
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt 
     FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`) AS st
INNER JOIN 
    `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ASDB_ICE_OP` AS mc
        ON st.asdb_member_key = mc.asdb_member_key
        AND st.asdb_plan_key = mc.asdb_plan_key
LEFT JOIN 
    `anbc-hcb-prod.cm_medicaid_hcb_prod.ICD10_X_ER_TYPE` AS nyu
        ON TRIM(mc.prindiag) = TRIM(nyu.dx_cd)
WHERE 
    CAST(mc.asdb_incurred_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 24 MONTH) 
                                          AND DATE_SUB(st.index_dt, INTERVAL 13 MONTH)
    AND CAST(mc.asdb_coe_id AS INT64) = 20100
    AND mc.event_ct = 1
;


/*==============================================================================
  STEP 2d: ED SUMMARY - YEAR 2
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ed_yr2`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ed_yr2`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
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
    -- Severity-specific costs (ADDED - was missing)
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "1-Low" THEN mc.cost ELSE 0 END) AS low_sev_ed_cost
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "2-Low/Med" THEN mc.cost ELSE 0 END) AS low_med_sev_ed_cost
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "3-Med" THEN mc.cost ELSE 0 END) AS med_sev_ed_cost
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "4-Med/High" THEN mc.cost ELSE 0 END) AS med_high_sev_ed_cost
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "5-High" THEN mc.cost ELSE 0 END) AS high_sev_ed_cost
FROM 
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt 
     FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`) AS st
LEFT JOIN 
    `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ed_cases_yr2` AS mc
        ON st.asdb_member_key = mc.asdb_member_key
        AND st.asdb_plan_key = mc.asdb_plan_key
GROUP BY 
    st.asdb_member_key, st.asdb_plan_key, st.index_dt
;


/*==============================================================================
  STEP 3a: IP CASES - YEAR 1
  
  Purpose: Extract Inpatient admissions with acute/non-acute classification.
           This is the historical IP utilization (predictor), NOT the outcome.
  
  Key Definitions:
  - asdb_coe_id IN (10200, 10700, 10800): Acute IP
    - 10200 = Acute Medical/Surgical
    - 10700 = Acute Behavioral Health
    - 10800 = Acute Rehabilitation
  - asdb_coe_id IN (10000, 10100, 10300): Maternity/Infant
  - Other IP codes: Non-Acute
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ip_cases_yr1`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ip_cases_yr1`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
SELECT 
    mc.asdb_member_key
    , mc.asdb_plan_key
    , st.index_dt
    , mc.asdb_event_start_dt
    , mc.asdb_event_end_dt
    , mc.final_discharge_dt
    , mc.prindiag
    -- IP Type Classification
    , CASE WHEN mc.asdb_coe_id IN (10200, 10700, 10800) THEN "Acute"
           WHEN mc.asdb_coe_id IN (10000, 10100, 10300) THEN "Maternity/Infant"
           ELSE "Non-Acute"
      END AS ip_type
    -- Length of Stay calculations
    , DATE_DIFF(mc.final_discharge_dt, mc.asdb_event_start_dt, DAY) AS calc_los
    , mc.event_ct
    , mc.admit_los
    , mc.paid_los
    , mc.cost AS ip_paid_amt
FROM
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt 
     FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`) AS st
INNER JOIN 
    `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ICE_IP` AS mc
        ON st.asdb_member_key = mc.asdb_member_key
        AND st.asdb_plan_key = mc.asdb_plan_key
WHERE 
    CAST(mc.asdb_event_start_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 12 MONTH) 
                                              AND DATE_SUB(st.index_dt, INTERVAL 1 DAY)
    AND mc.event_ct = 1  -- Primary events only
;


/*==============================================================================
  STEP 3b: IP SUMMARY - YEAR 1
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ip_yr1`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ip_yr1`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
WITH acute AS (
    SELECT asdb_member_key, asdb_plan_key, index_dt
        , CASE WHEN SUM(event_ct) > 0 THEN 1 ELSE 0 END AS acute_ip_flag
        , SUM(event_ct) AS sum_acute_ip_admits
        , SUM(calc_los) AS sum_acute_calc_los
        , SUM(admit_los) AS sum_acute_admit_los
        , SUM(paid_los) AS sum_acute_paid_los
        , SUM(ip_paid_amt) AS sum_acute_ip_cost
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ip_cases_yr1`
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
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ip_cases_yr1`
    WHERE ip_type = "Non-Acute"
    GROUP BY asdb_member_key, asdb_plan_key
),
maternity AS (
    SELECT asdb_member_key, asdb_plan_key
        , CASE WHEN SUM(event_ct) > 0 THEN 1 ELSE 0 END AS maternity_ip_flag
        , SUM(event_ct) AS sum_maternity_ip_admits
        , SUM(calc_los) AS sum_maternity_calc_los
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ip_cases_yr1`
    WHERE ip_type = "Maternity/Infant"
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
    , COALESCE(c.sum_maternity_calc_los, 0) AS sum_maternity_calc_los
FROM 
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt 
     FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`) AS st
LEFT JOIN acute AS a ON st.asdb_member_key = a.asdb_member_key AND st.asdb_plan_key = a.asdb_plan_key
LEFT JOIN nonacute AS b ON st.asdb_member_key = b.asdb_member_key AND st.asdb_plan_key = b.asdb_plan_key
LEFT JOIN maternity AS c ON st.asdb_member_key = c.asdb_member_key AND st.asdb_plan_key = c.asdb_plan_key
;


/*==============================================================================
  STEP 3c: IP CASES - YEAR 2
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ip_cases_yr2`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ip_cases_yr2`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
SELECT 
    mc.asdb_member_key, mc.asdb_plan_key, st.index_dt
    , mc.asdb_event_start_dt, mc.asdb_event_end_dt, mc.final_discharge_dt, mc.prindiag
    , CASE WHEN mc.asdb_coe_id IN (10200, 10700, 10800) THEN "Acute"
           WHEN mc.asdb_coe_id IN (10000, 10100, 10300) THEN "Maternity/Infant"
           ELSE "Non-Acute" END AS ip_type
    , DATE_DIFF(mc.final_discharge_dt, mc.asdb_event_start_dt, DAY) AS calc_los
    , mc.event_ct, mc.admit_los, mc.paid_los, mc.cost AS ip_paid_amt
FROM
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt 
     FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`) AS st
INNER JOIN `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ICE_IP` AS mc
    ON st.asdb_member_key = mc.asdb_member_key AND st.asdb_plan_key = mc.asdb_plan_key
WHERE CAST(mc.asdb_event_start_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 24 MONTH) 
                                                AND DATE_SUB(st.index_dt, INTERVAL 13 MONTH)
    AND mc.event_ct = 1
;


/*==============================================================================
  STEP 3d: IP SUMMARY - YEAR 2
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ip_yr2`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ip_yr2`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
WITH acute AS (
    SELECT asdb_member_key, asdb_plan_key, index_dt
        , CASE WHEN SUM(event_ct) > 0 THEN 1 ELSE 0 END AS acute_ip_flag
        , SUM(event_ct) AS sum_acute_ip_admits, SUM(calc_los) AS sum_acute_calc_los
        , SUM(admit_los) AS sum_acute_admit_los, SUM(paid_los) AS sum_acute_paid_los
        , SUM(ip_paid_amt) AS sum_acute_ip_cost
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ip_cases_yr2`
    WHERE ip_type = "Acute" GROUP BY asdb_member_key, asdb_plan_key, index_dt
),
nonacute AS (
    SELECT asdb_member_key, asdb_plan_key
        , CASE WHEN SUM(event_ct) > 0 THEN 1 ELSE 0 END AS non_acute_ip_flag
        , SUM(event_ct) AS sum_non_acute_ip_admits, SUM(calc_los) AS sum_non_acute_calc_los
        , SUM(admit_los) AS sum_non_acute_admit_los, SUM(paid_los) AS sum_non_acute_paid_los
        , SUM(ip_paid_amt) AS sum_non_acute_ip_cost
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ip_cases_yr2`
    WHERE ip_type = "Non-Acute" GROUP BY asdb_member_key, asdb_plan_key
),
maternity AS (
    SELECT asdb_member_key, asdb_plan_key
        , CASE WHEN SUM(event_ct) > 0 THEN 1 ELSE 0 END AS maternity_ip_flag
        , SUM(event_ct) AS sum_maternity_ip_admits, SUM(calc_los) AS sum_maternity_calc_los
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ip_cases_yr2`
    WHERE ip_type = "Maternity/Infant" GROUP BY asdb_member_key, asdb_plan_key
)
SELECT st.asdb_member_key, st.asdb_plan_key, st.index_dt
    , COALESCE(a.acute_ip_flag, 0) AS acute_ip_flag, COALESCE(a.sum_acute_ip_admits, 0) AS sum_acute_ip_admits
    , COALESCE(a.sum_acute_calc_los, 0) AS sum_acute_calc_los, COALESCE(a.sum_acute_admit_los, 0) AS sum_acute_admit_los
    , COALESCE(a.sum_acute_paid_los, 0) AS sum_acute_paid_los, COALESCE(a.sum_acute_ip_cost, 0) AS sum_acute_ip_cost
    , COALESCE(b.non_acute_ip_flag, 0) AS non_acute_ip_flag, COALESCE(b.sum_non_acute_ip_admits, 0) AS sum_non_acute_ip_admits
    , COALESCE(b.sum_non_acute_calc_los, 0) AS sum_non_acute_calc_los, COALESCE(b.sum_non_acute_admit_los, 0) AS sum_non_acute_admit_los
    , COALESCE(b.sum_non_acute_paid_los, 0) AS sum_non_acute_paid_los, COALESCE(b.sum_non_acute_ip_cost, 0) AS sum_non_acute_ip_cost
    , COALESCE(c.maternity_ip_flag, 0) AS maternity_ip_flag, COALESCE(c.sum_maternity_ip_admits, 0) AS sum_maternity_ip_admits
    , COALESCE(c.sum_maternity_calc_los, 0) AS sum_maternity_calc_los
FROM (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt 
      FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`) AS st
LEFT JOIN acute AS a ON st.asdb_member_key = a.asdb_member_key AND st.asdb_plan_key = a.asdb_plan_key
LEFT JOIN nonacute AS b ON st.asdb_member_key = b.asdb_member_key AND st.asdb_plan_key = b.asdb_plan_key
LEFT JOIN maternity AS c ON st.asdb_member_key = c.asdb_member_key AND st.asdb_plan_key = c.asdb_plan_key
;


/*==============================================================================
  STEP 4a: OP (OUTPATIENT) SUMMARY - YEAR 1
  
  Purpose: Count outpatient visits excluding ED and IP
  Definition: Outpatient = claims NOT (Inpatient OR Institutional OR Emergency)
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_op_yr1`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_op_yr1`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
WITH clm AS (
    SELECT *, CASE WHEN ROW_NUMBER() OVER(PARTITION BY asdb_member_key, asdb_plan_key, asdb_svc_prov_key, asdb_incurred_dt) = 1 
                   THEN 1 ELSE 0 END AS op_ct
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_med_claims_yr1`
    WHERE TRIM(asdb_coe_general_type) != "Inpatient" AND TRIM(emis_cat) != "Institutional Services" AND TRIM(emis_cat) != "Emergency"
)
SELECT st.asdb_member_key, st.asdb_plan_key, st.index_dt
    , COALESCE(SUM(clm.paid_amt), 0) AS sum_op_cost
    , COALESCE(SUM(clm.op_ct), 0) AS sum_op_visits
    , MAX(CASE WHEN clm.op_ct = 1 THEN 1 ELSE 0 END) AS op_flag
FROM (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`) AS st
LEFT JOIN clm ON st.asdb_member_key = clm.asdb_member_key AND st.asdb_plan_key = clm.asdb_plan_key
GROUP BY st.asdb_member_key, st.asdb_plan_key, st.index_dt
;


/*==============================================================================
  STEP 4b: OP SUMMARY - YEAR 2
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_op_yr2`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_op_yr2`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
WITH clm AS (
    SELECT *, CASE WHEN ROW_NUMBER() OVER(PARTITION BY asdb_member_key, asdb_plan_key, asdb_svc_prov_key, asdb_incurred_dt) = 1 
                   THEN 1 ELSE 0 END AS op_ct
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_med_claims_yr2`
    WHERE TRIM(asdb_coe_general_type) != "Inpatient" AND TRIM(emis_cat) != "Institutional Services" AND TRIM(emis_cat) != "Emergency"
)
SELECT st.asdb_member_key, st.asdb_plan_key, st.index_dt
    , COALESCE(SUM(clm.paid_amt), 0) AS sum_op_cost
    , COALESCE(SUM(clm.op_ct), 0) AS sum_op_visits
    , MAX(CASE WHEN clm.op_ct = 1 THEN 1 ELSE 0 END) AS op_flag
FROM (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`) AS st
LEFT JOIN clm ON st.asdb_member_key = clm.asdb_member_key AND st.asdb_plan_key = clm.asdb_plan_key
GROUP BY st.asdb_member_key, st.asdb_plan_key, st.index_dt
;


/*==============================================================================
  STEP 5a: COST & UTILIZATION FLAGS - YEAR 1
  
  Purpose: Create claim-level flags for EMIS and COE categories.
           These will be aggregated into utilization counts in Step 5b.
  
  EMIS Categories: Community-Based, Emergency, Home Health, Lab, etc.
  COE Categories: Inpatient Hospital, Laboratory, LTC, Physician, etc.
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_med_claims_flag_yr1`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_med_claims_flag_yr1`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
WITH clm AS (
    SELECT *
        , CASE WHEN TRIM(asdb_coe_general_type)="Inpatient" OR TRIM(emis_cat)="Institutional Services" THEN "Inpatient"
               WHEN TRIM(emis_cat)="Emergency" THEN "Emergency" ELSE "Outpatient" END AS plc_svc_ctg
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_med_claims_yr1`
)
SELECT clm.*, fac.prov_specialty
    -- COST METRICS (ADDED - was missing)
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
    -- EMIS utilization flags
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
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" THEN 1 ELSE 0 END AS ltc_clm
    -- COE category flags
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
    , CASE WHEN (TRIM(clm.revcode) IN ("0760","0761","0762","0769") 
            AND (TRIM(clm.billtype) LIKE "13%" OR TRIM(clm.billtype) LIKE "85%")
            AND TRIM(clm.servcode) IN ("99217","99218","99219","99202","G0378","G0379","")) THEN 1 ELSE 0 END AS obs_clm
FROM clm
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_SVC_PROV` AS fac
    ON clm.asdb_svc_prov_key = fac.asdb_svc_prov_key AND clm.asdb_plan_key = fac.asdb_plan_key
;


/*==============================================================================
  STEP 5b: COST & UTILIZATION SUMMARY - YEAR 1
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_other_cost_utilization_yr1`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_other_cost_utilization_yr1`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
SELECT st.asdb_member_key, st.asdb_plan_key, st.index_dt
    , COUNT(DISTINCT clm.claimid) AS claim_cnt
    , COUNT(*) AS claim_line_cnt
    , COALESCE(SUM(clm.paid_amt), 0) AS sum_paid_amt
    -- COST METRICS (ADDED - was missing)
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
    -- Utilization counts
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
FROM (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`) AS st
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_med_claims_flag_yr1` AS clm
    ON st.asdb_member_key = clm.asdb_member_key AND st.asdb_plan_key = clm.asdb_plan_key
GROUP BY st.asdb_member_key, st.asdb_plan_key, st.index_dt
;


/*==============================================================================
  STEP 5c: COST & UTILIZATION FLAGS - YEAR 2
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_med_claims_flag_yr2`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_med_claims_flag_yr2`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
WITH clm AS (
    SELECT *
        , CASE WHEN TRIM(asdb_coe_general_type)="Inpatient" OR TRIM(emis_cat)="Institutional Services" THEN "Inpatient"
               WHEN TRIM(emis_cat)="Emergency" THEN "Emergency" ELSE "Outpatient" END AS plc_svc_ctg
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_med_claims_yr2`
)
SELECT clm.*, fac.prov_specialty
    -- COST METRICS (ADDED - was missing)
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
    -- EMIS utilization flags
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
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" THEN 1 ELSE 0 END AS ltc_clm
    -- COE category flags
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
    , CASE WHEN (TRIM(clm.revcode) IN ("0760","0761","0762","0769") 
            AND (TRIM(clm.billtype) LIKE "13%" OR TRIM(clm.billtype) LIKE "85%")
            AND TRIM(clm.servcode) IN ("99217","99218","99219","99202","G0378","G0379","")) THEN 1 ELSE 0 END AS obs_clm
FROM clm
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_SVC_PROV` AS fac
    ON clm.asdb_svc_prov_key = fac.asdb_svc_prov_key AND clm.asdb_plan_key = fac.asdb_plan_key
;


/*==============================================================================
  STEP 5d: COST & UTILIZATION SUMMARY - YEAR 2
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_other_cost_utilization_yr2`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_other_cost_utilization_yr2`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
SELECT st.asdb_member_key, st.asdb_plan_key, st.index_dt
    , COUNT(DISTINCT clm.claimid) AS claim_cnt
    , COUNT(*) AS claim_line_cnt
    , COALESCE(SUM(clm.paid_amt), 0) AS sum_paid_amt
    -- COST METRICS (ADDED - was missing)
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
    -- Utilization counts
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
FROM (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`) AS st
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_med_claims_flag_yr2` AS clm
    ON st.asdb_member_key = clm.asdb_member_key AND st.asdb_plan_key = clm.asdb_plan_key
GROUP BY st.asdb_member_key, st.asdb_plan_key, st.index_dt
;


/*==============================================================================
  STEP 6: CONDITIONS (48 Chronic Condition Flags from PPM)
  
  Purpose: Extract chronic condition flags from PPM (Predictive Modeling) history.
           These flags indicate presence of major chronic conditions in the
           12 months preceding the index date.
  
  Source: ASDB_PPM_MEMBER_CONDITION_HISTORY
  
  Condition Categories:
  - Cardiovascular: CHF, CAD, HYP, PVD
  - Metabolic: DIA, OBE, HYC
  - Respiratory: COP, AST
  - Mental Health: DEP, ANX, substance, ALC, bipolar, psychoses
  - Cancer: Cancer, meta_cancer
  - Kidney: CRF, esrd
  - And 30+ more...
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_conditions`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_conditions`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
SELECT *
    -- Count of major chronic conditions
    , (abdominal_pain+AID+ANX+OST+AST+AUT+CHO+burns+cad+Cancer+narc+CBD+CHF+CRF+VNA+CHD+
       COP+CYS+DEP+DIA+EDO+esrd+EPL+CRO+MOH+HEM+HepC+immune+intel_dsblty+meta_cancer+
       liver_dis+MSS+OBE+oud+liver_other+paralysis+PAR+hmd+PVD+autoimmune+DEM+SCA+
       sleep_apnea+spinal_inj+back+substance+ALC+bipolar+psychoses) AS major_chronic_cnt
FROM (
    SELECT
        st.asdb_member_key, st.asdb_plan_key, st.index_dt
        , MIN(b.rpt_end_dt) AS first_rpt
        , MAX(b.rpt_end_dt) AS last_rpt
        -- Condition flags (cond_rank mapped to condition names)
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
        -- Behavioral health conditions
        , MAX(CASE WHEN b.cond_rank=22 THEN 1 ELSE 0 END) AS substance
        , MAX(CASE WHEN b.cond_rank=14 THEN 1 ELSE 0 END) AS ALC
        , MAX(CASE WHEN b.cond_rank=36 THEN 1 ELSE 0 END) AS bipolar 
        , MAX(CASE WHEN b.cond_rank=25 THEN 1 ELSE 0 END) AS psychoses
    FROM 
        (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt 
         FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`) AS st
    LEFT JOIN 
        `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_PPM_MEMBER_CONDITION_HISTORY` AS b 
            ON st.asdb_member_key = b.ppm_member_key
            AND st.asdb_plan_key = b.ppm_plan_key
            -- Condition must be within 12 months of index date
            AND DATE_TRUNC(st.index_dt, MONTH) BETWEEN 
                DATE_ADD(LAST_DAY(CAST(b.rpt_end_dt AS DATE), MONTH), INTERVAL 1 DAY) 
                AND DATE_ADD(LAST_DAY(CAST(b.rpt_end_dt AS DATE), MONTH), INTERVAL 12 MONTH)
    GROUP BY st.asdb_member_key, st.asdb_plan_key, st.index_dt
) tb
;


/*==============================================================================
  STEP 7a: RX CLAIMS - YEAR 1
  
  Purpose: Extract pharmacy claims with drug classifications.
  
  Key Features:
  - Claim counts, days supply
  - Drug counts (NDC, GPI, GPI4, GPI2)
  - Fill types (retail, mail order, generic, brand)
  - Therapeutic class scripts (antidiabetic, beta blockers, etc.)
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_rx_claims_yr1`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_rx_claims_yr1`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
SELECT DISTINCT
    st.asdb_member_key, st.asdb_plan_key, st.index_dt
    , rx.asdb_pharmacy_key, rx.prescriptionnum
    , CAST(rx.asdb_incurred_dt AS DATE) AS disp_dt
    , rx.days_supply, rx.script_ct
    , ROUND(CASE WHEN rx.days_supply >= 0.1 AND rx.days_supply < 30 THEN 30 ELSE rx.days_supply END/30) AS scripts
    , rx.ndcnum AS ndc_cd, rx.gpi AS adjudicated_gpi_cd
    , SUBSTR(rx.gpi,1,4) AS gpi4, SUBSTR(rx.gpi,1,2) AS gpi2
    , rx.billed_amt, rx.claim_adj_amt, rx.copay_amt
    , CASE WHEN rx.pharmacytype="R" THEN 1 ELSE 0 END AS retail_flag
    , CASE WHEN rx.pharmacytype="M" THEN 1 ELSE 0 END AS mail_order_flag
    , CASE WHEN rx.drugtype = 3 THEN 1 ELSE 0 END AS generic_fill_flag
    , CASE WHEN rx.drugtype = 2 THEN 1 ELSE 0 END AS branded_generic_fill_flag
    , CASE WHEN rx.drugtype = 4 THEN 1 ELSE 0 END AS otc_fill_flag
    , CASE WHEN rx.drugtype = 1 THEN 1 ELSE 0 END AS ss_brand_fill_flag
    , CASE WHEN rx.drugtype = 5 THEN 1 ELSE 0 END AS ms_brand_fill_flag
    , CASE WHEN rx.formularyflag="F" OR rx.drugtype = 3 THEN 1 ELSE 0 END AS formulary_fill_flag
    , CASE WHEN c.maint_drug_cd="X" THEN 1 ELSE 0 END AS maint_drug_flag
FROM (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`) AS st
INNER JOIN `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_RX_DATA_STAGE` AS rx
    ON st.asdb_member_key = rx.asdb_member_key AND st.asdb_plan_key = rx.asdb_plan_key
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_anbor_enrsrcv.EDW_DRUG` AS c ON TRIM(rx.ndcnum) = TRIM(c.ndc_cd)
WHERE rx.ClaimType = "P"
    AND CAST(rx.asdb_incurred_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 12 MONTH) AND DATE_SUB(st.index_dt, INTERVAL 1 DAY)
;


/*==============================================================================
  STEP 7b: RX SUMMARY - YEAR 1
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_rx_yr1`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_rx_yr1`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
WITH rx_tmp AS (
    SELECT asdb_member_key, asdb_plan_key, index_dt
        , COUNT(*) AS rx_claim_cnt
        , SUM(days_supply) AS days_supply_sum
        , COUNT(DISTINCT ndc_cd) AS ndc_cnt
        , COUNT(DISTINCT adjudicated_gpi_cd) AS gpi_cnt
        , COUNT(DISTINCT gpi4) AS gpi4_cnt
        , COUNT(DISTINCT gpi2) AS gpi2_cnt
        , SUM(retail_flag) AS retail_fills, SUM(mail_order_flag) AS mail_order_fills
        , SUM(generic_fill_flag) AS generic_fills, SUM(branded_generic_fill_flag) AS branded_generic_fills
        , SUM(otc_fill_flag) AS otc_fills, SUM(ss_brand_fill_flag) AS ss_brand_fills
        , SUM(ms_brand_fill_flag) AS ms_brand_fills, SUM(formulary_fill_flag) AS formulary_fills
        , SUM(maint_drug_flag) AS maint_drug_fills
        , SUM(CASE WHEN gpi2="27" THEN scripts ELSE 0 END) AS antidiabetic_scripts
        , SUM(CASE WHEN gpi2="27" THEN days_supply ELSE 0 END) AS antidiabetic_days_supply
        , SUM(CASE WHEN gpi2="33" THEN scripts ELSE 0 END) AS beta_blocker_scripts
        , SUM(CASE WHEN gpi2="33" THEN days_supply ELSE 0 END) AS beta_blocker_days_supply
        , SUM(CASE WHEN gpi2="36" THEN scripts ELSE 0 END) AS antihypertensive_scripts
        , SUM(CASE WHEN gpi2="36" THEN days_supply ELSE 0 END) AS antihypertensive_days_supply
        , SUM(CASE WHEN gpi2="39" THEN scripts ELSE 0 END) AS lipid_lowering_scripts
        , SUM(CASE WHEN gpi2="39" THEN days_supply ELSE 0 END) AS lipid_lowering_days_supply
        , SUM(CASE WHEN gpi2="34" THEN scripts ELSE 0 END) AS calcium_channel_blk_scripts
        , SUM(CASE WHEN gpi2="34" THEN days_supply ELSE 0 END) AS calcium_channel_blk_days_supply
        , SUM(CASE WHEN gpi2="37" THEN scripts ELSE 0 END) AS diuretic_scripts
        , SUM(CASE WHEN gpi2="37" THEN days_supply ELSE 0 END) AS diuretic_days_supply
        , SUM(CASE WHEN gpi2="32" THEN scripts ELSE 0 END) AS antianginal_agent_scripts
        , SUM(CASE WHEN gpi2="32" THEN days_supply ELSE 0 END) AS antianginal_agent_days_supply
        , SUM(CASE WHEN gpi2="58" THEN scripts ELSE 0 END) AS antidepressant_scripts
        , SUM(CASE WHEN gpi2="58" THEN days_supply ELSE 0 END) AS antidepressant_days_supply
        , SUM(CASE WHEN gpi2="59" THEN scripts ELSE 0 END) AS antipsychotic_scripts
        , SUM(CASE WHEN gpi2="59" THEN days_supply ELSE 0 END) AS antipsychotic_days_supply
        , SUM(CASE WHEN gpi2="57" THEN scripts ELSE 0 END) AS antianxiety_scripts
        , SUM(CASE WHEN gpi2="57" THEN days_supply ELSE 0 END) AS antianxiety_days_supply
        , SUM(CASE WHEN gpi2="72" THEN scripts ELSE 0 END) AS anticonvulsant_scripts
        , SUM(CASE WHEN gpi2="72" THEN days_supply ELSE 0 END) AS anticonvulsant_days_supply
        , SUM(CASE WHEN gpi4="4440" THEN scripts ELSE 0 END) AS inhaled_steroid_scripts
        , SUM(CASE WHEN gpi4="4440" THEN days_supply ELSE 0 END) AS inhaled_steroid_days_supply
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_rx_claims_yr1`
    GROUP BY asdb_member_key, asdb_plan_key, index_dt
)
SELECT st.asdb_member_key, st.asdb_plan_key, st.index_dt
    , COALESCE(rx.rx_claim_cnt, 0) AS rx_claim_cnt, COALESCE(rx.days_supply_sum, 0) AS days_supply_sum
    , COALESCE(rx.ndc_cnt, 0) AS ndc_cnt, COALESCE(rx.gpi_cnt, 0) AS gpi_cnt
    , COALESCE(rx.gpi4_cnt, 0) AS gpi4_cnt, COALESCE(rx.gpi2_cnt, 0) AS gpi2_cnt
    , COALESCE(rx.retail_fills, 0) AS retail_fills, COALESCE(rx.mail_order_fills, 0) AS mail_order_fills
    , COALESCE(rx.generic_fills, 0) AS generic_fills, COALESCE(rx.branded_generic_fills, 0) AS branded_generic_fills
    , COALESCE(rx.otc_fills, 0) AS otc_fills, COALESCE(rx.ss_brand_fills, 0) AS ss_brand_fills
    , COALESCE(rx.ms_brand_fills, 0) AS ms_brand_fills, COALESCE(rx.formulary_fills, 0) AS formulary_fills
    , COALESCE(rx.maint_drug_fills, 0) AS maint_drug_fills
    , COALESCE(rx.antidiabetic_scripts, 0) AS antidiabetic_scripts, COALESCE(rx.antidiabetic_days_supply, 0) AS antidiabetic_days_supply
    , COALESCE(rx.beta_blocker_scripts, 0) AS beta_blocker_scripts, COALESCE(rx.beta_blocker_days_supply, 0) AS beta_blocker_days_supply
    , COALESCE(rx.antihypertensive_scripts, 0) AS antihypertensive_scripts, COALESCE(rx.antihypertensive_days_supply, 0) AS antihypertensive_days_supply
    , COALESCE(rx.lipid_lowering_scripts, 0) AS lipid_lowering_scripts, COALESCE(rx.lipid_lowering_days_supply, 0) AS lipid_lowering_days_supply
    , COALESCE(rx.calcium_channel_blk_scripts, 0) AS calcium_channel_blk_scripts, COALESCE(rx.calcium_channel_blk_days_supply, 0) AS calcium_channel_blk_days_supply
    , COALESCE(rx.diuretic_scripts, 0) AS diuretic_scripts, COALESCE(rx.diuretic_days_supply, 0) AS diuretic_days_supply
    , COALESCE(rx.antianginal_agent_scripts, 0) AS antianginal_agent_scripts, COALESCE(rx.antianginal_agent_days_supply, 0) AS antianginal_agent_days_supply
    , COALESCE(rx.antidepressant_scripts, 0) AS antidepressant_scripts, COALESCE(rx.antidepressant_days_supply, 0) AS antidepressant_days_supply
    , COALESCE(rx.antipsychotic_scripts, 0) AS antipsychotic_scripts, COALESCE(rx.antipsychotic_days_supply, 0) AS antipsychotic_days_supply
    , COALESCE(rx.antianxiety_scripts, 0) AS antianxiety_scripts, COALESCE(rx.antianxiety_days_supply, 0) AS antianxiety_days_supply
    , COALESCE(rx.anticonvulsant_scripts, 0) AS anticonvulsant_scripts, COALESCE(rx.anticonvulsant_days_supply, 0) AS anticonvulsant_days_supply
    , COALESCE(rx.inhaled_steroid_scripts, 0) AS inhaled_steroid_scripts, COALESCE(rx.inhaled_steroid_days_supply, 0) AS inhaled_steroid_days_supply
FROM (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`) AS st
LEFT JOIN rx_tmp AS rx ON st.asdb_member_key = rx.asdb_member_key AND st.asdb_plan_key = rx.asdb_plan_key
;


/*==============================================================================
  STEP 7c: RX CLAIMS - YEAR 2
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_rx_claims_yr2`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_rx_claims_yr2`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
SELECT DISTINCT
    st.asdb_member_key, st.asdb_plan_key, st.index_dt
    , rx.asdb_pharmacy_key, rx.prescriptionnum
    , CAST(rx.asdb_incurred_dt AS DATE) AS disp_dt
    , rx.days_supply, rx.script_ct
    , ROUND(CASE WHEN rx.days_supply >= 0.1 AND rx.days_supply < 30 THEN 30 ELSE rx.days_supply END/30) AS scripts
    , rx.ndcnum AS ndc_cd, rx.gpi AS adjudicated_gpi_cd
    , SUBSTR(rx.gpi,1,4) AS gpi4, SUBSTR(rx.gpi,1,2) AS gpi2
    , rx.billed_amt, rx.claim_adj_amt, rx.copay_amt
    , CASE WHEN rx.pharmacytype="R" THEN 1 ELSE 0 END AS retail_flag
    , CASE WHEN rx.pharmacytype="M" THEN 1 ELSE 0 END AS mail_order_flag
    , CASE WHEN rx.drugtype = 3 THEN 1 ELSE 0 END AS generic_fill_flag
    , CASE WHEN rx.drugtype = 2 THEN 1 ELSE 0 END AS branded_generic_fill_flag
    , CASE WHEN rx.drugtype = 4 THEN 1 ELSE 0 END AS otc_fill_flag
    , CASE WHEN rx.drugtype = 1 THEN 1 ELSE 0 END AS ss_brand_fill_flag
    , CASE WHEN rx.drugtype = 5 THEN 1 ELSE 0 END AS ms_brand_fill_flag
    , CASE WHEN rx.formularyflag="F" OR rx.drugtype = 3 THEN 1 ELSE 0 END AS formulary_fill_flag
    , CASE WHEN c.maint_drug_cd="X" THEN 1 ELSE 0 END AS maint_drug_flag
FROM (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`) AS st
INNER JOIN `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_RX_DATA_STAGE` AS rx
    ON st.asdb_member_key = rx.asdb_member_key AND st.asdb_plan_key = rx.asdb_plan_key
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_anbor_enrsrcv.EDW_DRUG` AS c ON TRIM(rx.ndcnum) = TRIM(c.ndc_cd)
WHERE rx.ClaimType = "P"
    AND CAST(rx.asdb_incurred_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 24 MONTH) AND DATE_SUB(st.index_dt, INTERVAL 13 MONTH)
;


/*==============================================================================
  STEP 7d: RX SUMMARY - YEAR 2
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_rx_yr2`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_rx_yr2`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
WITH rx_tmp AS (
    SELECT asdb_member_key, asdb_plan_key, index_dt
        , COUNT(*) AS rx_claim_cnt, SUM(days_supply) AS days_supply_sum
        , COUNT(DISTINCT ndc_cd) AS ndc_cnt, COUNT(DISTINCT adjudicated_gpi_cd) AS gpi_cnt
        , COUNT(DISTINCT gpi4) AS gpi4_cnt, COUNT(DISTINCT gpi2) AS gpi2_cnt
        , SUM(retail_flag) AS retail_fills, SUM(mail_order_flag) AS mail_order_fills
        , SUM(generic_fill_flag) AS generic_fills, SUM(branded_generic_fill_flag) AS branded_generic_fills
        , SUM(otc_fill_flag) AS otc_fills, SUM(ss_brand_fill_flag) AS ss_brand_fills
        , SUM(ms_brand_fill_flag) AS ms_brand_fills, SUM(formulary_fill_flag) AS formulary_fills
        , SUM(maint_drug_flag) AS maint_drug_fills
        , SUM(CASE WHEN gpi2="27" THEN scripts ELSE 0 END) AS antidiabetic_scripts
        , SUM(CASE WHEN gpi2="27" THEN days_supply ELSE 0 END) AS antidiabetic_days_supply
        , SUM(CASE WHEN gpi2="33" THEN scripts ELSE 0 END) AS beta_blocker_scripts
        , SUM(CASE WHEN gpi2="33" THEN days_supply ELSE 0 END) AS beta_blocker_days_supply
        , SUM(CASE WHEN gpi2="36" THEN scripts ELSE 0 END) AS antihypertensive_scripts
        , SUM(CASE WHEN gpi2="36" THEN days_supply ELSE 0 END) AS antihypertensive_days_supply
        , SUM(CASE WHEN gpi2="39" THEN scripts ELSE 0 END) AS lipid_lowering_scripts
        , SUM(CASE WHEN gpi2="39" THEN days_supply ELSE 0 END) AS lipid_lowering_days_supply
        , SUM(CASE WHEN gpi2="34" THEN scripts ELSE 0 END) AS calcium_channel_blk_scripts
        , SUM(CASE WHEN gpi2="34" THEN days_supply ELSE 0 END) AS calcium_channel_blk_days_supply
        , SUM(CASE WHEN gpi2="37" THEN scripts ELSE 0 END) AS diuretic_scripts
        , SUM(CASE WHEN gpi2="37" THEN days_supply ELSE 0 END) AS diuretic_days_supply
        , SUM(CASE WHEN gpi2="32" THEN scripts ELSE 0 END) AS antianginal_agent_scripts
        , SUM(CASE WHEN gpi2="32" THEN days_supply ELSE 0 END) AS antianginal_agent_days_supply
        , SUM(CASE WHEN gpi2="58" THEN scripts ELSE 0 END) AS antidepressant_scripts
        , SUM(CASE WHEN gpi2="58" THEN days_supply ELSE 0 END) AS antidepressant_days_supply
        , SUM(CASE WHEN gpi2="59" THEN scripts ELSE 0 END) AS antipsychotic_scripts
        , SUM(CASE WHEN gpi2="59" THEN days_supply ELSE 0 END) AS antipsychotic_days_supply
        , SUM(CASE WHEN gpi2="57" THEN scripts ELSE 0 END) AS antianxiety_scripts
        , SUM(CASE WHEN gpi2="57" THEN days_supply ELSE 0 END) AS antianxiety_days_supply
        , SUM(CASE WHEN gpi2="72" THEN scripts ELSE 0 END) AS anticonvulsant_scripts
        , SUM(CASE WHEN gpi2="72" THEN days_supply ELSE 0 END) AS anticonvulsant_days_supply
        , SUM(CASE WHEN gpi4="4440" THEN scripts ELSE 0 END) AS inhaled_steroid_scripts
        , SUM(CASE WHEN gpi4="4440" THEN days_supply ELSE 0 END) AS inhaled_steroid_days_supply
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_rx_claims_yr2`
    GROUP BY asdb_member_key, asdb_plan_key, index_dt
)
SELECT st.asdb_member_key, st.asdb_plan_key, st.index_dt
    , COALESCE(rx.rx_claim_cnt, 0) AS rx_claim_cnt, COALESCE(rx.days_supply_sum, 0) AS days_supply_sum
    , COALESCE(rx.ndc_cnt, 0) AS ndc_cnt, COALESCE(rx.gpi_cnt, 0) AS gpi_cnt
    , COALESCE(rx.gpi4_cnt, 0) AS gpi4_cnt, COALESCE(rx.gpi2_cnt, 0) AS gpi2_cnt
    , COALESCE(rx.retail_fills, 0) AS retail_fills, COALESCE(rx.mail_order_fills, 0) AS mail_order_fills
    , COALESCE(rx.generic_fills, 0) AS generic_fills, COALESCE(rx.branded_generic_fills, 0) AS branded_generic_fills
    , COALESCE(rx.otc_fills, 0) AS otc_fills, COALESCE(rx.ss_brand_fills, 0) AS ss_brand_fills
    , COALESCE(rx.ms_brand_fills, 0) AS ms_brand_fills, COALESCE(rx.formulary_fills, 0) AS formulary_fills
    , COALESCE(rx.maint_drug_fills, 0) AS maint_drug_fills
    , COALESCE(rx.antidiabetic_scripts, 0) AS antidiabetic_scripts, COALESCE(rx.antidiabetic_days_supply, 0) AS antidiabetic_days_supply
    , COALESCE(rx.beta_blocker_scripts, 0) AS beta_blocker_scripts, COALESCE(rx.beta_blocker_days_supply, 0) AS beta_blocker_days_supply
    , COALESCE(rx.antihypertensive_scripts, 0) AS antihypertensive_scripts, COALESCE(rx.antihypertensive_days_supply, 0) AS antihypertensive_days_supply
    , COALESCE(rx.lipid_lowering_scripts, 0) AS lipid_lowering_scripts, COALESCE(rx.lipid_lowering_days_supply, 0) AS lipid_lowering_days_supply
    , COALESCE(rx.calcium_channel_blk_scripts, 0) AS calcium_channel_blk_scripts, COALESCE(rx.calcium_channel_blk_days_supply, 0) AS calcium_channel_blk_days_supply
    , COALESCE(rx.diuretic_scripts, 0) AS diuretic_scripts, COALESCE(rx.diuretic_days_supply, 0) AS diuretic_days_supply
    , COALESCE(rx.antianginal_agent_scripts, 0) AS antianginal_agent_scripts, COALESCE(rx.antianginal_agent_days_supply, 0) AS antianginal_agent_days_supply
    , COALESCE(rx.antidepressant_scripts, 0) AS antidepressant_scripts, COALESCE(rx.antidepressant_days_supply, 0) AS antidepressant_days_supply
    , COALESCE(rx.antipsychotic_scripts, 0) AS antipsychotic_scripts, COALESCE(rx.antipsychotic_days_supply, 0) AS antipsychotic_days_supply
    , COALESCE(rx.antianxiety_scripts, 0) AS antianxiety_scripts, COALESCE(rx.antianxiety_days_supply, 0) AS antianxiety_days_supply
    , COALESCE(rx.anticonvulsant_scripts, 0) AS anticonvulsant_scripts, COALESCE(rx.anticonvulsant_days_supply, 0) AS anticonvulsant_days_supply
    , COALESCE(rx.inhaled_steroid_scripts, 0) AS inhaled_steroid_scripts, COALESCE(rx.inhaled_steroid_days_supply, 0) AS inhaled_steroid_days_supply
FROM (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`) AS st
LEFT JOIN rx_tmp AS rx ON st.asdb_member_key = rx.asdb_member_key AND st.asdb_plan_key = rx.asdb_plan_key
;


/*==============================================================================
  STEP 8: DEMOGRAPHICS
  
  Purpose: Extract demographic features including age, gender, ethnicity,
           language, tenure, and geographic characteristics.
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_demographics`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_demographics`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
WITH mth AS (
    SELECT mnth.asdb_member_key, st.index_dt, CAST(mnth.asdb_elig_dt AS DATE) AS asdb_elig_dt
        , ROW_NUMBER() OVER(PARTITION BY mnth.asdb_member_key ORDER BY mnth.asdb_elig_dt) AS mnths
    FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ELIG_DATA_MBR_PER_MTH` AS mnth
    LEFT JOIN (SELECT asdb_member_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`) AS st
        ON st.asdb_member_key = mnth.asdb_member_key
    WHERE CAST(mnth.asdb_elig_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 12 MONTH) AND st.index_dt
),
mth2 AS (
    SELECT mnth.asdb_member_key, st.index_dt, CAST(mnth.asdb_elig_dt AS DATE) AS asdb_elig_dt
        , ROW_NUMBER() OVER(PARTITION BY mnth.asdb_member_key ORDER BY mnth.asdb_elig_dt) AS mnths
    FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ELIG_DATA_MBR_PER_MTH` AS mnth
    LEFT JOIN (SELECT asdb_member_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`) AS st
        ON st.asdb_member_key = mnth.asdb_member_key
    WHERE CAST(mnth.asdb_elig_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 24 MONTH) AND DATE_SUB(st.index_dt, INTERVAL 13 MONTH)
),
post AS (
    SELECT mnth.asdb_member_key, st.index_dt, CAST(mnth.asdb_elig_dt AS DATE) AS asdb_elig_dt
        , ROW_NUMBER() OVER(PARTITION BY mnth.asdb_member_key ORDER BY mnth.asdb_elig_dt) AS mnths
    FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ELIG_DATA_MBR_PER_MTH` AS mnth
    LEFT JOIN (SELECT asdb_member_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`) AS st
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
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index` AS st
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_MEMBER` AS mb ON st.asdb_member_key = mb.asdb_member_key
LEFT JOIN (SELECT asdb_member_key, MAX(mnths) AS tenure FROM mth GROUP BY asdb_member_key) AS mth ON st.asdb_member_key = mth.asdb_member_key
LEFT JOIN (SELECT asdb_member_key, MAX(mnths) AS tenure FROM mth2 GROUP BY asdb_member_key) AS mth2 ON st.asdb_member_key = mth2.asdb_member_key
LEFT JOIN (SELECT asdb_member_key, MAX(mnths) AS tenure FROM post GROUP BY asdb_member_key) AS post ON st.asdb_member_key = post.asdb_member_key
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_tra_ckd_phm_srcv.ZIP_CENSUS_USPS_URBRUR` AS zcuu ON TRIM(mb.member_zip) = TRIM(zcuu.zip_cd)
;


/*==============================================================================
  STEP 9: GEOID (Geographic Identifiers for ACS/CSDI)
  
  Purpose: Map members to census tract and block group for SDOH data linkage.
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_geoid`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_geoid`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
WITH maxdt AS (
    SELECT iodb_member_key, MAX(source_pstd_dts) AS source_pstd_dts
    FROM `anbc-hcb-prod.insights_share_hcb_prod.v_enriched_address_medicaid`
    GROUP BY iodb_member_key
)
SELECT DISTINCT 
    st.asdb_member_key, st.asdb_plan_key, st.index_dt
    , mb.iodb_member_key, id.ctfips, id.bgfips
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index` AS st
INNER JOIN (SELECT iodb_member_key, asdb_member_key, asdb_plan_key FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_MEMBER`) AS mb
    ON st.asdb_member_key = mb.asdb_member_key AND st.asdb_plan_key = mb.asdb_plan_key
INNER JOIN (SELECT block_code AS bgfips, CONCAT(fips_state_county_code, census_tract) AS ctfips, iodb_member_key, source_pstd_dts, geo_accuracy_code
            FROM `anbc-hcb-prod.insights_share_hcb_prod.v_enriched_address_medicaid`) AS id
    ON mb.iodb_member_key = id.iodb_member_key
INNER JOIN maxdt ON id.iodb_member_key = maxdt.iodb_member_key AND id.source_pstd_dts = maxdt.source_pstd_dts
WHERE TRIM(id.geo_accuracy_code) IN ("1", "2", "5", "6")
;


/*==============================================================================
  STEP 10: ACS (American Community Survey) Scores
  
  Purpose: Extract area-level social risk scores from ACS data.
  
  Scores: SDI (Social Deprivation Index), SVI (Social Vulnerability Index),
          ADI (Area Deprivation Index), Social Risk Score
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_acs`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_acs`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
SELECT st.asdb_member_key, st.asdb_plan_key, st.index_dt
    , COALESCE(MAX(b.social_risk_score), 0) AS social_risk_score
    , COALESCE(MAX(b.sdi_score), 0) AS sdi_score
    , COALESCE(MAX(b.svi_score), 0) AS svi_score
    , COALESCE(MAX(b.adi_score), 0) AS adi_score
FROM (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`) AS st
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_geoid` AS id
    ON st.asdb_member_key = id.asdb_member_key AND st.asdb_plan_key = id.asdb_plan_key
LEFT JOIN (SELECT * FROM `edp-prod-storage.edp_ent_sdoheir_srcv.srs_acs_block_group_allscores_historical_data` WHERE effective_year = 2022) AS b
    ON id.ctfips = b.ctfips
GROUP BY st.asdb_member_key, st.asdb_plan_key, st.index_dt
;


/*==============================================================================
  STEP 11: CSDI (Clinical & Social Determinants Intelligence) Risk Indices
  
  Purpose: Extract 22 CSDI risk indices covering multiple SDOH domains.
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_csdi`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_csdi`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
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
FROM (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt, bgfips FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_geoid`) AS st
LEFT JOIN (SELECT * FROM `edp-prod-storage.edp_ent_sdoheir_srcv.risk_index_block_group_historical_data` WHERE effective_year = 2023) AS c
    ON TRIM(st.bgfips) = TRIM(c.bgfips)
GROUP BY st.asdb_member_key, st.asdb_plan_key, st.index_dt
;


/*==============================================================================
  STEP 12a: PREVENTATIVE CARE FLAGS
  
  Purpose: Extract preventative care service claims with CMS screening flags.
  
  Categories: PCP visits, specialist visits, lab tests, vaccinations, screenings
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_preventative`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_preventative`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
WITH clm AS (
    SELECT *
        , CASE WHEN TRIM(asdb_coe_general_type)="Inpatient" OR TRIM(emis_cat)="Institutional Services" THEN "Inpatient"
               WHEN TRIM(emis_cat)="Emergency" THEN "Emergency" ELSE "Outpatient" END AS plc_svc_ctg
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_med_claims_flag_yr1`
)
SELECT asdb_member_key, asdb_plan_key, claimid, index_dt, asdb_incurred_dt
    -- PCP/Specialist/OB visits
    , CASE WHEN plc_svc_ctg = "Outpatient"
            AND (asdb_coe_id IN (63000, 63100, 63200, 63300, 63400, 63500, 63600, 63999)
                OR TRIM(prindiag) IN ("V20.2","V20.31","V20.32","V70.0","V70.3","V70.5","V70.6","V70.8","V70.9","V72.31","V72.3",
                    "Z00.110","Z00.111","Z00.129", "Z00.8","Z01.411","Z01.419","Z01.42"))
            AND TRIM(emis_cat) = "Primary Physician"
        THEN 1 ELSE 0 END AS pcp_op_visit
    , CASE WHEN plc_svc_ctg = "Outpatient"
            AND (asdb_coe_id IN (63000, 63100, 63200, 63300, 63400, 63500, 63600, 63999)
                OR TRIM(prindiag) IN ("V20.2","V20.31","V20.32","V70.0","V70.3","V70.5","V70.6","V70.8","V70.9","V72.31","V72.3",
                    "Z00.110","Z00.111","Z00.129", "Z00.8","Z01.411","Z01.419","Z01.42"))
            AND NOT (TRIM(emis_cat) = "Primary Physician" OR TRIM(LOWER(prov_specialty)) LIKE "%gynecol%")
        THEN 1 ELSE 0 END AS spec_op_visit
    , CASE WHEN plc_svc_ctg = "Outpatient"
            AND (TRIM(LOWER(prov_specialty)) LIKE "%midwif%" OR TRIM(LOWER(prov_specialty)) LIKE "%gynecol%")
        THEN 1 ELSE 0 END AS obgyn_mw_op_visit
    -- Lab tests
    , CASE WHEN TRIM(servcode) IN ("80061","83715","83716","83721","83718","83700","83701","83704","3048F","3049F","3050F")
        THEN 1 ELSE 0 END AS cholest_screen_claim
    , CASE WHEN TRIM(servcode) IN ("83036","83037","3044F","3045F","3046F")
        THEN 1 ELSE 0 END AS hba1c_test_claim
    -- CMS preventive screenings
    , CASE WHEN TRIM(servcode) IN ("G0442","G0443") THEN 1 ELSE 0 END AS cms_alcohol_misuse_screening_counseling
    , CASE WHEN TRIM(servcode) IN ("G0444") THEN 1 ELSE 0 END AS cms_depression_screening
    , CASE WHEN TRIM(servcode) IN ("82947","82950","82951") AND TRIM(prindiag) = "Z13.1" THEN 1 ELSE 0 END AS cms_diabetes_screening
    , CASE WHEN TRIM(servcode) IN ("G0499") THEN 1 ELSE 0 END AS cms_hep_b_virus_screening
    , CASE WHEN TRIM(servcode) IN ("90739","90740","90743","90744","90746","90747","G0010") AND TRIM(prindiag) = "Z23" THEN 1 ELSE 0 END AS cms_hep_b_virus_vax
    , CASE WHEN TRIM(servcode) IN ("G0446") THEN 1 ELSE 0 END AS cms_ibt_for_cvd
    -- Fixed: IBT Obesity - added obesity diagnosis filter per original script
    , CASE WHEN TRIM(servcode) IN ("G0447","G0473") 
            AND TRIM(prindiag) IN ("Z68.30","Z68.31","Z68.32","Z68.33","Z68.34","Z68.35","Z68.36","Z68.37","Z68.38","Z68.39",
                                   "Z68.41","Z68.42","Z68.43","Z68.44","Z68.45","E66.0","E66.01","E66.09","E66.1","E66.2","E66.8","E66.9")
        THEN 1 ELSE 0 END AS cms_ibt_for_obesity
    , CASE WHEN TRIM(servcode) IN ("90630","90653","90654","90655","90656","90657","90658","90660","90662","90672","90673","90674","90682","90685","90686","90687","90688","90689","90694","90756","Q2034","Q2035","Q2036","Q2037","Q2038","Q2039","G0008") AND TRIM(prindiag) = "Z23" THEN 1 ELSE 0 END AS cms_influenza_virus_vaccine
    , CASE WHEN TRIM(servcode) IN ("90670","90732","G0009") AND TRIM(prindiag) = "Z23" THEN 1 ELSE 0 END AS cms_pneumococcal_vaccine
    , CASE WHEN TRIM(servcode) IN ("G0476") THEN 1 ELSE 0 END AS cms_cervical_cancer_hpv
    , CASE WHEN TRIM(servcode) IN ("77063","77067") THEN 1 ELSE 0 END AS cms_screening_mammography
    , CASE WHEN TRIM(servcode) IN ("G0123","G0124","G0141","G0143","G0144","G0145","G0147","G0148","P3000","P3001") THEN 1 ELSE 0 END AS cms_screening_pap
    , CASE WHEN TRIM(servcode) = "G0101" THEN 1 ELSE 0 END AS cms_screening_pelvic_exams
    , CASE WHEN TRIM(servcode) IN ("G0108","G0109") THEN 1 ELSE 0 END AS cms_diabetes_self_management_training
    , CASE WHEN TRIM(servcode) = "G0102" AND TRIM(prindiag) = "Z12.5" THEN 1 ELSE 0 END AS cms_prostate_cancer_rectal_examination
    -- ADDED: Additional CMS preventive services that were missing
    , CASE WHEN TRIM(servcode) IN ("G0438","G0439") THEN 1 ELSE 0 END AS cms_annual_wellness_visit
    , CASE WHEN TRIM(servcode) IN ("G0402","G0403","G0404","G0405") THEN 1 ELSE 0 END AS cms_initial_prevent_exam
    -- Adult BMI Assessment (ADDED - was missing)
    , CASE WHEN TRIM(servcode) IN ("G0447","G0473","G8417","G8418","G8419","G8420","G8421","G8422")
            OR TRIM(prindiag) LIKE "Z68.%" THEN 1 ELSE 0 END AS adult_bmi_assess
    -- Fall Risk Assessment (ADDED - was missing)
    , CASE WHEN TRIM(servcode) IN ("G0442","G0443","G8941","3288F","1100F","1101F") 
            OR TRIM(prindiag) IN ("W19","W18","R29.6") THEN 1 ELSE 0 END AS screenforfall
    , CASE WHEN TRIM(servcode) IN ("G8941","1100F","1101F","3288F") THEN 1 ELSE 0 END AS screen_future_fallrisk
    -- Vaccinations (ADDED - was missing individual codes)
    , CASE WHEN TRIM(servcode) IN ("90630","90653","90654","90655","90656","90657","90658","90660","90662","90672","90673","90674","90682","90685","90686","90687","90688","90689","90694","90756","Q2034","Q2035","Q2036","Q2037","Q2038","Q2039","G0008") THEN 1 ELSE 0 END AS flu_vacc
    , CASE WHEN TRIM(servcode) IN ("90670","90732","G0009") THEN 1 ELSE 0 END AS pneumonia_vaccine
    , CASE WHEN TRIM(servcode) IN ("90698","90700","90702","90714","90715") THEN 1 ELSE 0 END AS tdap_vacc
    , CASE WHEN TRIM(servcode) IN ("90649","90650","90651") THEN 1 ELSE 0 END AS hpv_vacc
    , CASE WHEN TRIM(servcode) IN ("90680","90681") THEN 1 ELSE 0 END AS rotavirus_vacc
    -- Cancer Screening (ADDED - was missing)
    , CASE WHEN TRIM(servcode) IN ("82270","82274","G0328","G0104","G0105","G0106","G0120","G0121","G0122") THEN 1 ELSE 0 END AS colorectal_ca_screen
    , CASE WHEN TRIM(servcode) IN ("G0123","G0124","G0141","G0143","G0144","G0145","G0147","G0148","P3000","P3001","Q0091") THEN 1 ELSE 0 END AS cervical_ca_screen
    , CASE WHEN TRIM(servcode) IN ("77065","77066","77067","G0202","G0204","G0206") THEN 1 ELSE 0 END AS breast_ca_screen
    , CASE WHEN TRIM(servcode) IN ("G0102","G0103") THEN 1 ELSE 0 END AS prostate_ca_screen
    -- STD Screening (ADDED - was missing)
    , CASE WHEN TRIM(servcode) IN ("86631","86632","86780","87110","87270","87290","87320","87490","87491","87492","87590","87591","87592","87800","87801","87810","G0445","S0610") THEN 1 ELSE 0 END AS std_screening
    -- Tobacco Cessation (ADDED - was missing)
    , CASE WHEN TRIM(servcode) IN ("99406","99407","S9453","G0436","G0437","G9016") THEN 1 ELSE 0 END AS tobacco_cessation
FROM clm
;


/*==============================================================================
  STEP 12b: PREVENTATIVE CARE SUMMARY
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_preventative_summary`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_preventative_summary`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
SELECT asdb_member_key, asdb_plan_key, index_dt
    , MIN(asdb_incurred_dt) AS first_prv_dt, MAX(asdb_incurred_dt) AS last_prv_dt
    , SUM(pcp_op_visit) AS sum_pcp, SUM(spec_op_visit) AS sum_spec, SUM(obgyn_mw_op_visit) AS sum_ob
    , SUM(cholest_screen_claim) AS sum_chol_lab, SUM(hba1c_test_claim) AS sum_a1c_lab
    , SUM(cms_alcohol_misuse_screening_counseling) AS cms_alc_scrn
    , SUM(cms_depression_screening) AS cms_dep_scrn
    , SUM(cms_diabetes_screening) AS cms_t2d_scrn
    , SUM(cms_hep_b_virus_screening) AS cms_hepb_scrn
    , SUM(cms_hep_b_virus_vax) AS cms_hepb_vax
    , SUM(cms_ibt_for_cvd) AS cms_ibt_cvd
    , SUM(cms_ibt_for_obesity) AS cms_ibt_obese
    , SUM(cms_influenza_virus_vaccine) AS cms_flu_vax
    , SUM(cms_pneumococcal_vaccine) AS cms_pneum_vax
    , SUM(cms_cervical_cancer_hpv) AS cms_hpv_scrn
    , SUM(cms_screening_mammography) AS cms_mam_scrn
    , SUM(cms_screening_pap) AS cms_pap
    , SUM(cms_screening_pelvic_exams) AS cms_pelvic
    , SUM(cms_diabetes_self_management_training) AS cms_t2d_train
    , SUM(cms_prostate_cancer_rectal_examination) AS cms_prost_cancer_scrn
    -- ADDED: Additional preventative features
    , SUM(cms_annual_wellness_visit) AS cms_annual_wellness
    , SUM(cms_initial_prevent_exam) AS cms_initial_prevent
    , SUM(adult_bmi_assess) AS sum_bmi_assess
    , SUM(screenforfall) AS sum_fall_screen
    , SUM(screen_future_fallrisk) AS sum_future_fall_risk
    , SUM(flu_vacc) AS sum_flu_vacc
    , SUM(pneumonia_vaccine) AS sum_pneum_vacc
    , SUM(tdap_vacc) AS sum_tdap_vacc
    , SUM(hpv_vacc) AS sum_hpv_vacc
    , SUM(rotavirus_vacc) AS sum_rotavirus_vacc
    , SUM(colorectal_ca_screen) AS sum_colorectal_screen
    , SUM(cervical_ca_screen) AS sum_cervical_screen
    , SUM(breast_ca_screen) AS sum_breast_screen
    , SUM(prostate_ca_screen) AS sum_prostate_screen
    , SUM(std_screening) AS sum_std_screen
    , SUM(tobacco_cessation) AS sum_tobacco_cess
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_preventative`
GROUP BY asdb_member_key, asdb_plan_key, index_dt
;


/*==============================================================================
  STEP 13: FINAL NON-EMBEDDING FEATURE ASSEMBLY
  
  Purpose: Join all component feature tables into a single comprehensive
           non-embedding feature table (~300 features).
  
  Output: a964286_Medicaid_nonembed_feature_4_te_experiment_features
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_features`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_features`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
SELECT
    -- Member identifiers
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
    , st.coa_population_category
    , st.coa_population_group
    
    -- ED features yr1
    , ed1.sum_ed_visits AS sum_ed_visits_yr1, ed1.ed_flag AS ed_flag_yr1, ed1.sum_ed_cost AS sum_ed_cost_yr1
    , ed1.sum_avoidable AS sum_avoidable_yr1, ed1.sum_unnecessary AS sum_unnecessary_yr1, ed1.sum_preventable AS sum_preventable_yr1
    , ed1.low_sev_ed_visits AS low_sev_ed_visits_yr1, ed1.low_med_sev_ed_visits AS low_med_sev_ed_visits_yr1
    , ed1.med_sev_ed_visits AS med_sev_ed_visits_yr1, ed1.med_high_sev_ed_visits AS med_high_sev_ed_visits_yr1
    , ed1.high_sev_ed_visits AS high_sev_ed_visits_yr1
    , ed1.low_sev_ed_flag AS low_sev_ed_flag_yr1, ed1.low_med_sev_ed_flag AS low_med_sev_ed_flag_yr1
    , ed1.med_sev_ed_flag AS med_sev_ed_flag_yr1, ed1.med_high_sev_ed_flag AS med_high_sev_ed_flag_yr1
    , ed1.high_sev_ed_flag AS high_sev_ed_flag_yr1
    -- ED severity costs yr1 (ADDED)
    , ed1.low_sev_ed_cost AS low_sev_ed_cost_yr1, ed1.low_med_sev_ed_cost AS low_med_sev_ed_cost_yr1
    , ed1.med_sev_ed_cost AS med_sev_ed_cost_yr1, ed1.med_high_sev_ed_cost AS med_high_sev_ed_cost_yr1
    , ed1.high_sev_ed_cost AS high_sev_ed_cost_yr1
    
    -- ED features yr2
    , ed2.sum_ed_visits AS sum_ed_visits_yr2, ed2.ed_flag AS ed_flag_yr2, ed2.sum_ed_cost AS sum_ed_cost_yr2
    , ed2.sum_avoidable AS sum_avoidable_yr2, ed2.sum_unnecessary AS sum_unnecessary_yr2, ed2.sum_preventable AS sum_preventable_yr2
    , ed2.low_sev_ed_visits AS low_sev_ed_visits_yr2, ed2.low_med_sev_ed_visits AS low_med_sev_ed_visits_yr2
    , ed2.med_sev_ed_visits AS med_sev_ed_visits_yr2, ed2.med_high_sev_ed_visits AS med_high_sev_ed_visits_yr2
    , ed2.high_sev_ed_visits AS high_sev_ed_visits_yr2
    , ed2.low_sev_ed_flag AS low_sev_ed_flag_yr2, ed2.low_med_sev_ed_flag AS low_med_sev_ed_flag_yr2
    , ed2.med_sev_ed_flag AS med_sev_ed_flag_yr2, ed2.med_high_sev_ed_flag AS med_high_sev_ed_flag_yr2
    , ed2.high_sev_ed_flag AS high_sev_ed_flag_yr2
    -- ED severity costs yr2 (ADDED)
    , ed2.low_sev_ed_cost AS low_sev_ed_cost_yr2, ed2.low_med_sev_ed_cost AS low_med_sev_ed_cost_yr2
    , ed2.med_sev_ed_cost AS med_sev_ed_cost_yr2, ed2.med_high_sev_ed_cost AS med_high_sev_ed_cost_yr2
    , ed2.high_sev_ed_cost AS high_sev_ed_cost_yr2
    
    -- IP features yr1/yr2
    , ip1.acute_ip_flag AS acute_ip_flag_yr1, ip1.sum_acute_ip_admits AS sum_acute_ip_admits_yr1
    , ip1.sum_acute_calc_los AS sum_acute_calc_los_yr1
    , ip2.acute_ip_flag AS acute_ip_flag_yr2, ip2.sum_acute_ip_admits AS sum_acute_ip_admits_yr2
    , ip2.sum_acute_calc_los AS sum_acute_calc_los_yr2
    
    -- OP features
    , op1.sum_op_visits AS sum_op_visits_yr1
    , op2.sum_op_visits AS sum_op_visits_yr2
    
    -- Utilization features yr1
    , ut1.sum_paid_amt AS sum_paid_amt_yr1
    , ut1.emis_community_clm AS emis_community_clm_yr1, ut1.emis_ed_clm AS emis_ed_clm_yr1
    , ut1.emis_hh_clm AS emis_hh_clm_yr1, ut1.emis_home_clm AS emis_home_clm_yr1
    , ut1.emis_ip_clm AS emis_ip_clm_yr1, ut1.emis_ins_clm AS emis_ins_clm_yr1
    , ut1.emis_lab_clm AS emis_lab_clm_yr1, ut1.emis_mrx_clm AS emis_mrx_clm_yr1
    , ut1.emis_mh_clm AS emis_mh_clm_yr1, ut1.emis_misc_clm AS emis_misc_clm_yr1
    , ut1.emis_pcp_clm AS emis_pcp_clm_yr1, ut1.emis_radio_clm AS emis_radio_clm_yr1
    , ut1.emis_ambul_clm AS emis_ambul_clm_yr1, ut1.emis_spec_clm AS emis_spec_clm_yr1
    , ut1.ltc_clm AS ltc_clm_yr1
    , ut1.coe_ip_hos_clm AS coe_ip_hos_clm_yr1, ut1.coe_ip_non_hos_clm AS coe_ip_non_hos_clm_yr1
    , ut1.coe_lab_clm AS coe_lab_clm_yr1, ut1.coe_ltc_community_clm AS coe_ltc_community_clm_yr1
    , ut1.coe_ltc_home_clm AS coe_ltc_home_clm_yr1, ut1.coe_ltc_ins_clm AS coe_ltc_ins_clm_yr1
    , ut1.coe_other_clm AS coe_other_clm_yr1, ut1.coe_op_hos_clm AS coe_op_hos_clm_yr1
    , ut1.coe_op_non_hos_clm AS coe_op_non_hos_clm_yr1, ut1.coe_anesth_clm AS coe_anesth_clm_yr1
    , ut1.coe_eval_clm AS coe_eval_clm_yr1, ut1.coe_maternity_clm AS coe_maternity_clm_yr1
    , ut1.coe_mrx_clm AS coe_mrx_clm_yr1, ut1.coe_mh_clm AS coe_mh_clm_yr1
    , ut1.coe_phy_clm AS coe_phy_clm_yr1, ut1.coe_surg_clm AS coe_surg_clm_yr1
    , ut1.coe_radio_clm AS coe_radio_clm_yr1, ut1.uc_clm AS uc_clm_yr1, ut1.obs_clm AS obs_clm_yr1
    -- Cost features yr1 (ADDED)
    , ut1.inpatient_cost AS inpatient_cost_yr1, ut1.emergency_cost AS emergency_cost_yr1, ut1.outpatient_cost AS outpatient_cost_yr1
    , ut1.emis_community_cost AS emis_community_cost_yr1, ut1.emis_ed_cost AS emis_ed_cost_yr1
    , ut1.emis_hh_cost AS emis_hh_cost_yr1, ut1.emis_home_cost AS emis_home_cost_yr1
    , ut1.emis_ip_cost AS emis_ip_cost_yr1, ut1.emis_ins_cost AS emis_ins_cost_yr1
    , ut1.emis_lab_cost AS emis_lab_cost_yr1, ut1.emis_mrx_cost AS emis_mrx_cost_yr1
    , ut1.emis_mh_cost AS emis_mh_cost_yr1, ut1.emis_misc_cost AS emis_misc_cost_yr1
    , ut1.emis_pcp_cost AS emis_pcp_cost_yr1, ut1.emis_radio_cost AS emis_radio_cost_yr1
    , ut1.emis_ambul_cost AS emis_ambul_cost_yr1, ut1.emis_spec_cost AS emis_spec_cost_yr1
    , ut1.coe_ltc_cost AS coe_ltc_cost_yr1
    , ut1.coe_ip_hos_cost AS coe_ip_hos_cost_yr1, ut1.coe_ip_non_hos_cost AS coe_ip_non_hos_cost_yr1
    , ut1.coe_lab_cost AS coe_lab_cost_yr1, ut1.coe_ltc_community_cost AS coe_ltc_community_cost_yr1
    , ut1.coe_ltc_home_cost AS coe_ltc_home_cost_yr1, ut1.coe_ltc_ins_cost AS coe_ltc_ins_cost_yr1
    , ut1.coe_other_cost AS coe_other_cost_yr1, ut1.coe_op_hos_cost AS coe_op_hos_cost_yr1
    , ut1.coe_op_non_hos_cost AS coe_op_non_hos_cost_yr1, ut1.coe_anesth_cost AS coe_anesth_cost_yr1
    , ut1.coe_eval_cost AS coe_eval_cost_yr1, ut1.coe_maternity_cost AS coe_maternity_cost_yr1
    , ut1.coe_mrx_cost AS coe_mrx_cost_yr1, ut1.coe_mh_cost AS coe_mh_cost_yr1
    , ut1.coe_phy_cost AS coe_phy_cost_yr1, ut1.coe_surg_cost AS coe_surg_cost_yr1
    , ut1.coe_radio_cost AS coe_radio_cost_yr1, ut1.uc_cost AS uc_cost_yr1
    
    -- Utilization features yr2
    , ut2.sum_paid_amt AS sum_paid_amt_yr2
    , ut2.emis_community_clm AS emis_community_clm_yr2, ut2.emis_ed_clm AS emis_ed_clm_yr2
    , ut2.emis_hh_clm AS emis_hh_clm_yr2, ut2.emis_home_clm AS emis_home_clm_yr2
    , ut2.emis_ip_clm AS emis_ip_clm_yr2, ut2.emis_ins_clm AS emis_ins_clm_yr2
    , ut2.emis_lab_clm AS emis_lab_clm_yr2, ut2.emis_mrx_clm AS emis_mrx_clm_yr2
    , ut2.emis_mh_clm AS emis_mh_clm_yr2, ut2.emis_misc_clm AS emis_misc_clm_yr2
    , ut2.emis_pcp_clm AS emis_pcp_clm_yr2, ut2.emis_radio_clm AS emis_radio_clm_yr2
    , ut2.emis_ambul_clm AS emis_ambul_clm_yr2, ut2.emis_spec_clm AS emis_spec_clm_yr2
    , ut2.ltc_clm AS ltc_clm_yr2
    , ut2.coe_ip_hos_clm AS coe_ip_hos_clm_yr2, ut2.coe_ip_non_hos_clm AS coe_ip_non_hos_clm_yr2
    , ut2.coe_lab_clm AS coe_lab_clm_yr2, ut2.coe_ltc_community_clm AS coe_ltc_community_clm_yr2
    , ut2.coe_ltc_home_clm AS coe_ltc_home_clm_yr2, ut2.coe_ltc_ins_clm AS coe_ltc_ins_clm_yr2
    , ut2.coe_other_clm AS coe_other_clm_yr2, ut2.coe_op_hos_clm AS coe_op_hos_clm_yr2
    , ut2.coe_op_non_hos_clm AS coe_op_non_hos_clm_yr2, ut2.coe_anesth_clm AS coe_anesth_clm_yr2
    , ut2.coe_eval_clm AS coe_eval_clm_yr2, ut2.coe_maternity_clm AS coe_maternity_clm_yr2
    , ut2.coe_mrx_clm AS coe_mrx_clm_yr2, ut2.coe_mh_clm AS coe_mh_clm_yr2
    , ut2.coe_phy_clm AS coe_phy_clm_yr2, ut2.coe_surg_clm AS coe_surg_clm_yr2
    , ut2.coe_radio_clm AS coe_radio_clm_yr2, ut2.uc_clm AS uc_clm_yr2, ut2.obs_clm AS obs_clm_yr2
    -- Cost features yr2 (ADDED)
    , ut2.inpatient_cost AS inpatient_cost_yr2, ut2.emergency_cost AS emergency_cost_yr2, ut2.outpatient_cost AS outpatient_cost_yr2
    , ut2.emis_community_cost AS emis_community_cost_yr2, ut2.emis_ed_cost AS emis_ed_cost_yr2
    , ut2.emis_hh_cost AS emis_hh_cost_yr2, ut2.emis_home_cost AS emis_home_cost_yr2
    , ut2.emis_ip_cost AS emis_ip_cost_yr2, ut2.emis_ins_cost AS emis_ins_cost_yr2
    , ut2.emis_lab_cost AS emis_lab_cost_yr2, ut2.emis_mrx_cost AS emis_mrx_cost_yr2
    , ut2.emis_mh_cost AS emis_mh_cost_yr2, ut2.emis_misc_cost AS emis_misc_cost_yr2
    , ut2.emis_pcp_cost AS emis_pcp_cost_yr2, ut2.emis_radio_cost AS emis_radio_cost_yr2
    , ut2.emis_ambul_cost AS emis_ambul_cost_yr2, ut2.emis_spec_cost AS emis_spec_cost_yr2
    , ut2.coe_ltc_cost AS coe_ltc_cost_yr2
    , ut2.coe_ip_hos_cost AS coe_ip_hos_cost_yr2, ut2.coe_ip_non_hos_cost AS coe_ip_non_hos_cost_yr2
    , ut2.coe_lab_cost AS coe_lab_cost_yr2, ut2.coe_ltc_community_cost AS coe_ltc_community_cost_yr2
    , ut2.coe_ltc_home_cost AS coe_ltc_home_cost_yr2, ut2.coe_ltc_ins_cost AS coe_ltc_ins_cost_yr2
    , ut2.coe_other_cost AS coe_other_cost_yr2, ut2.coe_op_hos_cost AS coe_op_hos_cost_yr2
    , ut2.coe_op_non_hos_cost AS coe_op_non_hos_cost_yr2, ut2.coe_anesth_cost AS coe_anesth_cost_yr2
    , ut2.coe_eval_cost AS coe_eval_cost_yr2, ut2.coe_maternity_cost AS coe_maternity_cost_yr2
    , ut2.coe_mrx_cost AS coe_mrx_cost_yr2, ut2.coe_mh_cost AS coe_mh_cost_yr2
    , ut2.coe_phy_cost AS coe_phy_cost_yr2, ut2.coe_surg_cost AS coe_surg_cost_yr2
    , ut2.coe_radio_cost AS coe_radio_cost_yr2, ut2.uc_cost AS uc_cost_yr2
    
    -- Conditions (48 flags)
    , cond.abdominal_pain, cond.AID, cond.IDA, cond.ANX, cond.OST, cond.AST
    , cond.AUT, cond.CHO, cond.burns, cond.cad, cond.Cancer, cond.narc
    , cond.CBD, cond.CHF, cond.CRF, cond.VNA, cond.CHD, cond.COP
    , cond.CYS, cond.DEP, cond.DIA, cond.EDO, cond.esrd, cond.EPL
    , cond.CRO, cond.MOH, cond.HEM, cond.HepC, cond.HYP, cond.HYC
    , cond.immune, cond.intel_dsblty, cond.meta_cancer, cond.liver_dis
    , cond.MSS, cond.OBE, cond.oud, cond.liver_other, cond.paralysis
    , cond.PAR, cond.PUD, cond.hmd, cond.PVD, cond.autoimmune, cond.DEM
    , cond.SCA, cond.sleep_apnea, cond.spinal_inj, cond.back, cond.substance
    , cond.ALC, cond.bipolar, cond.psychoses, cond.major_chronic_cnt
    
    -- Rx features yr1
    , rx1.rx_claim_cnt AS rx_claim_cnt_yr1, rx1.days_supply_sum AS days_supply_sum_yr1
    , rx1.ndc_cnt AS ndc_cnt_yr1, rx1.gpi_cnt AS gpi_cnt_yr1
    , rx1.gpi4_cnt AS gpi4_cnt_yr1, rx1.gpi2_cnt AS gpi2_cnt_yr1
    , rx1.retail_fills AS retail_fills_yr1, rx1.mail_order_fills AS mail_order_fills_yr1
    , rx1.generic_fills AS generic_fills_yr1, rx1.branded_generic_fills AS branded_generic_fills_yr1
    , rx1.otc_fills AS otc_fills_yr1, rx1.ss_brand_fills AS ss_brand_fills_yr1
    , rx1.ms_brand_fills AS ms_brand_fills_yr1, rx1.formulary_fills AS formulary_fills_yr1
    , rx1.maint_drug_fills AS maint_drug_fills_yr1
    , rx1.antidiabetic_scripts AS antidiabetic_scripts_yr1, rx1.antidiabetic_days_supply AS antidiabetic_days_supply_yr1
    , rx1.beta_blocker_scripts AS beta_blocker_scripts_yr1, rx1.beta_blocker_days_supply AS beta_blocker_days_supply_yr1
    , rx1.antihypertensive_scripts AS antihypertensive_scripts_yr1, rx1.antihypertensive_days_supply AS antihypertensive_days_supply_yr1
    , rx1.lipid_lowering_scripts AS lipid_lowering_scripts_yr1, rx1.lipid_lowering_days_supply AS lipid_lowering_days_supply_yr1
    , rx1.calcium_channel_blk_scripts AS calcium_channel_blk_scripts_yr1, rx1.calcium_channel_blk_days_supply AS calcium_channel_blk_days_supply_yr1
    , rx1.diuretic_scripts AS diuretic_scripts_yr1, rx1.diuretic_days_supply AS diuretic_days_supply_yr1
    , rx1.antianginal_agent_scripts AS antianginal_agent_scripts_yr1, rx1.antianginal_agent_days_supply AS antianginal_agent_days_supply_yr1
    , rx1.antidepressant_scripts AS antidepressant_scripts_yr1, rx1.antidepressant_days_supply AS antidepressant_days_supply_yr1
    , rx1.antipsychotic_scripts AS antipsychotic_scripts_yr1, rx1.antipsychotic_days_supply AS antipsychotic_days_supply_yr1
    , rx1.antianxiety_scripts AS antianxiety_scripts_yr1, rx1.antianxiety_days_supply AS antianxiety_days_supply_yr1
    , rx1.anticonvulsant_scripts AS anticonvulsant_scripts_yr1, rx1.anticonvulsant_days_supply AS anticonvulsant_days_supply_yr1
    , rx1.inhaled_steroid_scripts AS inhaled_steroid_scripts_yr1, rx1.inhaled_steroid_days_supply AS inhaled_steroid_days_supply_yr1
    
    -- Rx features yr2
    , rx2.rx_claim_cnt AS rx_claim_cnt_yr2, rx2.days_supply_sum AS days_supply_sum_yr2
    , rx2.ndc_cnt AS ndc_cnt_yr2, rx2.gpi_cnt AS gpi_cnt_yr2
    , rx2.gpi4_cnt AS gpi4_cnt_yr2, rx2.gpi2_cnt AS gpi2_cnt_yr2
    , rx2.retail_fills AS retail_fills_yr2, rx2.mail_order_fills AS mail_order_fills_yr2
    , rx2.generic_fills AS generic_fills_yr2, rx2.branded_generic_fills AS branded_generic_fills_yr2
    , rx2.otc_fills AS otc_fills_yr2, rx2.ss_brand_fills AS ss_brand_fills_yr2
    , rx2.ms_brand_fills AS ms_brand_fills_yr2, rx2.formulary_fills AS formulary_fills_yr2
    , rx2.maint_drug_fills AS maint_drug_fills_yr2
    , rx2.antidiabetic_scripts AS antidiabetic_scripts_yr2, rx2.antidiabetic_days_supply AS antidiabetic_days_supply_yr2
    , rx2.beta_blocker_scripts AS beta_blocker_scripts_yr2, rx2.beta_blocker_days_supply AS beta_blocker_days_supply_yr2
    , rx2.antihypertensive_scripts AS antihypertensive_scripts_yr2, rx2.antihypertensive_days_supply AS antihypertensive_days_supply_yr2
    , rx2.lipid_lowering_scripts AS lipid_lowering_scripts_yr2, rx2.lipid_lowering_days_supply AS lipid_lowering_days_supply_yr2
    , rx2.calcium_channel_blk_scripts AS calcium_channel_blk_scripts_yr2, rx2.calcium_channel_blk_days_supply AS calcium_channel_blk_days_supply_yr2
    , rx2.diuretic_scripts AS diuretic_scripts_yr2, rx2.diuretic_days_supply AS diuretic_days_supply_yr2
    , rx2.antianginal_agent_scripts AS antianginal_agent_scripts_yr2, rx2.antianginal_agent_days_supply AS antianginal_agent_days_supply_yr2
    , rx2.antidepressant_scripts AS antidepressant_scripts_yr2, rx2.antidepressant_days_supply AS antidepressant_days_supply_yr2
    , rx2.antipsychotic_scripts AS antipsychotic_scripts_yr2, rx2.antipsychotic_days_supply AS antipsychotic_days_supply_yr2
    , rx2.antianxiety_scripts AS antianxiety_scripts_yr2, rx2.antianxiety_days_supply AS antianxiety_days_supply_yr2
    , rx2.anticonvulsant_scripts AS anticonvulsant_scripts_yr2, rx2.anticonvulsant_days_supply AS anticonvulsant_days_supply_yr2
    , rx2.inhaled_steroid_scripts AS inhaled_steroid_scripts_yr2, rx2.inhaled_steroid_days_supply AS inhaled_steroid_days_supply_yr2
    
    -- Demographics
    , demo.agenbr, demo.gender, demo.ethnicity_code, demo.primarylanguage_desc
    , demo.tenure_yr1, demo.tenure_yr2, demo.post_mnths
    , demo.urbsubr, demo.zip_weight_avg_medinc
    
    -- ACS scores
    , acs.social_risk_score AS acs_social_risk_score
    , acs.sdi_score, acs.svi_score, acs.adi_score
    
    -- CSDI indices
    , csdi.citizenship_index, csdi.education_index, csdi.food_access
    , csdi.health_access, csdi.health_habits, csdi.housing_desert
    , csdi.housing_ownership, csdi.housing_quality, csdi.income_index
    , csdi.income_inequality, csdi.language_score, csdi.natural_disaster
    , csdi.poverty_score, csdi.proactive_health, csdi.racial_diversity
    , csdi.social_isolation, csdi.technology_access, csdi.transport_access
    , csdi.unemployment_index, csdi.water_quality, csdi.disability_score
    , csdi.health_infra, csdi.social_risk_score AS csdi_social_risk_score
    
    -- Preventative care
    , prev.first_prv_dt, prev.last_prv_dt
    , prev.sum_pcp, prev.sum_spec, prev.sum_ob
    , prev.sum_chol_lab, prev.sum_a1c_lab
    , prev.cms_alc_scrn, prev.cms_dep_scrn, prev.cms_t2d_scrn
    , prev.cms_hepb_scrn, prev.cms_hepb_vax
    , prev.cms_ibt_cvd, prev.cms_ibt_obese, prev.cms_flu_vax
    , prev.cms_pneum_vax, prev.cms_hpv_scrn, prev.cms_mam_scrn
    , prev.cms_pap, prev.cms_pelvic, prev.cms_t2d_train
    , prev.cms_prost_cancer_scrn
    -- Additional preventative features (ADDED)
    , prev.cms_annual_wellness, prev.cms_initial_prevent
    , prev.sum_bmi_assess, prev.sum_fall_screen, prev.sum_future_fall_risk
    , prev.sum_flu_vacc, prev.sum_pneum_vacc, prev.sum_tdap_vacc
    , prev.sum_hpv_vacc, prev.sum_rotavirus_vacc
    , prev.sum_colorectal_screen, prev.sum_cervical_screen
    , prev.sum_breast_screen, prev.sum_prostate_screen
    , prev.sum_std_screen, prev.sum_tobacco_cess

FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index` AS st
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ed_yr1` AS ed1
    ON st.asdb_member_key = ed1.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ed_yr2` AS ed2
    ON st.asdb_member_key = ed2.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ip_yr1` AS ip1
    ON st.asdb_member_key = ip1.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_ip_yr2` AS ip2
    ON st.asdb_member_key = ip2.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_op_yr1` AS op1
    ON st.asdb_member_key = op1.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_op_yr2` AS op2
    ON st.asdb_member_key = op2.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_other_cost_utilization_yr1` AS ut1
    ON st.asdb_member_key = ut1.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_other_cost_utilization_yr2` AS ut2
    ON st.asdb_member_key = ut2.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_conditions` AS cond
    ON st.asdb_member_key = cond.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_rx_yr1` AS rx1
    ON st.asdb_member_key = rx1.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_rx_yr2` AS rx2
    ON st.asdb_member_key = rx2.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_demographics` AS demo
    ON st.asdb_member_key = demo.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_acs` AS acs
    ON st.asdb_member_key = acs.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_csdi` AS csdi
    ON st.asdb_member_key = csdi.asdb_member_key
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_preventative_summary` AS prev
    ON st.asdb_member_key = prev.asdb_member_key
;


/*==============================================================================
  STEP 14: JOIN NON-EMBEDDING FEATURES WITH OUTCOME TABLE
  
  Purpose: Create the final table with features joined to outcome labels
           for downstream model training/evaluation.
  
  Output: a964286_Medicaid_nonembed_feature_4_te_experiment_with_outcome
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_with_outcome`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_with_outcome`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"), ("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
SELECT
    -- Outcome labels (from outcome table)
    o.acute_ip_flag AS outcome_acute_ip_flag
    , o.sum_acute_ip_admits AS outcome_sum_acute_ip_admits
    , o.sum_acute_calc_los AS outcome_sum_acute_calc_los
    , o.sum_acute_ip_cost AS outcome_sum_acute_ip_cost
    
    -- All non-embedding features
    , f.*

FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_features` AS f
INNER JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_outcome_ip_4_te_experiment` AS o
    ON f.asdb_member_key = o.individual_id
    AND f.index_dt = o.index_dt
;


/*==============================================================================
  VALIDATION QUERIES
  
  Purpose: Verify data quality and alignment across tables.
  Run these queries after pipeline execution to validate results.
  
==============================================================================*/

-- Query 1: Check member counts across key tables
-- Expected: All counts should match the transformer cohort count
SELECT 
    'Transformer Cohort' AS table_name,
    COUNT(DISTINCT individual_id) AS member_count
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid`

UNION ALL

SELECT 
    'Member Index' AS table_name,
    COUNT(DISTINCT asdb_member_key) AS member_count
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_member_index`

UNION ALL

SELECT 
    'Non-Embedding Features' AS table_name,
    COUNT(DISTINCT asdb_member_key) AS member_count
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_features`

UNION ALL

SELECT 
    'Outcome Table' AS table_name,
    COUNT(DISTINCT individual_id) AS member_count
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_outcome_ip_4_te_experiment`

UNION ALL

SELECT 
    'Final With Outcome' AS table_name,
    COUNT(DISTINCT asdb_member_key) AS member_count
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_with_outcome`
;


-- Query 2: Check outcome distribution
SELECT 
    outcome_acute_ip_flag,
    COUNT(*) AS member_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_with_outcome`
GROUP BY outcome_acute_ip_flag
ORDER BY outcome_acute_ip_flag
;


-- Query 3: Check feature summary statistics
SELECT 
    AVG(agenbr) AS avg_age,
    AVG(sum_ed_visits_yr1) AS avg_ed_visits_yr1,
    AVG(acute_ip_flag_yr1) AS avg_acute_ip_yr1,
    AVG(rx_claim_cnt_yr1) AS avg_rx_claims_yr1,
    AVG(major_chronic_cnt) AS avg_chronic_cnt,
    AVG(sdi_score) AS avg_sdi_score,
    -- Check new cost features (ADDED)
    AVG(sum_ed_cost_yr1) AS avg_ed_cost_yr1,
    AVG(inpatient_cost_yr1) AS avg_inpatient_cost_yr1,
    AVG(emergency_cost_yr1) AS avg_emergency_cost_yr1,
    AVG(outpatient_cost_yr1) AS avg_outpatient_cost_yr1,
    AVG(sum_paid_amt_yr1) AS avg_total_paid_yr1,
    -- Check new ED severity cost features (ADDED)
    AVG(low_sev_ed_cost_yr1) AS avg_low_sev_ed_cost_yr1,
    AVG(high_sev_ed_cost_yr1) AS avg_high_sev_ed_cost_yr1,
    -- Check new preventative features (ADDED)
    AVG(sum_bmi_assess) AS avg_bmi_assess,
    AVG(sum_fall_screen) AS avg_fall_screen,
    AVG(sum_colorectal_screen) AS avg_colorectal_screen,
    AVG(sum_tobacco_cess) AS avg_tobacco_cessation
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_with_outcome`
;


-- Query 4: Check population group distribution with outcome rates
SELECT 
    coa_population_group,
    COUNT(*) AS member_count,
    SUM(outcome_acute_ip_flag) AS members_with_ip,
    ROUND(AVG(outcome_acute_ip_flag) * 100, 2) AS ip_rate_pct
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_with_outcome`
GROUP BY coa_population_group
ORDER BY ip_rate_pct DESC
;


-- Query 5: Verify index_dt alignment (should return 0 mismatches)
SELECT 
    'Index Date Mismatches' AS check_name,
    COUNT(*) AS mismatch_count
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_features` AS f
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_outcome_ip_4_te_experiment` AS o
    ON f.asdb_member_key = o.individual_id
WHERE f.index_dt != o.index_dt
;


-- Query 6: Verify feature column count (expected: ~400+ features with new additions)
SELECT
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS 
     WHERE table_name = 'a964286_Medicaid_nonembed_feature_4_te_experiment_features') AS feature_table_columns,
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS 
     WHERE table_name = 'a964286_Medicaid_nonembed_feature_4_te_experiment_with_outcome') AS final_table_columns
;


-- Query 7: Verify new cost features are populated (should have non-zero values)
SELECT
    'ED Severity Costs' AS feature_group,
    SUM(CASE WHEN low_sev_ed_cost_yr1 > 0 THEN 1 ELSE 0 END) AS members_with_values,
    AVG(low_sev_ed_cost_yr1 + med_sev_ed_cost_yr1 + high_sev_ed_cost_yr1) AS avg_total_sev_cost
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_with_outcome`
UNION ALL
SELECT
    'Utilization Costs' AS feature_group,
    SUM(CASE WHEN inpatient_cost_yr1 + emergency_cost_yr1 + outpatient_cost_yr1 > 0 THEN 1 ELSE 0 END),
    AVG(inpatient_cost_yr1 + emergency_cost_yr1 + outpatient_cost_yr1)
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_with_outcome`
UNION ALL
SELECT
    'New Preventative' AS feature_group,
    SUM(CASE WHEN sum_bmi_assess + sum_fall_screen + sum_colorectal_screen > 0 THEN 1 ELSE 0 END),
    AVG(sum_bmi_assess + sum_fall_screen + sum_colorectal_screen + 0.0)
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_with_outcome`
;


/*==============================================================================
  USAGE IN TRANSFORMER EMBEDDING EVALUATION
  
  After running this SQL pipeline, use the following Python workflow:
  
  ```python
  import pandas as pd
  from google.cloud import bigquery
  
  client = bigquery.Client()
  
  # 1. Load transformer embeddings
  embeddings_df = client.query('''
      SELECT individual_id, * 
      FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_o3_train_ending`
  ''').to_dataframe()
  
  # 2. Load non-embedding features with outcome
  nonembed_df = client.query('''
      SELECT * 
      FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_nonembed_feature_4_te_experiment_with_outcome`
  ''').to_dataframe()
  
  # 3. Merge on individual_id
  merged_df = embeddings_df.merge(
      nonembed_df, 
      left_on='individual_id', 
      right_on='asdb_member_key', 
      how='inner'
  )
  
  # 4. Experiment 1: Embedding Only
  X_embed = model.get_embeddings(merged_df)
  y = merged_df['outcome_acute_ip_flag'].values
  
  # 5. Experiment 2: Embedding + Non-Embedding Features
  X_nonembed = merged_df[nonembed_feature_columns].values
  X_combined = np.hstack([X_embed, X_nonembed])
  
  # 6. Train and evaluate models
  from sklearn.linear_model import LogisticRegression
  from sklearn.model_selection import cross_val_score
  
  # Embedding only
  clf_embed = LogisticRegression()
  scores_embed = cross_val_score(clf_embed, X_embed, y, cv=5, scoring='roc_auc')
  
  # Embedding + Non-Embedding
  clf_combined = LogisticRegression()
  scores_combined = cross_val_score(clf_combined, X_combined, y, cv=5, scoring='roc_auc')
  
  print(f"Embedding Only AUC: {scores_embed.mean():.3f}")
  print(f"Embedding + Non-Embedding AUC: {scores_combined.mean():.3f}")
  ```
  
==============================================================================*/
