/*==============================================================================
  MEDICARE IP OUTCOME GENERATION - EXP ROUND 10 / EXP2B

  Purpose: Generate IP outcome labels (ip6) and enrollment continuity flags
           (mon_6_include) for the exp_round10_exp2b Medicare cohort.

  This script generates OUTCOMES ONLY (no baseline features).
  - Baseline features (52) come from production table:
    anbc-hcb-prod.clin_analytics_hcb_prod.inpatient_me_features_history
  - Embeddings (256 dims) come from:
    edp-prod-storage.edp_ent_sdoheir_cns.a834793_exp_round10_exp2b_medicare_embeddings_20241120_20250930

  Temporal Periods (defined in downstream notebook):
  - Training + Validation + Test: index_dt IN [2024-07-01, 2025-06-30]
  - Out-of-Time (OOT) Test:      index_dt IN [2025-07-01, 2025-09-30]
  - This script covers the FULL range: [2024-07-01, 2025-09-30]

  Outcome Definition (same as production):
  - ip6: Binary flag — 1 if ANY qualifying acute IP admission within
         [index_dt + 1 day, index_dt + 180 days], 0 otherwise
  - Exclusions: Maternity, Major Trauma, Transplant/Non-impactible DRGs
  - mon_6_include: 1 if member had >= 5 monthly enrollment records
                   spanning >= 120 days in the 6-month post-period

  Temporal Alignment (NO DATA LEAKAGE):
  - Features end at:    feature_end_dt = index_dt - 90 days
  - 90-day buffer between features and outcomes
  - Outcomes start at:  index_dt + 1 day (6-month forward window)

  PREREQUISITES:
  1. Embedding table must exist:
     a834793_exp_round10_exp2b_medicare_embeddings_20241120_20250930
  2. Member-id mapping table must exist:
     a834793_Medicare_member_base_memberid
  3. Source tables accessible: MEDICAL_CASE, PRSPCTV_MEMBERSHIP

  TABLES CREATED (6 total):
  1. a834793_Medicare_member_base_exp_round10_exp2b (base cohort)
  2. a834793_Medicare_ip_admissions_post_6mo_exp_round10_exp2b (raw IP)
  3. a834793_Medicare_ip_admissions_filtered_6mo_exp_round10_exp2b (after exclusions)
  4. a834793_Medicare_outcome_6mo_exp_round10_exp2b (aggregated ip6)
  5. a834793_Medicare_post_status_exp_round10_exp2b (continuity flags)
  6. a834793_Medicare_outcome_6mo_final_exp_round10_exp2b (FINAL OUTPUT)

  FINAL OUTPUT TABLE: a834793_Medicare_outcome_6mo_final_exp_round10_exp2b
  Columns: individual_id, member_id, index_dt, ip6, sum_ip6_admits, sum_ip6_los, mon_6_include
  Retention: 180 days
  One row per: (individual_id, index_dt)

  Team: Clinical & Social Determinants Intelligence (CSDI)
  Owner: Pritha Ghosh (pritha.ghosh@cvshealth.com)
  Cost Center: 13070
  Last Updated: April 2026

==============================================================================*/


/*==============================================================================
  STEP 0: BASE COHORT FROM EXP_ROUND10_EXP2B EMBEDDING TABLE

  Purpose: Build base cohort from the new experiment's embedding table,
           filtered to the temporal periods of interest.

  Source: a834793_exp_round10_exp2b_medicare_embeddings_20241120_20250930
  - Contains: individual_id, index_dt, embedding_0 through embedding_255
  - Does NOT contain member_id

  member_id mapping: a834793_Medicare_member_base_memberid
  - Contains: individual_id, member_id, index_dt (1:1 mapping)
  - Created during transformer training pipeline
  - Provides correct member_id for joining to MEDICAL_CASE, PRSPCTV_MEMBERSHIP

  CRITICAL: Must join to base_memberid, NOT INDVDL_CUST_DIST
  - INDVDL_CUST_DIST has multiple member_ids per individual_id (up to 114!)
  - This causes cartesian product and duplicate rows

  Temporal Filter: index_dt BETWEEN 2024-07-01 AND 2025-09-30
  - Covers both training period (2024-07 to 2025-06) and OOT (2025-07 to 2025-09)

==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_exp_round10_exp2b`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_exp_round10_exp2b`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT DISTINCT
    CAST(emb.individual_id AS STRING) AS individual_id
    , base.member_id
    , CAST(emb.index_dt AS DATE) AS index_dt
    , DATE_SUB(CAST(emb.index_dt AS DATE), INTERVAL 90 DAY) AS feature_end_dt
    , 'ME' AS business_ln_cd
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_exp_round10_exp2b_medicare_embeddings_20241120_20250930` emb
INNER JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_memberid` base
    ON CAST(emb.individual_id AS STRING) = CAST(base.individual_id AS STRING)
WHERE CAST(emb.index_dt AS DATE) BETWEEN '2024-07-01' AND '2025-09-30';

-- VERIFICATION: Check base cohort
SELECT
    'STEP 0 - Base Cohort' AS check_name
    , COUNT(*) AS total_rows
    , COUNT(DISTINCT individual_id) AS unique_individuals
    , COUNT(DISTINCT CONCAT(individual_id, '|', CAST(index_dt AS STRING))) AS unique_pairs
    , MIN(index_dt) AS earliest_index_dt
    , MAX(index_dt) AS latest_index_dt
    , SUM(CASE WHEN index_dt BETWEEN '2024-07-01' AND '2025-06-30' THEN 1 ELSE 0 END) AS training_period_count
    , SUM(CASE WHEN index_dt BETWEEN '2025-07-01' AND '2025-09-30' THEN 1 ELSE 0 END) AS oot_period_count
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_exp_round10_exp2b`;

-- Check for duplicates (total_rows should equal unique_pairs)
SELECT
    'STEP 0 - Duplicate Check' AS check_name
    , COUNT(*) AS total_rows
    , COUNT(DISTINCT CONCAT(individual_id, '|', CAST(index_dt AS STRING))) AS unique_pairs
    , COUNT(*) - COUNT(DISTINCT CONCAT(individual_id, '|', CAST(index_dt AS STRING))) AS duplicate_count
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_exp_round10_exp2b`;


/*==============================================================================
  STEP 1: EXTRACT ACUTE IP ADMISSIONS IN POST-PERIOD (6-MONTH WINDOW)

  Purpose: Identify all acute inpatient admissions occurring AFTER index_dt

  Data Source: MEDICAL_CASE table
  - med_cs_ps_ctg_cd = 'I' -> Acute Inpatient (target)
  - med_cs_ps_ctg_cd = 'N' -> Non-Acute Inpatient (excluded)
  - med_cs_ps_ctg_cd = 'E' -> Emergency (excluded)

  Time Window: index_dt + 1 day to index_dt + 180 days (6 months)

  Quality Filters:
  - dummy_mbr_id_ind = 'N' (exclude test/dummy members)
  - med_case_start_dt must be in prediction window

==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_ip_admissions_post_6mo_exp_round10_exp2b`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_ip_admissions_post_6mo_exp_round10_exp2b`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT
    base.individual_id
    , base.member_id
    , base.index_dt
    , mc.medical_case_id
    , CAST(mc.med_case_start_dt AS DATE) AS med_case_start_dt
    , CAST(mc.med_case_stop_dt AS DATE) AS med_case_stop_dt
    , mc.med_cs_ps_ctg_cd
    , mc.los_day_cnt
    , mc.acu_pd_day_cnt
    , mc.birth_outcome_cd
    , mc.delivery_type_cd
    , mc.detain_newborn_cd
    , mc.drg_cd
    , mc.drg_type_cd
    , mc.dschrg_status_cd
    , mc.icd9_dx_cd
    , mc.icd9_dx_group_nbr
    , mc.mdc_cd
    , mc.med_cs_admit_ty_cd
    , mc.prcdr_cd
    , mc.prcdr_group_nbr
    , CASE WHEN mc.med_cs_ps_ctg_cd = 'I' THEN 1 ELSE 0 END AS acute_ip_admit
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_exp_round10_exp2b` base
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEDICAL_CASE` mc
    ON base.member_id = mc.member_id
WHERE mc.dummy_mbr_id_ind = 'N'
    AND mc.med_cs_ps_ctg_cd = 'I'
    AND CAST(mc.med_case_start_dt AS DATE) BETWEEN DATE_ADD(base.index_dt, INTERVAL 1 DAY)
        AND DATE_ADD(base.index_dt, INTERVAL 180 DAY);

-- VERIFICATION: Check IP admissions extracted
SELECT
    'STEP 1 - IP Admissions Extracted' AS check_name
    , COUNT(*) AS total_ip_admissions
    , COUNT(DISTINCT individual_id) AS unique_members_with_ip
    , COUNT(DISTINCT CONCAT(individual_id, '|', CAST(index_dt AS STRING))) AS unique_member_index_pairs
    , MIN(med_case_start_dt) AS earliest_admission
    , MAX(med_case_start_dt) AS latest_admission
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_ip_admissions_post_6mo_exp_round10_exp2b`;


/*==============================================================================
  STEP 2: APPLY EXCLUSION LOGIC (Maternity, Trauma, Transplant)

  Purpose: Filter out non-impactible IP admissions based on clinical criteria

  Exclusions (same as production model):
  1. Maternity/Delivery (birth_outcome_cd, delivery_type_cd, detain_newborn_cd)
     - 'N' = No maternity (KEEP)
     - NULL or '' = No maternity (KEEP)
     - Actual codes like 'L','S','V','C' = EXCLUDE
  2. Major Trauma (mdc_cd = '24' AND admit_ty_cd = '2')
  3. Transplant / Non-impactible (drg_cd in hard-coded exclusion list)

==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_ip_admissions_filtered_6mo_exp_round10_exp2b`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_ip_admissions_filtered_6mo_exp_round10_exp2b`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT *
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_ip_admissions_post_6mo_exp_round10_exp2b`
WHERE
    -- Exclude maternity/delivery cases
    -- 'N' = No maternity (KEEP), NULL = No maternity (KEEP), actual codes = EXCLUDE
    (birth_outcome_cd IS NULL OR TRIM(birth_outcome_cd) IN ('', 'N'))
    AND (delivery_type_cd IS NULL OR TRIM(delivery_type_cd) IN ('', 'N'))
    AND (detain_newborn_cd IS NULL OR TRIM(detain_newborn_cd) IN ('', 'N'))

    -- Exclude major trauma (MDC 24 and admit type 2 = trauma)
    AND NOT (TRIM(mdc_cd) = '24' AND TRIM(med_cs_admit_ty_cd) = '2')

    -- Exclude transplant / non-impactible DRGs (same list as production)
    AND (drg_cd IS NULL OR CAST(drg_cd AS STRING) NOT IN (
        '1', '2', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
        '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '480', '481', '482', '650', '651', '652',
        '653', '654', '655', '656', '657', '658', '659', '660', '661', '662', '663', '664', '665', '666',
        '667', '668', '669', '670', '671', '672', '673', '674', '675', '765', '766', '767', '768', '769',
        '770', '774', '799', '800', '801', '802', '803', '804', '805', '806', '807', '808', '809', '810',
        '811', '812', '813', '814', '815', '816', '817', '818', '819', '820', '821', '822', '823', '824',
        '825', '826', '827', '828', '829', '830', '831', '832', '833', '834', '835', '836', '837', '838',
        '839', '840', '841', '842', '843', '844', '845', '846', '847', '848', '849', '850', '851', '852',
        '853', '854', '855', '856', '857', '858', '876', '877', '878', '879', '880', '881', '882', '883',
        '884', '885', '886', '887', '927', '928', '929', '939', '940', '941', '955', '956', '957', '958',
        '959', '960', '961', '962', '963', '964', '965', '969', '970', '981', '982', '983', '984', '985',
        '986', '987', '988', '989', '998', '999'
    ));

-- VERIFICATION: Check filtered IP admissions (after exclusions)
SELECT
    'STEP 2 - IP Admissions After Exclusions' AS check_name
    , COUNT(*) AS total_filtered_admissions
    , COUNT(DISTINCT individual_id) AS unique_members_with_ip
    , ROUND(COUNT(*) * 100.0 / NULLIF((SELECT COUNT(*) FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_ip_admissions_post_6mo_exp_round10_exp2b`), 0), 2) AS pct_retained
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_ip_admissions_filtered_6mo_exp_round10_exp2b`;


/*==============================================================================
  STEP 3: CREATE MEMBER-LEVEL OUTCOME FLAGS (6-MONTH HORIZON)

  Purpose: Aggregate IP admissions to member level for binary classification

  Output Columns:
  - ip6: Binary flag (1 = ANY acute IP admission, 0 = none)
  - sum_ip6_admits: Total count of admissions in 6-month window
  - sum_ip6_los: Total length of stay (days) in 6-month window

  Logic:
  - LEFT JOIN from base cohort ensures all members are included
  - Members with NO filtered IP admissions -> ip6 = 0, counts = 0

==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_exp_round10_exp2b`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_exp_round10_exp2b`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH aggregated_outcomes AS (
    SELECT
        individual_id
        , member_id
        , index_dt
        , COUNT(DISTINCT medical_case_id) AS sum_ip6_admits
        , SUM(COALESCE(los_day_cnt, 0)) AS sum_ip6_los
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_ip_admissions_filtered_6mo_exp_round10_exp2b`
    GROUP BY individual_id, member_id, index_dt
)
SELECT
    base.individual_id
    , base.member_id
    , base.index_dt
    , CASE
        WHEN agg.sum_ip6_admits IS NULL OR agg.sum_ip6_admits = 0 THEN 0
        ELSE 1
      END AS ip6
    , COALESCE(agg.sum_ip6_admits, 0) AS sum_ip6_admits
    , COALESCE(agg.sum_ip6_los, 0) AS sum_ip6_los
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_exp_round10_exp2b` base
LEFT JOIN aggregated_outcomes agg
    ON base.individual_id = agg.individual_id
    AND base.member_id = agg.member_id
    AND base.index_dt = agg.index_dt;

-- VERIFICATION: Check outcome distribution
SELECT
    'STEP 3 - Outcome Distribution' AS check_name
    , COUNT(*) AS total_members
    , SUM(CASE WHEN ip6 = 1 THEN 1 ELSE 0 END) AS members_with_ip
    , SUM(CASE WHEN ip6 = 0 THEN 1 ELSE 0 END) AS members_without_ip
    , ROUND(SUM(CASE WHEN ip6 = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS ip_rate_pct
    , SUM(sum_ip6_admits) AS total_admissions
    , SUM(sum_ip6_los) AS total_los_days
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_exp_round10_exp2b`;

-- Check distribution by temporal period
SELECT
    'STEP 3 - Outcome by Period' AS check_name
    , CASE
        WHEN index_dt BETWEEN '2024-07-01' AND '2025-06-30' THEN 'Training (2024-07 to 2025-06)'
        WHEN index_dt BETWEEN '2025-07-01' AND '2025-09-30' THEN 'OOT (2025-07 to 2025-09)'
      END AS period
    , COUNT(*) AS total_members
    , SUM(CASE WHEN ip6 = 1 THEN 1 ELSE 0 END) AS members_with_ip
    , ROUND(SUM(CASE WHEN ip6 = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS ip_rate_pct
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_exp_round10_exp2b`
GROUP BY period
ORDER BY period;


/*==============================================================================
  STEP 4: POST-PERIOD MEMBERSHIP CONTINUITY FLAGS

  Purpose: Verify member remained enrolled for full prediction window

  Why This Matters:
  - If member disenrolled before 6 months -> Cannot observe full outcome
  - Use continuity flag to filter evaluation cohort
  - Ensures fair model comparison (same observation window for all)

  Logic (PRSPCTV_MEMBERSHIP has monthly records, not daily):
  - Count distinct eff_dt records in post-period
  - Calculate span: MAX(eff_dt) - MIN(eff_dt) in days
  - mon_6_include = 1 IF:
    * >= 5 monthly enrollment records (out of 6 months)
    * Coverage span >= 120 days (5 monthly records = ~120 day span)
  - mon_6_include = 0: Incomplete enrollment

  Expected: ~93% of members have 6-month continuity

==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_post_status_exp_round10_exp2b`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_post_status_exp_round10_exp2b`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH post_membership AS (
    SELECT
        base.individual_id
        , base.member_id
        , base.index_dt
        , COUNT(DISTINCT mbr.eff_dt) AS num_monthly_records
        , MIN(mbr.eff_dt) AS first_eff_dt
        , MAX(mbr.eff_dt) AS last_eff_dt
        , DATE_DIFF(MAX(mbr.eff_dt), MIN(mbr.eff_dt), DAY) AS coverage_span_days
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_exp_round10_exp2b` base
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PRSPCTV_MEMBERSHIP` mbr
        ON base.member_id = mbr.member_id
        AND mbr.eff_dt BETWEEN DATE_ADD(base.index_dt, INTERVAL 1 DAY)
            AND DATE_ADD(base.index_dt, INTERVAL 180 DAY)
        AND mbr.business_ln_cd LIKE 'ME%'
    GROUP BY base.individual_id, base.member_id, base.index_dt
)
SELECT
    individual_id
    , member_id
    , index_dt
    -- 3-month continuity: >= 3 monthly records AND span >= 60 days
    , CASE WHEN num_monthly_records >= 3 AND coverage_span_days >= 60 THEN 1 ELSE 0 END AS mon_3_include
    -- 6-month continuity: >= 5 monthly records AND span >= 120 days
    , CASE WHEN num_monthly_records >= 5 AND coverage_span_days >= 120 THEN 1 ELSE 0 END AS mon_6_include
    -- 12-month continuity: >= 11 monthly records AND span >= 300 days
    , CASE WHEN num_monthly_records >= 11 AND coverage_span_days >= 300 THEN 1 ELSE 0 END AS mon_12_include
FROM post_membership;

-- VERIFICATION: Check continuity distribution
SELECT
    'STEP 4 - Continuity Distribution' AS check_name
    , COUNT(*) AS total_members
    , SUM(mon_3_include) AS has_3mo_continuity
    , ROUND(SUM(mon_3_include) * 100.0 / COUNT(*), 2) AS pct_3mo
    , SUM(mon_6_include) AS has_6mo_continuity
    , ROUND(SUM(mon_6_include) * 100.0 / COUNT(*), 2) AS pct_6mo
    , SUM(mon_12_include) AS has_12mo_continuity
    , ROUND(SUM(mon_12_include) * 100.0 / COUNT(*), 2) AS pct_12mo
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_post_status_exp_round10_exp2b`;


/*==============================================================================
  STEP 5: FINAL OUTCOME TABLE

  Purpose: Combine outcomes and continuity flags into single output table

  This is the FINAL OUTPUT used as OUTCOMES_TABLE in the downstream notebook:
  dev/downstream/medicare/medicare_ip_model_training_full_downstream_eval_medicare_IP.ipynb

  Key Fields:
  - individual_id, index_dt: Join keys to embedding table and production features
  - member_id: Retained for reference
  - ip6: Binary outcome target (0/1)
  - sum_ip6_admits: Admission count in 6-month window
  - sum_ip6_los: Total LOS in 6-month window
  - mon_6_include: Continuity flag (filter evaluation cohort WHERE mon_6_include = 1)

  Retention: 180 days

==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_final_exp_round10_exp2b`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_final_exp_round10_exp2b`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
SELECT
    outcome.individual_id
    , outcome.member_id
    , outcome.index_dt
    , outcome.ip6
    , outcome.sum_ip6_admits
    , outcome.sum_ip6_los
    , COALESCE(cont.mon_6_include, 0) AS mon_6_include
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_exp_round10_exp2b` outcome
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_post_status_exp_round10_exp2b` cont
    ON outcome.individual_id = cont.individual_id
    AND outcome.member_id = cont.member_id
    AND outcome.index_dt = cont.index_dt;


/*==============================================================================
  VALIDATION QUERIES

  Run these after all steps complete to verify data quality.

==============================================================================*/

-- 1. Final outcome table summary
SELECT
    'FINAL - Overall Summary' AS check_name
    , COUNT(*) AS total_members
    , SUM(CASE WHEN ip6 = 1 THEN 1 ELSE 0 END) AS members_with_ip
    , SUM(CASE WHEN ip6 = 0 THEN 1 ELSE 0 END) AS members_without_ip
    , ROUND(SUM(CASE WHEN ip6 = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS ip_rate_pct
    , SUM(CASE WHEN mon_6_include = 1 THEN 1 ELSE 0 END) AS members_with_full_6mo
    , ROUND(SUM(CASE WHEN mon_6_include = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS continuity_pct
    , SUM(CASE WHEN ip6 = 1 AND mon_6_include = 1 THEN 1 ELSE 0 END) AS evaluable_with_ip
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_final_exp_round10_exp2b`;

-- 2. Outcome distribution by temporal period
SELECT
    'FINAL - By Period' AS check_name
    , CASE
        WHEN index_dt BETWEEN '2024-07-01' AND '2025-06-30' THEN 'Training (2024-07 to 2025-06)'
        WHEN index_dt BETWEEN '2025-07-01' AND '2025-09-30' THEN 'OOT (2025-07 to 2025-09)'
        ELSE 'Other'
      END AS period
    , COUNT(*) AS total_members
    , SUM(CASE WHEN ip6 = 1 THEN 1 ELSE 0 END) AS members_with_ip
    , ROUND(SUM(CASE WHEN ip6 = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS ip_rate_pct
    , SUM(CASE WHEN mon_6_include = 1 THEN 1 ELSE 0 END) AS with_6mo_continuity
    , ROUND(SUM(CASE WHEN mon_6_include = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS continuity_pct
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_final_exp_round10_exp2b`
GROUP BY period
ORDER BY period;

-- 3. Duplicate check (should be 0)
SELECT
    'FINAL - Duplicate Check' AS check_name
    , COUNT(*) AS total_rows
    , COUNT(DISTINCT CONCAT(individual_id, '|', CAST(index_dt AS STRING))) AS unique_pairs
    , COUNT(*) - COUNT(DISTINCT CONCAT(individual_id, '|', CAST(index_dt AS STRING))) AS duplicate_count
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_final_exp_round10_exp2b`;

-- 4. IP rate among evaluable members (mon_6_include = 1) by period
SELECT
    'FINAL - Evaluable IP Rate by Period' AS check_name
    , CASE
        WHEN index_dt BETWEEN '2024-07-01' AND '2025-06-30' THEN 'Training (2024-07 to 2025-06)'
        WHEN index_dt BETWEEN '2025-07-01' AND '2025-09-30' THEN 'OOT (2025-07 to 2025-09)'
        ELSE 'Other'
      END AS period
    , COUNT(*) AS evaluable_members
    , SUM(ip6) AS with_ip
    , ROUND(SUM(ip6) * 100.0 / COUNT(*), 2) AS ip_rate_pct
    , ROUND(AVG(sum_ip6_admits), 3) AS avg_admits_among_evaluable
    , ROUND(AVG(sum_ip6_los), 3) AS avg_los_among_evaluable
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_final_exp_round10_exp2b`
WHERE mon_6_include = 1
GROUP BY period
ORDER BY period;

-- 5. Join check: verify overlap with embedding table
SELECT
    'FINAL - Embedding Join Check' AS check_name
    , COUNT(DISTINCT outcome.individual_id) AS outcome_individuals
    , COUNT(DISTINCT emb.individual_id) AS embedding_individuals
    , COUNT(DISTINCT CASE WHEN outcome.individual_id IS NOT NULL AND emb.individual_id IS NOT NULL
            THEN outcome.individual_id END) AS matched_individuals
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_final_exp_round10_exp2b` outcome
FULL OUTER JOIN (
    SELECT DISTINCT CAST(individual_id AS STRING) AS individual_id
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_exp_round10_exp2b_medicare_embeddings_20241120_20250930`
    WHERE CAST(index_dt AS DATE) BETWEEN '2024-07-01' AND '2025-09-30'
) emb
    ON CAST(outcome.individual_id AS STRING) = CAST(emb.individual_id AS STRING);


/*==============================================================================
  USAGE IN DOWNSTREAM NOTEBOOK

  Set OUTCOMES_TABLE in the notebook to:
  edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_final_exp_round10_exp2b

  The notebook will query:
  SELECT individual_id, index_dt, ip6, mon_6_include
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_final_exp_round10_exp2b`
  WHERE mon_6_include = 1

  Then join to embeddings and production features on (individual_id, index_dt)

==============================================================================*/
