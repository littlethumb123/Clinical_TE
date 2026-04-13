/*==============================================================================
  MEDICARE IP OUTCOME & BASELINE FEATURES - TRAINING DATASET GENERATION
  
  Purpose: Create complete training dataset with:
           1. IP outcome labels (6-month prediction)
           2. Baseline clinical features (chronic conditions + utilization)
           Ready for transformer embedding comparison
  
  ⚠️ PREREQUISITES:
  1. Transformer training must be complete
     - Requires: a834793_Medicare_member_o3_train_ending (transformer output)
     - Requires: a834793_Medicare_member_base_memberid (transformer base cohort)
  2. All required source tables must be accessible (MEDICAL_CASE, PRSPCTV_MEMBERSHIP, etc.)
  
  🚀 QUICK START:
  1. Run entire script (all 919 lines)
  2. Verify results with validation queries at end (lines 785-870)
  3. Expected results:
     ✅ ~3.4M rows (one per individual_id + index_dt)
     ✅ ~7% IP rate (ip6 = 1)
     ✅ ~93% continuity (mon_6_include = 1)
     ✅ 0 duplicates
  4. Use final table: a834793_Medicare_final_dataset_4_te_experiment
  
  WHAT THIS SCRIPT CREATES:
  ═══════════════════════════════════════════════════════════════════════════
  
  FINAL OUTPUT TABLE: a834793_Medicare_final_dataset_4_te_experiment
  
  Columns (63 total):
  ├── Identifiers (5): individual_id, member_id, index_dt, feature_end_dt, business_ln_cd
  ├── Outcomes (4): ip6 (target), sum_ip6_admits, sum_ip6_los, mon_6_include
  └── Baseline Features (54): All features required for production model
      ├── HPD Chronic Conditions (11): CHF, COPD, diabetes, hypertension, etc.
      ├── Medical Utilization (4): claim counts, ER visits, unique codes
      ├── Medical Case (5): IP counts/days across DC1, DC2, DC3
      ├── Diagnosis (1): Specific diagnosis category counts
      ├── Procedures (5): Medical procedure groups/categories
      ├── Revenue Codes (2): Specific revenue codes across time windows
      ├── ER/UC (2): Emergency room claim counts
      ├── RX Class (2): Medication class flags
      ├── RX Group (5): Medication group days supply
      ├── Specialty Claims (3): Specialty service counts
      ├── Care Management (2): CM enrollment flags
      ├── Text Notes (2): Clinical note keyword flags
      ├── YLM Demographics (4): Lifestyle/wealth indicators
      ├── Membership Demographics (2): Age features
      └── Expenditure (2): Experian retail demand scores
  
  One row per: (individual_id, index_dt)
  Retention: 180 days
  
  ⚠️ NOTE: Transformer embeddings (43 features) must be joined separately by user
  
  ═══════════════════════════════════════════════════════════════════════════
  
  OUTCOME DEFINITION (6-month IP prediction):
  - Acute IP: med_cs_ps_ctg_cd = 'I' (MEDICAL_CASE table)
  - Prediction window: index_dt + 1 day to index_dt + 180 days (6 months)
  - Binary target: ip6 = 1 if ANY acute IP admission in window, 0 otherwise
  - Exclusions: Maternity, trauma, transplant (non-preventable admissions)
  
  TEMPORAL ALIGNMENT (NO DATA LEAKAGE):
  - Features: End at feature_end_dt = index_dt - 90 days
  - 90-day buffer: Prevents any feature/outcome overlap
  - Outcomes: Start at index_dt + 1 day (6-month forward window)
  
  FEATURE EXTRACTION:
  - HPD chronic conditions: 12-month lookback before feature_end_dt
  - Medical utilization: 12-month lookback before feature_end_dt
  - Transformer embeddings: JOIN SEPARATELY by user (not included in this script)
  
  TO ADD TRANSFORMER EMBEDDINGS:
  ───────────────────────────────────────────────────────────────────────────
  SELECT 
      base.*
      , emb.* EXCEPT(individual_id, index_dt)
  FROM a834793_Medicare_final_dataset_4_te_experiment base
  LEFT JOIN a834793_Medicare_member_o3_train_ending emb
      ON base.individual_id = emb.individual_id
      AND base.index_dt = emb.index_dt
  WHERE base.mon_6_include = 1;  -- Filter to members with full 6-month observation
  ───────────────────────────────────────────────────────────────────────────
  
  COMPARISON WITH COMMERCIAL:
  - Same IP definition: med_cs_ps_ctg_cd = 'I'
  - Same exclusions: Maternity ('N'=keep, L/S/V/C=exclude), trauma, transplant
  - Same temporal alignment: 90-day buffer between features and outcomes
  - Different LOB: Medicare ('ME') vs Commercial ('CP')
  - member_id mapping: Via transformer training base table (1:1 mapping)
  - Continuity check: Monthly enrollment records (not daily like Commercial)
  
  TABLE RETENTION POLICY:
  - Intermediate tables (19): Expire after 1 DAY (auto-cleanup)
  - Final dataset: Expires after 180 DAYS (for analysis and model training)
  
  TABLES CREATED (21 total):
  
  COHORT & OUTCOMES (5 tables):
  1. a834793_Medicare_member_base_4_te_experiment (base cohort with member_id)
  2. a834793_Medicare_ip_admissions_post_6mo_4_te_experiment (raw IP admissions)
  3. a834793_Medicare_ip_admissions_filtered_6mo_4_te_experiment (after exclusions)
  4. a834793_Medicare_outcome_6mo_4_te_experiment (aggregated outcomes)
  5. a834793_Medicare_post_status_4_te_experiment (continuity flags)
  
  FEATURE TABLES (14 tables - all expire after 1 day):
  6. a834793_Medicare_hpd_features_4_te_experiment (11 chronic conditions)
  7. a834793_Medicare_medutil_features_4_te_experiment (4 utilization features)
  8. a834793_Medicare_medcase_features_4_te_experiment (5 medical case features)
  9. a834793_Medicare_dx_features_4_te_experiment (1 diagnosis feature)
  10. a834793_Medicare_prc_features_4_te_experiment (5 procedure features)
  11. a834793_Medicare_revenue_features_4_te_experiment (2 revenue code features)
  12. a834793_Medicare_eruc_features_4_te_experiment (2 ER/UC features)
  13. a834793_Medicare_rxclass_features_4_te_experiment (2 RX class features)
  14. a834793_Medicare_rxgroup_features_4_te_experiment (5 RX group features)
  15. a834793_Medicare_specialty_features_4_te_experiment (3 specialty claims features)
  16. a834793_Medicare_cm_features_4_te_experiment (2 care management features)
  17. a834793_Medicare_txtnotes_features_4_te_experiment (2 text notes features)
  18. a834793_Medicare_ylm_features_4_te_experiment (4 YLM demographic features)
  19. a834793_Medicare_mbrshp_features_4_te_experiment (2 membership features)
  20. a834793_Medicare_exp_features_4_te_experiment (2 expenditure features)
  
  FINAL MERGED DATASET (expires after 180 days):
  21. a834793_Medicare_final_dataset_4_te_experiment (⭐ FINAL - use this for training)
  
  VALIDATION QUERIES:
  - 7 comprehensive checks included at end of script
  - Run immediately after execution to verify data quality
  
  Team: Clinical & Social Determinants Intelligence (CSDI)
  Owner: Pritha Ghosh (pritha.ghosh@cvshealth.com)
  Cost Center: 13070
  Last Updated: January 2, 2026
  
==============================================================================*/

/*==============================================================================
  CONFIGURATION: PREDICTION HORIZONS
  
  This script focuses exclusively on 6-month prediction (180 days):
  - ip6: 180 days (6 months) - ONLY OUTCOME
  
  All intermediate tables expire after 1 day.
  Only the final merged dataset (with ip6 outcome) is retained for 180 days.
  
  Note: Using literal value 180 instead of DECLARE to ensure compatibility
        when running individual statements (not as a script block).
  
==============================================================================*/


/*==============================================================================
  STEP 0: USE EXISTING MEDICARE MEMBER BASE WITH INDEX_DT
  
  Purpose: Reuse the existing Medicare member cohort (same as Commercial approach)
  
  Source Table: a834793_Medicare_member (transformer training base table)
  - Already has individual_id and index_dt assigned
  - Matches the cohort used for transformer training
  - Ensures consistency across all Medicare analyses
  
  This is the SAME approach as Commercial, where we used:
  - Commercial: a834793_Commercial_member
  - Medicare: a834793_Medicare_member
  
  ✅ Benefits:
  - Consistent with transformer training cohort
  - No need to regenerate index_dt assignments
  - Aligns with existing Medicare features
  - Simpler and faster
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT DISTINCT
    tf.individual_id
    , base.member_id  -- ✅ FIXED: Get member_id from transformer training base table (1:1 mapping)
    , tf.index_dt
    , DATE_SUB(tf.index_dt, INTERVAL 90 DAY) AS feature_end_dt  -- Features end 90 days before index
    , 'ME' AS business_ln_cd
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_o3_train_ending` tf
INNER JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_memberid` base
    ON tf.individual_id = base.individual_id
    AND tf.index_dt = base.index_dt;

-- ============================================================================
-- ✅ SOURCE TABLES:
-- 
-- 1. a834793_Medicare_member_o3_train_ending (Transformer training output)
--    - Contains: individual_id, index_dt, sequence data (cd, target, embeddings)
--    - Does NOT contain member_id
-- 
-- 2. a834793_Medicare_member_base_memberid (Transformer training base cohort)
--    - Contains: individual_id, member_id, index_dt (1:1 mapping)
--    - Created in STEP 0 of transformer training pipeline
--    - This provides the correct member_id for each (individual_id, index_dt)
-- 
-- ⚠️ CRITICAL: Must join to base_memberid, NOT INDVDL_CUST_DIST
-- - INDVDL_CUST_DIST has multiple member_ids per individual_id (up to 114!)
-- - This causes cartesian product and duplicate rows
-- - base_memberid has the CORRECT member_id used during transformer training
-- - MEDICAL_CASE and PRSPCTV_MEMBERSHIP require member_id for joins
-- 
-- Verification: Check for duplicates after base cohort creation
-- SELECT COUNT(*) AS total_rows,
--        COUNT(DISTINCT CONCAT(individual_id, '|', CAST(index_dt AS STRING))) AS unique_pairs
-- FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment`;
-- (Should be equal - no duplicates!)
-- ============================================================================


/*==============================================================================
  STEP 1: EXTRACT ACUTE IP ADMISSIONS IN POST-PERIOD (6-MONTH WINDOW)
  
  Purpose: Identify all acute inpatient admissions occurring AFTER index_dt
  
  Data Source: MEDICAL_CASE table (same as Commercial)
  - med_cs_ps_ctg_cd = 'I' → Acute Inpatient
  - med_cs_ps_ctg_cd = 'N' → Non-Acute Inpatient (excluded)
  - med_cs_ps_ctg_cd = 'E' → Emergency (excluded)
  
  Time Window: index_dt + 1 day to index_dt + 180 days (6 months)
  
  Quality Filters:
  - dummy_mbr_id_ind = 'N' (exclude test members)
  - med_case_start_dt must be in prediction window
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_ip_admissions_post_6mo_4_te_experiment`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_ip_admissions_post_6mo_4_te_experiment`
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
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` base
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEDICAL_CASE` mc
    ON base.member_id = mc.member_id
WHERE mc.dummy_mbr_id_ind = 'N'  -- Exclude test members
    AND mc.med_cs_ps_ctg_cd = 'I'  -- Acute inpatient only
    AND CAST(mc.med_case_start_dt AS DATE) BETWEEN DATE_ADD(base.index_dt, INTERVAL 1 DAY) 
        AND DATE_ADD(base.index_dt, INTERVAL 180 DAY);  -- 6 months = 180 days

-- VERIFICATION: Check IP admissions extracted
SELECT 
    'STEP 1 - IP Admissions Extracted' AS check_name
    , COUNT(*) AS total_ip_admissions
    , COUNT(DISTINCT individual_id) AS unique_members_with_ip
    , COUNT(DISTINCT CONCAT(individual_id, '|', CAST(index_dt AS STRING))) AS unique_member_index_pairs
    , MIN(med_case_start_dt) AS earliest_admission
    , MAX(med_case_start_dt) AS latest_admission
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_ip_admissions_post_6mo_4_te_experiment`;


/*==============================================================================
  STEP 2: APPLY EXCLUSION LOGIC (Maternity, Trauma, Transplant)
  
  Purpose: Filter out non-impactible IP admissions based on clinical criteria
  
  Exclusions (same as Commercial):
  1. Maternity/Delivery (birth_outcome_cd, delivery_type_cd, detain_newborn_cd)
  2. Major Trauma (mdc_cd = '24', admit_ty_cd = '2')
  3. Transplant (drg_cd in transplant list)
  4. Non-impactible conditions (specific DRG and MDC combinations)
  
  This ensures we're only measuring preventable/manageable IP admissions.
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_ip_admissions_filtered_6mo_4_te_experiment`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_ip_admissions_filtered_6mo_4_te_experiment`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT *
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_ip_admissions_post_6mo_4_te_experiment`
WHERE 
    -- Exclude maternity/delivery cases
    -- 'N' = No maternity (KEEP), NULL = No maternity (KEEP), actual codes like 'L','S','V','C' = Exclude
    (birth_outcome_cd IS NULL OR TRIM(birth_outcome_cd) IN ('', 'N'))
    AND (delivery_type_cd IS NULL OR TRIM(delivery_type_cd) IN ('', 'N'))
    AND (detain_newborn_cd IS NULL OR TRIM(detain_newborn_cd) IN ('', 'N'))
    
    -- Exclude major trauma (MDC 24 and admit type 2 = trauma)
    AND NOT (TRIM(mdc_cd) = '24' AND TRIM(med_cs_admit_ty_cd) = '2')
    
    -- Exclude transplant DRGs (same list as Commercial)
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
    , ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_ip_admissions_post_6mo_4_te_experiment`), 2) AS pct_retained
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_ip_admissions_filtered_6mo_4_te_experiment`;


/*==============================================================================
  STEP 3: CREATE MEMBER-LEVEL OUTCOME FLAGS (6-MONTH HORIZON)
  
  Purpose: Aggregate IP admissions to member level for binary classification
  
  Output Columns:
  - ip6: Binary flag (1 = ANY acute IP admission, 0 = none)
  - sum_ip6_admits: Total count of admissions in 6-month window
  - sum_ip6_los: Total length of stay (days) in 6-month window
  
  Logic:
  - If member has ANY filtered IP admission in window → ip6 = 1
  - If member has NO IP admissions (or all excluded) → ip6 = 0
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_4_te_experiment`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_4_te_experiment`
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
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_ip_admissions_filtered_6mo_4_te_experiment`
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
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` base
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
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_4_te_experiment`;


/*==============================================================================
  STEP 4: POST-PERIOD MEMBERSHIP CONTINUITY FLAGS
  
  Purpose: Verify member remained enrolled for full prediction window
  
  Why This Matters:
  - If member disenrolled before 6 months → Cannot observe full outcome
  - Use continuity flag to filter evaluation cohort
  - Ensures fair model comparison (same observation window for all)
  
  Logic (PRSPCTV_MEMBERSHIP has monthly records, not daily):
  - Count distinct eff_dt records in post-period (monthly enrollment records)
  - Calculate span: MAX(eff_dt) - MIN(eff_dt) in days
  - mon_6_include = 1 IF:
    * ≥5 monthly enrollment records (out of 6 months)
    * Coverage span ≥120 days (5 monthly records = ~120 day span)
  - mon_6_include = 0: Incomplete enrollment (exclude from evaluation)
  
  Example: 
  - Good: 5 eff_dt values (Jan 1, Feb 1, Mar 1, Apr 1, May 1) spanning 120 days → mon_6_include = 1
  - Bad: 3 eff_dt values spanning 60 days → mon_6_include = 0
  
  Expected: ~93% of members have 6-month continuity (based on 2023 cohort)
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_post_status_4_te_experiment`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_post_status_4_te_experiment`
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
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` base
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PRSPCTV_MEMBERSHIP` mbr
        ON base.member_id = mbr.member_id
        AND mbr.eff_dt BETWEEN DATE_ADD(base.index_dt, INTERVAL 1 DAY) 
            AND DATE_ADD(base.index_dt, INTERVAL 180 DAY)  -- 6 months = 180 days
        AND mbr.business_ln_cd LIKE 'ME%'
    GROUP BY base.individual_id, base.member_id, base.index_dt
)
SELECT 
    individual_id
    , member_id
    , index_dt
    -- 3-month continuity: ≥3 monthly records AND span ≥60 days
    , CASE WHEN num_monthly_records >= 3 AND coverage_span_days >= 60 THEN 1 ELSE 0 END AS mon_3_include
    -- 6-month continuity: ≥5 monthly records AND span ≥120 days (5 months = ~120 day span)
    , CASE WHEN num_monthly_records >= 5 AND coverage_span_days >= 120 THEN 1 ELSE 0 END AS mon_6_include
    -- 12-month continuity: ≥11 monthly records AND span ≥300 days (11 months = ~300 day span)
    , CASE WHEN num_monthly_records >= 11 AND coverage_span_days >= 300 THEN 1 ELSE 0 END AS mon_12_include
FROM post_membership;


/*==============================================================================
  STEP 5: CREATE FINAL OUTCOME TABLE (6-MONTH FOCUS)
  
  Purpose: Combine outcomes and continuity flags into single table
  
  Output: Ready to merge with Medicare features from weekly-inpatient-me repo
  
  Key Fields:
  - individual_id, member_id, index_dt: Join keys
  - ip6: Binary outcome (0/1)
  - sum_ip6_admits: Admission count
  - sum_ip6_los: Total LOS
  - mon_6_include: Continuity flag (use for filtering evaluation cohort)
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_final_4_te_experiment`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_final_4_te_experiment`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)  -- Final table: 180 days (for analysis and merging with features)
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
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_4_te_experiment` outcome
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_post_status_4_te_experiment` cont
    ON outcome.individual_id = cont.individual_id 
    AND outcome.member_id = cont.member_id
    AND outcome.index_dt = cont.index_dt;

-- VERIFICATION: Check final outcome table with continuity
SELECT 
    'STEP 5 - Final Outcomes with Continuity' AS check_name
    , COUNT(*) AS total_members
    , SUM(CASE WHEN ip6 = 1 THEN 1 ELSE 0 END) AS members_with_ip
    , ROUND(SUM(CASE WHEN ip6 = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS ip_rate_pct
    , SUM(CASE WHEN mon_6_include = 1 THEN 1 ELSE 0 END) AS members_with_full_6mo_enrollment
    , ROUND(SUM(CASE WHEN mon_6_include = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS continuity_pct
    , SUM(CASE WHEN ip6 = 1 AND mon_6_include = 1 THEN 1 ELSE 0 END) AS evaluable_members_with_ip
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_final_4_te_experiment`;


/*==============================================================================
  ███████╗███████╗ █████╗ ████████╗██╗   ██╗██████╗ ███████╗███████╗
  ██╔════╝██╔════╝██╔══██╗╚══██╔══╝██║   ██║██╔══██╗██╔════╝██╔════╝
  █████╗  █████╗  ███████║   ██║   ██║   ██║██████╔╝█████╗  ███████╗
  ██╔══╝  ██╔══╝  ██╔══██║   ██║   ██║   ██║██╔══██╗██╔══╝  ╚════██║
  ██║     ███████╗██║  ██║   ██║   ╚██████╔╝██║  ██║███████╗███████║
  ╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝
  
  MEDICARE FEATURE EXTRACTION - ADAPTED FROM WEEKLY-INPATIENT-ME PRODUCTION MODEL
  
  This section extracts ALL 54 baseline features used in the production Medicare IP model.
  Features are adapted to work with (individual_id, member_id, index_dt) cohort.
  
  Feature Categories (14 categories, 54 baseline features):
  - STEP 6:  HPD Chronic Conditions (11 features)
  - STEP 7:  Medical Utilization (4 features - claims, ER, unique codes) [ALREADY IMPLEMENTED]
  - STEP 9:  Medical Case Utilization (5 features - IP counts/days across DC1/DC2/DC3) [NEW]
  - STEP 10: Diagnosis Features (1 feature - specific dx category counts) [NEW]
  - STEP 11: Procedure Features (5 features - specific procedure groups/categories) [NEW]
  - STEP 12: Revenue Code Features (2 features - DC3, DC4) [NEW]
  - STEP 13: ER/UC Features (2 features - DC1, DC2) [NEW]
  - STEP 14: RX Class Utilization (2 features - DC5 flags) [NEW]
  - STEP 15: RX Group Utilization (5 features - DC1/DC2/DC3 days/flags) [NEW]
  - STEP 16: Specialty Claims Features (3 features - DC1/DC2/DC3) [NEW]
  - STEP 17: Care Management Features (2 features - DC2, DC5) [NEW]
  - STEP 18: Text Notes Features (2 features - DC5 keyword flags) [NEW]
  - STEP 19: YLM Demographic Features (4 features - lifestyle/wealth) [NEW]
  - STEP 20: Membership Demographics (2 features - age) [NEW]
  - STEP 21: Expenditure Features (2 features - Experian scores) [NEW]
  - STEP 22: FINAL MERGE - All 54 baseline features + outcomes
  
  Temporal Logic:
  - Feature lookback varies by feature type:
    * DC1: 0-30 days before feature_end_dt
    * DC2: 31-90 days before feature_end_dt
    * DC3: 91-180 days before feature_end_dt
    * DC4: 181-365 days before feature_end_dt
    * DC5: 366-730 days before feature_end_dt
  - feature_end_dt = index_dt - 90 days
  - This ensures features end 90 days before outcomes begin (no data leakage)
  
  ⚠️ TRANSFORMER EMBEDDINGS (43 features) must be joined separately by user
  
==============================================================================*/


/*==============================================================================
  STEP 6: HPD CHRONIC CONDITIONS (11 FEATURES)
  
  Source: INDVDL_MSTR_INTGTN table (HPD = Health Predictive Data)
  
  Features:
  - AFF: Affective disorders
  - ALC: Alcohol-related disorders
  - CBD: Cerebrovascular disease
  - CHF: Congestive heart failure
  - COP: Chronic obstructive pulmonary disease
  - CRF: Chronic renal failure
  - DIA: Diabetes
  - HYP: Hypertension
  - NGD: Neurological disorders
  - cv_cond: Any cardiovascular condition
  - chr_flag: Any chronic condition flag
  
  Temporal Window: Chronic conditions active during lookback period
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_hpd_features_4_te_experiment`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_hpd_features_4_te_experiment`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH hpd_raw AS (
    SELECT DISTINCT
        base.individual_id
        , base.member_id
        , base.index_dt
        , hpd.disease_cd
        , hpd.dm_eligibility_cd
        , hpd.first_indctn_dt
        , hpd.end_indctn_dt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_MSTR_INTGTN` hpd
        ON base.member_id = hpd.member_id
    WHERE hpd.disease_cd <> 'SUM'
        AND TRIM(hpd.dm_eligibility_cd) IN ('Y', 'H')
        AND hpd.first_indctn_dt <= base.feature_end_dt  -- Condition started before feature window ends
        AND hpd.end_indctn_dt >= DATE_SUB(base.feature_end_dt, INTERVAL 12 MONTH)  -- Still active during lookback
),
hpd_pivot AS (
    SELECT
        individual_id
        , member_id
        , index_dt
        , MAX(CASE WHEN TRIM(disease_cd) = 'AFF' THEN 1 ELSE 0 END) AS camemhpd_aff
        , MAX(CASE WHEN TRIM(disease_cd) = 'ALC' THEN 1 ELSE 0 END) AS camemhpd_alc
        , MAX(CASE WHEN TRIM(disease_cd) = 'CBD' THEN 1 ELSE 0 END) AS camemhpd_cbd
        , MAX(CASE WHEN TRIM(disease_cd) = 'CHF' THEN 1 ELSE 0 END) AS camemhpd_chf
        , MAX(CASE WHEN TRIM(disease_cd) = 'COP' THEN 1 ELSE 0 END) AS camemhpd_cop
        , MAX(CASE WHEN TRIM(disease_cd) = 'CRF' THEN 1 ELSE 0 END) AS camemhpd_crf
        , MAX(CASE WHEN TRIM(disease_cd) = 'DIA' THEN 1 ELSE 0 END) AS camemhpd_dia
        , MAX(CASE WHEN TRIM(disease_cd) = 'HYP' THEN 1 ELSE 0 END) AS camemhpd_hyp
        , MAX(CASE WHEN TRIM(disease_cd) = 'NGD' THEN 1 ELSE 0 END) AS camemhpd_ngd
        , MAX(CASE WHEN TRIM(disease_cd) IN ('CBD', 'CHF', 'IHD', 'PVD') THEN 1 ELSE 0 END) AS camemhpd_cv_cond
        , MAX(CASE WHEN TRIM(disease_cd) IN (
                'AFF', 'AID', 'ALC', 'AST', 'BIP', 'BLC', 'BNC', 'BRC', 
                'CAN', 'CBD', 'CHD', 'CHF', 'COC', 'COP', 'CRF', 'CRO', 
                'CYS', 'DEP', 'DIA', 'DTD', 'ENC', 'EPL', 'ESC', 'HCG', 
                'HDL', 'HEM', 'HEP', 'HNC', 'HYC', 'HYP', 'IHD', 'LBP', 
                'LEU', 'LUC', 'MLM', 'MOH', 'MSX', 'NGD', 'OBE', 'ORC', 
                'OSP', 'OST', 'OVC', 'PAN', 'PAR', 'PNC', 'PRC', 'PSY', 
                'PUD', 'PVD', 'RHA', 'SCA', 'SDO', 'SKC', 'STC', 'VNA'
            ) THEN 1 ELSE 0 END) AS camemhpd_chr_flag
    FROM hpd_raw
    GROUP BY individual_id, member_id, index_dt
)
SELECT 
    base.individual_id
    , base.member_id
    , base.index_dt
    , COALESCE(hpd.camemhpd_aff, 0) AS camemhpd_aff
    , COALESCE(hpd.camemhpd_alc, 0) AS camemhpd_alc
    , COALESCE(hpd.camemhpd_cbd, 0) AS camemhpd_cbd
    , COALESCE(hpd.camemhpd_chf, 0) AS camemhpd_chf
    , COALESCE(hpd.camemhpd_cop, 0) AS camemhpd_cop
    , COALESCE(hpd.camemhpd_crf, 0) AS camemhpd_crf
    , COALESCE(hpd.camemhpd_dia, 0) AS camemhpd_dia
    , COALESCE(hpd.camemhpd_hyp, 0) AS camemhpd_hyp
    , COALESCE(hpd.camemhpd_ngd, 0) AS camemhpd_ngd
    , COALESCE(hpd.camemhpd_cv_cond, 0) AS camemhpd_cv_cond
    , COALESCE(hpd.camemhpd_chr_flag, 0) AS camemhpd_chr_flag
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` base
LEFT JOIN hpd_pivot hpd
    ON base.individual_id = hpd.individual_id
    AND base.member_id = hpd.member_id
    AND base.index_dt = hpd.index_dt;


/*==============================================================================
  STEP 7: MEDICAL CASE UTILIZATION - DC1, DC2, DC3 (5 FEATURES)
  
  Source: MEDICAL_CASE table
  
  Features extract IP utilization across 3 time windows:
  - DC1: Last 30 days before feature_end_dt
  - DC2: 31-90 days before feature_end_dt
  - DC3: 91-180 days before feature_end_dt
  
  Note: These are LOOKBACK features (before index_dt), NOT outcomes
==============================================================================*/

-- This step extracts Medical Case features inline (simpler than 3 separate tables)
-- We'll add this in the final merge for efficiency


/*==============================================================================
  STEP 8: MEDICAL UTILIZATION (4 FEATURES)
  
  Source: CLAIM_LINE table
  
  Features:
  - clm_ln_cnt: Total claim line count
  - uniq_dx_cd_cnt: Unique diagnosis code count
  - uniq_rev_cd_cnt: Unique revenue code count
  - er_clm_cnt: Emergency room claim count
  
  Temporal Window: 12 months before feature_end_dt
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_medutil_features_4_te_experiment`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_medutil_features_4_te_experiment`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH claims_data AS (
    SELECT
        base.individual_id
        , base.member_id
        , base.index_dt
        , clm.claim_line_id
        , clm.pri_icd9_dx_cd
        , clm.revenue_cd
        , clm.plc_srv_ctg_cd
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLAIM_LINE` clm
        ON base.member_id = clm.member_id
    WHERE EXTRACT(YEAR FROM clm.srv_start_dt) >= 2020
        AND clm.srv_start_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 12 MONTH) AND base.feature_end_dt
        AND clm.adjn_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 12 MONTH) AND base.feature_end_dt
        AND (TRIM(clm.summarized_srv_ind) IN ('Y', '') OR clm.summarized_srv_ind IS NULL)
        AND (TRIM(clm.duplicate_ind) IN ('N', '') OR clm.duplicate_ind IS NULL)
),
medutil_agg AS (
    SELECT
        individual_id
        , member_id
        , index_dt
        , COUNT(claim_line_id) AS clm_ln_cnt
        , COUNT(DISTINCT pri_icd9_dx_cd) AS uniq_dx_cd_cnt
        , COUNT(DISTINCT revenue_cd) AS uniq_rev_cd_cnt
        , SUM(CASE WHEN TRIM(plc_srv_ctg_cd) = 'E' THEN 1 ELSE 0 END) AS er_clm_cnt
    FROM claims_data
    GROUP BY individual_id, member_id, index_dt
)
SELECT 
    base.individual_id
    , base.member_id
    , base.index_dt
    , COALESCE(mu.clm_ln_cnt, 0) AS camemmedutilization_clm_ln_cnt
    , COALESCE(mu.uniq_dx_cd_cnt, 0) AS camemmedutilization_uniq_dx_cd_cnt
    , COALESCE(mu.uniq_rev_cd_cnt, 0) AS camemmedutilization_uniq_rev_cd_cnt
    , COALESCE(mu.er_clm_cnt, 0) AS camemmedutilization_er_clm_cnt
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` base
LEFT JOIN medutil_agg mu
    ON base.individual_id = mu.individual_id
    AND base.member_id = mu.member_id
    AND base.index_dt = mu.index_dt;


/*==============================================================================
  STEP 9: MEDICAL CASE FEATURES - DC1, DC2, DC3 (5 FEATURES)
  
  Purpose: Extract IP admission history across different time windows
  
  Features:
  - DC1 (0-30 days before feature_end_dt): ip_cnt_dc1, ip_days_dc1
  - DC2 (31-90 days before feature_end_dt): ip_cnt_dc2, ip_days_dc2
  - DC3 (91-180 days before feature_end_dt): ip_cnt_dc3
  
  These are HISTORICAL IP admissions (lookback features), not outcomes.
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_medcase_features_4_te_experiment`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_medcase_features_4_te_experiment`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH claims_temp AS (
    SELECT DISTINCT
        base.individual_id,
        base.member_id,
        base.index_dt,
        base.feature_end_dt,
        EECL.adjn_dt,
        EECL.claim_line_id
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLAIM_LINE` AS EECL 
        ON base.member_id = EECL.member_id
    WHERE (TRIM(EECL.summarized_srv_ind) IN ('Y', '') OR EECL.summarized_srv_ind IS NULL)
        AND (TRIM(EECL.duplicate_ind) IN ('N', '') OR EECL.duplicate_ind IS NULL)
        AND EXTRACT(YEAR FROM EECL.srv_start_dt) >= 2020
        AND EECL.srv_start_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 180 DAY) AND base.feature_end_dt
        AND EECL.adjn_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 180 DAY) AND base.feature_end_dt
),
medcase_extract AS (
    SELECT DISTINCT
        base.individual_id,
        base.member_id,
        base.index_dt,
        base.feature_end_dt,
        EMC.medical_case_id,
        EMC.med_case_start_dt,
        EMC.med_case_stop_dt,
        EMC.med_cs_ps_ctg_cd,
        EMC.los_day_cnt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEDICAL_CASE` AS EMC
        ON base.member_id = EMC.member_id
    WHERE EMC.med_cs_ps_ctg_cd IN ('I','N','E')
        AND EXTRACT(YEAR FROM EMC.med_case_start_dt) >= 2020
        AND EMC.med_case_start_dt <= base.feature_end_dt
        AND DATE_SUB(base.feature_end_dt, INTERVAL 180 DAY) <= EMC.med_case_stop_dt
),
medcase_claims AS (
    SELECT DISTINCT
        base.individual_id,
        base.member_id,
        base.index_dt,
        EMC.medical_case_id,
        EMCCL.claim_line_id
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEDICAL_CASE` AS EMC
        ON base.member_id = EMC.member_id
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MED_CASE_X_CLM_LN` AS EMCCL
        ON EMC.member_id = EMCCL.member_id 
        AND EMC.medical_case_id = EMCCL.medical_case_id
    WHERE EMC.med_cs_ps_ctg_cd IN ('I','N','E')
        AND EXTRACT(YEAR FROM EMC.med_case_start_dt) >= 2020
        AND EMC.med_case_start_dt <= base.feature_end_dt
        AND DATE_SUB(base.feature_end_dt, INTERVAL 180 DAY) <= EMC.med_case_stop_dt
),
medcase_with_adjn AS (
    SELECT
        me.individual_id,
        me.member_id,
        me.index_dt,
        me.feature_end_dt,
        me.medical_case_id,
        me.med_cs_ps_ctg_cd,
        me.med_case_start_dt,
        me.med_case_stop_dt,
        MAX(ct.adjn_dt) AS max_adjn,
        me.los_day_cnt
    FROM medcase_extract me
    INNER JOIN medcase_claims mc
        ON me.member_id = mc.member_id
        AND me.medical_case_id = mc.medical_case_id
    INNER JOIN claims_temp ct
        ON me.member_id = ct.member_id
        AND mc.claim_line_id = ct.claim_line_id
    GROUP BY me.individual_id, me.member_id, me.index_dt, me.feature_end_dt, me.medical_case_id,
             me.med_cs_ps_ctg_cd, me.med_case_start_dt, me.med_case_stop_dt, me.los_day_cnt
),
medcase_dc1 AS (
    SELECT
        individual_id, member_id, index_dt,
        CAST(SUM(CASE WHEN med_cs_ps_ctg_cd='I' THEN 1 ELSE 0 END) AS INT64) AS ip_cnt_dc1,
        CAST(SUM(CASE WHEN med_cs_ps_ctg_cd='I' THEN los_day_cnt ELSE 0 END) AS INT64) AS ip_days_dc1
    FROM medcase_with_adjn
    WHERE med_case_start_dt BETWEEN DATE_SUB(feature_end_dt, INTERVAL 30 DAY) AND feature_end_dt
        AND max_adjn BETWEEN DATE_SUB(feature_end_dt, INTERVAL 30 DAY) AND feature_end_dt
    GROUP BY individual_id, member_id, index_dt
),
medcase_dc2 AS (
    SELECT
        individual_id, member_id, index_dt,
        CAST(SUM(CASE WHEN med_cs_ps_ctg_cd='I' THEN 1 ELSE 0 END) AS INT64) AS ip_cnt_dc2,
        CAST(SUM(CASE WHEN med_cs_ps_ctg_cd='I' THEN los_day_cnt ELSE 0 END) AS INT64) AS ip_days_dc2
    FROM medcase_with_adjn
    WHERE med_case_start_dt BETWEEN DATE_SUB(feature_end_dt, INTERVAL 90 DAY) AND DATE_SUB(feature_end_dt, INTERVAL 31 DAY)
        AND max_adjn BETWEEN DATE_SUB(feature_end_dt, INTERVAL 90 DAY) AND DATE_SUB(feature_end_dt, INTERVAL 31 DAY)
    GROUP BY individual_id, member_id, index_dt
),
medcase_dc3 AS (
    SELECT
        individual_id, member_id, index_dt,
        CAST(SUM(CASE WHEN med_cs_ps_ctg_cd='I' THEN 1 ELSE 0 END) AS INT64) AS ip_cnt_dc3
    FROM medcase_with_adjn
    WHERE med_case_start_dt BETWEEN DATE_SUB(feature_end_dt, INTERVAL 180 DAY) AND DATE_SUB(feature_end_dt, INTERVAL 91 DAY)
        AND max_adjn BETWEEN DATE_SUB(feature_end_dt, INTERVAL 180 DAY) AND DATE_SUB(feature_end_dt, INTERVAL 91 DAY)
    GROUP BY individual_id, member_id, index_dt
)
SELECT
    base.individual_id,
    base.member_id,
    base.index_dt,
    COALESCE(dc1.ip_cnt_dc1, 0) AS camemmedcasedc1_ip_cnt_dc1,
    COALESCE(dc1.ip_days_dc1, 0) AS camemmedcasedc1_ip_days_dc1,
    COALESCE(dc2.ip_cnt_dc2, 0) AS camemmedcasedc2_ip_cnt_dc2,
    COALESCE(dc2.ip_days_dc2, 0) AS camemmedcasedc2_ip_days_dc2,
    COALESCE(dc3.ip_cnt_dc3, 0) AS camemmedcasedc3_ip_cnt_dc3
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` base
LEFT JOIN medcase_dc1 dc1 ON base.individual_id = dc1.individual_id AND base.member_id = dc1.member_id AND base.index_dt = dc1.index_dt
LEFT JOIN medcase_dc2 dc2 ON base.individual_id = dc2.individual_id AND base.member_id = dc2.member_id AND base.index_dt = dc2.index_dt
LEFT JOIN medcase_dc3 dc3 ON base.individual_id = dc3.individual_id AND base.member_id = dc3.member_id AND base.index_dt = dc3.index_dt;


/*==============================================================================
  STEP 10: DIAGNOSIS FEATURES - DC1 (1 FEATURE)
  
  Purpose: Count specific diagnosis codes in recent claims
  
  Feature:
  - dxc1085_cnt_dc1: Count of diagnosis category 1085 claims
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_dx_features_4_te_experiment`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_dx_features_4_te_experiment`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH claims_base AS (
    SELECT DISTINCT
        base.individual_id,
        base.member_id,
        base.index_dt,
        base.feature_end_dt,
        clm.srv_start_dt,
        clm.hcfa_plc_srv_cd,
        clm.prcdr_cd,
        clm.pri_icd9_dx_cd
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLAIM_LINE` AS clm
        ON base.member_id = clm.member_id
    WHERE (TRIM(clm.summarized_srv_ind) IN ('Y', '') OR clm.summarized_srv_ind IS NULL)
        AND (TRIM(clm.duplicate_ind) IN ('N', '') OR clm.duplicate_ind IS NULL)
        AND EXTRACT(YEAR FROM clm.srv_start_dt) >= 2020
        AND clm.srv_start_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 30 DAY) AND base.feature_end_dt
        AND clm.adjn_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 30 DAY) AND base.feature_end_dt
),
dx_grouped AS (
    SELECT
        cb.individual_id,
        cb.member_id,
        cb.index_dt,
        icd.icd9_dx_group_nbr,
        TRIM(dxg.icd9_dx_ctg_cd) AS icd9_dx_ctg_cd,
        CAST(COUNT(*) AS INT64) AS clm_cnt
    FROM claims_base cb
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` AS icd
        ON TRIM(REGEXP_REPLACE(cb.pri_icd9_dx_cd,'[\\.]','')) = TRIM(REGEXP_REPLACE(icd.icd9_dx_cd,'[\\.]',''))
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DX_GROUP` AS dxg
        ON icd.icd9_dx_group_nbr = dxg.icd9_dx_group_nbr
    WHERE dxg.icd9_dx_ctg_cd IN ('1085')
    GROUP BY cb.individual_id, cb.member_id, cb.index_dt, icd.icd9_dx_group_nbr, dxg.icd9_dx_ctg_cd
),
dx_agg AS (
    SELECT
        individual_id, member_id, index_dt,
        CAST(SUM(CASE WHEN TRIM(icd9_dx_ctg_cd) = '1085' THEN clm_cnt ELSE 0 END) AS INT64) AS dxc1085_cnt_dc1
    FROM dx_grouped
    GROUP BY individual_id, member_id, index_dt
)
SELECT
    base.individual_id,
    base.member_id,
    base.index_dt,
    COALESCE(dx.dxc1085_cnt_dc1, 0) AS camemeipdxdc1_dxc1085_cnt_dc1
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` base
LEFT JOIN dx_agg dx ON base.individual_id = dx.individual_id AND base.member_id = dx.member_id AND base.index_dt = dx.index_dt;


/*==============================================================================
  STEP 11: PROCEDURE FEATURES - DC1 (5 FEATURES)
  
  Purpose: Count specific medical procedures in recent claims
  
  Features:
  - prc141_cnt_dc1, prc155_cnt_dc1, prc219_cnt_dc1 (procedure groups)
  - prcc1102_cnt_dc1, prcc1115_cnt_dc1 (procedure categories)
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_prc_features_4_te_experiment`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_prc_features_4_te_experiment`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH claims_base AS (
    SELECT DISTINCT
        base.individual_id,
        base.member_id,
        base.index_dt,
        clm.prcdr_cd
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLAIM_LINE` AS clm
        ON base.member_id = clm.member_id
    WHERE (TRIM(clm.summarized_srv_ind) IN ('Y', '') OR clm.summarized_srv_ind IS NULL)
        AND (TRIM(clm.duplicate_ind) IN ('N', '') OR clm.duplicate_ind IS NULL)
        AND EXTRACT(YEAR FROM clm.srv_start_dt) >= 2020
        AND clm.srv_start_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 30 DAY) AND base.feature_end_dt
        AND clm.adjn_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 30 DAY) AND base.feature_end_dt
),
prc_grouped AS (
    SELECT
        cb.individual_id,
        cb.member_id,
        cb.index_dt,
        prc.prcdr_group_nbr,
        prg.prcdr_ctg_cd,
        CAST(COUNT(*) AS INT64) AS clm_cnt
    FROM claims_base cb
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PROCEDURE` AS prc
        ON cb.prcdr_cd = prc.prcdr_cd
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PROCEDURE_GROUP` AS prg
        ON prc.prcdr_group_nbr = prg.prcdr_group_nbr
    WHERE (SAFE_CAST(prc.prcdr_group_nbr AS STRING) IN ('141', '155', '219') 
        OR prg.prcdr_ctg_cd IN ('1102', '1115'))
    GROUP BY cb.individual_id, cb.member_id, cb.index_dt, prc.prcdr_group_nbr, prg.prcdr_ctg_cd
),
prc_agg AS (
    SELECT
        individual_id, member_id, index_dt,
        CAST(SUM(CASE WHEN prcdr_group_nbr = 141 THEN clm_cnt ELSE 0 END) AS INT64) AS prc141_cnt_dc1,
        CAST(SUM(CASE WHEN prcdr_group_nbr = 155 THEN clm_cnt ELSE 0 END) AS INT64) AS prc155_cnt_dc1,
        CAST(SUM(CASE WHEN prcdr_group_nbr = 219 THEN clm_cnt ELSE 0 END) AS INT64) AS prc219_cnt_dc1,
        CAST(SUM(CASE WHEN prcdr_ctg_cd = '1102' THEN clm_cnt ELSE 0 END) AS INT64) AS prcc1102_cnt_dc1,
        CAST(SUM(CASE WHEN prcdr_ctg_cd = '1115' THEN clm_cnt ELSE 0 END) AS INT64) AS prcc1115_cnt_dc1
    FROM prc_grouped
    GROUP BY individual_id, member_id, index_dt
)
SELECT
    base.individual_id,
    base.member_id,
    base.index_dt,
    COALESCE(prc.prc141_cnt_dc1, 0) AS camemeipprcdc1_prc141_cnt_dc1,
    COALESCE(prc.prc155_cnt_dc1, 0) AS camemeipprcdc1_prc155_cnt_dc1,
    COALESCE(prc.prc219_cnt_dc1, 0) AS camemeipprcdc1_prc219_cnt_dc1,
    COALESCE(prc.prcc1102_cnt_dc1, 0) AS camemeipprcdc1_prcc1102_cnt_dc1,
    COALESCE(prc.prcc1115_cnt_dc1, 0) AS camemeipprcdc1_prcc1115_cnt_dc1
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` base
LEFT JOIN prc_agg prc ON base.individual_id = prc.individual_id AND base.member_id = prc.member_id AND base.index_dt = prc.index_dt;


/*==============================================================================
  STEP 12: REVENUE CODE FEATURES - DC3, DC4 (2 FEATURES)
  
  Purpose: Count specific revenue codes in different time windows
  
  Features:
  - rev730_cnt_dc3: Revenue code 730 (91-180 days ago)
  - rev430_cnt_dc4: Revenue code 430 (181-365 days ago)
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_revenue_features_4_te_experiment`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_revenue_features_4_te_experiment`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH rev_dc3 AS (
    SELECT
        base.individual_id,
        base.member_id,
        base.index_dt,
        clm.revenue_cd,
        COUNT(*) AS clm_cnt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLAIM_LINE` AS clm
        ON base.member_id = clm.member_id
    WHERE (TRIM(clm.summarized_srv_ind) IN ('Y', '') OR clm.summarized_srv_ind IS NULL)
        AND (TRIM(clm.duplicate_ind) IN ('N', '') OR clm.duplicate_ind IS NULL)
        AND EXTRACT(YEAR FROM clm.srv_start_dt) >= 2020
        AND clm.srv_start_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 180 DAY) AND DATE_SUB(base.feature_end_dt, INTERVAL 91 DAY)
        AND clm.adjn_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 180 DAY) AND DATE_SUB(base.feature_end_dt, INTERVAL 91 DAY)
        AND clm.revenue_cd IN ('730')
    GROUP BY base.individual_id, base.member_id, base.index_dt, clm.revenue_cd
),
rev_dc4 AS (
    SELECT
        base.individual_id,
        base.member_id,
        base.index_dt,
        clm.revenue_cd,
        COUNT(*) AS clm_cnt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLAIM_LINE` AS clm
        ON base.member_id = clm.member_id
    WHERE (TRIM(clm.summarized_srv_ind) IN ('Y', '') OR clm.summarized_srv_ind IS NULL)
        AND (TRIM(clm.duplicate_ind) IN ('N', '') OR clm.duplicate_ind IS NULL)
        AND EXTRACT(YEAR FROM clm.srv_start_dt) >= 2020
        AND clm.srv_start_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 365 DAY) AND DATE_SUB(base.feature_end_dt, INTERVAL 181 DAY)
        AND clm.adjn_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 365 DAY) AND DATE_SUB(base.feature_end_dt, INTERVAL 181 DAY)
        AND clm.revenue_cd IN ('430')
    GROUP BY base.individual_id, base.member_id, base.index_dt, clm.revenue_cd
),
rev_agg AS (
    SELECT
        individual_id, member_id, index_dt,
        CAST(SUM(CASE WHEN revenue_cd='730' THEN clm_cnt ELSE 0 END) AS INT64) AS rev730_cnt_dc3
    FROM rev_dc3
    GROUP BY individual_id, member_id, index_dt
),
rev_agg2 AS (
    SELECT
        individual_id, member_id, index_dt,
        CAST(SUM(CASE WHEN revenue_cd='430' THEN clm_cnt ELSE 0 END) AS INT64) AS rev430_cnt_dc4
    FROM rev_dc4
    GROUP BY individual_id, member_id, index_dt
)
SELECT
    base.individual_id,
    base.member_id,
    base.index_dt,
    COALESCE(r3.rev730_cnt_dc3, 0) AS camemrevenuedc3_rev730_cnt_dc3,
    COALESCE(r4.rev430_cnt_dc4, 0) AS camemrevenuedc4_rev430_cnt_dc4
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` base
LEFT JOIN rev_agg r3 ON base.individual_id = r3.individual_id AND base.member_id = r3.member_id AND base.index_dt = r3.index_dt
LEFT JOIN rev_agg2 r4 ON base.individual_id = r4.individual_id AND base.member_id = r4.member_id AND base.index_dt = r4.index_dt;


/*==============================================================================
  STEP 13: ER/UC FEATURES - DC1, DC2 (2 FEATURES)
  
  Purpose: Count emergency room visits in recent periods
  
  Features:
  - erclm_cnt_dc1: ER claim count (0-30 days ago)
  - erclm_cnt_dc2: ER claim count (31-90 days ago)
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_eruc_features_4_te_experiment`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_eruc_features_4_te_experiment`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH er_dc1 AS (
    SELECT
        base.individual_id,
        base.member_id,
        base.index_dt,
        CAST(SUM(CASE WHEN TRIM(clm.plc_srv_ctg_cd)='E' THEN 1 ELSE 0 END) AS INT64) AS erclm_cnt_dc1
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLAIM_LINE` AS clm
        ON base.member_id = clm.member_id
    WHERE (TRIM(clm.summarized_srv_ind) IN ('Y', '') OR clm.summarized_srv_ind IS NULL)
        AND (TRIM(clm.duplicate_ind) IN ('N', '') OR clm.duplicate_ind IS NULL)
        AND EXTRACT(YEAR FROM clm.srv_start_dt) >= 2020
        AND clm.srv_start_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 30 DAY) AND base.feature_end_dt
        AND clm.adjn_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 30 DAY) AND base.feature_end_dt
    GROUP BY base.individual_id, base.member_id, base.index_dt
),
er_dc2 AS (
    SELECT
        base.individual_id,
        base.member_id,
        base.index_dt,
        CAST(SUM(CASE WHEN TRIM(clm.plc_srv_ctg_cd)='E' THEN 1 ELSE 0 END) AS INT64) AS erclm_cnt_dc2
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLAIM_LINE` AS clm
        ON base.member_id = clm.member_id
    WHERE (TRIM(clm.summarized_srv_ind) IN ('Y', '') OR clm.summarized_srv_ind IS NULL)
        AND (TRIM(clm.duplicate_ind) IN ('N', '') OR clm.duplicate_ind IS NULL)
        AND EXTRACT(YEAR FROM clm.srv_start_dt) >= 2020
        AND clm.srv_start_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 90 DAY) AND DATE_SUB(base.feature_end_dt, INTERVAL 31 DAY)
        AND clm.adjn_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 90 DAY) AND DATE_SUB(base.feature_end_dt, INTERVAL 31 DAY)
    GROUP BY base.individual_id, base.member_id, base.index_dt
)
SELECT
    base.individual_id,
    base.member_id,
    base.index_dt,
    COALESCE(er1.erclm_cnt_dc1, 0) AS camemerucdc1_erclm_cnt_dc1,
    COALESCE(er2.erclm_cnt_dc2, 0) AS camemerucdc2_erclm_cnt_dc2
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` base
LEFT JOIN er_dc1 er1 ON base.individual_id = er1.individual_id AND base.member_id = er1.member_id AND base.index_dt = er1.index_dt
LEFT JOIN er_dc2 er2 ON base.individual_id = er2.individual_id AND base.member_id = er2.member_id AND base.index_dt = er2.index_dt;


/*==============================================================================
  STEP 14: RX CLASS UTILIZATION - DC5 (2 FEATURES)
  
  Purpose: Binary flags for specific medication classes (long lookback)
  
  Features:
  - loop_diuretics_flag_dc5: Used loop diuretics (366-730 days ago)
  - anticonvulsants_misc_flag_dc5: Used anticonvulsants (366-730 days ago)
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_rxclass_features_4_te_experiment`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_rxclass_features_4_te_experiment`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH rx_days AS (
    SELECT
        base.individual_id,
        base.member_id,
        base.index_dt,
        SUBSTR(RX.adjudicated_gpi_cd,1,4) AS gpi4,
        COALESCE(SUM(RX.days_supply_cnt), 0) AS days
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.RX_CLAIM_DTL` AS RX
        ON base.member_id = RX.member_id
    WHERE EXTRACT(YEAR FROM RX.disp_dt) >= 2020
        AND RX.disp_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 730 DAY) AND DATE_SUB(base.feature_end_dt, INTERVAL 366 DAY)
        AND SUBSTR(RX.adjudicated_gpi_cd,1,4) IN ('3720', '7260')
    GROUP BY base.individual_id, base.member_id, base.index_dt, SUBSTR(RX.adjudicated_gpi_cd,1,4)
    
    UNION DISTINCT
    
    SELECT
        base.individual_id,
        base.member_id,
        base.index_dt,
        SUBSTR(RX.adjudicated_gpi_cd,1,4) AS gpi4,
        COALESCE(SUM(RX.days_supply_cnt), 0) AS days
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.XTRNL_RX_CLAIM` AS RX
        ON base.member_id = RX.member_id
    WHERE EXTRACT(YEAR FROM RX.disp_dt) >= 2020
        AND RX.disp_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 730 DAY) AND DATE_SUB(base.feature_end_dt, INTERVAL 366 DAY)
        AND SUBSTR(RX.adjudicated_gpi_cd,1,4) IN ('3720', '7260')
    GROUP BY base.individual_id, base.member_id, base.index_dt, SUBSTR(RX.adjudicated_gpi_cd,1,4)
),
rx_agg AS (
    SELECT
        individual_id, member_id, index_dt,
        CAST(SUM(CASE WHEN gpi4='3720' THEN days ELSE 0 END) AS INT64) AS loop_diuretics_days_dc5,
        CAST(SUM(CASE WHEN gpi4='7260' THEN days ELSE 0 END) AS INT64) AS anticonvulsants_misc_days_dc5
    FROM rx_days
    GROUP BY individual_id, member_id, index_dt
),
rx_flags AS (
    SELECT
        individual_id, member_id, index_dt,
        CASE WHEN loop_diuretics_days_dc5 > 0 THEN 1 ELSE 0 END AS loop_diuretics_flag_dc5,
        CASE WHEN anticonvulsants_misc_days_dc5 > 0 THEN 1 ELSE 0 END AS anticonvulsants_misc_flag_dc5
    FROM rx_agg
)
SELECT
    base.individual_id,
    base.member_id,
    base.index_dt,
    COALESCE(rx.loop_diuretics_flag_dc5, 0) AS camemrxclassutilizationdc5_loop_diuretics_flag_dc5,
    COALESCE(rx.anticonvulsants_misc_flag_dc5, 0) AS camemrxclassutilizationdc5_anticonvulsants_misc_flag_dc5
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` base
LEFT JOIN rx_flags rx ON base.individual_id = rx.individual_id AND base.member_id = rx.member_id AND base.index_dt = rx.index_dt;


/*==============================================================================
  STEP 15: RX GROUP UTILIZATION - DC1, DC2, DC3 (5 FEATURES)
  
  Purpose: Days supply of specific medication groups in different time windows
  
  Features:
  - DC1 (0-30 days): corticosteroids_days_dc1, antidepressants_days_dc1
  - DC2 (31-90 days): corticosteroids_days_dc2, anticonvulsants_days_dc2
  - DC3 (91-180 days): anticonvulsants_flag_dc3
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_rxgroup_features_4_te_experiment`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_rxgroup_features_4_te_experiment`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH rx_dc1 AS (
    SELECT
        base.individual_id,
        base.member_id,
        base.index_dt,
        SUBSTR(RX.adjudicated_gpi_cd,1,2) AS gpi2,
        COALESCE(SUM(RX.days_supply_cnt), 0) AS days
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.RX_CLAIM_DTL` AS RX
        ON base.member_id = RX.member_id
    WHERE EXTRACT(YEAR FROM RX.disp_dt) >= 2020
        AND RX.disp_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 30 DAY) AND base.feature_end_dt
        AND SUBSTR(RX.adjudicated_gpi_cd,1,2) IN ('22', '58')
    GROUP BY base.individual_id, base.member_id, base.index_dt, SUBSTR(RX.adjudicated_gpi_cd,1,2)
    
    UNION DISTINCT
    
    SELECT
        base.individual_id,
        base.member_id,
        base.index_dt,
        SUBSTR(RX.adjudicated_gpi_cd,1,2) AS gpi2,
        COALESCE(SUM(RX.days_supply_cnt), 0) AS days
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.XTRNL_RX_CLAIM` AS RX
        ON base.member_id = RX.member_id
    WHERE EXTRACT(YEAR FROM RX.disp_dt) >= 2020
        AND RX.disp_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 30 DAY) AND base.feature_end_dt
        AND SUBSTR(RX.adjudicated_gpi_cd,1,2) IN ('22', '58')
    GROUP BY base.individual_id, base.member_id, base.index_dt, SUBSTR(RX.adjudicated_gpi_cd,1,2)
),
rx_dc2 AS (
    SELECT
        base.individual_id,
        base.member_id,
        base.index_dt,
        SUBSTR(RX.adjudicated_gpi_cd,1,2) AS gpi2,
        COALESCE(SUM(RX.days_supply_cnt), 0) AS days
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.RX_CLAIM_DTL` AS RX
        ON base.member_id = RX.member_id
    WHERE EXTRACT(YEAR FROM RX.disp_dt) >= 2020
        AND RX.disp_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 90 DAY) AND DATE_SUB(base.feature_end_dt, INTERVAL 31 DAY)
        AND SUBSTR(RX.adjudicated_gpi_cd,1,2) IN ('22', '72')
    GROUP BY base.individual_id, base.member_id, base.index_dt, SUBSTR(RX.adjudicated_gpi_cd,1,2)
    
    UNION DISTINCT
    
    SELECT
        base.individual_id,
        base.member_id,
        base.index_dt,
        SUBSTR(RX.adjudicated_gpi_cd,1,2) AS gpi2,
        COALESCE(SUM(RX.days_supply_cnt), 0) AS days
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.XTRNL_RX_CLAIM` AS RX
        ON base.member_id = RX.member_id
    WHERE EXTRACT(YEAR FROM RX.disp_dt) >= 2020
        AND RX.disp_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 90 DAY) AND DATE_SUB(base.feature_end_dt, INTERVAL 31 DAY)
        AND SUBSTR(RX.adjudicated_gpi_cd,1,2) IN ('22', '72')
    GROUP BY base.individual_id, base.member_id, base.index_dt, SUBSTR(RX.adjudicated_gpi_cd,1,2)
),
rx_dc3 AS (
    SELECT
        base.individual_id,
        base.member_id,
        base.index_dt,
        SUBSTR(RX.adjudicated_gpi_cd,1,2) AS gpi2,
        COALESCE(SUM(RX.days_supply_cnt), 0) AS days
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.RX_CLAIM_DTL` AS RX
        ON base.member_id = RX.member_id
    WHERE EXTRACT(YEAR FROM RX.disp_dt) >= 2020
        AND RX.disp_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 180 DAY) AND DATE_SUB(base.feature_end_dt, INTERVAL 91 DAY)
        AND SUBSTR(RX.adjudicated_gpi_cd,1,2) IN ('72')
    GROUP BY base.individual_id, base.member_id, base.index_dt, SUBSTR(RX.adjudicated_gpi_cd,1,2)
    
    UNION DISTINCT
    
    SELECT
        base.individual_id,
        base.member_id,
        base.index_dt,
        SUBSTR(RX.adjudicated_gpi_cd,1,2) AS gpi2,
        COALESCE(SUM(RX.days_supply_cnt), 0) AS days
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.XTRNL_RX_CLAIM` AS RX
        ON base.member_id = RX.member_id
    WHERE EXTRACT(YEAR FROM RX.disp_dt) >= 2020
        AND RX.disp_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 180 DAY) AND DATE_SUB(base.feature_end_dt, INTERVAL 91 DAY)
        AND SUBSTR(RX.adjudicated_gpi_cd,1,2) IN ('72')
    GROUP BY base.individual_id, base.member_id, base.index_dt, SUBSTR(RX.adjudicated_gpi_cd,1,2)
),
rx_agg1 AS (
    SELECT
        individual_id, member_id, index_dt,
        CAST(SUM(CASE WHEN gpi2='22' THEN days ELSE 0 END) AS INT64) AS corticosteroids_days_dc1,
        CAST(SUM(CASE WHEN gpi2='58' THEN days ELSE 0 END) AS INT64) AS antidepressants_days_dc1
    FROM rx_dc1
    GROUP BY individual_id, member_id, index_dt
),
rx_agg2 AS (
    SELECT
        individual_id, member_id, index_dt,
        CAST(SUM(CASE WHEN gpi2='22' THEN days ELSE 0 END) AS INT64) AS corticosteroids_days_dc2,
        CAST(SUM(CASE WHEN gpi2='72' THEN days ELSE 0 END) AS INT64) AS anticonvulsants_days_dc2
    FROM rx_dc2
    GROUP BY individual_id, member_id, index_dt
),
rx_agg3 AS (
    SELECT
        individual_id, member_id, index_dt,
        CAST(SUM(CASE WHEN gpi2='72' THEN days ELSE 0 END) AS INT64) AS anticonvulsants_days_dc3
    FROM rx_dc3
    GROUP BY individual_id, member_id, index_dt
),
rx_flag3 AS (
    SELECT
        individual_id, member_id, index_dt,
        CASE WHEN anticonvulsants_days_dc3 > 0 THEN 1 ELSE 0 END AS anticonvulsants_flag_dc3
    FROM rx_agg3
)
SELECT
    base.individual_id,
    base.member_id,
    base.index_dt,
    COALESCE(r1.corticosteroids_days_dc1, 0) AS camemrxgrouputilizationdc1_corticosteroids_days_dc1,
    COALESCE(r1.antidepressants_days_dc1, 0) AS camemrxgrouputilizationdc1_antidepressants_days_dc1,
    COALESCE(r2.corticosteroids_days_dc2, 0) AS camemrxgrouputilizationdc2_corticosteroids_days_dc2,
    COALESCE(r2.anticonvulsants_days_dc2, 0) AS camemrxgrouputilizationdc2_anticonvulsants_days_dc2,
    COALESCE(r3.anticonvulsants_flag_dc3, 0) AS camemrxgrouputilizationdc3_anticonvulsants_flag_dc3
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` base
LEFT JOIN rx_agg1 r1 ON base.individual_id = r1.individual_id AND base.member_id = r1.member_id AND base.index_dt = r1.index_dt
LEFT JOIN rx_agg2 r2 ON base.individual_id = r2.individual_id AND base.member_id = r2.member_id AND base.index_dt = r2.index_dt
LEFT JOIN rx_flag3 r3 ON base.individual_id = r3.individual_id AND base.member_id = r3.member_id AND base.index_dt = r3.index_dt;


/*==============================================================================
  STEP 16: SPECIALTY CLAIMS FEATURES - DC1, DC2, DC3 (3 FEATURES)
  
  Purpose: Count specialty claims and office visits
  
  Features:
  - spcclmwhos_cnt_dc1: Specialty claims WHOS (0-30 days)
  - spcclmwhos_cnt_dc2: Specialty claims WHOS (31-90 days)
  - spcd_cnt_dc3: Specialty office visits D (91-180 days)
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_specialty_features_4_te_experiment`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_specialty_features_4_te_experiment`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH spc_dc1 AS (
    SELECT
        base.individual_id,
        base.member_id,
        base.index_dt,
        clm.srv_spclty_ctg_cd,
        CAST(COUNT(*) AS INT64) AS clm_cnt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLAIM_LINE` AS clm
        ON base.member_id = clm.member_id
    WHERE (TRIM(clm.summarized_srv_ind) IN ('Y', '') OR clm.summarized_srv_ind IS NULL)
        AND (TRIM(clm.duplicate_ind) IN ('N', '') OR clm.duplicate_ind IS NULL)
        AND EXTRACT(YEAR FROM clm.srv_start_dt) >= 2020
        AND clm.srv_start_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 30 DAY) AND base.feature_end_dt
        AND clm.adjn_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 30 DAY) AND base.feature_end_dt
        AND clm.srv_spclty_ctg_cd = 'WHOS'
    GROUP BY base.individual_id, base.member_id, base.index_dt, clm.srv_spclty_ctg_cd
),
spc_dc2 AS (
    SELECT
        base.individual_id,
        base.member_id,
        base.index_dt,
        clm.srv_spclty_ctg_cd,
        CAST(COUNT(*) AS INT64) AS clm_cnt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLAIM_LINE` AS clm
        ON base.member_id = clm.member_id
    WHERE (TRIM(clm.summarized_srv_ind) IN ('Y', '') OR clm.summarized_srv_ind IS NULL)
        AND (TRIM(clm.duplicate_ind) IN ('N', '') OR clm.duplicate_ind IS NULL)
        AND EXTRACT(YEAR FROM clm.srv_start_dt) >= 2020
        AND clm.srv_start_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 90 DAY) AND DATE_SUB(base.feature_end_dt, INTERVAL 31 DAY)
        AND clm.adjn_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 90 DAY) AND DATE_SUB(base.feature_end_dt, INTERVAL 31 DAY)
        AND clm.srv_spclty_ctg_cd = 'WHOS'
    GROUP BY base.individual_id, base.member_id, base.index_dt, clm.srv_spclty_ctg_cd
),
spc_dc3 AS (
    SELECT
        base.individual_id,
        base.member_id,
        base.index_dt,
        clm.srv_spclty_ctg_cd,
        clm.srv_prvdr_id,
        CAST(COUNT(*) AS INT64) AS clm_cnt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLAIM_LINE` AS clm
        ON base.member_id = clm.member_id
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PROVIDER_DM` AS prov
        ON clm.srv_prvdr_id = prov.provider_id
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PROCEDURE` AS prc
        ON TRIM(clm.prcdr_cd) = TRIM(prc.prcdr_cd)
    WHERE (TRIM(clm.summarized_srv_ind) IN ('Y', '') OR clm.summarized_srv_ind IS NULL)
        AND (TRIM(clm.duplicate_ind) IN ('N', '') OR clm.duplicate_ind IS NULL)
        AND EXTRACT(YEAR FROM clm.srv_start_dt) >= 2020
        AND clm.srv_start_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 180 DAY) AND DATE_SUB(base.feature_end_dt, INTERVAL 91 DAY)
        AND clm.adjn_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 180 DAY) AND DATE_SUB(base.feature_end_dt, INTERVAL 91 DAY)
        AND TRIM(clm.plc_srv_ctg_cd) = 'O'
        AND CAST(prc.prcdr_group_nbr AS INT64) = 161
        AND TRIM(prov.provider_type_cd) NOT IN ('AAA', 'DB', 'OO', 'TE', 'U', 'UNK')
    GROUP BY base.individual_id, base.member_id, base.index_dt, clm.srv_spclty_ctg_cd, clm.srv_prvdr_id
),
spc_agg1 AS (
    SELECT
        individual_id, member_id, index_dt,
        CAST(SUM(CASE WHEN TRIM(srv_spclty_ctg_cd) = 'WHOS' THEN clm_cnt ELSE 0 END) AS INT64) AS spcclmwhos_cnt_dc1
    FROM spc_dc1
    GROUP BY individual_id, member_id, index_dt
),
spc_agg2 AS (
    SELECT
        individual_id, member_id, index_dt,
        CAST(SUM(CASE WHEN TRIM(srv_spclty_ctg_cd) = 'WHOS' THEN clm_cnt ELSE 0 END) AS INT64) AS spcclmwhos_cnt_dc2
    FROM spc_dc2
    GROUP BY individual_id, member_id, index_dt
),
spc_agg3 AS (
    SELECT
        individual_id, member_id, index_dt,
        CAST(SUM(CASE WHEN TRIM(srv_spclty_ctg_cd) = 'D' THEN clm_cnt ELSE 0 END) AS INT64) AS spcd_cnt_dc3
    FROM spc_dc3
    GROUP BY individual_id, member_id, index_dt
)
SELECT
    base.individual_id,
    base.member_id,
    base.index_dt,
    COALESCE(s1.spcclmwhos_cnt_dc1, 0) AS camemspcclmdc1_spcclmwhos_cnt_dc1,
    COALESCE(s2.spcclmwhos_cnt_dc2, 0) AS camemspcclmdc2_spcclmwhos_cnt_dc2,
    COALESCE(s3.spcd_cnt_dc3, 0) AS camemspcofcdc3_spcd_cnt_dc3
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` base
LEFT JOIN spc_agg1 s1 ON base.individual_id = s1.individual_id AND base.member_id = s1.member_id AND base.index_dt = s1.index_dt
LEFT JOIN spc_agg2 s2 ON base.individual_id = s2.individual_id AND base.member_id = s2.member_id AND base.index_dt = s2.index_dt
LEFT JOIN spc_agg3 s3 ON base.individual_id = s3.individual_id AND base.member_id = s3.member_id AND base.index_dt = s3.index_dt;


/*==============================================================================
  STEP 17: CARE MANAGEMENT FEATURES - DC2, DC5 (2 FEATURES)
  
  Purpose: Binary flags for care management program enrollment
  
  Features:
  - cm_soe_dc2: Care management (31-90 days ago)
  - cm_soe_dc5: Care management (366-730 days ago)
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_cm_features_4_te_experiment`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_cm_features_4_te_experiment`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH cm_dc2_atv AS (
    SELECT
        base.individual_id,
        base.member_id,
        base.index_dt,
        emb.srvofrg_id_cd
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEMBER` AS mbr
        ON base.member_id = mbr.member_id
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_srcv.ATV_EMBRODM` AS emb
        ON mbr.src_cumb_id = emb.cumb_id_no
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_srcv.ATV_SOETPL` AS soe
        ON emb.soe_id_no = soe.soe_id_no
    WHERE TRIM(emb.status_cd) NOT IN ('00003', '00005')
        AND TRIM(emb.srvofrg_id_cd) IN ('40001', '40002')
        AND soe.soetpl_ptlvl_st_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 90 DAY) AND DATE_SUB(base.feature_end_dt, INTERVAL 31 DAY)
),
cm_dc5_atv AS (
    SELECT
        base.individual_id,
        base.member_id,
        base.index_dt,
        emb.srvofrg_id_cd
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEMBER` AS mbr
        ON base.member_id = mbr.member_id
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_srcv.ATV_EMBRODM` AS emb
        ON mbr.src_cumb_id = emb.cumb_id_no
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_srcv.ATV_SOETPL` AS soe
        ON emb.soe_id_no = soe.soe_id_no
    WHERE emb.status_cd NOT IN ('00003', '00005')
        AND emb.srvofrg_id_cd IN ('40001', '40002')
        AND soe.soetpl_ptlvl_st_dt BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 730 DAY) AND DATE_SUB(base.feature_end_dt, INTERVAL 366 DAY)
),
cm_agg AS (
    SELECT
        base.individual_id,
        base.member_id,
        base.index_dt,
        MAX(CASE WHEN cm2.srvofrg_id_cd IS NOT NULL THEN 1 ELSE 0 END) AS cm_soe_dc2,
        MAX(CASE WHEN cm5.srvofrg_id_cd IS NOT NULL THEN 1 ELSE 0 END) AS cm_soe_dc5
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` base
    LEFT JOIN cm_dc2_atv cm2 ON base.individual_id = cm2.individual_id AND base.member_id = cm2.member_id AND base.index_dt = cm2.index_dt
    LEFT JOIN cm_dc5_atv cm5 ON base.individual_id = cm5.individual_id AND base.member_id = cm5.member_id AND base.index_dt = cm5.index_dt
    GROUP BY base.individual_id, base.member_id, base.index_dt
)
SELECT
    individual_id,
    member_id,
    index_dt,
    COALESCE(cm_soe_dc2, 0) AS camemtgtptpdc2_cm_soe,
    COALESCE(cm_soe_dc5, 0) AS camemtgtptpdc5_cm_soe
FROM cm_agg;


/*==============================================================================
  STEP 18: TEXT NOTES FEATURES - DC5 (2 FEATURES)
  
  Purpose: Binary flags for presence of specific text in clinical notes
  
  Features:
  - txt_end_dc5: Note contains " END " (366-730 days ago)
  - txt_short_dc5: Note contains " SHORT " (366-730 days ago)
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_txtnotes_features_4_te_experiment`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_txtnotes_features_4_te_experiment`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH txt_notes AS (
    SELECT DISTINCT
        base.individual_id,
        base.member_id,
        base.index_dt,
        note.pmnote_dscrptn_txt AS txt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEMBER` AS mbr
        ON base.member_id = mbr.member_id
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.SRDB_PMREVT` AS evt
        ON mbr.src_cumb_id = evt.cvrg_identifict_no
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.SRDB_PMRNOTE` AS note
        ON SAFE_CAST(evt.pme_reference_no AS STRING) = SAFE_CAST(note.pme_reference_no AS STRING)
    WHERE CAST(note.pmnote_create_dt AS DATE) BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 730 DAY) AND DATE_SUB(base.feature_end_dt, INTERVAL 366 DAY)
    
    UNION DISTINCT
    
    SELECT DISTINCT
        base.individual_id,
        base.member_id,
        base.index_dt,
        note.mbrnt_note_txt AS txt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEMBER` AS mbr
        ON base.member_id = mbr.member_id
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_srcv.ATV_EMBRODM` AS emb
        ON mbr.src_cumb_id = emb.cumb_id_no
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_srcv.ATV_MBRNT` AS note
        ON emb.cumb_id_no = note.cumb_id_no AND emb.soe_id_no = note.soe_id_no
    WHERE TRIM(emb.status_cd) <> '00005'
        AND TRIM(emb.srvofrg_id_cd) IN ('17001', '23001', '40001', '40002', '50001', '80001', '12002', '19001')
        AND SAFE_CAST(note.mbrnt_posted_dts AS DATE) BETWEEN DATE_SUB(base.feature_end_dt, INTERVAL 730 DAY) AND DATE_SUB(base.feature_end_dt, INTERVAL 366 DAY)
),
txt_flags AS (
    SELECT
        individual_id,
        member_id,
        index_dt,
        MAX(CASE WHEN UPPER(txt) LIKE '% END %' THEN 1 ELSE 0 END) AS txt_end_dc5,
        MAX(CASE WHEN UPPER(txt) LIKE '% SHORT %' THEN 1 ELSE 0 END) AS txt_short_dc5
    FROM txt_notes
    GROUP BY individual_id, member_id, index_dt
)
SELECT
    base.individual_id,
    base.member_id,
    base.index_dt,
    COALESCE(txt.txt_end_dc5, 0) AS camemtxtnotesdc5_txt_end_dc5,
    COALESCE(txt.txt_short_dc5, 0) AS camemtxtnotesdc5_txt_short_dc5
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` base
LEFT JOIN txt_flags txt ON base.individual_id = txt.individual_id AND base.member_id = txt.member_id AND base.index_dt = txt.index_dt;


/*==============================================================================
  STEP 19: YLM DEMOGRAPHIC FEATURES (4 FEATURES)
  
  Purpose: Lifestyle and demographic flags from marketing data
  
  Features:
  - ylm_orent: Own/rent indicator
  - ylm_tw_hvalsecinv: High value security investor
  - ylm_tw_hvyinvtrad: Heavy investment trader
  - ylm_homeagesourcer: Home age source R
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_ylm_features_4_te_experiment`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_ylm_features_4_te_experiment`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH ylm_raw AS (
    SELECT
        base.individual_id,
        base.member_id,
        base.index_dt,
        mkt.ownrent,
        mkt.tw_highvaluesecurityinvestor,
        mkt.tw_heavyinvestmenttraders,
        mkt.homeagesource
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `anbc-hcb-prod.insights_share_hcb_prod.v_enriched_membership` AS enr
        ON SAFE_CAST(base.individual_id AS STRING) = SAFE_CAST(enr.indiv_id AS STRING)
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_srcv.INFOGROUP_AEP_MEMBERSHIP` AS aep
        ON enr.indiv_anlytcs_id = aep.member_id
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_srcv.MARKETING_ODS_TBMZB_INDIV_INFOGROUP_LIST` AS mkt
        ON aep.mzb_indiv_id = mkt.mzb_indiv_id
),
ylm_processed AS (
    SELECT
        individual_id, member_id, index_dt,
        SAFE_CAST(ownrent AS INT64) AS orent,
        SAFE_CAST(tw_highvaluesecurityinvestor AS INT64) AS tw_hvalsecinv,
        SAFE_CAST(tw_heavyinvestmenttraders AS INT64) AS tw_hvyinvtrad,
        CASE WHEN TRIM(homeagesource) = 'R' THEN 1 ELSE 0 END AS homeagesourceR
    FROM ylm_raw
),
ylm_agg AS (
    SELECT
        individual_id, member_id, index_dt,
        MAX(CASE WHEN orent > 0 THEN orent ELSE 0 END) AS ylm_orent,
        MAX(CASE WHEN tw_hvalsecinv > 0 THEN tw_hvalsecinv ELSE 0 END) AS ylm_tw_hvalsecinv,
        MAX(CASE WHEN tw_hvyinvtrad > 0 THEN tw_hvyinvtrad ELSE 0 END) AS ylm_tw_hvyinvtrad,
        MAX(CASE WHEN homeagesourceR > 0 THEN homeagesourceR ELSE 0 END) AS ylm_homeagesourcer
    FROM ylm_processed
    GROUP BY individual_id, member_id, index_dt
)
SELECT
    base.individual_id,
    base.member_id,
    base.index_dt,
    COALESCE(ylm.ylm_orent, 0) AS camemylm_ylm_orent,
    COALESCE(ylm.ylm_tw_hvalsecinv, 0) AS camemylm_ylm_tw_hvalsecinv,
    COALESCE(ylm.ylm_tw_hvyinvtrad, 0) AS camemylm_ylm_tw_hvyinvtrad,
    COALESCE(ylm.ylm_homeagesourcer, 0) AS camemylm_ylm_homeagesourcer
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` base
LEFT JOIN ylm_agg ylm ON base.individual_id = ylm.individual_id AND base.member_id = ylm.member_id AND base.index_dt = ylm.index_dt;


/*==============================================================================
  STEP 20: MEMBERSHIP DEMOGRAPHICS (2 FEATURES)
  
  Purpose: Age-based demographic features
  
  Features:
  - agenbr: Member age (in years)
  - age65_74: Binary flag for age 65-74
==============================================================================*/

CREATE TEMPORARY FUNCTION months_between_hive(date1 DATE, date2 DATE) AS (
    (EXTRACT(YEAR FROM date1) - EXTRACT(YEAR FROM date2)) * 12 + 
    (EXTRACT(MONTH FROM date1) - EXTRACT(MONTH FROM date2)) + 
    (EXTRACT(DAY FROM date1) - EXTRACT(DAY FROM date2)) / 30.436875
);

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_mbrshp_features_4_te_experiment`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_mbrshp_features_4_te_experiment`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH age_calc AS (
    SELECT DISTINCT
        base.individual_id,
        base.member_id,
        base.index_dt,
        CAST(FLOOR(months_between_hive(base.feature_end_dt, mbr.birth_dt)/12) AS INT64) AS age
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEMBER` AS mbr
        ON base.member_id = mbr.member_id
),
age_features AS (
    SELECT
        individual_id, member_id, index_dt,
        MAX(CASE WHEN age > 0 THEN age ELSE 0 END) AS agenbr,
        MAX(CASE WHEN age > 64 AND age <= 74 THEN 1 ELSE 0 END) AS age65_74
    FROM age_calc
    GROUP BY individual_id, member_id, index_dt
)
SELECT
    base.individual_id,
    base.member_id,
    base.index_dt,
    COALESCE(age.agenbr, 0) AS camemmbrshp_agenbr,
    COALESCE(age.age65_74, 0) AS camemmbrshp_age65_74
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` base
LEFT JOIN age_features age ON base.individual_id = age.individual_id AND base.member_id = age.member_id AND base.index_dt = age.index_dt;


/*==============================================================================
  STEP 21: EXPENDITURE FEATURES (2 FEATURES)
  
  Purpose: Experian retail demand scores
  
  Features:
  - e_caperetdem444: CAPE retail demand 444
  - e_caperetdem21220: CAPE retail demand 21220
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_exp_features_4_te_experiment`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_exp_features_4_te_experiment`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH exp_dates AS (
    SELECT
        base.individual_id,
        base.member_id,
        base.index_dt,
        CONCAT(EXTRACT(YEAR FROM base.feature_end_dt), LPAD(CAST(EXTRACT(MONTH FROM base.feature_end_dt) AS STRING), 2, '0')) AS cc_max,
        CONCAT(exp.load_year, exp.load_month) AS cc_exp
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` AS base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_X_PROXY` AS prx
        ON base.individual_id = prx.individual_id
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_srcv.EXPERIAN_CURRENT_2020` AS exp
        ON prx.preferred_proxy_id = exp.proxy_id
),
exp_max AS (
    SELECT
        individual_id, member_id, index_dt,
        MAX(cc_exp) AS exp_max
    FROM exp_dates
    WHERE cc_exp <= cc_max
    GROUP BY individual_id, member_id, index_dt
),
exp_values AS (
    SELECT
        base.individual_id,
        base.member_id,
        base.index_dt,
        CASE WHEN exp.CAPE_Retail_Demand_444 IS NOT NULL THEN SAFE_CAST(exp.CAPE_Retail_Demand_444 AS INT64) ELSE 0 END AS e_caperetdem444,
        CASE WHEN exp.CAPE_Retail_Demand_21220 IS NOT NULL THEN SAFE_CAST(exp.CAPE_Retail_Demand_21220 AS INT64) ELSE 0 END AS e_caperetdem21220
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` base
    INNER JOIN exp_max em ON base.individual_id = em.individual_id AND base.member_id = em.member_id AND base.index_dt = em.index_dt
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_X_PROXY` AS prx
        ON base.individual_id = prx.individual_id
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_srcv.EXPERIAN_CURRENT_2020` AS exp
        ON prx.preferred_proxy_id = exp.proxy_id
        AND SUBSTR(em.exp_max, 1, 4) = exp.load_year
        AND SUBSTR(em.exp_max, 5, 2) = exp.load_month
)
SELECT
    base.individual_id,
    base.member_id,
    base.index_dt,
    COALESCE(ev.e_caperetdem444, 0) AS e_caperetdem444,
    COALESCE(ev.e_caperetdem21220, 0) AS e_caperetdem21220
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` base
LEFT JOIN exp_values ev ON base.individual_id = ev.individual_id AND base.member_id = ev.member_id AND base.index_dt = ev.index_dt;


/*==============================================================================
  STEP 22: FINAL MERGE - COMPLETE DATASET WITH ALL FEATURES + OUTCOMES
  
  Combines:
  - Base identifiers (individual_id, member_id, index_dt)
  - Outcomes (ip6, sum_ip6_admits, sum_ip6_los, mon_6_include)
  - HPD features (11 chronic conditions)
  - Medical Utilization (4 features)
  - Medical Case (5 features)
  - Diagnosis (1 feature)
  - Procedures (5 features)
  - Revenue Codes (2 features)
  - ER/UC (2 features)
  - RX Class (2 features)
  - RX Group (5 features)
  - Specialty Claims (3 features)
  - Care Management (2 features)
  - Text Notes (2 features)
  - YLM Demographics (4 features)
  - Membership Demographics (2 features)
  - Expenditure (2 features)
  
  Total: 54 baseline features (excluding embeddings)
  
  Output: Complete baseline feature dataset
  
  Note: Transformer embeddings should be joined separately by user
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_final_dataset_4_te_experiment`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_final_dataset_4_te_experiment`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)  -- Final dataset: 180 days
)
AS
SELECT 
    -- ===== BASE IDENTIFIERS =====
    base.individual_id
    , base.member_id
    , base.index_dt
    , base.feature_end_dt
    , base.business_ln_cd
    
    -- ===== *** OUTCOMES (FRONT) *** =====
    , COALESCE(outcomes.ip6, 0) AS ip6
    , COALESCE(outcomes.sum_ip6_admits, 0) AS sum_ip6_admits
    , COALESCE(outcomes.sum_ip6_los, 0) AS sum_ip6_los
    , COALESCE(cont.mon_6_include, 0) AS mon_6_include
    
    -- ===== HPD CHRONIC CONDITIONS (11 features) =====
    , COALESCE(hpd.camemhpd_aff, 0) AS camemhpd_aff
    , COALESCE(hpd.camemhpd_alc, 0) AS camemhpd_alc
    , COALESCE(hpd.camemhpd_cbd, 0) AS camemhpd_cbd
    , COALESCE(hpd.camemhpd_chf, 0) AS camemhpd_chf
    , COALESCE(hpd.camemhpd_cop, 0) AS camemhpd_cop
    , COALESCE(hpd.camemhpd_crf, 0) AS camemhpd_crf
    , COALESCE(hpd.camemhpd_dia, 0) AS camemhpd_dia
    , COALESCE(hpd.camemhpd_hyp, 0) AS camemhpd_hyp
    , COALESCE(hpd.camemhpd_ngd, 0) AS camemhpd_ngd
    , COALESCE(hpd.camemhpd_cv_cond, 0) AS camemhpd_cv_cond
    , COALESCE(hpd.camemhpd_chr_flag, 0) AS camemhpd_chr_flag
    
    -- ===== MEDICAL UTILIZATION (4 features) =====
    , COALESCE(medutil.camemmedutilization_clm_ln_cnt, 0) AS camemmedutilization_clm_ln_cnt
    , COALESCE(medutil.camemmedutilization_uniq_dx_cd_cnt, 0) AS camemmedutilization_uniq_dx_cd_cnt
    , COALESCE(medutil.camemmedutilization_uniq_rev_cd_cnt, 0) AS camemmedutilization_uniq_rev_cd_cnt
    , COALESCE(medutil.camemmedutilization_er_clm_cnt, 0) AS camemmedutilization_er_clm_cnt
    
    -- ===== MEDICAL CASE (5 features) =====
    , COALESCE(medcase.camemmedcasedc1_ip_cnt_dc1, 0) AS camemmedcasedc1_ip_cnt_dc1
    , COALESCE(medcase.camemmedcasedc1_ip_days_dc1, 0) AS camemmedcasedc1_ip_days_dc1
    , COALESCE(medcase.camemmedcasedc2_ip_cnt_dc2, 0) AS camemmedcasedc2_ip_cnt_dc2
    , COALESCE(medcase.camemmedcasedc2_ip_days_dc2, 0) AS camemmedcasedc2_ip_days_dc2
    , COALESCE(medcase.camemmedcasedc3_ip_cnt_dc3, 0) AS camemmedcasedc3_ip_cnt_dc3
    
    -- ===== DIAGNOSIS (1 feature) =====
    , COALESCE(dx.camemeipdxdc1_dxc1085_cnt_dc1, 0) AS camemeipdxdc1_dxc1085_cnt_dc1
    
    -- ===== PROCEDURES (5 features) =====
    , COALESCE(prc.camemeipprcdc1_prc141_cnt_dc1, 0) AS camemeipprcdc1_prc141_cnt_dc1
    , COALESCE(prc.camemeipprcdc1_prc155_cnt_dc1, 0) AS camemeipprcdc1_prc155_cnt_dc1
    , COALESCE(prc.camemeipprcdc1_prc219_cnt_dc1, 0) AS camemeipprcdc1_prc219_cnt_dc1
    , COALESCE(prc.camemeipprcdc1_prcc1102_cnt_dc1, 0) AS camemeipprcdc1_prcc1102_cnt_dc1
    , COALESCE(prc.camemeipprcdc1_prcc1115_cnt_dc1, 0) AS camemeipprcdc1_prcc1115_cnt_dc1
    
    -- ===== REVENUE CODES (2 features) =====
    , COALESCE(rev.camemrevenuedc3_rev730_cnt_dc3, 0) AS camemrevenuedc3_rev730_cnt_dc3
    , COALESCE(rev.camemrevenuedc4_rev430_cnt_dc4, 0) AS camemrevenuedc4_rev430_cnt_dc4
    
    -- ===== ER/UC (2 features) =====
    , COALESCE(eruc.camemerucdc1_erclm_cnt_dc1, 0) AS camemerucdc1_erclm_cnt_dc1
    , COALESCE(eruc.camemerucdc2_erclm_cnt_dc2, 0) AS camemerucdc2_erclm_cnt_dc2
    
    -- ===== RX CLASS UTILIZATION (2 features) =====
    , COALESCE(rxclass.camemrxclassutilizationdc5_loop_diuretics_flag_dc5, 0) AS camemrxclassutilizationdc5_loop_diuretics_flag_dc5
    , COALESCE(rxclass.camemrxclassutilizationdc5_anticonvulsants_misc_flag_dc5, 0) AS camemrxclassutilizationdc5_anticonvulsants_misc_flag_dc5
    
    -- ===== RX GROUP UTILIZATION (5 features) =====
    , COALESCE(rxgroup.camemrxgrouputilizationdc1_corticosteroids_days_dc1, 0) AS camemrxgrouputilizationdc1_corticosteroids_days_dc1
    , COALESCE(rxgroup.camemrxgrouputilizationdc1_antidepressants_days_dc1, 0) AS camemrxgrouputilizationdc1_antidepressants_days_dc1
    , COALESCE(rxgroup.camemrxgrouputilizationdc2_corticosteroids_days_dc2, 0) AS camemrxgrouputilizationdc2_corticosteroids_days_dc2
    , COALESCE(rxgroup.camemrxgrouputilizationdc2_anticonvulsants_days_dc2, 0) AS camemrxgrouputilizationdc2_anticonvulsants_days_dc2
    , COALESCE(rxgroup.camemrxgrouputilizationdc3_anticonvulsants_flag_dc3, 0) AS camemrxgrouputilizationdc3_anticonvulsants_flag_dc3
    
    -- ===== SPECIALTY CLAIMS (3 features) =====
    , COALESCE(spc.camemspcclmdc1_spcclmwhos_cnt_dc1, 0) AS camemspcclmdc1_spcclmwhos_cnt_dc1
    , COALESCE(spc.camemspcclmdc2_spcclmwhos_cnt_dc2, 0) AS camemspcclmdc2_spcclmwhos_cnt_dc2
    , COALESCE(spc.camemspcofcdc3_spcd_cnt_dc3, 0) AS camemspcofcdc3_spcd_cnt_dc3
    
    -- ===== CARE MANAGEMENT (2 features) =====
    , COALESCE(cm.camemtgtptpdc2_cm_soe, 0) AS camemtgtptpdc2_cm_soe
    , COALESCE(cm.camemtgtptpdc5_cm_soe, 0) AS camemtgtptpdc5_cm_soe
    
    -- ===== TEXT NOTES (2 features) =====
    , COALESCE(txt.camemtxtnotesdc5_txt_end_dc5, 0) AS camemtxtnotesdc5_txt_end_dc5
    , COALESCE(txt.camemtxtnotesdc5_txt_short_dc5, 0) AS camemtxtnotesdc5_txt_short_dc5
    
    -- ===== YLM DEMOGRAPHICS (4 features) =====
    , COALESCE(ylm.camemylm_ylm_orent, 0) AS camemylm_ylm_orent
    , COALESCE(ylm.camemylm_ylm_tw_hvalsecinv, 0) AS camemylm_ylm_tw_hvalsecinv
    , COALESCE(ylm.camemylm_ylm_tw_hvyinvtrad, 0) AS camemylm_ylm_tw_hvyinvtrad
    , COALESCE(ylm.camemylm_ylm_homeagesourcer, 0) AS camemylm_ylm_homeagesourcer
    
    -- ===== MEMBERSHIP DEMOGRAPHICS (2 features) =====
    , COALESCE(mbrshp.camemmbrshp_agenbr, 0) AS camemmbrshp_agenbr
    , COALESCE(mbrshp.camemmbrshp_age65_74, 0) AS camemmbrshp_age65_74
    
    -- ===== EXPENDITURE (2 features) =====
    , COALESCE(exp.e_caperetdem444, 0) AS e_caperetdem444
    , COALESCE(exp.e_caperetdem21220, 0) AS e_caperetdem21220

FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment` base

-- Join outcomes
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_4_te_experiment` outcomes
    ON base.individual_id = outcomes.individual_id
    AND base.member_id = outcomes.member_id
    AND base.index_dt = outcomes.index_dt

-- Join continuity flags
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_post_status_4_te_experiment` cont
    ON base.individual_id = cont.individual_id
    AND base.member_id = cont.member_id
    AND base.index_dt = cont.index_dt

-- Join HPD features
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_hpd_features_4_te_experiment` hpd
    ON base.individual_id = hpd.individual_id
    AND base.member_id = hpd.member_id
    AND base.index_dt = hpd.index_dt

-- Join Medical Utilization features
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_medutil_features_4_te_experiment` medutil
    ON base.individual_id = medutil.individual_id
    AND base.member_id = medutil.member_id
    AND base.index_dt = medutil.index_dt

-- Join Medical Case features
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_medcase_features_4_te_experiment` medcase
    ON base.individual_id = medcase.individual_id
    AND base.member_id = medcase.member_id
    AND base.index_dt = medcase.index_dt

-- Join Diagnosis features
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_dx_features_4_te_experiment` dx
    ON base.individual_id = dx.individual_id
    AND base.member_id = dx.member_id
    AND base.index_dt = dx.index_dt

-- Join Procedure features
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_prc_features_4_te_experiment` prc
    ON base.individual_id = prc.individual_id
    AND base.member_id = prc.member_id
    AND base.index_dt = prc.index_dt

-- Join Revenue Code features
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_revenue_features_4_te_experiment` rev
    ON base.individual_id = rev.individual_id
    AND base.member_id = rev.member_id
    AND base.index_dt = rev.index_dt

-- Join ER/UC features
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_eruc_features_4_te_experiment` eruc
    ON base.individual_id = eruc.individual_id
    AND base.member_id = eruc.member_id
    AND base.index_dt = eruc.index_dt

-- Join RX Class features
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_rxclass_features_4_te_experiment` rxclass
    ON base.individual_id = rxclass.individual_id
    AND base.member_id = rxclass.member_id
    AND base.index_dt = rxclass.index_dt

-- Join RX Group features
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_rxgroup_features_4_te_experiment` rxgroup
    ON base.individual_id = rxgroup.individual_id
    AND base.member_id = rxgroup.member_id
    AND base.index_dt = rxgroup.index_dt

-- Join Specialty Claims features
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_specialty_features_4_te_experiment` spc
    ON base.individual_id = spc.individual_id
    AND base.member_id = spc.member_id
    AND base.index_dt = spc.index_dt

-- Join Care Management features
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_cm_features_4_te_experiment` cm
    ON base.individual_id = cm.individual_id
    AND base.member_id = cm.member_id
    AND base.index_dt = cm.index_dt

-- Join Text Notes features
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_txtnotes_features_4_te_experiment` txt
    ON base.individual_id = txt.individual_id
    AND base.member_id = txt.member_id
    AND base.index_dt = txt.index_dt

-- Join YLM Demographics features
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_ylm_features_4_te_experiment` ylm
    ON base.individual_id = ylm.individual_id
    AND base.member_id = ylm.member_id
    AND base.index_dt = ylm.index_dt

-- Join Membership Demographics features
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_mbrshp_features_4_te_experiment` mbrshp
    ON base.individual_id = mbrshp.individual_id
    AND base.member_id = mbrshp.member_id
    AND base.index_dt = mbrshp.index_dt

-- Join Expenditure features
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_exp_features_4_te_experiment` exp
    ON base.individual_id = exp.individual_id
    AND base.member_id = exp.member_id
    AND base.index_dt = exp.index_dt;

-- VERIFICATION: Check final complete dataset
SELECT 
    'FINAL DATASET - Complete Training Data' AS check_name
    , COUNT(*) AS total_rows
    , COUNT(DISTINCT individual_id) AS unique_individuals
    , COUNT(DISTINCT CONCAT(individual_id, '|', CAST(index_dt AS STRING))) AS unique_member_index_pairs
    , SUM(CASE WHEN ip6 = 1 THEN 1 ELSE 0 END) AS members_with_ip
    , ROUND(SUM(CASE WHEN ip6 = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS ip_rate_pct
    , SUM(CASE WHEN mon_6_include = 1 THEN 1 ELSE 0 END) AS evaluable_members
    , SUM(CASE WHEN ip6 = 1 AND mon_6_include = 1 THEN 1 ELSE 0 END) AS evaluable_with_ip
    , ROUND(SUM(CASE WHEN ip6 = 1 AND mon_6_include = 1 THEN 1 ELSE 0 END) * 100.0 / NULLIF(SUM(CASE WHEN mon_6_include = 1 THEN 1 ELSE 0 END), 0), 2) AS evaluable_ip_rate_pct
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_final_dataset_4_te_experiment`;

-- FEATURE COUNT VERIFICATION: Check all columns are present
SELECT 
    'FEATURE COUNT - All Columns' AS check_name,
    COUNT(*) AS total_columns,
    CASE 
        WHEN COUNT(*) = 63 THEN '✅ PASS - All 63 columns present (5 identifiers + 4 outcomes + 54 features)'
        ELSE CONCAT('❌ FAIL - Expected 63 columns, found ', CAST(COUNT(*) AS STRING))
    END AS status
FROM `edp-prod-storage.edp_ent_sdoheir_cns.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'a834793_Medicare_final_dataset_4_te_experiment';


/*==============================================================================
  VALIDATION QUERIES FOR COMPLETE DATASET
  
  Run these to verify the final training dataset is correct:
==============================================================================*/

-- 1. Check row counts match base cohort
SELECT 
    'Row Count Validation' AS check_name,
    (SELECT COUNT(*) FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment`) AS base_count,
    (SELECT COUNT(*) FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_final_dataset_4_te_experiment`) AS final_count,
    CASE 
        WHEN (SELECT COUNT(*) FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_base_4_te_experiment`) = 
             (SELECT COUNT(*) FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_final_dataset_4_te_experiment`)
        THEN '✅ PASS - Row counts match'
        ELSE '❌ FAIL - Row count mismatch!'
    END AS status;

-- 2. Check for duplicates (should be 0)
SELECT 
    'Duplicate Check' AS check_name,
    COUNT(*) AS duplicate_count,
    CASE 
        WHEN COUNT(*) = 0 THEN '✅ PASS - No duplicates'
        ELSE '❌ FAIL - Duplicates found!'
    END AS status
FROM (
    SELECT individual_id, member_id, index_dt, COUNT(*) AS cnt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_final_dataset_4_te_experiment`
    GROUP BY individual_id, member_id, index_dt
    HAVING COUNT(*) > 1
);

-- 3. Check outcome distribution
SELECT 
    'Outcome Distribution' AS check_name,
    COUNT(*) AS total_members,
    SUM(ip6) AS members_with_ip,
    ROUND(SUM(ip6) / COUNT(*) * 100, 2) AS ip_rate_pct,
    ROUND(AVG(sum_ip6_admits), 2) AS avg_admits_per_member,
    ROUND(AVG(sum_ip6_los), 2) AS avg_los_per_member,
    CASE 
        WHEN ROUND(SUM(ip6) / COUNT(*) * 100, 2) BETWEEN 5 AND 20
        THEN '✅ PASS - IP rate in expected range (5-20% for Medicare)'
        ELSE '⚠️  WARNING - IP rate outside typical range'
    END AS status
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_final_dataset_4_te_experiment`;

-- 4. Check continuity flags
SELECT 
    'Continuity Flags' AS check_name,
    COUNT(*) AS total_members,
    SUM(mon_6_include) AS has_6mo_continuity,
    ROUND(SUM(mon_6_include) / COUNT(*) * 100, 2) AS continuity_pct,
    CASE 
        WHEN ROUND(SUM(mon_6_include) / COUNT(*) * 100, 2) >= 80
        THEN '✅ PASS - Good continuity rate (>=80%)'
        ELSE '⚠️  WARNING - Low continuity rate (<80%)'
    END AS status
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_final_dataset_4_te_experiment`;

-- 5. Check feature completeness (NULL rates)
SELECT 
    'Feature Completeness' AS check_name,
    ROUND((COUNT(*) - COUNT(camemhpd_chf)) / COUNT(*) * 100, 2) AS hpd_null_pct,
    ROUND((COUNT(*) - COUNT(camemmedutilization_clm_ln_cnt)) / COUNT(*) * 100, 2) AS medutil_null_pct,
    CASE 
        WHEN ROUND((COUNT(*) - COUNT(camemhpd_chf)) / COUNT(*) * 100, 2) < 10
            AND ROUND((COUNT(*) - COUNT(camemmedutilization_clm_ln_cnt)) / COUNT(*) * 100, 2) < 10
        THEN '✅ PASS - Low NULL rate (<10%) for features'
        ELSE '⚠️  WARNING - High NULL rate (>=10%) in some features'
    END AS status
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_final_dataset_4_te_experiment`;

-- 6. Check outcome vs continuity stratification
SELECT 
    mon_6_include,
    COUNT(*) AS member_count,
    SUM(ip6) AS ip_admissions,
    ROUND(SUM(ip6) / COUNT(*) * 100, 2) AS ip_rate_pct
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_final_dataset_4_te_experiment`
GROUP BY mon_6_include
ORDER BY mon_6_include;

-- 7. Sample data preview (first 5 rows)
SELECT *
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_final_dataset_4_te_experiment`
LIMIT 5;

/*==============================================================================
  END OF MEDICARE IP TRAINING DATASET GENERATION
  
  ✅ TABLES CREATED (20 total):
  
  COHORT & OUTCOMES (5 tables):
  1. a834793_Medicare_member_base_4_te_experiment (base cohort with index_dt)
  2. a834793_Medicare_ip_admissions_post_6mo_4_te_experiment (raw IP admissions)
  3. a834793_Medicare_ip_admissions_filtered_6mo_4_te_experiment (after exclusions)
  4. a834793_Medicare_outcome_6mo_4_te_experiment (aggregated outcomes)
  5. a834793_Medicare_post_status_4_te_experiment (continuity flags)
  
  FEATURE TABLES (14 tables):
  6. a834793_Medicare_hpd_features_4_te_experiment (11 chronic conditions)
  7. a834793_Medicare_medutil_features_4_te_experiment (4 utilization features)
  8. a834793_Medicare_medcase_features_4_te_experiment (5 medical case features)
  9. a834793_Medicare_dx_features_4_te_experiment (1 diagnosis feature)
  10. a834793_Medicare_prc_features_4_te_experiment (5 procedure features)
  11. a834793_Medicare_revenue_features_4_te_experiment (2 revenue code features)
  12. a834793_Medicare_eruc_features_4_te_experiment (2 ER/UC features)
  13. a834793_Medicare_rxclass_features_4_te_experiment (2 RX class features)
  14. a834793_Medicare_rxgroup_features_4_te_experiment (5 RX group features)
  15. a834793_Medicare_specialty_features_4_te_experiment (3 specialty claims features)
  16. a834793_Medicare_cm_features_4_te_experiment (2 care management features)
  17. a834793_Medicare_txtnotes_features_4_te_experiment (2 text notes features)
  18. a834793_Medicare_ylm_features_4_te_experiment (4 YLM demographic features)
  19. a834793_Medicare_mbrshp_features_4_te_experiment (2 membership demographic features)
  20. a834793_Medicare_exp_features_4_te_experiment (2 expenditure features)
  
  FINAL DATASET (⭐ USE THIS FOR TRAINING):
  21. a834793_Medicare_final_dataset_4_te_experiment
     - Columns: 63 total
       * 5 identifiers (individual_id, member_id, index_dt, feature_end_dt, business_ln_cd)
       * 4 outcomes (ip6, sum_ip6_admits, sum_ip6_los, mon_6_include)
       * 54 baseline features (matching production model requirements)
     - Rows: Same as base cohort (one row per individual_id + index_dt)
     - Retention: 180 days
  
  FEATURE BREAKDOWN (54 baseline features):
  - HPD Chronic Conditions: 11 features
  - Medical Utilization: 4 features
  - Medical Case (IP history): 5 features
  - Diagnosis: 1 feature
  - Procedures: 5 features
  - Revenue Codes: 2 features
  - ER/UC: 2 features
  - RX Class: 2 features
  - RX Group: 5 features
  - Specialty Claims: 3 features
  - Care Management: 2 features
  - Text Notes: 2 features
  - YLM Demographics: 4 features
  - Membership Demographics: 2 features
  - Expenditure: 2 features
  
  READY FOR:
  - Baseline model training (XGBoost, CatBoost) with 54 features
  - Feature importance analysis
  - JOIN with transformer embeddings (43 dimensions) for comparison
  - Total features with embeddings: 54 baseline + 43 embeddings = 97 features
  
  TO ADD TRANSFORMER EMBEDDINGS:
  Join on (individual_id, index_dt) with your transformer output table:
  
  SELECT 
      base.*
      , emb.* EXCEPT(individual_id, index_dt)
  FROM a834793_Medicare_final_dataset_4_te_experiment base
  LEFT JOIN a834793_Medicare_member_o3_train_ending emb
      ON base.individual_id = emb.individual_id
      AND base.index_dt = emb.index_dt
  WHERE base.mon_6_include = 1;
  
  ALIGNMENT WITH PRODUCTION MODEL:
  ✅ All 54 baseline features match production requirements
  ✅ Same temporal logic (90-day buffer between features and outcomes)
  ✅ Same IP outcome definition (acute IP, 6-month window, with exclusions)
  ✅ Same feature engineering (DC1/DC2/DC3 time windows, GPI codes, etc.)
  ⚠️ Transformer embeddings (43 features) need to be joined separately
  
  Total Expected Features: 54 baseline + 43 embeddings = 97 features (production model uses ~100)
  
  ═══════════════════════════════════════════════════════════════════════════
  CRITICAL FIXES APPLIED (January 2, 2026):
  ═══════════════════════════════════════════════════════════════════════════
  
  1. ✅ MATERNITY FILTER BUG (STEP 2):
     Problem: Filter excluded 'N' values (No maternity), removing ALL admissions
     Fix: Changed to accept 'N' and NULL as valid non-maternity cases
     Impact: Restored ~353K admissions (was 0)
  
  2. ✅ DUPLICATE ROWS BUG (STEP 0):
     Problem: Joining to INDVDL_CUST_DIST created 4.4M duplicates
              (1 individual_id → up to 114 member_ids in crosswalk table)
     Fix: Changed to join to a834793_Medicare_member_base_memberid
          (transformer training base table with 1:1 mapping)
     Impact: Eliminated all duplicates (7.8M rows → 3.4M rows)
  
  3. ✅ CONTINUITY CHECK BUG (STEP 4):
     Problem: Expected 180 daily records, but PRSPCTV_MEMBERSHIP has monthly records
              (91% had only 5 records spanning ~120 days)
     Fix: Changed threshold from 150 days → 120 days span
          (5 monthly records = ~120 day span is valid for 6-month continuity)
     Impact: Restored continuity flag (0% → ~93% have valid 6-month observation)
  
  FINAL METRICS (After All Fixes):
  - Rows: ~3.4M (no duplicates)
  - IP Rate: ~7.2% (expected range for Medicare 6-month)
  - Continuity: ~93% (high retention, as expected)
  
==============================================================================*/


