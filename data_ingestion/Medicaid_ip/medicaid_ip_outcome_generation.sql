/*==============================================================================
  MEDICAID CLINICAL TRANSFORMER - IP OUTCOME CREATION
  
  Purpose: Create acute inpatient admission outcome labels for downstream
           classification evaluation of transformer embeddings (linear probe)
  
  Timeline Alignment with Transformer Features (medicaid_data_prep.sql):
  - Uses SAME membership base: a834793_Medicaid (2023 calendar year)
  - Uses SAME index_dt per member (ONE random eligibility month)
  - Features lookback: 36 months before index_dt (ends 90 days before index_dt)
  - Outcome window: 181 days AFTER index_dt (starts 1 day after index_dt)
  
  Methodology from Medicaid IP Model (IP_model_outcome.sql):
  - Acute IP: COE IDs 10200, 10700, 10800
  - Event window: index_dt + 1 day to index_dt + 181 days
  - Binary flag: 1 if ANY acute IP admission in window, 0 otherwise
  
  Owner: Pritha Ghosh (pritha.ghosh@cvshealth.com)
  Team: Clinical & Social Determinants Intelligence (CSDI)
  
  ⚠️ RETRAINING: When retraining transformer, regenerate this table AFTER
     regenerating a834793_Medicaid to ensure index_dt alignment.
  
==============================================================================*/

/*==============================================================================
  STEP 1: EXTRACT ACUTE IP ADMISSION CASES
  
  Purpose: Identify all acute inpatient admissions within 181-day outcome window
  
  Data Source: ASDB_ICE_IP (Inpatient Clinical Events)
  
  Acute IP Definition (from Medicaid IP model):
  - asdb_coe_id IN (10200, 10700, 10800)
  - 10200 = Acute Medical/Surgical
  - 10700 = Acute Behavioral Health  
  - 10800 = Acute Rehabilitation
  
  Timeline:
  - Start: index_dt + 1 day (no overlap with features)
  - End: index_dt + 181 days (6-month prediction horizon)
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_outcome_ip_cases_4_te_experiment`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_outcome_ip_cases_4_te_experiment`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
SELECT 
    -- Member identifier (matches transformer cohort)
    st.individual_id
    , st.index_dt
    
    -- IP admission details
    , mc.asdb_event_start_dt
    , mc.asdb_event_end_dt
    , mc.final_discharge_dt
    , mc.prindiag
    
    -- Acute vs Non-Acute classification (IP model methodology)
    , CASE WHEN mc.asdb_coe_id IN (10200, 10700, 10800) THEN 'Acute'
           ELSE 'Non-Acute'
      END AS ip_type
    
    -- Length of stay metrics
    , DATE_DIFF(mc.final_discharge_dt, mc.asdb_event_start_dt, DAY) AS calc_los
    , mc.event_ct
    , mc.admit_los
    , mc.paid_los
    
    -- Cost
    , mc.cost AS ip_paid_amt

FROM
    -- Transformer membership base (ONE random index_dt per member)
    (SELECT DISTINCT 
        individual_id
        , index_dt 
     FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid`
    ) AS st

INNER JOIN 
    -- Inpatient Clinical Events table (Medicaid)
    `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ICE_IP` AS mc
        ON st.individual_id = mc.asdb_member_key

WHERE 
    -- OUTCOME WINDOW: 181 days AFTER index_dt
    -- Start: 1 day after index_dt (ensures no overlap with features)
    -- End: 181 days after index_dt (6-month prediction horizon)
    CAST(mc.asdb_event_start_dt AS DATE) 
        BETWEEN DATE_ADD(st.index_dt, INTERVAL 1 DAY) 
            AND DATE_ADD(st.index_dt, INTERVAL 181 DAY)
    
    -- Only primary events (avoid double-counting)
    AND mc.event_ct = 1
;


/*==============================================================================
  STEP 2: CREATE MEMBER-LEVEL OUTCOME FLAGS
  
  Purpose: Aggregate IP cases into member-level binary flags and summary metrics
  
  Output columns:
  - individual_id: Member identifier (matches transformer cohort)
  - index_dt: Prediction point (matches transformer features)
  - acute_ip_flag: Binary (1 = any acute IP admission, 0 = none)
  - sum_acute_ip_admits: Count of acute admissions
  - sum_acute_calc_los: Total length of stay
  - sum_acute_ip_cost: Total IP cost
  
  Note: LEFT JOIN ensures all members in transformer cohort are included,
        even those with no IP admissions (acute_ip_flag = 0)
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_outcome_ip_4_te_experiment`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_outcome_ip_4_te_experiment`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
WITH acute AS (
    -- Aggregate acute IP cases per member
    SELECT
        individual_id
        
        -- Binary outcome flag (primary target for linear probe)
        , CASE WHEN SUM(event_ct) > 0 THEN 1 
               ELSE 0 
          END AS acute_ip_flag
        
        -- Count of admissions
        , SUM(event_ct) AS sum_acute_ip_admits
        
        -- Length of stay metrics
        , SUM(calc_los) AS sum_acute_calc_los
        , SUM(admit_los) AS sum_acute_admit_los
        , SUM(paid_los) AS sum_acute_paid_los
        
        -- Cost
        , SUM(ip_paid_amt) AS sum_acute_ip_cost
        
    FROM  
        `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_outcome_ip_cases`
    WHERE 
        ip_type = 'Acute'
    GROUP BY 
        individual_id
)
SELECT 
    -- Member identifier (matches transformer training data)
    st.individual_id
    , st.index_dt
    
    -- PRIMARY OUTCOME: Binary acute IP flag (for linear probe evaluation)
    , COALESCE(a.acute_ip_flag, 0) AS acute_ip_flag
    
    -- SECONDARY OUTCOMES: Admission count, LOS, cost (for regression tasks)
    , COALESCE(a.sum_acute_ip_admits, 0) AS sum_acute_ip_admits
    , COALESCE(a.sum_acute_calc_los, 0) AS sum_acute_calc_los
    , COALESCE(a.sum_acute_admit_los, 0) AS sum_acute_admit_los
    , COALESCE(a.sum_acute_paid_los, 0) AS sum_acute_paid_los
    , COALESCE(a.sum_acute_ip_cost, 0) AS sum_acute_ip_cost

FROM 
    -- Full transformer cohort (ensures all members included)
    (SELECT DISTINCT 
        individual_id
        , index_dt 
     FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid`
    ) AS st

LEFT JOIN 
    acute AS a
        ON st.individual_id = a.individual_id
;


/*==============================================================================
  STEP 3: VALIDATION QUERIES (Run after table creation)
  
  Purpose: Verify outcome table quality and distribution
  
==============================================================================*/

-- Query 1: Check outcome distribution (expected ~5-10% positive rate for Medicaid)
-- SELECT 
--     COUNT(*) AS total_members,
--     SUM(acute_ip_flag) AS members_with_ip,
--     AVG(acute_ip_flag) AS ip_rate,
--     AVG(sum_acute_ip_admits) AS avg_admits_overall,
--     AVG(CASE WHEN acute_ip_flag = 1 THEN sum_acute_ip_admits ELSE NULL END) AS avg_admits_among_ip
-- FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_outcome_ip`;

-- Query 2: Verify member count matches transformer cohort
-- SELECT 
--     (SELECT COUNT(DISTINCT individual_id) FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid`) AS transformer_cohort_count,
--     (SELECT COUNT(DISTINCT individual_id) FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_outcome_ip`) AS outcome_count,
--     CASE WHEN 
--         (SELECT COUNT(DISTINCT individual_id) FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid`) =
--         (SELECT COUNT(DISTINCT individual_id) FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_outcome_ip`)
--     THEN 'MATCH ✓' ELSE 'MISMATCH ✗' END AS alignment_check;

-- Query 3: Verify index_dt alignment (should return 0 mismatches)
-- SELECT COUNT(*) AS mismatched_index_dt
-- FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid` AS base
-- LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_outcome_ip` AS outcome
--     ON base.individual_id = outcome.individual_id
-- WHERE base.index_dt != outcome.index_dt;

-- Query 4: Verify no temporal overlap (outcome should start AFTER features end)
-- Features end at: index_dt - 90 days
-- Outcome starts at: index_dt + 1 day
-- Gap should be 91 days minimum
-- SELECT 
--     'Features end' AS boundary,
--     MIN(DATE_SUB(index_dt, INTERVAL 90 DAY)) AS earliest,
--     MAX(DATE_SUB(index_dt, INTERVAL 90 DAY)) AS latest
-- FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_outcome_ip`
-- UNION ALL
-- SELECT 
--     'Outcome starts' AS boundary,
--     MIN(asdb_event_start_dt) AS earliest,
--     MAX(asdb_event_start_dt) AS latest
-- FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_outcome_ip_cases`;


/*==============================================================================
  USAGE IN TRANSFORMER LINEAR PROBE EVALUATION
  
  After running this SQL, join the outcome to transformer embeddings:
  
  Python pseudocode:
  
  # Load outcome data
  outcome_df = pd.read_gbq(
      "SELECT individual_id, acute_ip_flag FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_outcome_ip`"
  )
  
  # Load transformer embeddings (from o3_train_ending)
  train_df = pd.read_gbq(
      "SELECT individual_id, ... FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_o3_train_ending`"  
  )
  
  # Join on individual_id
  merged_df = train_df.merge(outcome_df, on='individual_id', how='inner')
  
  # Extract embeddings and train linear probe
  X = model.get_embeddings(merged_df)  # [N, embedding_dim]
  y = merged_df['acute_ip_flag'].values  # [N,]
  
  from sklearn.linear_model import LogisticRegression
  probe = LogisticRegression()
  probe.fit(X, y)
  
==============================================================================*/