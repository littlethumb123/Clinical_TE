/*==============================================================================
  MEDICARE HOLDOUT SET WITH FEATURES - NOT IN TRANSFORMER TRAINING (10% SAMPLE)
  
  Purpose: Create a holdout cohort (90%) that was NOT used in transformer training,
           including RAW transformer features + baseline IP features + outcomes
  
  Workflow:
  1. Get 90% of Medicare members NOT in transformer training (10% sample)
  2. Include their RAW transformer input features (will be used to generate embeddings)
  3. Join with baseline IP model features (54) + outcomes from feature generation
  4. Sample 30% from this holdout set
  5. Run 30% through transformer to GENERATE embeddings
  6. Train downstream Medicare IP model with baseline + newly generated embeddings
  
  Why This Approach:
  - Transformer was trained on 10% sample → Need fresh 90% holdout data
  - This 90% holdout has raw transformer features (NOT embeddings yet)
  - From 90% holdout, take 30% for embedding generation via transformer inference
  - Result: ~27% of total Medicare population for downstream IP model training
  
  Input Tables:
  - a834793_Combined_All_LOB_o3_train_10pct_sample (10% used for transformer training)
  - a834793_Medicare_member_o3_train_ending (ALL Medicare members with RAW transformer input features)
  - a834793_Medicare_final_dataset_4_te_experiment (baseline IP features + outcomes)
  
  Output Table: a834793_Medicare_holdout_members_with_features
  
  Expected Result: ~90% of Medicare members (~3M rows)
  - RAW transformer input features (from a834793_Medicare_member_o3_train_ending)
  - 54 baseline IP model features (from a834793_Medicare_final_dataset_4_te_experiment)
  - 4 outcomes (ip6, sum_ip6_admits, sum_ip6_los, mon_6_include)
  - 5 identifiers (individual_id, member_id, index_dt, feature_end_dt, business_ln_cd)
  
  Next Steps After This Script:
  1. Sample 30% from this holdout table → ~900K members
  2. Use raw transformer features as input to transformer model to generate embeddings
  3. Join newly generated embeddings with baseline IP features
  4. Train Medicare IP model (XGBoost/CatBoost) with baseline + embeddings
  
  Team: Clinical & Social Determinants Intelligence (CSDI)
  Owner: Pritha Ghosh (pritha.ghosh@cvshealth.com)
  Cost Center: 13070
  Last Updated: January 2026
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_holdout_members_with_features`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_holdout_members_with_features`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
WITH training_sample AS (
    -- Get Medicare members in the 10% training sample
    SELECT DISTINCT
        individual_id,
        index_dt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_10pct_sample`
    WHERE lob = 'Medicare'
),
holdout_members AS (
    -- Get ALL columns (raw transformer features) for holdout members NOT in training sample
    SELECT *
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_o3_train_ending`
    WHERE NOT EXISTS (
        SELECT 1
        FROM training_sample ts
        WHERE ts.individual_id = `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_o3_train_ending`.individual_id
          AND ts.index_dt = `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_o3_train_ending`.index_dt
    )
)
-- Join holdout members (with raw transformer features) with baseline IP features and outcomes
SELECT 
    feat.*,
    hm.* EXCEPT(individual_id, index_dt)  -- Get all raw transformer features except duplicate keys
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_final_dataset_4_te_experiment` feat
INNER JOIN holdout_members hm
    ON feat.individual_id = hm.individual_id
    AND feat.index_dt = hm.index_dt
;


/*==============================================================================
  VALIDATION QUERIES
==============================================================================*/

-- 1. Check holdout set size and IP rate
SELECT 
    'Holdout Set with Baseline IP + Raw Transformer Features' AS check_name,
    COUNT(*) AS holdout_members,
    COUNT(DISTINCT individual_id) AS unique_individuals,
    (SELECT COUNT(*) FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_o3_train_ending`) AS total_medicare_members,
    (SELECT COUNT(*) FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_10pct_sample` WHERE lob = 'Medicare') AS training_sample_size,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_o3_train_ending`), 1) AS holdout_pct,
    SUM(CASE WHEN ip6 = 1 THEN 1 ELSE 0 END) AS members_with_ip,
    ROUND(SUM(CASE WHEN ip6 = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS ip_rate_pct,
    SUM(mon_6_include) AS evaluable_members,
    ROUND(SUM(mon_6_include) * 100.0 / COUNT(*), 1) AS evaluable_pct
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_holdout_members_with_features`;

-- 2. Verify no overlap with training sample (should be 0)
SELECT
    'No Overlap Check' AS check_name,
    COUNT(*) AS overlap_count,
    CASE
        WHEN COUNT(*) = 0 THEN '✅ PASS - No overlap with training sample'
        ELSE '❌ FAIL - Found overlap with training sample!'
    END AS status
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_holdout_members_with_features` h
INNER JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_10pct_sample` t
    ON h.individual_id = t.individual_id
    AND h.index_dt = t.index_dt
WHERE t.lob = 'Medicare';

-- 3. Check feature completeness (baseline IP features + raw transformer features + outcomes + identifiers)
SELECT
    'Feature Completeness' AS check_name,
    COUNT(*) AS total_columns,
    CONCAT('Table has ', CAST(COUNT(*) AS STRING), ' columns (baseline IP + raw transformer features)') AS description
FROM `edp-prod-storage.edp_ent_sdoheir_cns.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'a834793_Medicare_holdout_members_with_features';

-- 3b. Verify key columns are present
SELECT
    'Key Columns Check' AS check_name,
    SUM(CASE WHEN column_name = 'individual_id' THEN 1 ELSE 0 END) AS has_individual_id,
    SUM(CASE WHEN column_name = 'index_dt' THEN 1 ELSE 0 END) AS has_index_dt,
    SUM(CASE WHEN column_name = 'ip6' THEN 1 ELSE 0 END) AS has_outcome,
    SUM(CASE WHEN column_name = 'camemhpd_chf' THEN 1 ELSE 0 END) AS has_baseline_features,
    CASE
        WHEN SUM(CASE WHEN column_name = 'individual_id' THEN 1 ELSE 0 END) = 1
             AND SUM(CASE WHEN column_name = 'ip6' THEN 1 ELSE 0 END) = 1
             AND SUM(CASE WHEN column_name = 'camemhpd_chf' THEN 1 ELSE 0 END) = 1
        THEN '✅ PASS - Key columns present'
        ELSE '❌ FAIL - Missing key columns!'
    END AS status
FROM `edp-prod-storage.edp_ent_sdoheir_cns.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'a834793_Medicare_holdout_members_with_features';

-- 4. Preview first 5 rows
SELECT *
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_holdout_members_with_features`
LIMIT 5;


/*==============================================================================
  NEXT STEP: SAMPLE 30% FOR EMBEDDING GENERATION
  
  This table contains 90% holdout members with baseline IP features + RAW transformer features.
  Sample 30% for embedding generation and downstream IP model training:
  
  -- Step 1: Sample 30% from holdout
  CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_holdout_30pct_sample`
  OPTIONS (
      labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
      , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
  )
  AS
  SELECT *
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_holdout_members_with_features`
  WHERE RAND() < 0.30
    AND mon_6_include = 1;  -- Only members with full 6-month observation
  
  Expected Sample Size: ~900K members (30% of ~3M holdout)
  
  -- Step 2: Run transformer inference
  Use the raw transformer features in the 30% sample as input to the transformer model
  to generate embeddings (43 dimensions)
  
  -- Step 3: Join embeddings with baseline features
  Once embeddings are generated, join them back to create final training dataset:
  
  SELECT 
      sample.*,
      embeddings.emb6, embeddings.emb7, ..., embeddings.emb254
  FROM a834793_Medicare_holdout_30pct_sample sample
  INNER JOIN <newly_generated_embeddings_table> embeddings
      ON sample.individual_id = embeddings.individual_id
      AND sample.index_dt = embeddings.index_dt
  
  -- Step 4: Train downstream Medicare IP model
  - Baseline only: 54 features
  - Embeddings only: 43 features  
  - Combined: 54 + 43 = 97 features
  
==============================================================================*/

