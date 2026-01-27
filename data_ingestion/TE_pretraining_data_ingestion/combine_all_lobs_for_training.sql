/*==============================================================================
  COMBINED MULTI-LOB TRANSFORMER TRAINING DATASET
  
  Purpose: Combine Commercial, Medicare, and Medicaid transformer input tables
           into a single unified dataset for model retraining
  
  Developer: Pritha Ghosh (pritha.ghosh@cvshealth.com)
  Created: 2024
  Cost Center: 13070
  
  ============================================================================
  
  📊 INPUT TABLES:
  
  1. Commercial: edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_o3_train_ending
     - Columns: individual_id, index_dt, gender_cd, age_in_months, cd, target, dt_cnt
     - Population: ~X members from 2022 enrollment
     
  2. Medicare: edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_o3_train_ending
     - Columns: individual_id, index_dt, gender_cd, age_in_months, cd, target, dt_cnt
     - Population: ~X members from 2022 enrollment
     
  3. Medicaid: edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_o3_train_ending
     - Columns: individual_id, index_dt, gender_cd, age_in_months, cd, target, dt_cnt
     - Population: ~X members from 2022 enrollment
     - ⚠️ NOTE: index_dt column now included (added in recent update)
  
  ============================================================================
  
  🎯 OUTPUT TABLE:
  
  Table: edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_ending
  
  Schema:
  - individual_id: STRING (patient identifier)
  - lob: STRING (Line of Business: 'Commercial', 'Medicare', 'Medicaid')
  - index_dt: DATE (reference date for scoring)
  - gender_cd: STRING (temporal sequence: "1*1*0*1*0*...")
  - age_in_months: STRING (temporal sequence: "540*541*542*...")
  - cd: STRING (INPUT medical code sequences: "123,456*789,101*..." ~84k vocab)
  - target: STRING (TARGET code sequences for next-day prediction: "45*67*89*..." ~5k vocab)
  - dt_cnt: INT64 (number of days in sequence, ≤200)
  
  Expected Output: ~[Commercial + Medicare + Medicaid] total members
  
  ============================================================================
  
  ⚠️ IMPORTANT NOTES:
  
  1. All three tables now include index_dt column
     - Recent update added index_dt to Medicaid training pipeline
     - All LOBs now have consistent schema
  
  2. All three tables share the SAME w2ind vocabulary
     - Input codes: a834793_member_w2ind (~84k codes)
     - Target codes: a834793_member_w2ind_target (~5k codes)
     - This ensures code indices are consistent across LOBs
     - Critical for combined model training
  
  3. Member IDs may overlap between LOBs (dual-eligible members)
     - individual_id + lob combination is unique
     - Same person in multiple LOBs will appear as separate rows
  
  4. Target column enables next-day prediction training
     - Input (cd): Codes from day N
     - Target (target): Codes from day N+1
     - Model learns to predict tomorrow's healthcare events
  
  ============================================================================
  
  📋 DATA QUALITY CHECKS (Run After Creation):
  
  1. Check record counts by LOB:
  
     SELECT 
         lob
         , COUNT(*) as member_count
         , AVG(dt_cnt) as avg_days_per_member
         , MIN(dt_cnt) as min_days
         , MAX(dt_cnt) as max_days
     FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_ending`
     GROUP BY lob
     ORDER BY lob;
  
  2. Check for duplicate members within same LOB:
  
     SELECT 
         lob
         , individual_id
         , COUNT(*) as duplicate_count
     FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_ending`
     GROUP BY lob, individual_id
     HAVING COUNT(*) > 1;
     -- Should return 0 rows
  
  3. Check for dual-eligible members (same individual_id in multiple LOBs):
  
     SELECT 
         individual_id
         , STRING_AGG(lob, ', ' ORDER BY lob) as lobs
         , COUNT(DISTINCT lob) as lob_count
     FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_ending`
     GROUP BY individual_id
     HAVING COUNT(DISTINCT lob) > 1
     ORDER BY lob_count DESC, individual_id;
  
  4. Validate code sequences (no empty sequences):
  
     SELECT 
         lob
         , COUNTIF(cd IS NULL OR cd = '') as empty_code_sequences
         , COUNTIF(target IS NULL OR target = '') as empty_target_sequences
         , COUNTIF(gender_cd IS NULL OR gender_cd = '') as empty_gender_sequences
         , COUNTIF(age_in_months IS NULL OR age_in_months = '') as empty_age_sequences
     FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_ending`
     GROUP BY lob;
     -- All counts should be 0
  
  5. Check index_dt distribution:
  
     SELECT 
         lob
         , COUNTIF(index_dt IS NULL) as null_index_dt_count
         , COUNTIF(index_dt IS NOT NULL) as non_null_index_dt_count
         , MIN(index_dt) as earliest_index_dt
         , MAX(index_dt) as latest_index_dt
     FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_ending`
     GROUP BY lob;
     -- All LOBs should have non-NULL index_dt (Medicaid updated to include it)
  
  ============================================================================
  
  🎓 USAGE FOR TRANSFORMER RETRAINING:
  
  Python Data Loading Example:
  
  ```python
  from google.cloud import bigquery
  
  # Load combined dataset
  query = '''
  SELECT 
      individual_id
      , lob
      , index_dt
      , gender_cd
      , age_in_months
      , cd
      , target
      , dt_cnt
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_ending`
  WHERE dt_cnt >= 10  -- Filter to members with at least 10 days of history
  '''
  
  client = bigquery.Client()
  df = client.query(query).to_dataframe()
  
  # Split by LOB if needed
  df_commercial = df[df['lob'] == 'Commercial']
  df_medicare = df[df['lob'] == 'Medicare']
  df_medicaid = df[df['lob'] == 'Medicaid']
  
  # Or train on all LOBs combined
  print(f"Total members for training: {len(df):,}")
  print(f"Commercial: {len(df_commercial):,}")
  print(f"Medicare: {len(df_medicare):,}")
  print(f"Medicaid: {len(df_medicaid):,}")
  ```
  
  ============================================================================
  
  ⚠️ CONSIDERATIONS FOR MODEL RETRAINING:
  
  1. SHARED VOCABULARY:
     - All LOBs use same w2ind lookup table
     - Code indices are consistent across LOBs
     - Same medical code = same index in all LOBs
     - This enables unified model training
  
  2. LOB-SPECIFIC PATTERNS:
     - Different LOBs may have different code distributions
     - Consider LOB as additional feature during training
     - Or stratify sampling by LOB for balanced training
  
  3. DUAL-ELIGIBLE MEMBERS:
     - Same person in multiple LOBs appears as separate rows
     - Their medical histories may overlap or be complementary
     - Consider deduplication strategy if needed
  
  4. NEXT-DAY PREDICTION TRAINING:
     - Input (cd): Current day medical codes
     - Target (target): Next day medical codes to predict
     - Use supervised learning with teacher forcing
     - Evaluate using accuracy, F1, or AUROC on target codes
  
  5. SAMPLE SIZE CONSIDERATIONS:
     - Check if LOBs are balanced
     - May need stratified sampling to ensure representation
     - Consider LOB-specific weights in training
  
  ============================================================================
==============================================================================*/


-- ============================================================================
-- MAIN QUERY: Combine Commercial, Medicare, and Medicaid datasets
-- ============================================================================

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_ending`;

CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_ending`
CLUSTER BY lob, individual_id
OPTIONS (
    labels=[("owner", "pritha_ghosh_cvshealth_com"), ("costcenter", "13070")],
    description="Combined Commercial, Medicare, and Medicaid transformer training dataset with ~200 days lookback per member",
    expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 365 DAY)
) AS

-- Commercial members
SELECT 
    individual_id,
    'Commercial' AS lob,
    index_dt,
    gender_cd,
    age_in_months,
    cd,
    target,  -- Next-day prediction target codes
    dt_cnt
FROM 
    `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_o3_train_ending`

UNION ALL

-- Medicare members  
SELECT 
    individual_id,
    'Medicare' AS lob,
    index_dt,
    gender_cd,
    age_in_months,
    cd,
    target,  -- Next-day prediction target codes
    dt_cnt
FROM 
    `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_member_o3_train_ending`

UNION ALL

-- Medicaid members (now includes index_dt from updated pipeline)
SELECT 
    individual_id,
    'Medicaid' AS lob,
    index_dt,  -- Now included in Medicaid training pipeline
    gender_cd,
    age_in_months,
    cd,
    target,  -- Next-day prediction target codes
    dt_cnt
FROM 
    `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_o3_train_ending`
;


/*==============================================================================
  ✅ COMBINED TABLE CREATED SUCCESSFULLY
  
  Next Steps:
  
  1. Run Data Quality Checks (see queries above in header comments)
  
  2. Verify record counts match expectations:
     - Compare to individual LOB table counts
     - Confirm UNION ALL captured all records
  
  3. Export to Python for transformer training:
     - Use BigQuery Python client or pandas.read_gbq()
     - Process sequences into token arrays
     - Train unified transformer model
  
  4. Consider creating LOB-specific subsets if needed:
     - Separate models per LOB (if patterns differ significantly)
     - Or single model with LOB as additional feature
  
  5. Document training decisions:
     - Whether to include dual-eligible members
     - How to handle NULL index_dt for Medicaid
     - Sampling strategy (balanced vs. proportional)
  
  ============================================================================
  
  📊 ESTIMATED TABLE SIZE:
  
  Assuming:
  - Commercial: ~100K members
  - Medicare: ~100K members  
  - Medicaid: ~50K members
  
  Total: ~250K members
  
  Storage: Each row contains ~1-5 KB of string data (sequences)
  Estimated size: ~500 MB - 1 GB (compressed in BigQuery)
  
  Query cost: Full table scan ~$0.005 per GB scanned
  
  ============================================================================
==============================================================================*/
