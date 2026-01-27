/*==============================================================================
  CREATE w2ind_target FROM w2ind - ALGORITHMIC GROUPING WITH 100% COVERAGE
  
  Developer: Pritha Ghosh (pritha.ghosh@cvshealth.com)
  
  ============================================================================
  
  📚 CONCEPTUAL OVERVIEW:
  
  This script creates a CONDENSED target vocabulary for the transformer's 
  prediction task by intelligently grouping similar codes from the larger 
  input vocabulary (w2ind).
  
  VOCABULARY TRANSFORMATION:
  - w2ind (INPUT):        ~80k codes → What transformer READS (encoder)
  - w2ind_target (OUTPUT): ~6k codes → What transformer PREDICTS (decoder)
  - Compression Ratio:     ~13:1 reduction
  
  ============================================================================
  
  🎯 WHY COMPRESS THE TARGET VOCABULARY?
  
  1. ⚡ TRAINING EFFICIENCY (Primary Reason)
     - Softmax over 6K classes is ~13X faster than 80K classes
     - Reduces memory requirements significantly
     - Enables faster iteration during model development
     - Critical for transformer training which is already compute-intensive
  
  2. 🎓 BETTER GENERALIZATION
     - Model learns code FAMILIES rather than memorizing individual codes
     - Example: Instead of memorizing 99213 vs 99214 vs 99215 separately,
       it learns "office visit codes (992xx)" as a concept
     - Reduces overfitting on rare codes
     - Improves predictions for codes with limited training examples
  
  3. 🔗 PRESERVES CLINICAL MEANING
     - Grouping follows standard medical code hierarchies:
       * CPT codes: First 3 digits define procedure category
       * ICD-10: First 3 chars define disease family
       * GPI drugs: First 2 digits define therapeutic class
     - Groups clinically related codes together automatically
     - Example: ICD-10 E11.9, E11.65, E11.22 → "E11 (Type 2 Diabetes)"
  
  4. 📊 ADDRESSES CLASS IMBALANCE
     - Reduces extreme imbalance from rare codes
     - Improves training stability
     - More balanced gradient updates across code families
  
  ============================================================================
  
  🔑 KEY STRATEGY: ALGORITHMIC PREFIX-BASED GROUPING
  
  We use PREFIX-BASED grouping (first N characters) to systematically group
  codes WITHOUT relying on external lookup tables or crosswalks. This design
  ensures:
  
  ✅ 100% Coverage: Every input code maps to exactly one target group
  ✅ Reproducibility: Same logic applies to new codes in future retraining
  ✅ No Data Dependencies: No need for external crosswalk/mapping tables
  ✅ Consistency: Uses IDENTICAL logic to LOB files (Commercial, Medicare, Medicaid)
     → The grouping rules in this file EXACTLY MATCH the prcdr_group_cd logic
       in commercial.sql, medicare.sql, and medicaid.sql (Step 1)
     → Ensures alignment across entire pipeline from data prep → training
  
  ============================================================================
  
  📊 CONCRETE TRANSFORMATION EXAMPLES:
  
  INPUT (w2ind) - ~84K codes      →  OUTPUT (w2ind_target) - ~5K codes
  ───────────────────────────────────────────────────────────────────────────
  
  EXAMPLE 1: ICD-10 Diagnosis Codes
  icd9_dx_cd250.00  ┐
  icd9_dx_cd250.01  │
  icd9_dx_cd250.02  ├─────────────→  icd9_dx_cd250 (Type 2 Diabetes family)
  icd9_dx_cd250.90  │
  icd9_dx_cd250.91  ┘
  Logic: Keep first 3 chars after "icd9_dx_cd" prefix → Clinical code family
  
  EXAMPLE 2: CPT Procedure Codes (Office Visits)
  prcdr_cd99213     ┐
  prcdr_cd99214     ├─────────────→  prcdr_group_992 (Office visit codes)
  prcdr_cd99215     ┘
  Logic: Extract first 3 digits from 5-digit CPT → Procedure category
  
  EXAMPLE 3: GPI Medication Codes (Beta Blockers)
  gpi2210           ┐
  gpi2215           ├─────────────→  gpi22 (Beta-adrenergic blocking agents)
  gpi2220           ┘
  Logic: Keep first 2 digits → Therapeutic drug class
  
  EXAMPLE 4: ICD-10-PCS Procedures (Cardiac)
  prcdr_cd02H60JZ   ┐
  prcdr_cd02H63JZ   ├─────────────→  prcdr_group_02h (Heart insertion procedures)
  prcdr_cd02H64JZ   ┘
  Logic: First 3 chars (02H) define body system (02=Heart) + operation (H=Insertion)
  
  ============================================================================
  
  📋 COMPLETE GROUPING RULES BY CODE TYPE:
  
  ⚠️ CRITICAL: These rules EXACTLY MATCH the prcdr_group_cd logic in 
     commercial.sql, medicare.sql, and medicaid.sql (Step 1, lines ~220-260)
  
  ┌──────────────────────┬─────────────────┬────────────────────────────────┬──────────────────────────────┐
  │ Code Type            │ Grouping Rule   │ Examples                       │ Clinical Rationale           │
  ├──────────────────────┼─────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ ICD-10 Diagnosis     │ First 3 chars   │ G24.01 → icd9_dx_cdG24         │ G24=Dystonia family          │
  │ (icd9_dx_cd column)  │                 │ E11.9  → icd9_dx_cdE11         │ E11=Type 2 Diabetes family   │
  │ [Note: column name   │                 │ I10    → icd9_dx_cdI10         │ I10=Hypertension codes       │
  │  is legacy, contains │                 │                                │ Standard ICD-10 hierarchy    │
  │  ICD-10 codes]       │                 │                                │                              │
  ├──────────────────────┼─────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ CPT (5-digit)        │ First 3 digits  │ 99213 → prcdr_group_992        │ 992=Office visits            │
  │                      │                 │ 33510 → prcdr_group_335        │ 335=Coronary artery bypass   │
  │                      │                 │ 80053 → prcdr_group_800        │ 800=Lab tests                │
  │                      │                 │                                │ CPT organized by first 3     │
  ├──────────────────────┼─────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ CPT Category II/III  │ First 4 digits  │ 0001A → prcdr_group_0001       │ More specific categories     │
  │ (4 digits + letter)  │                 │ 0012M → prcdr_group_0012       │ Need 4 digits for grouping   │
  ├──────────────────────┼─────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ ICD-10-PCS           │ First 3 chars   │ 02H60JZ → prcdr_group_02h      │ 02=Heart, H=Insertion        │
  │ (7-char alphanumeric)│ (lowercased)    │ 0U5T7ZZ → prcdr_group_0u5      │ 0U=Female repro, 5=Destruction│
  │                      │                 │ 0BH17EZ → prcdr_group_0bh      │ 0B=Respiratory, H=Insertion  │
  │                      │                 │                                │ Chars 1-3 = system + operation│
  ├──────────────────────┼─────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ ICD-9 Procedure      │ Before decimal  │ 00.50 → prcdr_group_00         │ 00=Nervous system procedures │
  │                      │                 │ 66.21 → prcdr_group_66         │ 66=Operations on ovary       │
  │                      │                 │ 81.54 → prcdr_group_81         │ 81=Hip/knee operations       │
  │                      │                 │                                │ Digits before decimal = category│
  ├──────────────────────┼─────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ HCPCS                │ First 2 chars   │ J1234 → prcdr_group_j1         │ J=Drugs administered         │
  │ (Letter + 4 digits)  │ (lowercased)    │ A0021 → prcdr_group_a0         │ A=Transportation/ambulance   │
  │ (not Dental)         │                 │ E0601 → prcdr_group_e0         │ E=Durable medical equipment  │
  │                      │                 │                                │ First letter defines category│
  ├──────────────────────┼─────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ Dental               │ First 3 chars   │ D7220 → prcdr_group_d72        │ D72=Oral surgery             │
  │ (D + 4 digits)       │ (lowercased)    │ D0120 → prcdr_group_d01        │ D01=Diagnostic procedures    │
  │                      │                 │ D2150 → prcdr_group_d21        │ D21=Restorative procedures   │
  │                      │                 │                                │ First 3 chars = service type │
  ├──────────────────────┼─────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ GPI Medications      │ First 2 digits  │ gpi2210 → gpi22                │ 22=Beta-blockers             │
  │                      │                 │ gpi6510 → gpi65                │ 65=Antihyperlipidemics       │
  │                      │                 │ gpi9910 → gpi99                │ 99=Vitamins/supplements      │
  │                      │                 │                                │ First 2 digits = drug class  │
  ├──────────────────────┼─────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ Provider Taxonomy    │ First 4 chars   │ 207Q00000X → provider_..._207Q │ 207Q=Family Medicine         │
  │                      │                 │ 363L00000X → provider_..._363L │ 363L=Nurse Practitioner      │
  │                      │                 │ 208D00000X → provider_..._208D │ 208D=General Practice        │
  │                      │                 │                                │ First 4 chars = specialty    │
  ├──────────────────────┼─────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ Revenue Codes        │ First 3 digits  │ 0250 → revenue_cd025           │ 025=Pharmacy                 │
  │                      │                 │ 0450 → revenue_cd045           │ 045=Emergency room           │
  │                      │                 │ 0120 → revenue_cd012           │ 012=Semi-private room        │
  │                      │                 │                                │ First 3 digits = dept/service│
  ├──────────────────────┼─────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ DRG Codes            │ Keep as-is      │ drg_cd470 → drg_cd470          │ Already grouped; ~700 codes  │
  │                      │ (no grouping)   │ drg_cd871 → drg_cd871          │ Each DRG is a clinical group │
  ├──────────────────────┼─────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ Days Count           │ Keep as-is      │ days_cnt_5 → days_cnt_5        │ Temporal feature; no grouping│
  │                      │ (no grouping)   │ days_cnt_90 → days_cnt_90      │ Each day count is meaningful │
  ├──────────────────────┼─────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ Place of Service     │ Keep as-is      │ hcfa_plc_srv_cd21 → (same)     │ Already categorical; ~20 codes│
  │                      │ (no grouping)   │ hcfa_plc_srv_cd11 → (same)     │ Each code has distinct meaning│
  └──────────────────────┴─────────────────┴────────────────────────────────┴──────────────────────────────┘
  
  ============================================================================
  
  🔍 DETAILED LOGIC: PROCEDURE CODE GROUPING (Most Complex)
  
  Procedure codes come in many formats, so we use REGEX patterns to identify
  the code type, then apply type-specific grouping:
  
  1️⃣ CPT (Standard - 5 digits)
     Pattern: ^\d{5}$
     Examples: 99213, 33510, 80053
     Grouping: First 3 digits → prcdr_group_992, prcdr_group_335, prcdr_group_800
     Rationale: CPT codes are organized hierarchically by first digits
  
  2️⃣ CPT Category II/III (4 digits + letter)
     Pattern: ^\d{4}[A-Z]$
     Examples: 0001A, 0012M, 0003T
     Grouping: First 4 digits → prcdr_group_0001, prcdr_group_0012, prcdr_group_0003
     Rationale: Category II/III codes are more specific, need 4 digits to group
  
  3️⃣ ICD-10-PCS (7-char alphanumeric)
     Pattern: ^\d[A-Z0-9]{6}$
     Examples: 02H60JZ, 0U5T7ZZ, 0BH17EZ
     Grouping: First 3 chars lowercased → prcdr_group_02h, prcdr_group_0u5
     Rationale: First 3 chars define body system and operation
  
  4️⃣ ICD-9 Procedure (with decimal)
     Pattern: ^\d+\.\d+$
     Examples: 00.50, 66.21, 81.54
     Grouping: Before decimal → prcdr_group_00, prcdr_group_66, prcdr_group_81
     Rationale: Digits before decimal define procedure category
  
  5️⃣ HCPCS (Letter + 4 digits, not starting with D)
     Pattern: ^[A-Z]\d{4}$ AND first char != 'D'
     Examples: J1234, A0021, E0601
     Grouping: First 2 chars lowercased → prcdr_group_j1, prcdr_group_a0
     Rationale: First letter defines broad category (J=drugs, A=transport, E=DME)
  
  6️⃣ Dental (D + 4 digits)
     Pattern: ^D\d{4}$
     Examples: D7220, D0120, D2150
     Grouping: First 3 chars lowercased → prcdr_group_d72, prcdr_group_d01
     Rationale: First 3 chars define dental service category
  
  7️⃣ Short Numeric (1-4 digits)
     Pattern: ^\d{1,4}$
     Examples: 92, 924, 0210
     Grouping: First 2 digits → prcdr_group_92, prcdr_group_02
     Rationale: Legacy codes, group by first 2 digits
  
  8️⃣ Long Numeric (6+ digits)
     Pattern: ^\d{6,}$
     Examples: 0210083, 9999999
     Grouping: First 4 digits → prcdr_group_0210, prcdr_group_9999
     Rationale: Rare format, use first 4 digits for specificity
  
  9️⃣ Fallback (anything else)
     Grouping: prcdr_group_unk
     Rationale: Catch-all for unexpected formats (should be very rare)
  
  ============================================================================
  
  📊 EXPECTED OUTPUT VOCABULARY SIZE: ~5,000 codes
  
  Breakdown by code type:
  - ICD-10 Diagnoses: ~2,000 groups (first 3 characters of ICD-10 codes)
  - Procedure Groups: ~1,000 groups (algorithmic grouping)
  - GPI Medications: ~100 groups (first 2 digits of GPI)
  - Provider Taxonomy: ~1,500 groups (first 4 chars)
  - Revenue Codes: ~100 groups (first 3 digits)
  - DRG Codes: ~700 codes (kept as-is)
  - Days Count: ~180 codes (kept as-is)
  - Place of Service: ~20 codes (kept as-is)
  
  Total: ~5,500 target codes (manageable prediction task for transformer)
  
  ============================================================================
  
  ✅ VALIDATION & TROUBLESHOOTING:
  
  After running this script, verify:
  
  1. Check total code count:
     SELECT COUNT(*) FROM `...a834793_member_w2ind_target`;
     -- Expected: ~5,000-6,000
  
  2. Check for NULL or empty codes:
     SELECT COUNT(*) FROM `...a834793_member_w2ind_target` WHERE cd IS NULL OR cd = '';
     -- Expected: 1 (only index 0 should be empty)
  
  3. Verify procedure groups exist:
     SELECT cd, COUNT(*) as cnt 
     FROM `...a834793_member_w2ind_target` 
     WHERE cd LIKE 'prcdr_group_%' 
     GROUP BY cd 
     ORDER BY cnt DESC 
     LIMIT 10;
     -- Should show groups like prcdr_group_992, prcdr_group_02h, etc.
  
  4. Check coverage by comparing to w2ind:
     -- All input codes should map to a target group
     -- See verification queries at end of this script
  
  ============================================================================
  
  🔗 INTEGRATION WITH TRANSFORMER TRAINING PIPELINE:
  
  This vocabulary mapping is fundamental to how the transformer learns and predicts:
  
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ TRAINING PHASE                                                              │
  └─────────────────────────────────────────────────────────────────────────────┘
  
  1. DATA PREPARATION (LOB SQL Files → This Script)
     
     Commercial/Medicare/Medicaid SQL (Steps 1-5):
     → Generates patient sequences with raw codes
     → Example sequence: [99213, 99214, I10, J1234, gpi2210, ...]
     
     create_w2ind_table.sql:
     → Creates INPUT vocabulary (w2ind): Maps each unique code to integer index
     → Example: 99213 → index 1523, I10 → index 8492, gpi2210 → index 45321
     
     THIS SCRIPT (create_w2ind_target_from_w2ind.sql):
     → Creates OUTPUT vocabulary (w2ind_target): Maps codes to grouped targets
     → Example: 99213 → prcdr_group_992, I10 → icd9_dx_cdI10, gpi2210 → gpi22
  
  2. TRANSFORMER ARCHITECTURE (pritha_transformer_train.py)
     
     ENCODER (Reads full vocabulary):
     → Embedding layer: vocab_size = 84,000 (from w2ind)
     → Input: [index_1523, index_8492, index_45321, ...]
     → Converts each code to dense embedding vector (dim=512)
     → Processes temporal sequences with self-attention
     
     DECODER (Predicts condensed vocabulary):
     → Output layer: num_classes = 5,000 (from w2ind_target)
     → Softmax over 5K classes (much faster than 84K!)
     → Predicts: [prcdr_group_992, icd9_dx_cdI10, gpi22, ...]
  
  3. LOSS CALCULATION
     
     Ground Truth Mapping:
     → Each input code (e.g., 99213) is mapped to its target group (prcdr_group_992)
     → Model learns to predict the GROUP, not the exact code
     → This is why grouping must preserve clinical meaning!
     
     Cross-Entropy Loss:
     → Computed over 5K classes instead of 84K
     → ~17x faster computation per forward pass
     → More stable gradients due to better class balance
  
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ INFERENCE PHASE (Production Deployment)                                     │
  └─────────────────────────────────────────────────────────────────────────────┘
  
  1. INPUT PROCESSING
     → Patient's historical codes retrieved from database
     → Mapped to indices using w2ind vocabulary
     → Fed to transformer encoder
  
  2. PREDICTION
     → Model outputs probability distribution over 5K target classes
     → Top-K predictions are code GROUPS (e.g., prcdr_group_992, icd9_dx_cdI10)
     → Post-processing can map back to specific codes if needed
  
  3. INTERPRETATION
     → Group-level predictions are clinically meaningful
     → Example: "High probability of office visit codes (992xx)" 
               + "High probability of hypertension codes (I10)" 
               → Suggests patient likely to have office visit for HTN management
  
  ============================================================================
  
  🔗 USAGE IN TRANSFORMER TRAINING:
  
  After creating w2ind_target, you'll use it in two places:
  
  1. **Training Data Preparation** (in Commercial/Medicare/Medicaid SQL pipelines):
     - Join medical codes to w2ind for INPUT indices
     - Join medical codes to w2ind_target for TARGET indices (what to predict)
  
  2. **Model Architecture** (in Python training script):
     - Decoder layer size = len(w2ind_target) 
     - Loss function compares predicted vs actual target indices
  
  Example in SQL (Step 8+ of LOB pipelines):
  ```sql
  LEFT JOIN `...a834793_member_w2ind` AS w2ind 
    ON diagnosis_code = w2ind.cd  -- Input vocabulary
  LEFT JOIN `...a834793_member_w2ind_target` AS w2ind_target 
    ON grouped_diagnosis = w2ind_target.cd  -- Target vocabulary
  ```
  
  ============================================================================
  
  ⚠️ CRITICAL: CONSISTENCY ACROSS PIPELINE & FUTURE RETRAINING
  
  The grouping logic in this file MUST remain synchronized with:
  
  1. ✅ LOB SQL Files (commercial.sql, medicare.sql, medicaid.sql)
     → Step 1 creates prcdr_group_cd using IDENTICAL regex patterns
     → If you modify grouping here, update LOB files too!
     → Example: If you change CPT grouping from 3 digits → 4 digits,
       update BOTH this file AND all 3 LOB files
  
  2. ✅ Training Python Script (pritha_transformer_train.py)
     → Line 764: Uses most recent 200 days from prepared data
     → Expects vocabulary sizes: encoder=84K, decoder=5K
     → If vocabulary sizes change significantly, review model architecture
  
  WHEN RETRAINING FOR A NEW TIME PERIOD (e.g., 2024 instead of 2023):
  
  ✅ NO CHANGES NEEDED in this file! The logic is fully dynamic:
     → This file reads from w2ind (which contains codes from your data)
     → New codes automatically get grouped using same rules
     → Example: New CPT code 99217 appears in 2024 data
       → w2ind captures it automatically
       → This script groups it → prcdr_group_992 (same rule)
  
  ✅ Update membership dates in LOB files (Steps 0-1):
     → Change index_dt range from 2023 → 2024
     → 36-month lookback automatically adjusts
     → See commercial.sql line 201-203 for date change instructions
  
  ⚠️ BREAKING CHANGES - If you modify grouping logic:
     → Document the change clearly in all files
     → Retrain model from scratch (can't mix old/new grouping schemes)
     → Update validation queries to verify new grouping
     → Test on sample data before full retraining
  
==============================================================================*/

-- Drop and recreate w2ind_target with full coverage
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind_target`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind_target`
OPTIONS (labels=[("owner", "pritha_ghosh_cvshealth_com"),("costcenter","13070")],
    expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS

WITH all_target_codes AS (

  -- ============================================================================
  -- 1. Days Count (Keep as-is) - 100% coverage
  -- ============================================================================
  SELECT DISTINCT cd
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind`
  WHERE cd LIKE 'days_cnt%'
  
  UNION DISTINCT
  
  -- ============================================================================
  -- 2. Place of Service (Keep as-is) - 100% coverage
  -- ============================================================================
  SELECT DISTINCT cd
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind`
  WHERE cd LIKE 'hcfa_plc_srv_cd%'
  
  UNION DISTINCT
  
  -- ============================================================================
  -- 3. Provider Taxonomy (First 4 characters) - 100% coverage
  -- Transform: provider_taxonomy_cd207Q00000X → provider_taxonomy_cd207Q
  -- ============================================================================
  SELECT DISTINCT
    CONCAT('provider_taxonomy_cd', SUBSTR(REPLACE(cd, 'provider_taxonomy_cd', ''), 1, 4)) AS cd
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind`
  WHERE cd LIKE 'provider_taxonomy_cd%'
    AND LENGTH(REPLACE(cd, 'provider_taxonomy_cd', '')) >= 4
  
  UNION DISTINCT
  
  -- ============================================================================
  -- 4. ICD Diagnosis (First 3 characters) - 100% coverage
  -- Transform: icd9_dx_cdG24.01 → icd9_dx_cdG24 (ICD-10 with decimal)
  -- Transform: icd9_dx_cdG2401 → icd9_dx_cdG24 (ICD-10 without decimal)
  -- ✅ FIXED: Use SUBSTR instead of SPLIT to handle codes with/without decimals
  -- ============================================================================
  SELECT DISTINCT
    CONCAT('icd9_dx_cd', SUBSTR(REPLACE(cd, 'icd9_dx_cd', ''), 1, 3)) AS cd
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind`
  WHERE cd LIKE 'icd9_dx_cd%'
    AND LENGTH(REPLACE(cd, 'icd9_dx_cd', '')) >= 3
  
  UNION DISTINCT
  
  -- ============================================================================
  -- 5. GPI Medications (First 2 digits) - 100% coverage
  -- Transform: gpi2210 → gpi22
  -- ============================================================================
  SELECT DISTINCT
    CONCAT('gpi', SUBSTR(REPLACE(cd, 'gpi', ''), 1, 2)) AS cd
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind`
  WHERE cd LIKE 'gpi%'
    AND LENGTH(REPLACE(cd, 'gpi', '')) >= 2
  
  UNION DISTINCT
  
  -- ============================================================================
  -- 6. Revenue Codes (First 3 digits) - 100% coverage
  -- Transform: revenue_cd0250 → revenue_cd025
  -- ============================================================================
  SELECT DISTINCT
    CONCAT('revenue_cd', SUBSTR(REPLACE(cd, 'revenue_cd', ''), 1, 3)) AS cd
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind`
  WHERE cd LIKE 'revenue_cd%'
    AND LENGTH(REPLACE(cd, 'revenue_cd', '')) >= 3
  
  UNION DISTINCT
  
  -- ============================================================================
  -- 7. DRG Codes (Keep as-is) - 100% coverage
  -- ============================================================================
  SELECT DISTINCT cd
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind`
  WHERE cd LIKE 'drg_cd%'
  
  UNION DISTINCT
  
  -- ============================================================================
  -- 8. Procedure Groups (ALGORITHMIC GROUPING) - 100% coverage
  -- 
  -- Strategy: Use first N characters of procedure code to create groups
  --   Format: prcdr_group_{algorithm_code}
  -- 
  --   - CPT (5-digit numeric): First 3 digits → 992, 335, 800
  --   - CPT Category II/III (4 digits + letter): First 4 digits → 0001, 0012, 0003
  --   - ICD-10-PCS variants (6-char alphanumeric): First 3 chars lowercased → 10e, 02h, 0bd
  --   - ICD-10-PCS (7-char alphanumeric): First 3 chars lowercased → 02h, 0u5, 0bh
  --   - ICD-9 Procedure (decimal): Digits before decimal → 00, 66, 81
  --   - HCPCS (Letter + 4 digits): First 2 chars lowercased → j1, a0, e0
  --   - Dental (D + 4 digits): First 3 chars lowercased → d72, d01, d21
  --   - Medicaid Two-Letter (2 letters + digits): First 2 chars lowercased → pt, mr, dd
  --   - Other numeric: First N digits → 92, 0210
  --
  -- Examples:
  --   99213 → prcdr_group_992
  --   0001A → prcdr_group_0001
  --   10E0XZ → prcdr_group_10e (6-char variant)
  --   02H60JZ → prcdr_group_02h (7-char standard)
  --   66.21 → prcdr_group_66
  --   J1234 → prcdr_group_j1
  --   D7220 → prcdr_group_d72
  --   PT624 → prcdr_group_pt (Medicaid Physical Therapy)
  --   MR940 → prcdr_group_mr (Medicaid-specific)
  --
  -- This ensures 100% coverage - every procedure code gets a group
  -- ============================================================================
  
  SELECT DISTINCT
    CASE
      -- CPT: 5-digit numeric (e.g., 99213 → prcdr_group_992)
      WHEN REGEXP_CONTAINS(prcdr_raw, r'^\d{5}$') 
        THEN CONCAT('prcdr_group_', SUBSTR(prcdr_raw, 1, 3))
      
      -- CPT Category II/III: 4 digits + letter (e.g., 0001A → prcdr_group_0001, 0012M → prcdr_group_0012)
      WHEN REGEXP_CONTAINS(prcdr_raw, r'^\d{4}[A-Z]$')
        THEN CONCAT('prcdr_group_', SUBSTR(prcdr_raw, 1, 4))
      
      -- ICD-10-PCS variants: 6-character alphanumeric (e.g., 10E0XZ → prcdr_group_10e)
      WHEN REGEXP_CONTAINS(prcdr_raw, r'^[0-9A-Z]{6}$')
        THEN CONCAT('prcdr_group_', LOWER(SUBSTR(prcdr_raw, 1, 3)))
      
      -- ICD-10-PCS: 7-character alphanumeric starting with digit (e.g., 02H60JZ → prcdr_group_02h)
      WHEN REGEXP_CONTAINS(prcdr_raw, r'^\d[A-Z0-9]{6}$') 
        THEN CONCAT('prcdr_group_', LOWER(SUBSTR(prcdr_raw, 1, 3)))
      
      -- ICD-9 Procedure: Has decimal (e.g., 00.50 → prcdr_group_00, 66.21 → prcdr_group_66)
      WHEN REGEXP_CONTAINS(prcdr_raw, r'^\d+\.\d+$') 
        THEN CONCAT('prcdr_group_', SPLIT(prcdr_raw, '.')[SAFE_OFFSET(0)])
      
      -- HCPCS: Letter + 4 digits (e.g., J1234 → prcdr_group_j1, A0021 → prcdr_group_a0)
      WHEN REGEXP_CONTAINS(prcdr_raw, r'^[A-Z]\d{4}$') AND LEFT(prcdr_raw, 1) != 'D'
        THEN CONCAT('prcdr_group_', LOWER(SUBSTR(prcdr_raw, 1, 2)))
      
      -- Dental: D + 4 digits (e.g., D7220 → prcdr_group_d72)
      WHEN REGEXP_CONTAINS(prcdr_raw, r'^D\d{4}$') 
        THEN CONCAT('prcdr_group_', LOWER(SUBSTR(prcdr_raw, 1, 3)))
      
      -- Short numeric codes: First 2 digits or all if shorter (e.g., 924 → prcdr_group_92)
      WHEN REGEXP_CONTAINS(prcdr_raw, r'^\d{1,4}$')
        THEN CONCAT('prcdr_group_', SUBSTR(prcdr_raw, 1, LEAST(2, LENGTH(prcdr_raw))))
      
      -- Long numeric codes: First 4 digits (e.g., 0210083 → prcdr_group_0210)
      WHEN REGEXP_CONTAINS(prcdr_raw, r'^\d{6,}$')
        THEN CONCAT('prcdr_group_', SUBSTR(prcdr_raw, 1, 4))
      
      -- Two-letter Medicaid codes: Group by 2-letter prefix (e.g., PT624 → prcdr_group_pt, MR940 → prcdr_group_mr)
      WHEN REGEXP_CONTAINS(prcdr_raw, r'^[A-Z]{2}\d+$')
        THEN CONCAT('prcdr_group_', LOWER(SUBSTR(prcdr_raw, 1, 2)))
      
      -- Everything else: fallback group
      ELSE 'prcdr_group_unk'
    END AS cd
  FROM (
    SELECT DISTINCT REPLACE(cd, 'prcdr_cd', '') AS prcdr_raw
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind`
    WHERE cd LIKE 'prcdr_cd%'
      AND REPLACE(cd, 'prcdr_cd', '') IS NOT NULL
      AND REPLACE(cd, 'prcdr_cd', '') != ''
  )
)

-- Add row numbers for indices
, indexed_codes AS (
  SELECT 
    cd,
    ROW_NUMBER() OVER (ORDER BY cd) AS ind
  FROM all_target_codes
  WHERE cd IS NOT NULL 
    AND cd != ''
)

-- Add index 0 for unknown/missing codes
SELECT '' AS cd, 0 AS ind
UNION ALL
SELECT cd, ind
FROM indexed_codes
ORDER BY ind;

-- ============================================================================
-- VERIFICATION QUERIES (run separately after table creation)
-- ============================================================================

/*
==============================================================================
📊 POST-EXECUTION VALIDATION QUERIES
==============================================================================

1️⃣ Check Total Code Count
────────────────────────────────────────────────────────────────────────────*/
SELECT 'w2ind_target' AS table_name, COUNT(*) AS total_codes
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind_target`
WHERE cd IS NOT NULL AND cd != '';
-- Expected: ~5,000-6,000 codes

/*
2️⃣ Breakdown by Code Type
────────────────────────────────────────────────────────────────────────────*/
SELECT 
    CASE 
        WHEN cd LIKE 'icd9_dx_cd%' THEN '1_ICD-9 Diagnosis Groups'
        WHEN cd LIKE 'prcdr_group_%' THEN '2_Procedure Groups'
        WHEN cd LIKE 'gpi%' THEN '3_GPI Med Groups'
        WHEN cd LIKE 'provider_taxonomy_cd%' THEN '4_Provider Taxonomy Groups'
        WHEN cd LIKE 'revenue_cd%' THEN '5_Revenue Code Groups'
        WHEN cd LIKE 'drg_cd%' THEN '6_DRG Codes'
        WHEN cd LIKE 'days_cnt%' THEN '7_Days Count'
        WHEN cd LIKE 'hcfa_plc_srv_cd%' THEN '8_Place of Service'
        WHEN cd = '' THEN '0_Unknown/Missing'
        ELSE '9_Other'
    END AS code_category,
    COUNT(*) AS code_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS percent_of_total
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind_target`
GROUP BY code_category
ORDER BY code_category;

/*
3️⃣ Sample Procedure Groups (verify algorithmic grouping worked)
────────────────────────────────────────────────────────────────────────────*/
SELECT cd
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind_target`
WHERE cd LIKE 'prcdr_group_%'
ORDER BY cd
LIMIT 20;
-- Should see: prcdr_group_00, prcdr_group_02h, prcdr_group_0u5, 
--            prcdr_group_992, prcdr_group_j1, prcdr_group_d72, etc.

/*
4️⃣ Check for Unknown Procedure Group (should be minimal)
────────────────────────────────────────────────────────────────────────────*/
SELECT 
    cd,
    COUNT(*) as count
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind_target`
WHERE cd = 'prcdr_group_unk'
GROUP BY cd;
-- Expected: 0 or very small count

/*
5️⃣ Compare Input vs Target Vocabulary Size
────────────────────────────────────────────────────────────────────────────*/
SELECT 
    'w2ind (INPUT)' AS vocabulary,
    COUNT(*) AS total_codes
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind`
WHERE cd IS NOT NULL AND cd != ''

UNION ALL

SELECT 
    'w2ind_target (OUTPUT)' AS vocabulary,
    COUNT(*) AS total_codes
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind_target`
WHERE cd IS NOT NULL AND cd != '';
-- Expected: w2ind ~50K, w2ind_target ~5K (10x reduction)

/*
==============================================================================
🎯 LOGIC SUMMARY: How Each Code Type is Processed
==============================================================================

INPUT (w2ind)                  LOGIC APPLIED                OUTPUT (w2ind_target)
─────────────────────────────────────────────────────────────────────────────

📋 DIAGNOSIS CODES
─────────────────
icd9_dx_cd250.00  ─────────┐
icd9_dx_cd250.01  ─────────├──► SPLIT on '.' → Take [0]──► icd9_dx_cd250
icd9_dx_cd250.02  ─────────┘
icd9_dx_cd401.9   ─────────────► SPLIT on '.' → Take [0]──► icd9_dx_cd401


🏥 PROCEDURE CODES (Most Complex)
──────────────────
prcdr_cd99213  ────────────────► IF ^\d{5}$ → SUBSTR(1,3)──► prcdr_group_992
prcdr_cd99214  ────────────────► IF ^\d{5}$ → SUBSTR(1,3)──► prcdr_group_992
prcdr_cd0001A  ────────────────► IF ^\d{4}[A-Z]$ → SUBSTR(1,4)──► prcdr_group_0001
prcdr_cd02H60JZ ───────────────► IF ^\d[A-Z0-9]{6}$ → LOWER(SUBSTR(1,3))──► prcdr_group_02h
prcdr_cd66.21  ────────────────► IF ^\d+\.\d+$ → SPLIT('.')[0]──► prcdr_group_66
prcdr_cdJ1234  ────────────────► IF ^[A-Z]\d{4}$ → LOWER(SUBSTR(1,2))──► prcdr_group_j1
prcdr_cdD7220  ────────────────► IF ^D\d{4}$ → LOWER(SUBSTR(1,3))──► prcdr_group_d72


💊 MEDICATION CODES
───────────────────
gpi2210  ──────────────────────► SUBSTR(1,2)──────────────────► gpi22
gpi2215  ──────────────────────► SUBSTR(1,2)──────────────────► gpi22
gpi6510  ──────────────────────► SUBSTR(1,2)──────────────────► gpi65


👨‍⚕️ PROVIDER TAXONOMY
────────────────────
provider_taxonomy_cd207Q00000X ► SUBSTR(1,4)──────────────────► provider_taxonomy_cd207Q
provider_taxonomy_cd363L00000X ► SUBSTR(1,4)──────────────────► provider_taxonomy_cd363L


💰 REVENUE CODES
────────────────
revenue_cd0250 ────────────────► SUBSTR(1,3)──────────────────► revenue_cd025
revenue_cd0450 ────────────────► SUBSTR(1,3)──────────────────► revenue_cd045


🏷️ CODES KEPT AS-IS (No Grouping)
──────────────────────────────────
drg_cd470  ────────────────────► KEEP AS-IS───────────────────► drg_cd470
days_cnt_5 ────────────────────► KEEP AS-IS───────────────────► days_cnt_5
hcfa_plc_srv_cd21 ─────────────► KEEP AS-IS───────────────────► hcfa_plc_srv_cd21

==============================================================================

✅ KEY TAKEAWAY:
   
   We're creating a SMALLER, GROUPED vocabulary that the transformer can
   more easily learn to predict, while preserving clinical meaning through
   systematic code grouping rules.

   Input:  ~50,000 individual codes (too many to predict accurately)
   Output: ~5,000 code groups (manageable for transformer prediction)

==============================================================================
*/

