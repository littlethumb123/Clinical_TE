/*==============================================================================
  MEDICAID CLINICAL TRANSFORMER - TRAINING DATA PREPARATION PIPELINE
  
  Purpose: Complete end-to-end data preparation pipeline for clinical transformer
           model training using Medicaid claims, procedures, diagnoses, and medications
  
  Original Developer: Ellen Palmer (palmere1@aetna.com) - Complex Care Solutions (Medicaid SME)
  Current Owner: Pritha Ghosh (pritha.ghosh@cvshealth.com) - CSDI
  Team: Clinical & Social Determinants Intelligence (CSDI)
  
  Current Training Period: 2023 (Calendar Year)
  Lookback Window: 36 months
  
  ⚠️⚠️⚠️ RETRAINING CHECKLIST - UPDATE THESE WHEN RETRAINING ⚠️⚠️⚠️
  
  When retraining the model with new data, update the following:
  
  1. MEMBERSHIP DATES (Line ~183-184):
     - Current: 20230101 to 20231231 (2023 calendar year)
     - Change to: Your new index period (e.g., 20240101 to 20241231)
  
  2. PHARMACY DATES:
     - ✅ NOW DYNAMIC: Uses 36-month lookback from index_dt automatically
     - No hardcoded dates to update - fully dynamic like Commercial/Medicare
  
  3. PROVIDER CROSSWALK TABLE (Line ~407):
     - Current: a834793_provider_db_x_walk_20251013
     - Change to: New crosswalk table with updated date suffix -- Not a blocker
  
  4. w2ind LOOKUP TABLE (a834793_member_w2ind):
     - ⚠️ CRITICAL: Ensure w2ind contains mappings for DRG, GPI4, and Provider Taxonomy codes
     - Required patterns:
       • DRG: 'drg_cd' + drg_code (e.g., 'drg_cd470', 'drg_cd871')
       • GPI4: 'gpi' + 4_digit_code (e.g., 'gpi2210', 'gpi4600')
       • Provider Taxonomy: 'provider_taxonomy_cd' + taxonomy_code (e.g., 'provider_taxonomy_cd207Q00000X')
     - Without these mappings, codes will default to index 0
     - See lines 740-910 for full list of required code patterns
  
  5. TABLE NAME SUFFIXES:
     - Table names do NOT include year (consistent with Commercial/Medicare)
     - Pattern: a834793_Medicaid_*
     - Tables are reused for each training period
  
  ============================================================================
  
  PIPELINE OVERVIEW - 8 MAIN STEPS (Updated for Transformer Training):
  
  STEP 1: Create Membership Base (a834793_Medicaid)
          - Extract eligible members for training period
          - Set index_dt: ONE random month per member (matches Commercial/Medicare)
          - Prevents data leakage and ensures diverse training examples
  
  STEP 2: Prepare Member Table (a834793_Medicaid_member_train_ending)
          - Create clean member table for joins
          - One row per individual_id (already ensured by Step 1)
          - Adds member_id alias for compatibility
  
  STEP 3: Extract Claims & Procedures (a834793_Medicaid_d1a_train_ending)
          - Pull medical claims with procedures, revenue codes
          - Calculate demographics (age, gender)
          - Map place of service and provider taxonomy
          - Add DRG codes for inpatient stays
          - 🎯 NEW: Add unified procedure groups (prcdr_group_cd - ALGORITHMIC)
          - Lookback: 36 months from index_dt
          - Quality filters: Final, paid, non-denied claims only
  
  STEP 4: Extract Diagnosis Codes (a834793_Medicaid_d1b_train_ending)
          - Extract ICD-9 diagnosis codes (primary + 3 secondary)
          - Standardize format: XXX.XX (3 digits + 2 decimal places)
  
  STEP 5: Extract Medications (a834793_Medicaid_d1c_train_ending)
          - Extract GPI codes (first 4 digits = drug class)
          - Format: 'gpiXXXX' (includes 'gpi' prefix)
  
  STEP 6: Map Input Codes to Indices (a834793_Medicaid_o1_train_ending)
          - Map all 9 code types to w2ind indices (detailed vocabulary, ~84k codes)
          - Aggregate by individual + date
          - Output: One row per member per day with comma-separated indices
  
  STEP 6b: Map Target Codes to Indices 🎯 NEW!
           (a834793_Medicaid_o1_train_ending_target)
          - Map 8 grouped code types to w2ind_target indices (~5k codes)
          - Uses clinical groupings (e.g., ICD 3-digit, GPI 2-digit, procedure groups)
          - Output: One row per member per day with target indices
  
  STEP 7: Create Temporal Sequences with Next-Day Targets 🎯 UPDATED!
          (a834793_Medicaid_o3_train_ending)
          - Aggregate both input (cd) and target (target) sequences
          - Apply LEAD function for next-day prediction
          - Output: One row per member with asterisk-separated sequences
          - Format: cd="15,42*7,88*12,55*..." | target="101*67*23*..."
          - Features: individual_id, gender_cd, age_in_months, cd, target, dt_cnt
          - Ready for supervised transformer training!
  
  ============================================================================
  
  DATA SOURCES:
  
  Claims & Clinical:
  - ASDB_CLM_DATA_STAGE: Main claims data
  - ASDB_CLAIMDIAGSUMMARY: Diagnosis codes (ICD-9/10)
  - ASDB_CLAIMICDPROCSUMMARY: ICD procedure codes
  - ASDB_DRG: Diagnosis Related Group codes
  - ASDB_RX_DATA_STAGE: Pharmacy claims
  
  Demographics & Reference:
  - ASDB_MEMBER: Patient demographics
  - ASDB_ELIG_DATA_MBR_PER_MTH: Eligibility/enrollment
  - ASDB_SVC_PROV: Provider information
  - ASDB_TYPE_OF_SERVICE: Place of service categories
  
  Public Datasets:
  - bigquery-public-data.nppes.npi_raw: NPPES National Provider Identifier database
    (for healthcare_provider_taxonomy_code_1)
  
  Lookup Tables:
  - a834793_member_w2ind: Code-to-index mapping for transformer (SHARED across LOBs)
    ⚠️ MUST include mappings for DRG, GPI4, and Provider Taxonomy codes
  - a834793_provider_db_x_walk_20251013: Provider specialty crosswalk (includes NPI)
  
  ============================================================================
  
  OUTPUT FORMAT FOR TRANSFORMER:
  
  Final table columns:
  - individual_id: Unique patient identifier
  - dt_cnt: Number of days in sequence (≤200)
  - gender_cd: "*"-separated gender sequence per day
  - age_in_months: "*"-separated age sequence per day
  - cd: "*"-separated medical code sequences per day
       Within each day: ","-separated code indices (≤80)
  
  Sequence Limits:
  - Maximum 200 days per patient
  - Maximum 80 code indices per day
  - Codes ordered chronologically (earliest to latest)
  
  ============================================================================
  
  PIPELINE HISTORY:
  - Originally developed by Ellen Palmer (Medicaid SME) for transformer training
  - Transitioned to CSDI organization for ongoing maintenance and enhancement
  - Current owner: Pritha Ghosh
  
  ============================================================================
  
  TABLE EXPIRATION:
  - Most tables: 180 days (auto-delete after 6 months)
  - Temp table d1a: 1 day (auto-delete after 24 hours)
  
  ============================================================================
==============================================================================*/

----Build simple table for Medicaid membership (currently configured for 2023 data)
-- ✅ UPDATED: Now selects ONE random index_dt per member (matches Commercial/Medicare)
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)  -- Auto-delete after 180 days
)
AS
  SELECT
     -- Patient unique identifier (required for transformer)
     asdb_member_key AS individual_id
     
     -- Index date: ONE random month per member from 2023 eligibility period
     -- This prevents data leakage and creates diverse training examples
     -- Matches Commercial/Medicare training approach
     , ARRAY_AGG(CAST(asdb_elig_dt AS DATE) ORDER BY RAND() LIMIT 1)[SAFE_OFFSET(0)] AS index_dt
FROM 
     -- Source: ASDB eligibility data, one record per member per month
     `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ELIG_DATA_MBR_PER_MTH` -- STABLE GCP VIEW
WHERE 
    -- ⚠️ RETRAINING DATES: Update these dates when retraining the model
    -- Current: 2023 calendar year (20230101 to 20231231)
    -- For retraining: Change to your target period
    -- Format: "YYYYMMDD" (e.g., "20240101" for 2024 full year)
    -- Example: "20240101" to "20241231"
    CAST(asdb_elig_dt AS DATE) BETWEEN PARSE_DATE("%Y%m%d", CAST("20230101" AS STRING)) 
                                   AND PARSE_DATE("%Y%m%d", CAST("20231231" AS STRING))
GROUP BY asdb_member_key  -- ✅ ONE row per member (consistent with Commercial/Medicare)
ORDER BY individual_id
;


/*==============================================================================
  STEP 2: DEDUPLICATE AND CREATE MEMBERSHIP BASE TABLE FOR CLAIMS EXTRACTION
  
  Purpose: Create a clean member table with one record per individual_id 
           to be used for joining with claims data.
  
  Key Operations:
  - Creates member_id alias for compatibility with claims tables
  - Maintains index_dt for claims date filtering
  - ✅ UPDATED: No deduplication needed - Step 1 already ensures ONE row per member
  
  Input:  a834793_Medicaid (membership base with ONE random index_dt per member)
  Output: a834793_Medicaid_member_train_ending (clean member table)
  
==============================================================================*/

----------------------------------
--- Membership for transformer ---
----------------------------------

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_member_train_ending`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_member_train_ending`
OPTIONS (
  labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
  , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)  
)
AS
    SELECT 
    -- Unique patient identifier (primary key)
        individual_id
    
    -- Index date for date filtering in claims queries
    -- All claims must be between (index_dt - 36 months) and index_dt
    -- ✅ ONE random index_dt per member (already selected in Step 1)
        , index_dt 
    
    -- Alias for member_id (used in joins with claims tables)
    -- Note: In this Medicaid data, individual_id = member_id
    , individual_id AS member_id
    FROM 
    `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid`
    -- ✅ GROUP BY kept for safety, but should already have ONE row per member from Step 1
    GROUP BY
        individual_id
        , index_dt
;



-------------------------------------------
--- CPT procedure codes for transformer ---
-------------------------------------------

/*==============================================================================
  STEP 3: EXTRACT CLAIMS DATA WITH PROCEDURES AND METADATA (d1a table)
  
  Purpose: Extract medical claims with procedure codes, revenue codes, place of
           service, and provider specialty for transformer training.
  
  Data Sources:
  - ASDB_CLM_DATA_STAGE: Main claims data (procedures, revenue codes, dates)
  - ASDB_MEMBER: Patient demographics (gender, date of birth)
  - ASDB_SVC_PROV: Provider information (specialty, NPI)
  - ASDB_TYPE_OF_SERVICE: Category of care/place of service
  - ASDB_CLAIMICDPROCSUMMARY: ICD procedure codes
  - bigquery-public-data.nppes.npi_raw: NPPES provider taxonomy codes
  
  Crosswalk Tables:
  - a834793_provider_db_x_walk_20251013: Maps provider specialty to standard codes (includes NPI)
  
==============================================================================*/

-------------------------------------------
--- CPT procedure codes for transformer ---
-------------------------------------------


DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1a_train_ending`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1a_train_ending`
OPTIONS (
  labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
  , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY) 
)
AS
WITH clm AS (
    -- Step 3a: Extract claims with basic filtering
    -- Only pull claims within 36-month lookback window from index date
    -- Filters to final, paid, non-denied claims only for quality
    SELECT
        base.individual_id
        , base.index_dt
        , base.member_id
        , clm.claimid
        , clm.asdb_incurred_dt                -- Service date
        , clm.ip_paid_days_ct                 -- Length of stay for inpatient
        , clm.revcode                         -- Revenue code
        , clm.location                        -- Place of service code
        , clm.servcode                        -- Procedure code (CPT/HCPCS)
        , clm.asdb_svc_prov_key              -- Provider key for specialty lookup
        , clm.asdb_coe_id_dev                -- Category of care ID
        , clm.final_claim
        , clm.status_header
        , clm.status_detail

    FROM 
        -- Base table: Deduplicated members with index dates
        `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_member_train_ending` AS base 
    LEFT JOIN
        `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_CLM_DATA_STAGE` AS clm
            ON base.member_id = clm.asdb_member_key
    WHERE 1=1
        -- CLAIM QUALITY FILTERS: Only final, paid, non-denied claims
        AND clm.final_claim = 1                                           -- Only final adjudicated claims
        AND TRIM(UPPER(clm.status_header)) = "PAID"                      -- Claim was paid
        AND TRIM(UPPER(clm.status_detail)) NOT IN ("DENY", "DENIED")    -- Not denied
        
        -- DATE RANGE FILTERS: Dynamic 36-month lookback with 90-day claims lag
        -- ⚠️ NOTE: These filters automatically adjust based on index_dt from membership table
        -- 90-day lag allows claims to finalize before index_dt (consistent with Commercial/Medicare)
        AND CAST(clm.asdb_incurred_dt AS DATE) > DATE_SUB(CAST(base.index_dt AS DATE), INTERVAL 36 MONTH)  -- Start: 36 months before index_dt
        AND CAST(clm.asdb_incurred_dt AS DATE) < DATE_SUB(CAST(base.index_dt AS DATE), INTERVAL 90 DAY)    -- End: 90 days before index (claims lag)
        AND CAST(clm.asdb_paid_dt AS DATE) < DATE_SUB(CAST(base.index_dt AS DATE), INTERVAL 90 DAY)        -- Paid before 90-day lag
)
-- Step 3b: Transform and enrich claims with demographics and standardized codes
SELECT
    -- PATIENT IDENTIFIERS
    clm.individual_id
    , clm.index_dt
    , clm.member_id
    , clm.claimid AS claim_line_id
    
    -- SERVICE DATE (used for day-level aggregation)
    , CAST(clm.asdb_incurred_dt AS DATE) AS dt
    
    -- DAYS COUNT: Length of stay (for inpatient claims)
    -- Null/negative -> 99, >10 days -> 11 (capped), else actual value
    , CASE WHEN (clm.ip_paid_days_ct IS NULL OR clm.ip_paid_days_ct < 0) THEN 99 
        WHEN clm.ip_paid_days_ct > 10 THEN 11 
        ELSE clm.ip_paid_days_ct END AS days_cnt
    
    -- GENDER: Encoded as 0=Female, 1=Male, 2=Unknown/Other
    -- Calculated per claim date for consistency with transformer requirements
    , CASE WHEN TRIM(member.gender) = 'M' THEN 1
        WHEN TRIM(member.gender) = 'F' THEN 0 
        ELSE 2 END AS gender_cd
    
    -- AGE IN MONTHS: Age at service date (required by transformer)
    -- Calculated from service date to DOB, with standardization to handle data quality issues
    , CASE
        WHEN DATE_DIFF(CAST(clm.asdb_incurred_dt AS DATE), CAST(member.dob AS DATE), MONTH) < 0 THEN 0  -- Negative ages → 0
        WHEN DATE_DIFF(CAST(clm.asdb_incurred_dt AS DATE), CAST(member.dob AS DATE), MONTH) > 1439 THEN 1439  -- Cap at 1439 months
        ELSE DATE_DIFF(CAST(clm.asdb_incurred_dt AS DATE), CAST(member.dob AS DATE), MONTH)
      END AS age_in_months
    
    -- REVENUE CODE: UB-04 revenue code (kept as STRING for w2ind lookup)
    -- ✅ VALIDATION ADDED: Only keep 3-4 digit numeric revenue codes
    , CASE WHEN TRIM(clm.revcode) = '' OR clm.revcode IS NULL THEN NULL
           WHEN REGEXP_CONTAINS(CAST(clm.revcode AS STRING), r'^[0-9]{3,4}$') 
               THEN UPPER(TRIM(clm.revcode))  -- ✅ Valid: 3-4 digits
           ELSE NULL  -- ❌ Invalid: special characters, wrong length
      END AS revenue_cd
    
    -- PLACE OF SERVICE: HCFA/CMS place of service codes
    -- If location exists AND is valid, use it; otherwise derive from category of care logic
    -- Maps ASDB category of care to standard CMS place of service codes:
    -- ✅ UPDATED: Only accept valid numeric 1-2 digit POS codes from location field
    , CASE WHEN TRIM(clm.location) IS NOT NULL 
                AND TRIM(clm.location) != ''
                AND REGEXP_CONTAINS(CAST(clm.location AS STRING), r'^[0-9]{1,2}$')
            THEN CAST(clm.location AS STRING)  -- ✅ Valid location: use it
        -- 11 = Office (Outpatient non-hospital ambulatory)
        WHEN coe.asdb_coe_general_type = "Outpatient" 
            AND coe.asdb_coe_sub_cat = "Non Hospital" 
            AND coe.emis_cat = "Selected Ambulatory Facility"
            THEN CAST(11 AS STRING)
        -- 22 = On Campus-Outpatient Hospital (Hospital outpatient services)
        WHEN coe.asdb_coe_general_type = "Outpatient" 
            AND coe.asdb_coe_sub_cat = "Hospital" 
            AND coe.emis_cat IN ("Laboratory", "Medical Pharmacy", "Mental Health", "Radiology", "Selected Ambulatory Facility") 
            THEN CAST(22 AS STRING)
        -- 21 = Inpatient Hospital (Acute care hospital inpatient)
        WHEN coe.asdb_coe_general_type = "Inpatient" 
            AND coe.asdb_coe_sub_cat = "Hospital" 
            AND coe.emis_cat IN ("Inpatient Facility", "Inpatient Facility (or Institutional Services)") 
            THEN CAST(21 AS STRING)
        -- 23 = Emergency Room - Hospital (Hospital emergency department)
        WHEN coe.asdb_coe_general_type = "Outpatient" 
            AND coe.asdb_coe_sub_cat = "Hospital" 
            AND coe.emis_cat = "Emergency" 
            THEN CAST(23 AS STRING)   
        -- 81 = Independent Laboratory
        WHEN coe.asdb_coe_general_type = "Laboratory" 
            AND coe.asdb_coe_sub_cat = "Professional" 
            AND coe.emis_cat = "Laboratory"
            THEN CAST(81 AS STRING)
        -- 12 = Home (Patient's home)
        WHEN coe.asdb_coe_general_type = "Long Term Care, Other, Outpatient" 
            AND coe.asdb_coe_sub_cat = "Home Based Services, Professional, Non Hospital" 
            AND coe.emis_cat IN ("Home-Based Services", "Home Health", "Home Health") 
            THEN CAST(12 AS STRING)
        -- 19 = Off Campus-Outpatient Hospital (Inpatient facility non-hospital)
        WHEN coe.asdb_coe_general_type = "Inpatient" 
            AND coe.asdb_coe_sub_cat = "Non Hospital" 
            AND coe.emis_cat = "Inpatient Facility"
            THEN CAST(19 AS STRING)   
        -- 16 = Birthing Center (Long term care institution)
        WHEN coe.asdb_coe_general_type = "Long Term Care" 
            AND coe.asdb_coe_sub_cat = "Institution" 
            AND coe.emis_cat = "Institutional Services"
            THEN CAST(16 AS STRING)   
        -- 51 = Inpatient Psychiatric Facility
        WHEN coe.asdb_coe_general_type = "Inpatient" 
            AND coe.asdb_coe_sub_cat = "Hospital" 
            AND coe.emis_cat = "Mental Health"
            THEN CAST(51 AS STRING)   
        ELSE NULL END AS hcfa_plc_srv_cd
    
    -- PROVIDER SPECIALTY: Mapped from ASDB provider specialty to standard codes
    -- ⚠️ COMMENTED OUT: Replaced by provider_taxonomy_cd from NPPES
    -- Uses crosswalk table a834793_provider_db_x_walk_20251013 (updated Oct 2025)
    -- , map.src_specialty_cd
    
    -- PROVIDER TAXONOMY CODE: Healthcare provider taxonomy classification from NPPES
    -- Source: National Plan and Provider Enumeration System (NPPES) public dataset
    -- ✅ VALIDATION ADDED: Only keep 10-character alphanumeric taxonomy codes (NPPES standard)
    , CASE 
        WHEN TRIM(nppes.healthcare_provider_taxonomy_code_1) = '' OR nppes.healthcare_provider_taxonomy_code_1 IS NULL THEN NULL
        WHEN REGEXP_CONTAINS(TRIM(nppes.healthcare_provider_taxonomy_code_1), r'^[A-Z0-9]{10}$') 
            THEN UPPER(TRIM(nppes.healthcare_provider_taxonomy_code_1))  -- ✅ Valid: 10-char alphanumeric
        ELSE NULL  -- ❌ Invalid: wrong length, special chars, lowercase
      END AS provider_taxonomy_cd
    
    -- PROCEDURE CODE: CPT/HCPCS procedure code from claim
    -- ✅ VALIDATION ADDED: Filter codes < 4 chars (garbage like 'A', 'RX', etc.)
    , CASE WHEN TRIM(clm.servcode) = '' THEN NULL
           WHEN LENGTH(TRIM(clm.servcode)) < 4 THEN NULL  -- ✅ Filter too-short codes
           ELSE TRIM(clm.servcode) END AS prcdr_cd
    
    -- ICD PROCEDURE CODE: ICD-10-PCS procedure code (primary position)
    -- ⚠️ NOTE: Column named icd9_prcdr_cd but contains ICD-10-PCS codes
    -- ✅ VALIDATION ADDED: Filter codes < 4 chars
    , CASE WHEN TRIM(icd.icdpx1) = '' THEN NULL
           WHEN LENGTH(TRIM(icd.icdpx1)) < 4 THEN NULL  -- ✅ Filter too-short codes
           ELSE TRIM(icdpx1) END AS icd9_prcdr_cd

    -- DRG CODE: DRG code from claim
    -- ✅ UPDATED: Only keep numeric DRG codes AND strip leading zeros for consistency
    -- Examples: "0885" → 885 → "885", "001" → 1 → "1", "885" → "885" (unchanged)
    , CASE WHEN TRIM(drg.drg) = '' OR drg.drg IS NULL THEN NULL
           WHEN REGEXP_CONTAINS(CAST(drg.drg AS STRING), r'^[0-9]+$') 
               THEN CAST(CAST(drg.drg AS INT64) AS STRING)  -- ✅ Strip leading zeros for standardization
           ELSE NULL  -- ❌ Invalid: contains special characters
      END AS drg_cd
    
    -- ============================================================================
    -- ALGORITHMIC PROCEDURE GROUPING (for target vocabulary)
    -- ============================================================================
    -- Apply algorithmic grouping to BOTH CPT and ICD procedure codes
    -- Then merge with COALESCE (prioritize CPT if both exist)
    , COALESCE(
        -- CPT/HCPCS procedure group
        CASE
          WHEN TRIM(clm.servcode) IS NOT NULL AND REGEXP_CONTAINS(UPPER(TRIM(clm.servcode)), r'^\d{5}$') 
            THEN CONCAT('prcdr_group_', SUBSTR(UPPER(TRIM(clm.servcode)), 1, 3))
          WHEN TRIM(clm.servcode) IS NOT NULL AND REGEXP_CONTAINS(UPPER(TRIM(clm.servcode)), r'^\d{4}[A-Z]$')
            THEN CONCAT('prcdr_group_', SUBSTR(UPPER(TRIM(clm.servcode)), 1, 4))
          WHEN TRIM(clm.servcode) IS NOT NULL AND REGEXP_CONTAINS(UPPER(TRIM(clm.servcode)), r'^[0-9A-Z]{6}$')
            THEN CONCAT('prcdr_group_', LOWER(SUBSTR(UPPER(TRIM(clm.servcode)), 1, 3)))
          WHEN TRIM(clm.servcode) IS NOT NULL AND REGEXP_CONTAINS(UPPER(TRIM(clm.servcode)), r'^\d[A-Z0-9]{6}$') 
            THEN CONCAT('prcdr_group_', LOWER(SUBSTR(UPPER(TRIM(clm.servcode)), 1, 3)))
          WHEN TRIM(clm.servcode) IS NOT NULL AND REGEXP_CONTAINS(UPPER(TRIM(clm.servcode)), r'^\d+\.\d+$') 
            THEN CONCAT('prcdr_group_', SPLIT(UPPER(TRIM(clm.servcode)), '.')[SAFE_OFFSET(0)])
          WHEN TRIM(clm.servcode) IS NOT NULL AND REGEXP_CONTAINS(UPPER(TRIM(clm.servcode)), r'^[A-Z]\d{4}$') AND LEFT(UPPER(TRIM(clm.servcode)), 1) != 'D'
            THEN CONCAT('prcdr_group_', LOWER(SUBSTR(UPPER(TRIM(clm.servcode)), 1, 2)))
          WHEN TRIM(clm.servcode) IS NOT NULL AND REGEXP_CONTAINS(UPPER(TRIM(clm.servcode)), r'^D\d{4}$') 
            THEN CONCAT('prcdr_group_', LOWER(SUBSTR(UPPER(TRIM(clm.servcode)), 1, 3)))
          WHEN TRIM(clm.servcode) IS NOT NULL AND REGEXP_CONTAINS(UPPER(TRIM(clm.servcode)), r'^\d{1,4}$')
            THEN CONCAT('prcdr_group_', SUBSTR(UPPER(TRIM(clm.servcode)), 1, LEAST(2, LENGTH(UPPER(TRIM(clm.servcode))))))
          WHEN TRIM(clm.servcode) IS NOT NULL AND REGEXP_CONTAINS(UPPER(TRIM(clm.servcode)), r'^\d{6,}$')
            THEN CONCAT('prcdr_group_', SUBSTR(UPPER(TRIM(clm.servcode)), 1, 4))
          WHEN TRIM(clm.servcode) IS NOT NULL AND REGEXP_CONTAINS(UPPER(TRIM(clm.servcode)), r'^[A-Z]{2}\d+$')
            THEN CONCAT('prcdr_group_', LOWER(SUBSTR(UPPER(TRIM(clm.servcode)), 1, 2)))  -- ✅ Two-letter Medicaid codes (PT, MR, DD, etc.)
          WHEN TRIM(clm.servcode) IS NOT NULL AND TRIM(clm.servcode) != ''
            THEN 'prcdr_group_unk'
          ELSE NULL
        END,
        -- ICD procedure group (fallback if CPT is null)
        CASE
          WHEN TRIM(icd.icdpx1) IS NOT NULL AND REGEXP_CONTAINS(UPPER(TRIM(icd.icdpx1)), r'^\d{5}$') 
            THEN CONCAT('prcdr_group_', SUBSTR(UPPER(TRIM(icd.icdpx1)), 1, 3))
          WHEN TRIM(icd.icdpx1) IS NOT NULL AND REGEXP_CONTAINS(UPPER(TRIM(icd.icdpx1)), r'^\d{4}[A-Z]$')
            THEN CONCAT('prcdr_group_', SUBSTR(UPPER(TRIM(icd.icdpx1)), 1, 4))
          WHEN TRIM(icd.icdpx1) IS NOT NULL AND REGEXP_CONTAINS(UPPER(TRIM(icd.icdpx1)), r'^[0-9A-Z]{6}$')
            THEN CONCAT('prcdr_group_', LOWER(SUBSTR(UPPER(TRIM(icd.icdpx1)), 1, 3)))
          WHEN TRIM(icd.icdpx1) IS NOT NULL AND REGEXP_CONTAINS(UPPER(TRIM(icd.icdpx1)), r'^\d[A-Z0-9]{6}$') 
            THEN CONCAT('prcdr_group_', LOWER(SUBSTR(UPPER(TRIM(icd.icdpx1)), 1, 3)))
          WHEN TRIM(icd.icdpx1) IS NOT NULL AND REGEXP_CONTAINS(UPPER(TRIM(icd.icdpx1)), r'^\d+\.\d+$') 
            THEN CONCAT('prcdr_group_', SPLIT(UPPER(TRIM(icd.icdpx1)), '.')[SAFE_OFFSET(0)])
          WHEN TRIM(icd.icdpx1) IS NOT NULL AND REGEXP_CONTAINS(UPPER(TRIM(icd.icdpx1)), r'^[A-Z]\d{4}$') AND LEFT(UPPER(TRIM(icd.icdpx1)), 1) != 'D'
            THEN CONCAT('prcdr_group_', LOWER(SUBSTR(UPPER(TRIM(icd.icdpx1)), 1, 2)))
          WHEN TRIM(icd.icdpx1) IS NOT NULL AND REGEXP_CONTAINS(UPPER(TRIM(icd.icdpx1)), r'^D\d{4}$') 
            THEN CONCAT('prcdr_group_', LOWER(SUBSTR(UPPER(TRIM(icd.icdpx1)), 1, 3)))
          WHEN TRIM(icd.icdpx1) IS NOT NULL AND REGEXP_CONTAINS(UPPER(TRIM(icd.icdpx1)), r'^\d{1,4}$')
            THEN CONCAT('prcdr_group_', SUBSTR(UPPER(TRIM(icd.icdpx1)), 1, LEAST(2, LENGTH(UPPER(TRIM(icd.icdpx1))))))
          WHEN TRIM(icd.icdpx1) IS NOT NULL AND REGEXP_CONTAINS(UPPER(TRIM(icd.icdpx1)), r'^\d{6,}$')
            THEN CONCAT('prcdr_group_', SUBSTR(UPPER(TRIM(icd.icdpx1)), 1, 4))
          WHEN TRIM(icd.icdpx1) IS NOT NULL AND REGEXP_CONTAINS(UPPER(TRIM(icd.icdpx1)), r'^[A-Z]{2}\d+$')
            THEN CONCAT('prcdr_group_', LOWER(SUBSTR(UPPER(TRIM(icd.icdpx1)), 1, 2)))  -- ✅ Two-letter Medicaid codes (PT, MR, DD, etc.)
          WHEN TRIM(icd.icdpx1) IS NOT NULL AND TRIM(icd.icdpx1) != ''
            THEN 'prcdr_group_unk'
          ELSE NULL
        END
      ) AS prcdr_group_cd  -- Algorithmically generated procedure group code
    
FROM clm

-- JOIN 1: ASDB_MEMBER - Demographics (gender, date of birth)
-- Needed for: gender_cd and age_in_months calculations
LEFT JOIN 
    (SELECT asdb_member_key, gender, dob FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_MEMBER`) AS member
        ON clm.member_id = member.asdb_member_key

-- JOIN 2: ASDB_SVC_PROV - Provider Information
-- Needed for: Provider specialty (intermediate for specialty code mapping) and NPI for taxonomy lookup
LEFT JOIN
    (SELECT asdb_svc_prov_key, prov_specialty, npi FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_SVC_PROV`) AS prv
        ON clm.asdb_svc_prov_key = prv.asdb_svc_prov_key

-- JOIN 3: ASDB_TYPE_OF_SERVICE - Category of Care/Place of Service
-- Needed for: Deriving hcfa_plc_srv_cd when location is not populated
LEFT JOIN 
    (SELECT asdb_coe_id, asdb_coe_general_type, asdb_coe_sub_cat, emis_cat 
     FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_TYPE_OF_SERVICE`) AS coe
        ON CAST(clm.asdb_coe_id_dev AS INT) = CAST(coe.asdb_coe_id AS INT)

-- JOIN 4: Provider Specialty Crosswalk (a834793_provider_db_x_walk_20251013)
-- Needed for: Mapping ASDB provider specialty to standard src_specialty_cd and NPI lookup
-- This table provides the transformer-compatible specialty codes (updated Oct 2025)
-- ⚠️ NOTE: Update crosswalk table date suffix when retraining with new data -- not absolutely necessary
LEFT JOIN
    (SELECT src_specialty_cd, asdb_svc_prov_key, npi 
     FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_provider_db_x_walk_20251013`) AS map
        ON prv.asdb_svc_prov_key = map.asdb_svc_prov_key

-- JOIN 5: NPPES NPI Dataset - Provider Taxonomy Codes
-- Needed for: healthcare_provider_taxonomy_code_1 (primary taxonomy classification)
-- Source: Public BigQuery dataset with National Provider Identifier (NPI) information
LEFT JOIN
    (SELECT npi, healthcare_provider_taxonomy_code_1 
     FROM `bigquery-public-data.nppes.npi_raw`) AS nppes
        ON CAST(map.npi AS STRING) = nppes.npi  -- Cast to STRING to match NPPES data type

-- JOIN 6: ASDB_CLAIMICDPROCSUMMARY - ICD Procedure Codes
-- Needed for: icd9_prcdr_cd (primary ICD procedure code)
-- Source: edp_hcb_mdcd_core_srcv (secure view with row-level access)
LEFT JOIN
    (SELECT claimid, icdpx1 FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_CLAIMICDPROCSUMMARY`) AS icd
        ON clm.claimid = icd.claimid

-- JOIN 7: ASDB_DRG - Diagnosis Related Group Codes
-- Needed for: drg_cd (DRG classification for inpatient stays)
-- Source: Main DRG table with one DRG per claim
LEFT JOIN 
    (SELECT claimid, drg FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_DRG`) as drg
        ON clm.claimid = drg.claimid

-- ============================================================================
-- NOTE: Procedure groups are now generated ALGORITHMICALLY in the SELECT
-- No BASE_PROCEDURE lookup table needed - see prcdr_group_cd column above
-- ============================================================================

;

---------------------------------
--- ICD codes for transformer ---
---------------------------------

-- ============================================================================
-- Table: a834793_Medicaid_d1b_train_ending
-- Purpose: Extract and standardize ICD-9 diagnosis codes from Medicaid claims
-- Description: This table processes primary and secondary diagnosis codes from 
--              claim diagnosis summary data, splits and formats them into 
--              standardized ICD-9 format (XXX.XX)
-- Data Range: Dynamic - inherits date filtering from d1a_train_ending (Step 3)
-- Table expires: 180 days from creation
-- ============================================================================

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1b_train_ending`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1b_train_ending`
OPTIONS (
  labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
  , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
WITH 
-- Base CTE: Pull base member and claim identifiers from previous processing stage
base AS (
    SELECT 
        base.individual_id
        , base.member_id
        , base.claim_line_id
        , base.dt
        , base.index_dt
        , member.dob
        , member.gender
    FROM
        `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1a_train_ending` AS base 
    LEFT JOIN
        (SELECT asdb_member_key, gender, dob FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_MEMBER`) AS member
            ON base.individual_id = member.asdb_member_key
)
-- Primary Diagnosis CTE: Extract and split primary diagnosis codes (icddxpri)
-- Splits diagnosis code on '.' to separate code parts (e.g., "250.00" -> "250" and "00")
, p AS (
    SELECT 
        base.individual_id
        , base.member_id
        , base.claim_line_id
        , base.dt
        , base.index_dt
        -- Calculate gender_cd
        , CASE WHEN TRIM(base.gender) = 'M' THEN 1
               WHEN TRIM(base.gender) = 'F' THEN 0 
               ELSE 2 END AS gender_cd
        -- Calculate age_in_months with standardization
        , CASE
            WHEN DATE_DIFF(base.dt, CAST(base.dob AS DATE), MONTH) < 0 THEN 0  -- Negative ages → 0
            WHEN DATE_DIFF(base.dt, CAST(base.dob AS DATE), MONTH) > 1439 THEN 1439  -- Cap at 1439 months
            ELSE DATE_DIFF(base.dt, CAST(base.dob AS DATE), MONTH)
          END AS age_in_months
        , SPLIT(TRIM(b.icddxpri), '.')[offset(0)] AS x_0      -- First part before decimal (e.g., "250")
        , SPLIT(TRIM(b.icddxpri), '.')[safe_offset(1)] AS x_1 -- Second part after decimal (e.g., "00")
    FROM
        base 
    INNER JOIN
        (SELECT claimid, icddxpri FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_CLAIMDIAGSUMMARY`) AS b 
            ON base.claim_line_id = b.claimid
)
-- Secondary Diagnosis 1 CTE: Extract and split first secondary diagnosis code (icddxsec1)
, s1 AS (
    SELECT 
        base.individual_id
        , base.member_id
        , base.claim_line_id
        , base.dt
        , base.index_dt
        -- Calculate gender_cd
        , CASE WHEN TRIM(base.gender) = 'M' THEN 1
               WHEN TRIM(base.gender) = 'F' THEN 0 
               ELSE 2 END AS gender_cd
        -- Calculate age_in_months with standardization
        , CASE
            WHEN DATE_DIFF(base.dt, CAST(base.dob AS DATE), MONTH) < 0 THEN 0
            WHEN DATE_DIFF(base.dt, CAST(base.dob AS DATE), MONTH) > 1439 THEN 1439
            ELSE DATE_DIFF(base.dt, CAST(base.dob AS DATE), MONTH)
          END AS age_in_months
        , SPLIT(TRIM(b.icddxsec1), '.')[offset(0)] AS x_0      -- First part of secondary dx 1
        , SPLIT(TRIM(b.icddxsec1), '.')[safe_offset(1)] AS x_1 -- Second part of secondary dx 1
    FROM
        base 
    INNER JOIN
        (SELECT claimid, icddxsec1 FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_CLAIMDIAGSUMMARY`) AS b 
            ON base.claim_line_id = b.claimid
)
-- Secondary Diagnosis 2 CTE: Extract and split second secondary diagnosis code (icddxsec2)
, s2 AS (
    SELECT 
        base.individual_id
        , base.member_id
        , base.claim_line_id
        , base.dt
        , base.index_dt
        -- Calculate gender_cd
        , CASE WHEN TRIM(base.gender) = 'M' THEN 1
               WHEN TRIM(base.gender) = 'F' THEN 0 
               ELSE 2 END AS gender_cd
        -- Calculate age_in_months with standardization
        , CASE
            WHEN DATE_DIFF(base.dt, CAST(base.dob AS DATE), MONTH) < 0 THEN 0
            WHEN DATE_DIFF(base.dt, CAST(base.dob AS DATE), MONTH) > 1439 THEN 1439
            ELSE DATE_DIFF(base.dt, CAST(base.dob AS DATE), MONTH)
          END AS age_in_months
        , SPLIT(TRIM(b.icddxsec2), '.')[offset(0)] AS x_0      -- First part of secondary dx 2
        , SPLIT(TRIM(b.icddxsec2), '.')[safe_offset(1)] AS x_1 -- Second part of secondary dx 2
    FROM
        base 
    INNER JOIN
        (SELECT claimid, icddxsec2 FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_CLAIMDIAGSUMMARY`) AS b 
            ON base.claim_line_id = b.claimid
)
-- Secondary Diagnosis 3 CTE: Extract and split third secondary diagnosis code (icddxsec3)
, s3 AS (
    SELECT 
        base.individual_id
        , base.member_id
        , base.claim_line_id
        , base.dt
        , base.index_dt
        -- Calculate gender_cd
        , CASE WHEN TRIM(base.gender) = 'M' THEN 1
               WHEN TRIM(base.gender) = 'F' THEN 0 
               ELSE 2 END AS gender_cd
        -- Calculate age_in_months with standardization
        , CASE
            WHEN DATE_DIFF(base.dt, CAST(base.dob AS DATE), MONTH) < 0 THEN 0
            WHEN DATE_DIFF(base.dt, CAST(base.dob AS DATE), MONTH) > 1439 THEN 1439
            ELSE DATE_DIFF(base.dt, CAST(base.dob AS DATE), MONTH)
          END AS age_in_months
        , SPLIT(TRIM(b.icddxsec3), '.')[offset(0)] AS x_0      -- First part of secondary dx 3
        , SPLIT(TRIM(b.icddxsec3), '.')[safe_offset(1)] AS x_1 -- Second part of secondary dx 3
    FROM
        base 
    INNER JOIN
        (SELECT claimid, icddxsec3 FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_CLAIMDIAGSUMMARY`) AS b 
            ON base.claim_line_id = b.claimid
)
-- Union CTE: Combine all diagnosis codes (primary + 3 secondary) into one dataset
-- UNION DISTINCT removes duplicate diagnosis codes for the same claim
, x1 AS (
    SELECT 
        * 
    FROM 
        p 
    UNION DISTINCT
        SELECT * FROM s1 
    UNION DISTINCT
        SELECT * FROM s2 
    UNION DISTINCT
        SELECT * FROM s3 
)
-- Final SELECT: Format ICD-9 codes into standardized format
-- Logic: 
--   - If no decimal part exists (x_1 IS NULL), use only the first part (x_0)
--   - If decimal part exists, combine as XXX.XX (truncated to 2 decimal places)
-- ⚠️ NOTE: Column named icd9_dx_cd but contains ICD-10 codes
-- ✅ UPDATED: Only keep valid ICD-10 format (filters out legacy ICD-9, lowercase, etc.)
-- Valid ICD-10: Letter + 2 alphanumeric + optional decimal/alphanumeric (e.g., I10, E11.9, Z3A.39)
SELECT 
    individual_id
    , member_id
    , claim_line_id
    , dt
    , gender_cd
    , age_in_months
    , index_dt
    , CASE
        WHEN (CASE WHEN x_1 IS NULL THEN x_0 ELSE CONCAT(x_0, '.', SUBSTR(x_1, 1, 2)) END) IS NULL THEN NULL
        WHEN REGEXP_CONTAINS(
                UPPER(CASE WHEN x_1 IS NULL THEN x_0 ELSE CONCAT(x_0, '.', SUBSTR(x_1, 1, 2)) END),
                r'^[A-Z][0-9A-Z]{2}[\.\w]*$'
             )
            THEN UPPER(CASE WHEN x_1 IS NULL THEN x_0 ELSE CONCAT(x_0, '.', SUBSTR(x_1, 1, 2)) END)  -- ✅ Valid ICD-10
        ELSE NULL  -- ❌ Invalid: legacy ICD-9 (250.00), lowercase (f43.22), etc.
      END AS icd9_dx_cd
    FROM 
        x1
;
-- ============================================================================
-- STEP 4: EXTRACT MEDICATION DATA (GPI CODES) - d1c table
-- 
-- Purpose: Extract Generic Product Identifier (GPI) codes from pharmacy claims
--          for medication features in transformer model
-- 
-- Data Source: ASDB_RX_DATA_STAGE (pharmacy claims)
-- 
-- GPI Code: Generic Product Identifier - standardized drug classification system
--           First 4 digits identify the drug class (e.g., gpi2210 = Statins)
-- 

-- ============================================================================

----------------------------
--- GPIs for transformer ---
----------------------------
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1c_train_ending`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1c_train_ending`
OPTIONS (
  labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
  , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
WITH x0 AS (
    SELECT 
        base.individual_id
        , base.member_id
        , base.index_dt
        -- Gender encoding: 1=Male, 0=Female, 2=Unknown
        , CASE WHEN TRIM(member.gender) = 'M' THEN 1 
            WHEN TRIM(member.gender) = 'F' THEN 0 
            ELSE 2 END AS gender_cd
        -- Age in months at dispensing date (with standardization)
        , CASE
            WHEN DATE_DIFF(CAST(rx.disp_dt AS DATE), CAST(member.dob AS DATE), MONTH) < 0 THEN 0  -- Negative ages → 0
            WHEN DATE_DIFF(CAST(rx.disp_dt AS DATE), CAST(member.dob AS DATE), MONTH) > 1439 THEN 1439  -- Cap at 1439 months
            ELSE DATE_DIFF(CAST(rx.disp_dt AS DATE), CAST(member.dob AS DATE), MONTH)
          END AS age_in_months
        -- Dispensing/fill date
        , rx.disp_dt AS dt
        -- GPI code (first 4 digits): Drug class identifier
        -- Format: 'gpiXXXX' where XXXX is the therapeutic class (e.g., 'gpi2210')
        -- NOTE: This column includes 'gpi' prefix - no additional concat needed in w2ind lookup
        -- ✅ UPDATED: Validate that source has at least 4 digits before creating gpi4
        , CASE
            WHEN TRIM(rx.adjudicated_gpi_cd) IS NULL OR TRIM(rx.adjudicated_gpi_cd) = '' THEN NULL
            WHEN LENGTH(TRIM(rx.adjudicated_gpi_cd)) >= 4 
                AND REGEXP_CONTAINS(SUBSTR(TRIM(rx.adjudicated_gpi_cd), 1, 4), r'^[0-9]{4}$')
                THEN CONCAT('gpi', SUBSTR(TRIM(rx.adjudicated_gpi_cd), 1, 4))  -- ✅ Valid: at least 4 digits
            ELSE NULL  -- ❌ Invalid: too short or non-numeric
        END AS gpi4
    FROM 
        `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_member_train_ending` AS base 
    INNER JOIN
        (SELECT asdb_member_key, CAST(asdb_incurred_dt AS DATE) AS disp_dt, gpi AS adjudicated_gpi_cd 
         FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_RX_DATA_STAGE`) AS rx 
            ON base.member_id = rx.asdb_member_key
    INNER JOIN 
        (SELECT asdb_member_key, gender, dob FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_MEMBER`) AS member
            ON base.member_id = member.asdb_member_key
    WHERE 
        -- Dynamic 36-month lookback logic (same as claims)
        -- Start: 36 months before index_dt
        rx.disp_dt > DATE_SUB(CAST(base.index_dt AS DATE), INTERVAL 36 MONTH)
        -- End: Dispense date must be before index date
        AND CAST(base.index_dt AS DATE) > rx.disp_dt
)
SELECT 
    * 
FROM 
    x0
;

------------------------------------------------------------------------------------
--- MERGE ALL FOR TRANSFORMER PRE-TABLE - one row per day with claims per member ---
------------------------------------------------------------------------------------

-- ============================================================================
-- Table: a834793_Medicaid_o1_train_ending
-- Purpose: Create transformer-ready dataset with one row per member per day
-- Description: Combines demographics with medical event sequences (encoded as indices)
--              for input into transformer models. Each medical code (diagnosis, 
--              procedure, revenue, specialty, GPI) is mapped to a numeric index,
--              and up to 80 indices per day are concatenated into a sequence string.
-- Output Format: One row per member per day with comma-separated code indices
-- Table expires: 180 days from creation
-- ============================================================================

DROP TABLE IF EXISTS  `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_o1_train_ending`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_o1_train_ending`
OPTIONS (
  labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
  , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
WITH 
-- Root0 CTE: Pull demographic data and apply age bounds
-- Age is capped: negative ages set to 0, ages over 1439 months capped at 1439
root0 AS (
    SELECT
        individual_id
        , index_dt
        , member_id
        , dt
        , gender_cd
        , CASE WHEN age_in_months < 0 THEN 0 
            WHEN age_in_months > 1439 THEN 1439 
            ELSE age_in_months end AS age_in_months
    FROM 
        `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1a_train_ending`
/*
    -- Optional: Combine with scoring dataset (currently commented out)
    UNION DISTINCT
        SELECT 
            individual_id
            , member_id
            , dt
            , gender_cd
            , CASE WHEN age_in_months < 0 THEN 0 
                WHEN age_in_months > 1439 THEN 1439 
                ELSE age_in_months end AS age_in_months
    FROM `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_Medicaid_2022_d1c_train_ending`
*/
)
-- Root1 CTE: Add row numbers to identify duplicates per individual per day
, root1 AS (
    SELECT 
        *
        , ROW_NUMBER() OVER (PARTITION BY individual_id, dt) AS seqno
    from 
        root0
)
-- Root2 CTE: Deduplicate - keep only first record per individual per day
, root2 AS (
    SELECT 
        individual_id
        , index_dt
        , member_id
        , dt
        , gender_cd
        , age_in_months 
    FROM 
        root1 
    WHERE 
        seqno = 1
)
-- X0 CTE: Map all medical codes to transformer indices using lookup table (w2ind)
-- ⚠️ CRITICAL REQUIREMENT: The w2ind lookup table MUST contain mappings for ALL code types below
--    If any code type is missing from w2ind, those codes will default to index 0
--    
-- Code patterns in w2ind table should be:
--   - 'days_cnt' + days_count_value
--   - 'hcfa_plc_srv_cd' + place_of_service_code
--   - 'provider_taxonomy_cd' + taxonomy_code  ⚠️ REPLACES src_specialty_cd
--   - 'icd9_dx_cd' + diagnosis_code
--   - 'revenue_cd' + revenue_code
--   - 'prcdr_cd' + procedure_code
--   - 'drg_cd' + drg_code  ⚠️ Ensure DRG codes are in w2ind
--   - 'gpi' + 4_digit_gpi_code  ⚠️ Ensure GPI4 codes are in w2ind
--
-- This CTE combines 9 code types via UNION DISTINCT:
--   1. days_cnt (length of stay)
--   2. hcfa_plc_srv_cd (place of service)
--   3. provider_taxonomy_cd (provider taxonomy from NPPES - REPLACES provider specialty)
--   4. icd9_dx_cd (diagnosis codes)
--   5. revenue_cd (revenue codes)
--   6. prcdr_cd (procedure codes)
--   7. icd9_prcdr_cd (ICD-9 procedure codes)
--   8. drg_cd (Diagnosis Related Group codes)
--   9. gpi4 (Generic Product Identifier for medications) 
, x0 as (
    -- Map 1: Days count codes
    SELECT 
        base.individual_id
        , member_id
        , base.dt
        , CASE WHEN w2ind.ind IS NULL THEN 0 
            ELSE w2ind.ind END AS ind  -- Default to 0 if code not found in lookup
    FROM 
        `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1a_train_ending` AS base 
    LEFT JOIN 
        `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind` AS w2ind
            ON CONCAT('days_cnt', CAST(days_cnt AS STRING)) = w2ind.cd
    WHERE 
        days_cnt IS NOT NULL
    UNION DISTINCT
        -- Map 2: Place of service codes
        SELECT 
            base.individual_id
            , member_id
            , base.dt
            , CASE WHEN w2ind.ind IS NULL THEN 0 
                ELSE w2ind.ind END AS ind
        FROM
            `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1a_train_ending` AS base 
        LEFT JOIN
            `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind` AS w2ind
                ON concat('hcfa_plc_srv_cd', cast(hcfa_plc_srv_cd AS string)) = w2ind.cd
        WHERE hcfa_plc_srv_cd IS NOT NULL
    -- ⚠️ COMMENTED OUT: Provider specialty codes replaced by provider taxonomy codes
    -- UNION DISTINCT
    --     -- Map 3: Provider specialty codes
    --     SELECT 
    --         base.individual_id
    --         , member_id
    --         , base.dt
    --         , CASE WHEN w2ind.ind IS NULL THEN 0 
    --             ELSE w2ind.ind END AS ind
    --     FROM 
    --         `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1a_train_ending` AS base 
    --     LEFT JOIN
    --         `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind` AS w2ind
    --             ON CONCAT('src_specialty_cd', CAST(src_specialty_cd AS STRING)) = w2ind.cd
    --     WHERE
    --         src_specialty_cd IS NOT NULL
    UNION DISTINCT
        -- Map 3: Provider taxonomy codes (from NPPES) - REPLACES provider specialty
        -- Maps healthcare provider taxonomy classifications to indices
        -- Pattern in w2ind.cd column: 'provider_taxonomy_cd' + TAXONOMY_CODE
        SELECT 
            base.individual_id
            , member_id
            , base.dt
            , CASE WHEN w2ind.ind IS NULL THEN 0 
                ELSE w2ind.ind END AS ind
        FROM 
            `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1a_train_ending` AS base 
        LEFT JOIN
            `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind` AS w2ind
                ON CONCAT('provider_taxonomy_cd', CAST(provider_taxonomy_cd AS STRING)) = w2ind.cd
        WHERE
            provider_taxonomy_cd IS NOT NULL
    UNION DISTINCT
        -- Map 4: ICD-9 diagnosis codes
        SELECT 
            base.individual_id
            , member_id
            , base.dt
            , CASE WHEN w2ind.ind IS NULL THEN 0 
                ELSE w2ind.ind END AS ind
        FROM 
            `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1b_train_ending` AS base 
        LEFT JOIN
            `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind` AS w2ind
                ON CONCAT('icd9_dx_cd', CAST(icd9_dx_cd AS STRING)) = w2ind.cd
        WHERE 
            icd9_dx_cd IS NOT NULL
    UNION DISTINCT
        -- Map 5: Revenue codes
        SELECT 
            base.individual_id
            , member_id
            , base.dt
            , CASE WHEN w2ind.ind IS NULL THEN 0 
                ELSE w2ind.ind END AS ind
        FROM 
            `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1a_train_ending` AS base 
        LEFT JOIN
            `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind` AS w2ind
                ON CONCAT('revenue_cd', CAST(revenue_cd AS STRING)) = w2ind.cd
        WHERE 
            revenue_cd IS NOT NULL
    UNION DISTINCT
        -- Map 6: Procedure codes
        SELECT
            base.individual_id
            , member_id
            , base.dt
            , case when w2ind.ind IS NULL THEN 0 
                else w2ind.ind END AS ind
        FROM
            `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1a_train_ending` AS base 
        LEFT JOIN
            `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind` AS w2ind
                ON CONCAT('prcdr_cd', CAST(prcdr_cd AS STRING)) = w2ind.cd
        WHERE 
            prcdr_cd IS NOT NULL
    UNION DISTINCT
        -- Map 7: ICD-9 procedure codes
        SELECT 
            base.individual_id
            , member_id
            , base.dt
            , CASE WHEN w2ind.ind IS NULL THEN 0 
                ELSE w2ind.ind END AS ind
        FROM 
            `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1a_train_ending` AS base 
        LEFT JOIN
            `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind` AS w2ind
                ON CONCAT('prcdr_cd', CAST(icd9_prcdr_cd AS STRING)) = w2ind.cd
        WHERE 
            icd9_prcdr_cd IS NOT NULL
    UNION DISTINCT
        -- Map 8: DRG codes (Diagnosis Related Group for inpatient stays)
        -- ⚠️ REQUIREMENT: w2ind table must contain DRG mappings
        -- Pattern in w2ind.cd column: 'drg_cd' + DRG_CODE (e.g., 'drg_cd470')
        -- If DRG codes are missing from w2ind, they will be mapped to index 0
        SELECT 
            base.individual_id
            , member_id
            , base.dt
            , CASE WHEN w2ind.ind IS NULL THEN 0 
                ELSE w2ind.ind END AS ind
        FROM 
            `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1a_train_ending` AS base 
        LEFT JOIN
            `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind` AS w2ind
                ON CONCAT('drg_cd', CAST(drg_cd AS STRING)) = w2ind.cd
        WHERE 
            drg_cd IS NOT NULL

    -- Map 9: GPI medication codes (pharmacy data)
    -- ⚠️ REQUIREMENT: w2ind table must contain GPI4 mappings
    -- Pattern in w2ind.cd column: 'gpi' + 4_DIGIT_GPI (e.g., 'gpi2210')
    -- If GPI4 codes are missing from w2ind, they will be mapped to index 0
    -- NOTE: gpi4 column already has 'gpi' prefix from line 634, so no CONCAT needed
    UNION DISTINCT
        SELECT
            base.individual_id
            , member_id
            , base.dt
            , CASE WHEN w2ind.ind IS NULL THEN 0 
                ELSE w2ind.ind END AS ind
        FROM 
            `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1c_train_ending` AS base 
        LEFT JOIN
            `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind` AS w2ind
                ON CAST(gpi4 AS STRING) = w2ind.cd  -- No concat needed, gpi4 already has 'gpi' prefix
        WHERE 
            gpi4 IS NOT NULL

)
-- X1 CTE: Remove duplicate indices per individual per day
-- Groups to ensure each unique index appears only once per member per day
, x1 AS (
    SELECT 
        individual_id
        , dt
        , ind
    FROM 
        x0 
    GROUP BY
        individual_id
        , dt
        , ind 
    )
-- X2 CTE: Assign sequence numbers to indices within each individual-day
-- Orders the indices for each member on each day (used to limit to 80 indices)
, x2 AS (
    SELECT 
        *
        , ROW_NUMBER() OVER (PARTITION BY individual_id,dt) AS seqno
    FROM 
        x1
)
-- X3 CTE: Aggregate indices into comma-separated string (limited to 80 per day)
-- Creates the final sequence string for transformer input
-- Example output: "15,42,103,7,88,..."
, x3 AS (
    SELECT 
        individual_id
        , dt
        , STRING_AGG(CAST(ind AS STRING), ',') AS cd
    FROM 
        x2 
    WHERE 
        seqno<=80  -- Limit to first 80 indices per day for transformer input size constraints
    GROUP BY 
        individual_id
        , dt
)
-- Final SELECT: Combine demographics with encoded medical event sequences
-- Output: One row per member per day with all relevant features
SELECT 
    root2.individual_id
    , root2.index_dt
    , root2.member_id
    , root2.dt
    , root2.gender_cd
    , root2.age_in_months
    , x3.cd  -- Comma-separated string of up to 80 medical code indices
FROM 
    root2 
INNER JOIN
    x3 
        ON root2.individual_id = x3.individual_id 
        AND root2.dt = x3.dt
;


/*==============================================================================
  STEP 6b: MAP CODES TO TARGET INDICES 🎯 NEW!
  
  Purpose: Map grouped codes to target vocabulary for next-day prediction labels
  
  8 Target Code Types (~5k codes vs ~84k input codes):
  1. Place of Service (keep as-is)
  2. Procedure Groups - ALGORITHMIC (prcdr_group_cd for both CPT and ICD procedures)
  3. ICD Diagnosis (first 3 digits: 250.00 → 250)
  4. GPI Medications (first 2 digits: gpi2210 → gpi22)
  5. Revenue Code (first 3 digits: 0250 → 025)
  6. DRG codes (keep as-is)
  7. Provider Taxonomy (first 4 chars: 207Q00000X → 207Q)
  8. Days Count (keep as-is: 0-11, 99)
  
  Note: Procedure groups are unified - both CPT (e.g., 99213) and ICD (e.g., 0RJD4ZZ)
        procedures map to the same prcdr_group_cd space (e.g., prcdr_group_992, prcdr_group_02h)
  
  Output: a834793_Medicaid_o1_train_ending_target (patient-date-target_index)
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_o1_train_ending_target`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_o1_train_ending_target`
 OPTIONS (labels=[("owner", "pritha_ghosh_cvshealth_com"),("cost_center","13070")],
    expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)) AS
      
      -- Target Type 1: Place of Service (keep as-is)
      -- Maps: hcfa_plc_srv_cd → target index
      select 
          base.individual_id,
          base.index_dt,
          base.dt,
          case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from
          `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1a_train_ending` base
              left join (select ind, cd from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind_target`) w2ind
            on concat('hcfa_plc_srv_cd', cast (base.hcfa_plc_srv_cd as string)) = w2ind.cd
      where base.hcfa_plc_srv_cd is not null
      
      UNION ALL
      
      -- Target Type 2: Unified Procedure Groups (CPT + ICD) - ALGORITHMIC
      -- Maps: prcdr_group_cd → target index (e.g., prcdr_group_992 = office visits, prcdr_group_02h = cardiac procedures)
      -- Handles both CPT/HCPCS and ICD procedure groups in single unified column with algorithmic grouping
      select 
          base.individual_id,
          base.index_dt,
          base.dt,
          case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from
          `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1a_train_ending` base
              left join (select ind, cd from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind_target`) w2ind
            on base.prcdr_group_cd = w2ind.cd
      where base.prcdr_group_cd is not null
      
      UNION ALL
      
      -- Target Type 3: ICD Diagnosis (First 3 digits)
      -- Maps: icd9_dx_cd → first 3 digits → target index
      -- Example: 250.00 → 250 (all diabetes)
      -- Extract first 3 digits on-the-fly: split on '.', take first part
      select 
          base.individual_id,
          base.index_dt,
          base.dt,
          case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from
          `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1b_train_ending` base
              left join (select ind, cd from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind_target`) w2ind
            on concat('icd9_dx_cd', SPLIT(cast(base.icd9_dx_cd as string), '.')[SAFE_OFFSET(0)]) = w2ind.cd
      where base.icd9_dx_cd is not null
      
      UNION ALL
      
      -- Target Type 4: GPI Medications (First 2 digits)
      -- Maps: gpi4 → extract first 2 digits → target index
      -- Example: gpi2210 → gpi22 (all insulins)
      -- Extract first 2 digits from gpi4: gpi4='gpi1234' → take chars 4-5 (first 2 digits after 'gpi')
      select 
          base.individual_id,
          base.index_dt,
          base.dt,
          case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from
          `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1c_train_ending` base
              left join (select ind, cd from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind_target`) w2ind
            on concat('gpi', SUBSTR(REPLACE(base.gpi4, 'gpi', ''), 1, 2)) = w2ind.cd
      where base.gpi4 is not null
      
      UNION ALL
      
      -- Target Type 5: Revenue Code Groups (First 3 digits)
      -- Maps: revenue_cd → target index (first 3 digits)
      -- Example: 0250 → 025 (all pharmacy services)
      select 
          base.individual_id,
          base.index_dt,
          base.dt,
          case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from
          `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1a_train_ending` base
              left join (select ind, cd from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind_target`) w2ind
            on concat('revenue_cd', SUBSTR(cast(base.revenue_cd as string), 1, 3)) = w2ind.cd
      where base.revenue_cd is not null
      
      UNION ALL
      
      -- Target Type 6: DRG Codes (Keep as-is)
      -- Maps: drg_cd → target index (already grouped)
      select 
          base.individual_id,
          base.index_dt,
          base.dt,
          case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from
          `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1a_train_ending` base
              left join (select ind, cd from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind_target`) w2ind
            on concat('drg_cd', cast(base.drg_cd as string)) = w2ind.cd
      where base.drg_cd is not null
      
      UNION ALL
      
      -- Target Type 7: Provider Taxonomy Groups (First 4 characters)
      -- Maps: provider_taxonomy_cd → target index (first 4 chars)
      -- Example: 207Q00000X → 207Q (all family medicine)
      select 
          base.individual_id,
          base.index_dt,
          base.dt,
          case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from
          `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1a_train_ending` base
              left join (select ind, cd from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind_target`) w2ind
            on concat('provider_taxonomy_cd', SUBSTR(TRIM(CAST(base.provider_taxonomy_cd AS STRING)), 1, 4)) = w2ind.cd
      where base.provider_taxonomy_cd is not null
        and TRIM(CAST(base.provider_taxonomy_cd AS STRING)) != ''
        and LENGTH(TRIM(CAST(base.provider_taxonomy_cd AS STRING))) >= 4
      
      UNION ALL
      
      -- Target Type 8: Days Count (Keep as-is)
      -- Maps: days_cnt → target index (already bucketed 0-11, 99)
      select 
          base.individual_id,
          base.index_dt,
          base.dt,
          case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from
          `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_d1a_train_ending` base
              left join (select ind, cd from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind_target`) w2ind
            on concat('days_cnt', cast(base.days_cnt as string)) = w2ind.cd
      where base.days_cnt is not null
;


/*==============================================================================
  STEP 7: CREATE TEMPORAL SEQUENCES WITH NEXT-DAY PREDICTION TARGETS 🎯 UPDATED!
  
  Purpose: Aggregate daily codes into temporal sequences and apply next-day shift
           for transformer training
  
  Changes from original:
  - Now aggregates BOTH input codes (cd) and target codes (target)
  - Applies LEAD function to shift targets by 1 day for next-day prediction
  - Model learns: Given codes on day N, predict target codes on day N+1
  
  Output: a834793_Medicaid_o3_train_ending (ONE ROW PER MEMBER)
  Fields: individual_id, gender_cd, age_in_months, cd, target, dt_cnt
  
==============================================================================*/

DROP TABLE if exists `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_o3_train_ending`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_o3_train_ending`
OPTIONS (
  labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
  , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
WITH 
-- X1 CTE: Assign chronological sequence numbers (input codes)
-- 🔧 FIXED: Changed ORDER BY dt ASC → DESC to select MOST RECENT 200 days (not oldest)
x1 AS (
    SELECT 
        *
        , ROW_NUMBER() OVER (PARTITION BY individual_id ORDER BY dt DESC) AS seqno  -- ✅ DESC = newest first
    FROM 
        `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_o1_train_ending`
)
-- X2 CTE: Limit to most recent 200 days per individual
, x2 AS (
    SELECT * 
    FROM x1 
    WHERE seqno <= 200
)
-- X3 CTE: Aggregate input codes by individual+date (comma-separated within day)
-- Note: o1_train_ending already has cd aggregated, so we just pass it through
, x3 AS (
    SELECT 
        individual_id
        , index_dt
        , dt
        , gender_cd
        , age_in_months
        , cd
        , seqno
    FROM x2
)
-- Y1 CTE: Assign chronological sequence numbers (target codes)
-- 🔧 FIXED: Changed ORDER BY dt ASC → DESC to select MOST RECENT 200 days (not oldest)
, y1 AS (
    SELECT 
        *
        , ROW_NUMBER() OVER (PARTITION BY individual_id, index_dt ORDER BY dt DESC) AS seqno  -- ✅ DESC = newest first, partition by index_dt
    FROM 
        `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_o1_train_ending_target`
)
-- Y2 CTE: Limit to most recent 200 days per individual
, y2 AS (
    SELECT * 
    FROM y1 
    WHERE seqno <= 200
)
-- Y3 CTE: Aggregate target codes by individual+date (comma-separated within day)
, y3 AS (
    SELECT 
        individual_id
        , index_dt
        , dt
        , STRING_AGG(CAST(ind AS STRING), ',' ORDER BY ind) AS target
    FROM y2
    GROUP BY individual_id, index_dt, dt  -- ✅ FIXED: Removed seqno from GROUP BY, added index_dt
)
-- Z1 CTE: Join input and target codes by individual+date
, z1 AS (
    SELECT 
        x3.individual_id
        , x3.index_dt
        , x3.dt
        , x3.gender_cd
        , x3.age_in_months
        , x3.cd
        , y3.target
        , x3.seqno
    FROM x3
    LEFT JOIN y3
        ON x3.individual_id = y3.individual_id
        AND x3.dt = y3.dt
        AND x3.index_dt = y3.index_dt  -- ✅ ADDED: Match on index_dt for proper join
)
-- Z2 CTE: Apply LEAD for next-day prediction (shift target by 1 day)
-- The model learns: Given codes on day N (cd), predict codes on day N+1 (target)
, z2 AS (
    SELECT 
        individual_id
        , index_dt
        , dt
        , gender_cd
        , age_in_months
        , cd
        , LEAD(target, 1) OVER (PARTITION BY individual_id, index_dt ORDER BY dt ASC) AS target_next_day
        , seqno
    FROM z1
)
-- Z3 CTE: Filter out last day per member (no next-day target available)
, z3 AS (
    SELECT 
        individual_id
        , index_dt
        , dt
        , gender_cd
        , age_in_months
        , cd
        , target_next_day AS target
        , seqno
    FROM z2
    WHERE target_next_day IS NOT NULL  -- Remove last day (no prediction target)
)
-- Z4 CTE: Re-sequence after filtering
, z4 AS (
    SELECT 
        *
        , ROW_NUMBER() OVER (PARTITION BY individual_id, index_dt ORDER BY dt ASC) AS seqno2
    FROM z3
)
-- Z5 CTE: Final aggregation into temporal sequences (asterisk-separated days)
, z5 AS (
    SELECT 
        individual_id
        , index_dt
        , STRING_AGG(CAST(gender_cd AS STRING), '*' ORDER BY seqno2) AS gender_cd
        , STRING_AGG(CAST(age_in_months AS STRING), '*' ORDER BY seqno2) AS age_in_months
        , STRING_AGG(CAST(cd AS STRING), '*' ORDER BY seqno2) AS cd
        , STRING_AGG(CAST(target AS STRING), '*' ORDER BY seqno2) AS target
        , COUNT(*) AS dt_cnt
    FROM z4
    GROUP BY individual_id, index_dt
)
-- Final output: One row per member with both input and target sequences
SELECT 
    individual_id
    , index_dt
    , gender_cd
    , age_in_months
    , cd
    , target
    , dt_cnt
FROM z5
;

/*==============================================================================
  ✅ PIPELINE COMPLETE - TRANSFORMER TRAINING DATA READY 🎯
  
  Final Output Table: a834793_Medicaid_o3_train_ending
  
  OUTPUT FORMAT (ONE ROW PER MEMBER):
  - individual_id: Member identifier
  - gender_cd: Temporal gender sequence (asterisk-separated)
  - age_in_months: Temporal age sequence (asterisk-separated)
  - cd: INPUT CODES - Detailed medical codes (~84k vocabulary)
  - target: TARGET CODES - Grouped codes for prediction (~5k vocabulary)
  - dt_cnt: Number of days in sequence
  
  TRANSFORMER TRAINING TASK:
  Next-Day Prediction: Given codes on day N (cd), predict codes on day N+1 (target)
  
  INPUT VOCABULARY (cd field - 9 code types, ~84k codes):
  1. ✅ Days count (length of stay: 0-11, 99)
  2. ✅ Place of service (HCFA codes)
  3. ✅ Provider taxonomy codes (full NPPES taxonomy)
  4. ✅ ICD diagnosis codes (full precision: XXX.XX)
  5. ✅ Revenue codes (full 4-digit codes)
  6. ✅ CPT/HCPCS procedure codes (individual codes)
  7. ✅ ICD-9/10 procedure codes (individual codes)
  8. ✅ DRG codes (Diagnosis Related Groups)
  9. ✅ GPI4 medication codes (4-digit drug classes)
  
  TARGET VOCABULARY (target field - 8 code types, ~5k codes):
  1. ✅ Days count (keep as-is: 0-11, 99)
  2. ✅ Place of service (keep as-is)
  3. ✅ Procedure groups (unified CPT+ICD → prcdr_group_cd - ALGORITHMIC)
  4. ✅ ICD diagnosis groups (first 3 digits: 250.00 → 250)
  5. ✅ GPI medication groups (first 2 digits: gpi2210 → gpi22)
  6. ✅ Revenue code groups (first 3 digits: 0250 → 025)
  7. ✅ DRG codes (keep as-is)
  8. ✅ Provider taxonomy groups (first 4 chars: 207Q00000X → 207Q)
  
  ⚠️ REQUIRED LOOKUP TABLES:
  1. a834793_member_w2ind (input vocabulary mapping)
  2. a834793_member_w2ind_target (target vocabulary mapping)
  
  Both tables must be created BEFORE running this pipeline!
  See: create_w2ind_target_from_w2ind.sql
  
  ⚠️ For retraining with new data:
  - See RETRAINING CHECKLIST at the top of this file (lines 14-43)
  - Update all date ranges and table name suffixes
  - Rebuild w2ind and w2ind_target tables with new data
  - Re-run this entire script from top to bottom
  
  Contact: Pritha Ghosh (pritha.ghosh@cvshealth.com)
  
==============================================================================*/