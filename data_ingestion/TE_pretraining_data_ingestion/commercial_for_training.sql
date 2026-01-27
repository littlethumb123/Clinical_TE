/*==============================================================================
  COMMERCIAL CLINICAL TRANSFORMER - TRAINING DATA PREPARATION PIPELINE
  
  Owner: Pritha Ghosh (pritha.ghosh@cvshealth.com)
  Training Period: 2023 (36-month lookback: 2020-2023)
  
  Purpose: Prepare temporal sequences of medical codes for transformer training
           - Input sequences: ~84k detailed codes (procedures, diagnoses, medications)
           - Target sequences: ~5k grouped codes for next-day prediction
           - Output: 256-dim patient embeddings for risk stratification
  
  Key Features:
  - Real procedure groups from BASE_PROCEDURE (not approximations)
  - Unified procedure groups (CPT + ICD in same target space)
  - Next-day prediction using LEAD logic
  - 8 target code types for comprehensive prediction
  - Ready for transformer model training
  
  ============================================================================
  
  PIPELINE OVERVIEW (14 Steps):
  
  Step 0:  member_base_memberid             - Base membership
  Step 1:  member_monthly_claims             - Medical claims + procedure groups
  Step 2:  member_monthly_rx                 - Prescriptions  
  Step 3:  member_d1a_train_ending_tmp       - Procedures intermediate
  Step 4:  member_d1c_train_ending_tmp       - Prescriptions intermediate
  Step 5:  member_monthly_clm_ln             - Claims + diagnoses
  Step 6:  member_d1b_train_ending_tmp       - Diagnoses intermediate
  Step 7:  member_root                       - Patient-date calendar
  Step 8:  member_get_cd                     - INPUT code mapping (~84k codes)
  Step 8b: member_get_cd_target              - TARGET code mapping (~5k codes) 🎯
  Step 9:  member_o1_train_ending_tmp        - Aggregated with cd + target 🎯
  Step 10: member_o3_train_ending_tmp_1      - Filter 200 dates
  Step 11: member_o3_train_ending_tmp_ordered - LEAD applied (next-day) 🎯
  Step 12: member_o3_train_ending            - FINAL OUTPUT (sequences) 🎯
  
  Prefix: a834793_Commercial_member
  Dataset: edp-prod-storage.edp_ent_sdoheir_cns
  Expiration: 180 days
  
==============================================================================*/

/*==============================================================================
  STEP 0: CREATE BASE MEMBERSHIP
  
  Purpose: Random index date per member for 2023 Commercial cohort
  Source: EMIS_MEMBERSHIP (2023-01-01 to 2023-12-31)
  Output: individual_id, member_id, index_dt
  
==============================================================================*/

-- Create base membership: one random index_dt per member for 2023  
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_current_individual_id`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_current_individual_id`
OPTIONS (labels=[("owner", "pritha_ghosh_cvshealth_com"),("costcenter","13070")],
    expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)) AS
SELECT 
  member_id, 
  ARRAY_AGG(eff_dt ORDER BY RAND() LIMIT 1)[SAFE_OFFSET(0)] AS index_dt
FROM `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_MEMBERSHIP`
WHERE eff_dt BETWEEN DATE('2023-01-01') AND DATE('2023-12-31')
  AND business_ln_cd = 'CP'
  AND file_id <> 'C2'
GROUP BY member_id
ORDER BY member_id;


DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_memberid`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_memberid`
    CLUSTER BY member_id
    OPTIONS (labels=[("owner", "pritha_ghosh_cvshealth_com"),("costcenter","13070")],
    expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
as
-- Step 0b: Join input cohort with membership and create final base table
-- ✅ UPDATED: Added logic to remove restricted self-insured (SI) clients
WITH base_with_membership AS (
    -- Get membership details for restriction filtering
    SELECT DISTINCT 
        m.member_id, 
        x.individual_id, 
        m.index_dt,
        em.group_nbr,
        em.fund_ctg_cd
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_current_individual_id` m
    
    -- Map individual_id to member_id
    JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` x
        ON m.member_id = x.member_id
    
    -- Get group and funding category for restriction filtering
    JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_MEMBERSHIP` em
        ON m.member_id = em.member_id
        AND m.index_dt = em.eff_dt
        AND em.file_id <> 'C2'
        AND em.business_ln_cd = 'CP'
),

-- Identify restricted self-insured clients
restricted_SI AS (
    SELECT DISTINCT g.ps_unique_id
    FROM base_with_membership a
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GROUP_CONTROL` AS g
        ON a.group_nbr = g.group_nbr
    LEFT JOIN `anbc-hcb-prod.insights_share_hcb_prod.v_enriched_customer` AS cum
        ON g.ps_unique_id = SAFE_CAST(cum.psuid AS INT64)
    WHERE a.fund_ctg_cd = 'B'  -- Self-insured only
        AND cum.rstrctd_client_ind = 'Y'  -- Restricted flag
),

-- Identify termed self-insured clients (terminated before 24 months)
termed_SI AS (
    SELECT DISTINCT g.ps_unique_id
    FROM base_with_membership a
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GROUP_CONTROL` AS g
        ON a.group_nbr = g.group_nbr
    WHERE a.fund_ctg_cd = 'B'  -- Self-insured only
        AND g.renewal_dt < DATE_SUB(CURRENT_DATE(), INTERVAL 24 MONTH)  -- Termed before 24 months
)

-- Final selection: Exclude restricted and termed SI clients
SELECT DISTINCT 
    base.member_id, 
    base.individual_id, 
    base.index_dt
FROM base_with_membership base
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GROUP_CONTROL` AS g
    ON base.group_nbr = g.group_nbr
WHERE 
    -- Keep all non-self-insured (fully insured, split insured, etc.)
    (base.fund_ctg_cd != 'B')
    OR 
    -- For self-insured, exclude restricted and termed clients
    (base.fund_ctg_cd = 'B' 
     AND g.ps_unique_id NOT IN (SELECT ps_unique_id FROM restricted_SI)
     AND g.ps_unique_id NOT IN (SELECT ps_unique_id FROM termed_SI))
;


/*==============================================================================
  STEP 1: EXTRACT MEDICAL CLAIMS + ALGORITHMIC PROCEDURE GROUPS
  
  Purpose: Extract claims with 90-day lag, add algorithmic procedure groups
  Lookback: 36 months (2020-2023), 90-day claims finalization lag
  
  Key Additions:
  - prcdr_group_cd: ALGORITHMIC procedure groups based on code structure:
    * CPT (5-digit): First 3 digits → prcdr_group_992, prcdr_group_335, etc.
    * CPT Cat II/III (4-digit+letter): First 4 digits → prcdr_group_0001, prcdr_group_0012, etc.
    * ICD-10-PCS (7-char): First 3 chars → prcdr_group_02h, prcdr_group_0u5, etc.
    * ICD-9 Proc (decimal): Before decimal → prcdr_group_00, prcdr_group_66, etc.
    * HCPCS (letter+4 digits): First 2 chars → prcdr_group_j1, prcdr_group_a0, etc.
    * Dental (D+4 digits): First 3 chars → prcdr_group_d72, prcdr_group_d01, etc.
  - provider_taxonomy_cd: From NPPES (replaces src_specialty_cd)
  
  Output: member_monthly_claims (claim-level with algorithmic procedure groups)
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_monthly_claims`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_monthly_claims`
CLUSTER BY individual_id,member_id,index_dt
OPTIONS (labels=[("owner", "pritha_ghosh_cvshealth_com"),("costcenter","13070")],
    expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
as
with base as (
    -- Step 1a: Join base membership with demographics
    -- Handles cases where 1 individual_id maps to multiple member_ids over time
    SELECT rm.individual_id,
           rm.member_id,
           rm.index_dt,
           m.gender_cd,      -- Patient gender
           m.birth_dt        -- Date of birth for age calculation
    	   -- Priority order: most recent coverage effective date, then HMO conversion date, then highest member_id
    	   , row_number() over (partition by rm.individual_id 
    	                        order by orig_covg_eff_dt desc, hmo_to_trad_conv_dt desc, rm.member_id desc) as ord
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_memberid` rm
    JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEMBER` m
    ON rm.member_id = m.member_id
)
, b_dt as (
	-- Step 1b: Deduplicate demographics - ensure 1 individual_id has only 1 gender_cd and 1 birth_dt
	-- Takes the most recent member_id's demographics (ord = 1)
	select *
	from base
	where ord = 1
)
-- Step 1c: Extract claims with demographics and code transformations
select
        -- PATIENT IDENTIFIERS
base.individual_id, 
base.member_id,
        base.index_dt,
clm.claim_line_id,
        
        -- SERVICE DATE (used for day-level aggregation)
clm.srv_start_dt as dt,
        
        -- DAYS COUNT: Length of stay (for inpatient claims)
        -- Null/negative → 99, >10 days → 11 (capped), else actual value
        case when (clm.days_cnt is null or clm.days_cnt < 0) then 99 
             when clm.days_cnt > 10 then 11 
             else clm.days_cnt end as days_cnt,
        
        -- GENDER: Encoded as 0=Female, 1=Male, 2=Unknown/Other
        -- Uses deduplicated demographics from b_dt CTE
        case when trim(b_dt.gender_cd)='M' then 1 
             when trim(b_dt.gender_cd)='F' then 0 
             else 2 end as gender_cd,
        
        -- AGE IN MONTHS: Age at service date
        -- Using standard BigQuery DATE_DIFF instead of custom UDF
        -- Alternative: `anbc-hcb-prod.clin_analytics_share_hcb_prod.months_between_floor`(clm.srv_start_dt, b_dt.birth_dt)
        DATE_DIFF(CAST(clm.srv_start_dt AS DATE), CAST(b_dt.birth_dt AS DATE), MONTH) as age_in_months,
        
        -- MEDICAL CODES: All cleaned (trimmed, uppercased) and null if empty
        -- Revenue code with validation to exclude garbage values
        -- ✅ UPDATED: Only keep 3-4 digit numeric revenue codes
        CASE 
            WHEN TRIM(clm.revenue_cd) = '' OR clm.revenue_cd IS NULL THEN NULL
            WHEN REGEXP_CONTAINS(CAST(clm.revenue_cd AS STRING), r'^[0-9]{3,4}$') THEN UPPER(TRIM(clm.revenue_cd))  -- ✅ Valid: 3-4 digits
            ELSE NULL  -- ❌ Invalid: special characters, wrong length
        END AS revenue_cd,
        
        -- Place of Service with validation
        -- ✅ UPDATED: Only keep 1-2 digit numeric POS codes (filters out U, N, XX, empty strings, etc.)
        CASE 
            WHEN TRIM(clm.hcfa_plc_srv_cd) IS NULL OR TRIM(clm.hcfa_plc_srv_cd) = '' THEN NULL
            WHEN REGEXP_CONTAINS(CAST(clm.hcfa_plc_srv_cd AS STRING), r'^[0-9]{1,2}$') THEN UPPER(TRIM(clm.hcfa_plc_srv_cd))  -- ✅ Valid: 1-2 digits
            ELSE NULL  -- ❌ Invalid: letters (U, N), special chars, wrong length
        END AS hcfa_plc_srv_cd,
        -- ⚠️ COMMENTED OUT: Replaced by provider_taxonomy_cd from NPPES
        -- case when trim(clm.src_specialty_cd)='' then null else upper(trim(clm.src_specialty_cd)) end as src_specialty_cd, -- Provider specialty
        -- Provider Taxonomy with validation
        -- ✅ VALIDATION ADDED: Only keep 10-character alphanumeric taxonomy codes (NPPES standard)
        CASE 
            WHEN TRIM(nppes.healthcare_provider_taxonomy_code_1) = '' OR nppes.healthcare_provider_taxonomy_code_1 IS NULL THEN NULL
            WHEN REGEXP_CONTAINS(TRIM(nppes.healthcare_provider_taxonomy_code_1), r'^[A-Z0-9]{10}$') 
                THEN UPPER(TRIM(nppes.healthcare_provider_taxonomy_code_1))  -- ✅ Valid: 10-char alphanumeric
            ELSE NULL  -- ❌ Invalid: wrong length, special chars, lowercase
        END AS provider_taxonomy_cd,
        -- Procedure codes with validation: filter out too-short codes (e.g., single letter 'A')
        case when trim(clm.prcdr_cd)='' then null 
             when length(trim(clm.prcdr_cd)) < 4 then null  -- ✅ Filter codes < 4 chars (garbage)
             else upper(trim(clm.prcdr_cd)) end as prcdr_cd,  -- CPT/HCPCS procedure
        case when trim(icd_prc.icd9_prcdr_cd)='' then null 
             when length(trim(icd_prc.icd9_prcdr_cd)) < 4 then null  -- ✅ Filter codes < 4 chars
             else upper(trim(icd_prc.icd9_prcdr_cd)) end as icd9_prcdr_cd, -- ⚠️ NOTE: Column named icd9_prcdr_cd but contains ICD-10-PCS codes
        
        -- ============================================================================
        -- ALGORITHMIC PROCEDURE GROUPING (for target vocabulary)
        -- ============================================================================
        -- Apply algorithmic grouping to BOTH CPT and ICD procedure codes
        -- Then merge with COALESCE (prioritize CPT if both exist)
        COALESCE(
          -- CPT/HCPCS procedure group
          CASE
            WHEN REGEXP_CONTAINS(UPPER(TRIM(clm.prcdr_cd)), r'^\d{5}$') 
              THEN CONCAT('prcdr_group_', SUBSTR(UPPER(TRIM(clm.prcdr_cd)), 1, 3))
            WHEN REGEXP_CONTAINS(UPPER(TRIM(clm.prcdr_cd)), r'^\d{4}[A-Z]$')
              THEN CONCAT('prcdr_group_', SUBSTR(UPPER(TRIM(clm.prcdr_cd)), 1, 4))
            WHEN REGEXP_CONTAINS(UPPER(TRIM(clm.prcdr_cd)), r'^[0-9A-Z]{6}$')
              THEN CONCAT('prcdr_group_', LOWER(SUBSTR(UPPER(TRIM(clm.prcdr_cd)), 1, 3)))
            WHEN REGEXP_CONTAINS(UPPER(TRIM(clm.prcdr_cd)), r'^\d[A-Z0-9]{6}$') 
              THEN CONCAT('prcdr_group_', LOWER(SUBSTR(UPPER(TRIM(clm.prcdr_cd)), 1, 3)))
            WHEN REGEXP_CONTAINS(UPPER(TRIM(clm.prcdr_cd)), r'^\d+\.\d+$') 
              THEN CONCAT('prcdr_group_', SPLIT(UPPER(TRIM(clm.prcdr_cd)), '.')[SAFE_OFFSET(0)])
            WHEN REGEXP_CONTAINS(UPPER(TRIM(clm.prcdr_cd)), r'^[A-Z]\d{4}$') AND LEFT(UPPER(TRIM(clm.prcdr_cd)), 1) != 'D'
              THEN CONCAT('prcdr_group_', LOWER(SUBSTR(UPPER(TRIM(clm.prcdr_cd)), 1, 2)))
            WHEN REGEXP_CONTAINS(UPPER(TRIM(clm.prcdr_cd)), r'^D\d{4}$') 
              THEN CONCAT('prcdr_group_', LOWER(SUBSTR(UPPER(TRIM(clm.prcdr_cd)), 1, 3)))
            WHEN REGEXP_CONTAINS(UPPER(TRIM(clm.prcdr_cd)), r'^\d{1,4}$')
              THEN CONCAT('prcdr_group_', SUBSTR(UPPER(TRIM(clm.prcdr_cd)), 1, LEAST(2, LENGTH(UPPER(TRIM(clm.prcdr_cd))))))
            WHEN REGEXP_CONTAINS(UPPER(TRIM(clm.prcdr_cd)), r'^\d{6,}$')
              THEN CONCAT('prcdr_group_', SUBSTR(UPPER(TRIM(clm.prcdr_cd)), 1, 4))
            WHEN UPPER(TRIM(clm.prcdr_cd)) IS NOT NULL AND UPPER(TRIM(clm.prcdr_cd)) != ''
              THEN 'prcdr_group_unk'
            ELSE NULL
          END,
          -- ICD procedure group (fallback if CPT is null)
          CASE
            WHEN REGEXP_CONTAINS(UPPER(TRIM(icd_prc.icd9_prcdr_cd)), r'^\d{5}$') 
              THEN CONCAT('prcdr_group_', SUBSTR(UPPER(TRIM(icd_prc.icd9_prcdr_cd)), 1, 3))
            WHEN REGEXP_CONTAINS(UPPER(TRIM(icd_prc.icd9_prcdr_cd)), r'^\d{4}[A-Z]$')
              THEN CONCAT('prcdr_group_', SUBSTR(UPPER(TRIM(icd_prc.icd9_prcdr_cd)), 1, 4))
            WHEN REGEXP_CONTAINS(UPPER(TRIM(icd_prc.icd9_prcdr_cd)), r'^[0-9A-Z]{6}$')
              THEN CONCAT('prcdr_group_', LOWER(SUBSTR(UPPER(TRIM(icd_prc.icd9_prcdr_cd)), 1, 3)))
            WHEN REGEXP_CONTAINS(UPPER(TRIM(icd_prc.icd9_prcdr_cd)), r'^\d[A-Z0-9]{6}$') 
              THEN CONCAT('prcdr_group_', LOWER(SUBSTR(UPPER(TRIM(icd_prc.icd9_prcdr_cd)), 1, 3)))
            WHEN REGEXP_CONTAINS(UPPER(TRIM(icd_prc.icd9_prcdr_cd)), r'^\d+\.\d+$') 
              THEN CONCAT('prcdr_group_', SPLIT(UPPER(TRIM(icd_prc.icd9_prcdr_cd)), '.')[SAFE_OFFSET(0)])
            WHEN REGEXP_CONTAINS(UPPER(TRIM(icd_prc.icd9_prcdr_cd)), r'^[A-Z]\d{4}$') AND LEFT(UPPER(TRIM(icd_prc.icd9_prcdr_cd)), 1) != 'D'
              THEN CONCAT('prcdr_group_', LOWER(SUBSTR(UPPER(TRIM(icd_prc.icd9_prcdr_cd)), 1, 2)))
            WHEN REGEXP_CONTAINS(UPPER(TRIM(icd_prc.icd9_prcdr_cd)), r'^D\d{4}$') 
              THEN CONCAT('prcdr_group_', LOWER(SUBSTR(UPPER(TRIM(icd_prc.icd9_prcdr_cd)), 1, 3)))
            WHEN REGEXP_CONTAINS(UPPER(TRIM(icd_prc.icd9_prcdr_cd)), r'^\d{1,4}$')
              THEN CONCAT('prcdr_group_', SUBSTR(UPPER(TRIM(icd_prc.icd9_prcdr_cd)), 1, LEAST(2, LENGTH(UPPER(TRIM(icd_prc.icd9_prcdr_cd))))))
            WHEN REGEXP_CONTAINS(UPPER(TRIM(icd_prc.icd9_prcdr_cd)), r'^\d{6,}$')
              THEN CONCAT('prcdr_group_', SUBSTR(UPPER(TRIM(icd_prc.icd9_prcdr_cd)), 1, 4))
            WHEN UPPER(TRIM(icd_prc.icd9_prcdr_cd)) IS NOT NULL AND UPPER(TRIM(icd_prc.icd9_prcdr_cd)) != ''
              THEN 'prcdr_group_unk'
            ELSE NULL
          END
        ) AS prcdr_group_cd,  -- Algorithmically generated procedure group code
        
        -- DRG code with validation to exclude garbage values
        -- ✅ UPDATED: Only keep numeric DRG codes AND strip leading zeros for consistency
        -- Examples: "0885" → 885 → "885", "001" → 1 → "1", "885" → "885" (unchanged)
        CASE 
            WHEN TRIM(drg.drg_cd) = '' OR drg.drg_cd IS NULL THEN NULL
            WHEN REGEXP_CONTAINS(CAST(drg.drg_cd AS STRING), r'^[0-9]+$') 
                THEN CAST(CAST(drg.drg_cd AS INT64) AS STRING)  -- ✅ Strip leading zeros for standardization
            ELSE NULL  -- ❌ Invalid: contains special characters (*, -, ., /, #, etc.)
        END AS drg_cd
        
    from base
    
    -- Join deduplicated demographics
    join b_dt
    	on base.individual_id = b_dt.individual_id
    
    -- Join claims data (current, not archives)
    inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_CLAIM_LINE` clm
        on base.member_id = clm.member_id
    
    -- Join ICD procedure codes (optional - may not exist for all claims)
    left join `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_PRCD` icd_prc
        on clm.claim_line_id = icd_prc.claim_line_id 
        and clm.member_id = icd_prc.member_id
    
    -- Join DRG codes (Diagnosis Related Group for inpatient stays)
    left join `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_DRG_TYPE` drg
        on clm.claim_line_id = drg.claim_line_id and clm.member_id = drg.member_id
    
    -- Join NPPES NPI Dataset - Provider Taxonomy Codes
    -- Needed for: healthcare_provider_taxonomy_code_1 (primary taxonomy classification)
    -- Source: Public BigQuery dataset with National Provider Identifier (NPI) information
    left join
        (SELECT npi, healthcare_provider_taxonomy_code_1 
         FROM `bigquery-public-data.nppes.npi_raw`) AS nppes
        on CAST(clm.srv_prvdr_npi_nbr AS STRING) = nppes.npi
    
    -- ============================================================================
    -- NOTE: Procedure groups are now generated ALGORITHMICALLY in the SELECT
    -- No BASE_PROCEDURE lookup table needed - see prcdr_group_cd column above
    -- ============================================================================
    
    where 
        -- DATE FILTERS: Dynamic 36-month lookback per member
        -- 🔧 UPDATED: Removed hardcoded 2020-01-01 to use fully dynamic lookback
    	clm.srv_start_dt > DATE_SUB(base.index_dt, INTERVAL 36 MONTH)  -- Start: 36 months before index_dt
	and clm.srv_start_dt < DATE_SUB(base.index_dt, INTERVAL 90 DAY)    -- End: 90 days before index (claims lag)
        
        -- PAID/ADJUDICATION DATE FILTER: Ensures claim was processed before 90-day lag
        -- If paid_dt is valid (after 1900-01-01), use it; otherwise use adjudication date
        and IF(clm.paid_dt > DATE("1900-01-01"), clm.paid_dt, clm.adjn_dt) < DATE_SUB(base.index_dt, INTERVAL 90 DAY)
        
        -- QUALITY FILTERS:
        and clm.duplicate_ind = 'N'           -- Exclude duplicate claims
	and clm.summarized_srv_ind = 'Y'      -- Only summarized services (not individual line items)
	and clm.file_id <> 'C4'               -- Exclude C4 file type (⚠️ TODO: document what C4 is)
	and clm.reversal_cd <> 'R'            -- Exclude reversed claims
;

/*==============================================================================
  STEP 2: EXTRACT PRESCRIPTIONS
  
  Purpose: Extract pharmacy claims with GPI codes (first 4 digits)
  Lookback: 36 months (2020-2023), NO 90-day lag (pharmacy processes faster)
  
  Output: member_monthly_rx
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_monthly_rx`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_monthly_rx`
    CLUSTER BY member_id
OPTIONS (labels=[("owner", "pritha_ghosh_cvshealth_com"),("costcenter","13070")],
    expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)) AS
with base as (
    -- Step 2a: Join base membership with demographics
    -- Handles cases where 1 individual_id maps to multiple member_ids over time
    SELECT rm.individual_id,
           rm.member_id,
           rm.index_dt,
           m.gender_cd,      -- Patient gender
           m.birth_dt,       -- Date of birth for age calculation
           -- Priority order: most recent coverage effective date, then HMO conversion date, then highest member_id
           row_number() over (partition by rm.individual_id 
                              order by orig_covg_eff_dt desc, hmo_to_trad_conv_dt desc, rm.member_id desc) as ord
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_memberid` rm
    JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEMBER` m
        ON rm.member_id = m.member_id
)
, b_dt as (
	-- Step 2b: Deduplicate demographics - ensure 1 individual_id has only 1 gender_cd and 1 birth_dt
	-- Takes the most recent member_id's demographics (ord = 1)
	select *
	from base
	where ord = 1
)
, rx_combined as (
    -- Step 2c: Combine all 3 RX data sources
    -- Source 1: RX_CLAIM_DTL (primary pharmacy claims)
    SELECT member_id, disp_dt, process_dt, adjudicated_gpi_cd, file_id
    FROM `edp-prod-hcbstorage.edp_hcb_core_cnsv.RX_CLAIM_DTL`
    
    UNION ALL
    
    -- Source 2: XTRNL_RX_CLAIM (external pharmacy claims)
    SELECT member_id, disp_dt, process_dt, adjudicated_gpi_cd, file_id
    FROM `edp-prod-hcbstorage.edp_hcb_core_cnsv.XTRNL_RX_CLAIM`
    
    UNION ALL
    
    -- Source 3: RX_CLAIM_DTL_UNFLTRD (unfiltered pharmacy claims)
    SELECT member_id, disp_dt, process_dt, adjudicated_gpi_cd, file_id
    FROM `edp-prod-hcbstorage.edp_hcb_hcbdw_rxclmrestrict_v.RX_CLAIM_DTL_UNFLTRD`
)
-- Step 2d: Extract prescriptions with demographics and GPI code transformation
select
        -- PATIENT IDENTIFIERS
base.individual_id, 
        base.index_dt,
base.member_id,
        
        -- GENDER: Encoded as 0=Female, 1=Male, 2=Unknown/Other
        -- Uses deduplicated demographics from b_dt CTE
        case when trim(b_dt.gender_cd) = 'M' then 1
             when trim(b_dt.gender_cd) = 'F' then 0
             else 2 end as gender_cd,
        
        -- AGE IN MONTHS: Age at dispense date --DOUBLE CHECK THIS -- eg birthday is 01/31 and dispense date is 02/01, the months should be accurately calculated = 0
        -- Using standard BigQuery DATE_DIFF
        DATE_DIFF(CAST(rx.disp_dt AS DATE), CAST(b_dt.birth_dt AS DATE), MONTH) as age_in_months,
        
        -- DISPENSE DATE (used for day-level aggregation)
        rx.disp_dt as dt,
        
        -- GPI CODE: First 4 digits of Generic Product Identifier
        -- Format: 'gpi1234' where 1234 is the therapeutic class
        -- ✅ UPDATED: Validate that source has at least 4 digits before creating gpi4
        CASE
            WHEN TRIM(rx.adjudicated_gpi_cd) IS NULL OR TRIM(rx.adjudicated_gpi_cd) = '' THEN NULL
            WHEN LENGTH(TRIM(rx.adjudicated_gpi_cd)) >= 4 
                AND REGEXP_CONTAINS(SUBSTR(TRIM(rx.adjudicated_gpi_cd), 1, 4), r'^[0-9]{4}$')
                THEN CONCAT('gpi', SUBSTR(TRIM(rx.adjudicated_gpi_cd), 1, 4))  -- ✅ Valid: at least 4 digits
            ELSE NULL  -- ❌ Invalid: too short or non-numeric
        END AS gpi4

from base

    -- Join deduplicated demographics
    join b_dt
        on base.individual_id = b_dt.individual_id
    
    -- Join combined pharmacy claims data from all 3 sources
    inner join rx_combined rx
        on base.member_id = rx.member_id

where 
    -- DATE FILTERS: Dynamic 36-month lookback per member
    -- 🔧 UPDATED: Removed hardcoded 2020-01-01 to use fully dynamic lookback
    rx.disp_dt > DATE_SUB(base.index_dt, INTERVAL 36 MONTH)  -- Start: 36 months before index_dt
    and rx.disp_dt <= base.index_dt  -- End: index_dt (⚠️ NO 90-DAY LAG for pharmacy, unlike medical claims)
    and rx.process_dt <= base.index_dt  -- Prescription processed before index date
    
    -- QUALITY FILTERS:
    and rx.file_id <> 'C5'  -- Exclude C5 file type (⚠️ TODO: document what C5 is)
;


/*==============================================================================
  STEP 3: PREPARE CLAIMS FOR MAPPING
  
  Purpose: Intermediate table with procedures (diagnoses added in Step 5)
  Output: member_d1a_train_ending_tmp
  
  Data Flow:
  - Input: a834793_Commercial_member_monthly_claims (from Step 1)
  - Output: a834793_Commercial_member_d1a_train_ending_tmp (intermediate - medical claims)
  
  Fields Included:
  - Patient identifiers: individual_id, member_id, index_dt
  - Claim identifiers: claim_line_id
  - Temporal: dt (service date)
  - Demographics: gender_cd, age_in_months
  - Claim attributes: days_cnt (length of stay)
  - Procedure codes:
    * revenue_cd: Revenue code
    * hcfa_plc_srv_cd: Place of service code
    * provider_taxonomy_cd: Provider taxonomy code (from NPPES - replaces src_specialty_cd)
    * prcdr_cd: CPT/HCPCS procedure code
    * icd9_prcdr_cd: ICD-10-PCS procedure code (legacy column name)
    * drg_cd: DRG code (Diagnosis Related Group)
  
  Fields NOT Included:
  - Diagnosis codes (will be joined separately from diagnosis table)
  
  Next Steps: 
  - Step 4: Create similar intermediate table for prescription data
  - Later: Join with diagnosis codes and map all codes to indices using w2ind lookup table
  
==============================================================================*/

-- Step 3: Create intermediate claims table with procedure codes only
-- Pattern: DROP TABLE IF EXISTS + CREATE TABLE (safer than CREATE OR REPLACE)
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1a_train_ending_tmp`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1a_train_ending_tmp`
    CLUSTER BY member_id
OPTIONS (labels=[("owner", "pritha_ghosh_cvshealth_com"),("costcenter","13070")],
    expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)) AS
select 
        -- PATIENT IDENTIFIERS
        individual_id,       -- Unique patient identifier
        member_id,           -- Member ID for joining with other tables
        index_dt,            -- Reference date for lookback window
        
        -- CLAIM IDENTIFIERS
        claim_line_id,       -- Unique identifier for each claim line
        
        -- TEMPORAL DATA
        dt,                  -- Service date (for day-level aggregation)
        
        -- CLAIM ATTRIBUTES
        days_cnt,            -- Length of stay (99=null/negative, 11=capped at 10+)
        
        -- DEMOGRAPHICS
        gender_cd,           -- Gender (0=Female, 1=Male, 2=Unknown)
        age_in_months,       -- Age at service date in months
        
        -- PROCEDURE CODES (will be mapped to indices in next step)
        revenue_cd,          -- Revenue code
        hcfa_plc_srv_cd,     -- HCFA place of service code
        -- ⚠️ COMMENTED OUT: Replaced by provider_taxonomy_cd from NPPES
        -- src_specialty_cd,    -- Provider specialty code
        provider_taxonomy_cd, -- Provider taxonomy code (from NPPES)
        prcdr_cd,            -- CPT/HCPCS procedure code
        icd9_prcdr_cd,       -- ICD-9 procedure code
        prcdr_group_cd,      -- NEW! Algorithmic procedure group for CPT+ICD (for target vocab)
        drg_cd               -- DRG code

from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_monthly_claims`
;


/*==============================================================================
  STEP 4: PREPARE PRESCRIPTIONS FOR MAPPING
  
  Purpose: Intermediate table with medications (parallel to Step 3)
  Output: member_d1c_train_ending_tmp
  
  Data Flow:
  - Input: a834793_Commercial_member_monthly_rx (from Step 2)
  - Output: a834793_Commercial_member_d1c_train_ending_tmp (intermediate - prescriptions)
  
  Fields Included (all fields from Step 2):
  - Patient identifiers: individual_id, member_id, index_dt
  - Temporal: dt (dispense date)
  - Demographics: gender_cd, age_in_months
  - Medication codes:
    * gpi4: First 4 digits of GPI code (Generic Product Identifier)
  
  Key Differences from Step 3 (Medical Claims):
  - Clustered by individual_id (vs member_id for claims)
  - Contains GPI medication codes (vs procedure codes for claims)
  - No claim_line_id (prescriptions identified differently)
  - No days_cnt or place of service (not applicable to pharmacy)
  
  Next Steps:
  - Later: Map GPI codes to indices using w2ind lookup table
  - Eventually: Combine with medical claim data for transformer input
  
==============================================================================*/

-- Step 4: Create intermediate prescription table with GPI codes
-- Pattern: DROP TABLE IF EXISTS + CREATE TABLE (safer than CREATE OR REPLACE)
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1c_train_ending_tmp`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1c_train_ending_tmp`
    CLUSTER BY individual_id  -- Note: Clustered by individual_id (different from Step 3)
    OPTIONS (labels=[("owner", "pritha_ghosh_cvshealth_com"),("costcenter","13070")],
    expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)) AS
select 
        -- All prescription fields from Step 2
        -- No field-by-field selection needed - all fields are relevant for code mapping
        * 
from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_monthly_rx`
;


/*==============================================================================
  STEP 5: ADD DIAGNOSIS CODES
  
  Purpose: Join up to 3 diagnoses per claim, standardize ICD format (XXX.XX)
  Output: member_monthly_clm_ln
  
  Data Flow:
  - Input 1: a834793_Commercial_member_monthly_claims (from Step 1)
  - Input 2: CLM_LN_X_ICD9_DX (diagnosis code table)
  - Output: a834793_Commercial_member_monthly_clm_ln (claims + diagnoses)
  
  ICD-9 Code Standardization:
  - Original format varies: "250.00", "250", "250.0", "250.001"
  - Standardized format: "XXX.XX" (left part + '.' + first 2 chars of right part)
  - Example: "250.001" → "250.00", "250" → "250" (no decimal if no right part)
  - Uppercased for consistency
  
  Key Logic:
  1. Only first 3 diagnosis codes per claim (sequence_id < 4)
     - Commercial data typically has multiple diagnosis codes
     - First 3 are usually most clinically relevant
  2. UNNEST expands array: 1 claim with 3 diagnoses → 3 rows
     - Each diagnosis becomes its own record
     - This allows easier aggregation by diagnosis code later
  3. String manipulation:
     - Split on '.' to get left and right parts
     - Take first 2 characters of decimal portion
     - Reconstruct as "left.right"
  
  Fields Output:
  - From base claims: individual_id, index_dt, member_id, claim_line_id, dt, 
                      days_cnt, gender_cd, age_in_months
  - NEW: icd9_dx_cd (standardized diagnosis code)
  
  Note: This table has MORE ROWS than the input because of UNNEST
  - Input: 1 row per claim line
  - Output: Multiple rows per claim line (one for each diagnosis code)
  
  Next Steps:
  - Map diagnosis codes to indices using w2ind lookup table
  - Aggregate by date for transformer input
  
==============================================================================*/

-- Step 5: Join diagnosis codes to medical claims and standardize ICD-9 format
-- Pattern: DROP TABLE IF EXISTS + CREATE TABLE (safer than CREATE OR REPLACE)
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_monthly_clm_ln`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_monthly_clm_ln`
    CLUSTER BY individual_id
 OPTIONS (labels=[("owner", "pritha_ghosh_cvshealth_com"),("costcenter","13070")],
    expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)) AS
    select distinct
        -- PATIENT & CLAIM IDENTIFIERS (from base claims table)
base.individual_id, 
        base.index_dt,
base.member_id,
base.claim_line_id,
        
        -- TEMPORAL DATA
        base.dt,  -- Service date
        
        -- CLAIM ATTRIBUTES
        base.days_cnt,  -- Length of stay
        
        -- DEMOGRAPHICS
        base.gender_cd,
        base.age_in_months,
        
        -- DIAGNOSIS CODE (NEW - from diagnosis table)
        -- ⚠️ NOTE: Column named icd9_dx_cd but contains ICD-10 codes
        -- ✅ UPDATED: Only keep valid ICD-10 format (filters out legacy ICD-9, lowercase, etc.)
        -- Valid ICD-10: Letter + 2 alphanumeric + optional decimal/alphanumeric (e.g., I10, E11.9, Z3A.39)
        CASE
            WHEN ARRAY_TO_STRING([var_st.l,var_st.r],'.') = '' OR ARRAY_TO_STRING([var_st.l,var_st.r],'.') IS NULL THEN NULL
            WHEN REGEXP_CONTAINS(UPPER(ARRAY_TO_STRING([var_st.l,var_st.r],'.')), r'^[A-Z][0-9A-Z]{2}[\.\w]*$')
                THEN UPPER(ARRAY_TO_STRING([var_st.l,var_st.r],'.'))  -- ✅ Valid ICD-10
            ELSE NULL  -- ❌ Invalid: legacy ICD-9 (250.00), lowercase (f43.22), etc.
        END AS icd9_dx_cd
             
    -- Base medical claims table (from Step 1)
    from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_monthly_claims` base
    
    -- Join with diagnosis codes (produces multiple rows per claim if multiple diagnoses)
    inner join
        ( 
          -- Subquery: Extract and parse diagnosis codes
select 
member_id,
claim_line_id,
              -- Create array structure: split ICD-9 code on '.' and truncate decimal to 2 digits
              -- ARRAY[STRUCT(...)] creates a single-element array that will be UNNESTed
              ARRAY[STRUCT(
                    split(trim(icd9_dx_cd),'.')[safe_offset(0)] as l,  -- Left part (before decimal)
                    substr(split(trim(icd9_dx_cd),'.')[safe_offset(1)],1,2) as r  -- Right part (first 2 chars after decimal)
              )] AS icd_array
          from `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_DX`
          where 
              cast(sequence_id as int) < 4  -- Only first 3 diagnosis codes (position 0, 1, 2)
              and icd9_dx_cd is not NULL    -- Exclude null codes
              and trim(icd9_dx_cd)!=""      -- Exclude empty codes
        ) b
        on base.member_id = b.member_id 
        and base.claim_line_id = b.claim_line_id,  -- Join on claim identifier
    
    -- UNNEST: Expand the array so each diagnosis code becomes its own row
    UNNEST(icd_array) as var_st
;


/*==============================================================================
  STEP 6: PREPARE DIAGNOSES FOR MAPPING
  
  Purpose: Intermediate table with diagnoses (parallel to Steps 3-4)
  Output: member_d1b_train_ending_tmp
  
  Data Flow:
  - Input: a834793_Commercial_member_monthly_clm_ln (from Step 5 - claims with diagnoses)
  - Output: a834793_Commercial_member_d1b_train_ending_tmp (intermediate - diagnoses)
  
  Fields Included (all fields from Step 5):
  - Patient identifiers: individual_id, member_id, index_dt
  - Claim identifiers: claim_line_id
  - Temporal: dt (service date)
  - Demographics: gender_cd, age_in_months
  - Claim attributes: days_cnt (length of stay)
  - Diagnosis codes: icd9_dx_cd (standardized format)
  
  Relationship to Other Intermediate Tables:
  - Step 3 → d1a_train_ending_tmp: Medical claims with PROCEDURES only
  - Step 4 → d1c_train_ending_tmp: Prescriptions with GPI codes
  - Step 6 → d1b_train_ending_tmp: Medical claims with DIAGNOSES only ← YOU ARE HERE
  
  Why Separate Tables for Procedures vs Diagnoses?
  - Procedures and diagnoses are stored differently in source data
  - Allows independent processing and code mapping
  - Later steps will merge all three streams (procedures, diagnoses, medications)
  - Easier debugging and data quality checks
  
  Note: This table has the SAME row count as Step 5 output
  - Already expanded by UNNEST in Step 5
  - Each diagnosis code is already its own row
  
  Next Steps:
  - Map diagnosis codes (icd9_dx_cd) to indices using w2ind lookup table
  - Eventually merge with procedure codes (d1a) and medications (d1c)
  - Aggregate by date for transformer input
  
==============================================================================*/

-- Step 6: Create intermediate table for claims with diagnosis codes
-- Pattern: DROP TABLE IF EXISTS + CREATE TABLE (safer than CREATE OR REPLACE)
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1b_train_ending_tmp`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1b_train_ending_tmp`
    CLUSTER BY individual_id
 OPTIONS (labels=[("owner", "pritha_ghosh_cvshealth_com"),("costcenter","13070")],
    expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)) AS
 select 
        -- All fields from Step 5 (claims + diagnoses)
        -- No field-by-field selection needed - all fields are relevant for code mapping
        * 
 from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_monthly_clm_ln`
;


/*==============================================================================
  STEP 7: CREATE ROOT CALENDAR
  
  Purpose: Merge claims + prescriptions into patient-date calendar
  Output: member_root (master timeline of all healthcare events)
  
  Data Flow:
  - Input 1: a834793_Commercial_member_d1a_train_ending_tmp (procedures - Step 3)
  - Input 2: a834793_Commercial_member_d1c_train_ending_tmp (prescriptions - Step 4)
  - Output: a834793_Commercial_member_root (unique patient-date records)
  
  Age Standardization Logic:
  - Medical claims (d1a): Cap at 1439 months (119 years, 11 months)
  - Prescriptions (d1c): Cap at 1440 months (120 years exactly)
  - ⚠️ Note: Slight inconsistency between streams (1439 vs 1440)
  - Negative ages → 0 (data quality issue handling)
  
  Deduplication Logic:
  1. root0 CTE: Union medical and prescription dates with age standardization
  2. root1 CTE: Add row_number() to identify duplicates within same individual-date
  3. Main query: Keep only first record (seqno = 1) for each unique individual-date
  
  Result: ONE row per patient per date, regardless of how many events happened that day
  - If patient had 5 claims + 2 prescriptions on same date → 1 row
  - Demographics (gender, age) captured once per date
  - Actual codes will be joined/aggregated in next step
  
  Note: Diagnosis codes (d1b) are NOT included in this union
  - Diagnoses are linked to claims via claim_line_id
  - They'll be joined later during code mapping step
  
  Next Steps:
  - Map all medical codes to indices using w2ind lookup table
  - Join root table with mapped codes
  - Aggregate codes by date
  
==============================================================================*/

-- Step 7: Create root patient-date table by merging medical claims and prescriptions
-- Pattern: DROP TABLE IF EXISTS + CREATE TABLE (safer than CREATE OR REPLACE)
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_root`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_root`
    CLUSTER BY individual_id
     OPTIONS (labels=[("owner", "pritha_ghosh_cvshealth_com"),("costcenter","13070")],
    expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)) AS
with root0 as (
    -- Union Step 1: Medical claims (procedures) with age capping
    select
        individual_id,
        index_dt,
member_id,
        dt,  -- Service date
gender_cd,
        -- Age standardization for medical claims: cap at 1439 months (119.9 years)
        case
            when age_in_months < 0 then 0      -- Handle negative ages (data quality issue)
            when age_in_months > 1439 then 1439  -- Cap at 1439 months
            else age_in_months 
        end as age_in_months
     from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1a_train_ending_tmp`
     
    union all  -- Combines medical and prescription dates, deduplication handled in root1
    
    -- Union Step 2: Prescriptions with age capping
    select
        individual_id,
        index_dt,
member_id,
        dt,  -- Dispense date
gender_cd,
        -- Age standardization for prescriptions: cap at 1439 months (119.9 years)
        case
            when age_in_months < 0 then 0      -- Handle negative ages
            when age_in_months > 1439 then 1439  -- Cap at 1439 months
            else age_in_months 
        end as age_in_months
    from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1c_train_ending_tmp`
),

root1 as (
    -- Deduplication: Add sequence number to identify duplicates within same patient-date
    select
        r0.*,
        row_number() over (partition by r0.individual_id, r0.index_dt, r0.dt 
                          order by r0.member_id, r0.gender_cd) as seqno  -- Deterministic ordering
     from root0 r0
)
    -- Final selection: Keep only first record for each unique patient-date combination
    select DISTINCT
        individual_id,
        index_dt,
        member_id,
        dt,           -- Unique date per patient
        gender_cd,
        age_in_months
     from root1
     where seqno = 1  -- Deduplication: one row per patient per date
;


/*==============================================================================
  STEP 8: MAP CODES TO INPUT INDICES
  
  Purpose: Map all medical codes to w2ind (~84k vocab) for transformer input
  
  9 Code Types Mapped:
  1. days_cnt, 2. place of service, 3. provider taxonomy
  4. ICD diagnoses, 5. revenue codes, 6. CPT procedures
  7. ICD procedures, 8. DRG codes, 9. GPI medications
  
  Lookup: a834793_Commercial_member_w2ind
  Unknown codes → index 0
  
  Output: member_get_cd (patient-date-index)
  
  Special Case: prcdr_cd (Procedures)
  - Uses UNION DISTINCT to combine CPT codes and ICD-9 procedure codes
  - Both map to 'prcdr_cd' prefix in w2ind lookup
  - Deduplicates if same code appears as both CPT and ICD-9
  
  Output Structure:
  - individual_id, index_dt, member_id, dt: Patient and date identifiers
  - cd: Original code (with prefix)
  - ind: Numerical index from w2ind lookup
  
  Result: Tall table with one row per patient-date-code combination
  - If a date has 10 codes → 10 rows
  - Next step will aggregate these into comma-separated strings
  
==============================================================================*/

-- Step 8: Map all medical codes to numerical indices using w2ind lookup
-- ✅ UPDATED: Airflow variables replaced with actual table names
-- ✅ FIXED: Table name inconsistencies (_score_ending_tmp → _train_ending_tmp)
-- ✅ IMPORTANT: Using LOCAL w2ind copy instead of shared production table
--    Local table: edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_w2ind
--    This ensures version consistency and faster joins
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_get_cd`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_get_cd`
 OPTIONS (labels=[("owner", "pritha_ghosh_cvshealth_com"),("costcenter","13070")],
    expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)) AS
      -- Code Type 1: days_cnt (Length of Stay)
      -- Maps: 0-11, 99 (missing) → indices
      -- Example: days_cnt5 → index from w2ind
      select distinct
            base.individual_id,
            base.index_dt,
            base.member_id,
            base.dt,
            w2ind.cd,  -- Original code with prefix (e.g., "days_cnt5")
            case when w2ind.ind is null then 0 else w2ind.ind end as ind  -- Index 0 if not found
      from
          `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1a_train_ending_tmp` base
            -- ✅ Using LOCAL w2ind copy (not shared prod table)
            left join (select ind, cd from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind`) w2ind
            on concat('days_cnt', cast (days_cnt as string)) = w2ind.cd  -- Concat prefix + value
      where days_cnt is not null
      
    union all
    
      -- Code Type 2: hcfa_plc_srv_cd (Place of Service)
      -- Maps: 11=Office, 21=Inpatient Hospital, 23=ER, etc. → indices
      -- Example: hcfa_plc_srv_cd21 → index from w2ind
      select distinct
          base.individual_id,
          base.index_dt,
          base.member_id,
          base.dt,
          w2ind.cd,
          case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from
          `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1a_train_ending_tmp` base
          left join (select ind, cd from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind`) w2ind
            on concat('hcfa_plc_srv_cd', cast (base.hcfa_plc_srv_cd as string)) = w2ind.cd
      where base.hcfa_plc_srv_cd is not null
      
    -- ⚠️ COMMENTED OUT: Provider specialty replaced by provider_taxonomy_cd from NPPES
    -- union all
    --   -- Code Type 3: src_specialty_cd (Provider Specialty)
    --   -- Maps: Provider specialties (cardiology, oncology, etc.) → indices
    --   -- Example: src_specialty_cd08 → index from w2ind
    --   select distinct
    --       base.individual_id,
    --       base.index_dt,
    --       base.member_id,
    --       base.dt,
    --       w2ind.cd,
    --       case when w2ind.ind is null then 0 else w2ind.ind end as ind
    --   from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1a_train_ending_tmp` base
    --            left join (select ind, cd from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_w2ind`) w2ind
    --         on concat('src_specialty_cd', cast (base.src_specialty_cd as string)) = w2ind.cd
    --   where base.src_specialty_cd is not null
      
    union all
    
      -- Code Type 3: provider_taxonomy_cd (Provider Taxonomy from NPPES - REPLACES provider specialty)
      -- Maps: NPPES taxonomy codes → indices
      -- Example: provider_taxonomy_cd207Q00000X → index from w2ind
      -- ⚠️ REQUIREMENT: w2ind table must contain provider taxonomy mappings
      select distinct
          base.individual_id,
          base.index_dt,
          base.member_id,
          base.dt,
          w2ind.cd,
          case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1a_train_ending_tmp` base
               left join (select ind, cd from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind`) w2ind
            on concat('provider_taxonomy_cd', cast (base.provider_taxonomy_cd as string)) = w2ind.cd
      where base.provider_taxonomy_cd is not null
      
    union all
    
      -- Code Type 4: icd9_dx_cd (Diagnosis Codes) 🔥 FROM d1b TABLE
      -- Maps: ICD-9 diagnosis codes (250.00=diabetes, 401.9=hypertension, etc.) → indices
      -- Example: icd9_dx_cd250.00 → index from w2ind
      -- ⚠️ NOTE: This is the ONLY union that uses d1b (diagnosis table)
      select distinct
          base.individual_id,
          base.index_dt,
          base.member_id,
          base.dt,
          w2ind.cd,
          case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from
          `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1b_train_ending_tmp` base
              left join (select ind, cd from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind`) w2ind
            on concat('icd9_dx_cd', cast (base.icd9_dx_cd as string)) = w2ind.cd
      where base.icd9_dx_cd is not null
      
    union all
    
      -- Code Type 5: revenue_cd (Revenue Codes)
      -- Maps: Hospital revenue codes (billing categories) → indices
      -- Example: revenue_cd0450 → index from w2ind
      select distinct
          base.individual_id,
          base.index_dt,
          base.member_id,
          base.dt,
          w2ind.cd,
          case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from
          `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1a_train_ending_tmp` base
              left join (select ind, cd from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind`) w2ind
            on concat('revenue_cd', cast (base.revenue_cd as string)) = w2ind.cd
      where base.revenue_cd is not null
      
    union all
    
      -- Code Types 6 & 7: prcdr_cd (Procedure Codes - TWO TYPES COMBINED)
      -- ⚠️ SPECIAL CASE: Uses UNION DISTINCT to combine CPT and ICD-9 procedure codes
      -- Both are mapped with 'prcdr_cd' prefix in w2ind lookup
        (
          -- Code Type 6a: CPT/HCPCS Procedure Codes
          -- Maps: CPT codes (99213=office visit, 45378=colonoscopy, etc.) → indices
          -- Example: prcdr_cd99213 → index from w2ind
          select distinct
              base.individual_id,
              base.index_dt,
              base.member_id,
              base.dt,
              w2ind.cd,
              case when w2ind.ind is null then 0 else w2ind.ind end as ind
          from
              `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1a_train_ending_tmp` base
                  left join (select ind, cd from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind`) w2ind
                on concat('prcdr_cd', cast (base.prcdr_cd as string)) = w2ind.cd
          where base.prcdr_cd is not null
          
        union distinct  -- Deduplicates if same code appears as both CPT and ICD-9
        
          -- Code Type 6b: ICD-9 Procedure Codes
          -- Maps: ICD-9 procedure codes (45.23=colonoscopy, etc.) → indices
          -- Example: prcdr_cd45.23 → index from w2ind
          -- ⚠️ NOTE: Both CPT and ICD-9 procedures use 'prcdr_cd' prefix
          select distinct
              base.individual_id,
              base.index_dt,
              base.member_id,
              base.dt,
              w2ind.cd,
              case when w2ind.ind is null then 0 else w2ind.ind end as ind
          from
              `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1a_train_ending_tmp` base
                  left join (select ind, cd from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind`) w2ind
                on concat('prcdr_cd', cast (base.icd9_prcdr_cd as string)) = w2ind.cd
          where base.icd9_prcdr_cd is not null
        )
        
    union all
    
      -- Code Type 8: drg_cd (Diagnosis Related Group Codes)
      -- Maps: DRG codes for inpatient stays → indices
      -- Example: drg_cd470 → index from w2ind
      -- ⚠️ REQUIREMENT: w2ind table must contain DRG mappings
      -- Pattern in w2ind.cd column: 'drg_cd' + DRG_CODE (e.g., 'drg_cd470')
      select distinct
          base.individual_id,
          base.index_dt,
          base.member_id,
          base.dt,
          w2ind.cd,
          case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from
          `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1a_train_ending_tmp` base
              left join (select ind, cd from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind`) w2ind
            on concat('drg_cd', cast (base.drg_cd as string)) = w2ind.cd
      where base.drg_cd is not null
        
    union all
    
      -- Code Type 9: gpi4 (Generic Product Identifier - Medications) 🔥 FROM d1c TABLE
      -- Maps: First 4 digits of GPI codes (drug therapeutic classes) → indices
      -- Example: gpi1234 → index from w2ind
      -- ⚠️ NOTE: This is the ONLY union that uses d1c (prescription table)
      select distinct
          base.individual_id,
          base.index_dt,
          base.member_id,
          base.dt,
          w2ind.cd,
          case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from
          `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1c_train_ending_tmp` base
              left join (select ind, cd from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind`) w2ind
            on cast(base.gpi4 as string) = w2ind.cd  -- No concat needed, gpi4 already has 'gpi' prefix
      where base.gpi4 is not null
;


/*==============================================================================
  STEP 8b: MAP CODES TO TARGET INDICES 🎯 NEW!
  
  Purpose: Map grouped codes to target vocabulary for next-day prediction labels
  
  8 Target Code Types (~5k codes vs ~84k input codes):
  1. Place of Service (keep as-is)
  2. Procedure Groups - ALGORITHMIC (prcdr_group_cd: algorithmic groups like prcdr_group_992, prcdr_group_02h)
  3. ICD Diagnosis (first 3 digits: 250.00 → 250)
  4. GPI Medications (first 2 digits: gpi2210 → gpi22)
  5. Revenue Code (first 3 digits: 0250 → 025)
  6. DRG codes (keep as-is)
  7. Provider Taxonomy (first 4 chars: 207Q00000X → 207Q)
  8. Days Count (keep as-is: 0-11, 99)
  
  Note: Procedure groups use algorithmic prefixes - CPT (992, 335), ICD-10-PCS (02h, 0u5), 
        HCPCS (j1, a0), ICD-9 (66, 81), Dental (d72) - see prcdr_group_cd generation in Step 1
  
  Output: member_get_cd_target (patient-date-target_index)
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_get_cd_target`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_get_cd_target`
 OPTIONS (labels=[("owner", "pritha_ghosh_cvshealth_com"),("costcenter","13070")],
    expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)) AS
      
      -- Target Type 1: Place of Service (keep as-is)
      -- Maps: hcfa_plc_srv_cd → target index
      select 
          base.individual_id,
          base.index_dt,
          base.dt,
          case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from
          `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1a_train_ending_tmp` base
              left join (select ind, cd from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind_target`) w2ind
            on concat('hcfa_plc_srv_cd', cast (base.hcfa_plc_srv_cd as string)) = w2ind.cd
      where base.hcfa_plc_srv_cd is not null
      
      UNION ALL
      
      -- Target Type 2: Algorithmic Procedure Groups (CPT + ICD)
      -- Maps: prcdr_group_cd → target index (e.g., prcdr_group_992 = office visits, prcdr_group_02h = heart procedures)
      -- Handles both CPT/HCPCS and ICD procedure groups in single unified column with algorithmic grouping
      select 
          base.individual_id,
          base.index_dt,
          base.dt,
          case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from
          `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1a_train_ending_tmp` base
              left join (select ind, cd from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind_target`) w2ind
            on base.prcdr_group_cd = w2ind.cd  -- Direct string match (no concat needed - already formatted)
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
          `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1b_train_ending_tmp` base
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
          `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1c_train_ending_tmp` base
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
          `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1a_train_ending_tmp` base
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
          `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1a_train_ending_tmp` base
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
          `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1a_train_ending_tmp` base
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
          `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_d1a_train_ending_tmp` base
              left join (select ind, cd from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind_target`) w2ind
            on concat('days_cnt', cast(base.days_cnt as string)) = w2ind.cd
      where base.days_cnt is not null
;


/*==============================================================================
  STEP 9: AGGREGATE CODES BY DATE 🎯 UPDATED!
  
  Purpose: Aggregate codes into comma-separated strings per patient-date
  
  Two parallel aggregations:
  1. Input codes (cd): ~84k vocab, limit 80 codes/date for transformer input
  2. Target codes (target): ~3.5k vocab, no limit for training labels 🎯 NEW!
  
  Output: member_o1_train_ending_tmp (patient-date-cd-target-demographics)
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_o1_train_ending_tmp`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_o1_train_ending_tmp`
OPTIONS (labels=[("owner", "pritha_ghosh_cvshealth_com"),("costcenter","13070")],
    expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)) AS
with x1 as (
    -- Deduplicate: Remove duplicate codes for same patient-date
    -- If same code appears multiple times → keep 1
    select
        individual_id,
        index_dt,
        dt,
        ind  -- Code index from w2ind lookup
    from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_get_cd` x0
    group by individual_id,index_dt, dt, ind
    order by dt, ind
),

x2 as (
    -- Rank codes by index value (lowest index gets seqno=1)
    -- This allows us to prioritize lower-indexed codes if we need to truncate
    select
        *,
        row_number() over (partition by individual_id,index_dt,dt order by ind) as seqno
    from x1
),

x3 as (
    -- Aggregate codes into array, limit to first 80 codes per date
    -- ⚠️ IMPORTANT: 80-code limit prevents excessively long sequences 
    --IN THE PAST 80 WAS CHOSEN AS A LIMIT. WE CAN REVIEW IT AND SEE IF WE SHOULD INCREASE IT.
    --MAKE THE ORDER OF THE CODES REPEATABLE.
    select
        individual_id ,
        index_dt,
        dt,
        ARRAY_AGG(cast(ind as string) order by ind) as cd_arr  -- Aggregate indices into array
    from x2
    where seqno<=80  -- Keep only first 80 codes (by lowest index value)
    group by individual_id, index_dt,dt
),

-- ============================================================================
-- 🎯 NEW: TARGET CODE AGGREGATION (y1-y2 CTEs)
-- ============================================================================

y1 as (
    -- Deduplicate TARGET codes (same patient-date-target → 1 row)
    -- Note: No ranking/limiting for targets - we keep ALL target codes
    select
        individual_id,
        index_dt,
        dt,
        ind  -- Target code index from w2ind_target lookup
    from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_get_cd_target` y0
    group by individual_id, index_dt, dt, ind
    order by dt, ind
),

y2 as (
    -- Aggregate TARGET codes into comma-separated string
    -- No 80-code limit! Targets are already grouped/aggregated codes
    -- Example: "45,67,89" (place of service, procedure group, diagnosis group)
    select
        individual_id,
        index_dt,
        dt,
        ARRAY_TO_STRING(ARRAY_AGG(cast(ind as string) order by ind), ',') as target
    from y1
    group by individual_id, index_dt, dt
)

-- Final join: Combine INPUT codes, TARGET codes, and demographics
select
  root2.individual_id,
  root2.index_dt,
  root2.dt,
  root2.gender_cd,       -- Demographics from root table
  root2.age_in_months,   -- Demographics from root table
  ARRAY_TO_STRING(x3.cd_arr, ',') as cd,  -- INPUT codes (detailed, 80 max)
  y2.target              -- 🎯 TARGET codes (grouped, no limit)
from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_root` root2
inner join x3
    on root2.individual_id = x3.individual_id 
    and root2.dt = x3.dt 
    and root2.index_dt=x3.index_dt
inner join y2  -- 🎯 Join TARGET codes
    on root2.individual_id = y2.individual_id
    and root2.dt = y2.dt
    and root2.index_dt = y2.index_dt
;


/*==============================================================================
  STEPS 10-12: CREATE FINAL TRANSFORMER INPUT SEQUENCES
  
  🎯 FINAL GOAL: Transform patient healthcare data into temporal sequences for transformer model
  
  Overview:
  These three steps convert daily healthcare events into asterisk-separated sequences:
  - Step 10: Filter to most recent 200 dates per patient
  - Step 11: Re-order dates chronologically (oldest → newest)
  - Step 12: Aggregate all dates into asterisk-separated sequences
  
  Why These Steps?
  - Clinical transformers require temporal sequences (not tabular data)
  - Sequences must be chronologically ordered (oldest first)
  - Limited to 200 dates to fit transformer input constraints
  - Asterisk delimiter separates different dates
  - Comma delimiter separates codes within same date
  
  Final Output Format (Example):
  One row per patient with 3 sequence fields:
  - gender_cd: "1*1*1*0*0" (gender at each date)
  - age_in_months: "540*541*542*543*544" (age at each date)
  - cd: "123,456*789,101*202,303*404,505*606" (codes for each date)
  
  This is EXACTLY the format the clinical transformer model expects!
  
==============================================================================*/


/*==============================================================================
  STEP 10: FILTER TO 200 MOST RECENT DATES
  
  Purpose: Keep 200 most recent dates per patient (transformer input limit)
  Output: member_o3_train_ending_tmp_1
  
  Logic:
  - Rank dates by recency (most recent = seqno 1)
  - Order: "order by dt desc" → newest dates first
  - Keep first 200 (seqno<=200)
  - Note: These are in REVERSE chronological order (will fix in Step 11)
  
  ✅ ✅ ✅ AIRFLOW VARIABLES - ALREADY REPLACED ✅ ✅ ✅
  - All table references now use actual names
  - Table naming fixed: _score_ending → _train_ending for consistency
  
==============================================================================*/

-- Step 10: Filter to 200 most recent dates per patient
-- ✅ UPDATED: Airflow variables replaced with actual table names
-- ✅ FIXED: Table name consistency (_score_ending → _train_ending)
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_o3_train_ending_tmp_1`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_o3_train_ending_tmp_1`
CLUSTER BY individual_id
OPTIONS (labels=[("owner", "pritha_ghosh_cvshealth_com"),("costcenter","13070")],
    expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)) AS
with x1 as (
    -- Rank dates by recency: most recent date = seqno 1
    select *,
        row_number() over (partition by individual_id,index_dt order by dt desc) as seqno
    from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_o1_train_ending_tmp`
)
    -- Keep only 200 most recent dates
    select * from x1 where seqno<=200 -- DOUBLE CHECK THIS. WE CAN REVIEW IT AND SEE IF WE SHOULD INCREASE IT.
;


/*==============================================================================
  STEP 11: APPLY NEXT-DAY PREDICTION SHIFT 🎯 CRITICAL!
  
  Purpose: Shift targets by 1 day for next-day prediction
  
  LEAD Logic: Join seqno with seqno+1 to get tomorrow's target
  - Input: codes from today (date N)
  - Target: codes from tomorrow (date N+1)
  - Creates supervised pairs: predict(tomorrow) given today
  - Re-rank oldest→newest (seqno2) for chronological sequence
  
  Example: seqno 1 gets target from seqno 2 (next day)
  
  Output: member_o3_train_ending_tmp_ordered
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_o3_train_ending_tmp_ordered`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_o3_train_ending_tmp_ordered`
OPTIONS (labels=[("owner", "pritha_ghosh_cvshealth_com"),("costcenter","13070")],
expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)) AS

with x1 as (
    -- Get all data from Step 10 (200 most recent dates, newest first)
    select *
    from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_o3_train_ending_tmp_1`
),

x2 as (
    -- 🎯 LEAD LOGIC: Join current row with next row to get next-day target
    -- Current row (x1a) gets the target from next row (x1b where seqno = x1a.seqno - 1)
    -- This shifts targets forward by 1 day for next-day prediction
    select 
        x1a.individual_id,
        x1a.index_dt,
        x1a.dt,
        x1a.seqno,
        x1a.gender_cd,
        x1a.age_in_months,
        x1a.cd,                  -- INPUT codes (from current day)
        x1b.target               -- 🎯 TARGET codes (from NEXT day! seqno-1 = next day)
    from x1 as x1a
    inner join x1 as x1b
        on x1a.individual_id = x1b.individual_id
        and x1a.index_dt = x1b.index_dt
        and x1a.seqno = x1b.seqno + 1  -- 🎯 KEY: Get target from next day (seqno is descending!)
)

-- Re-rank chronologically: oldest date = seqno2 1
select *,
    row_number() over (partition by individual_id, index_dt order by dt) as seqno2
from x2
order by individual_id, index_dt, seqno2
;


/*==============================================================================
  STEP 12: FINAL TRANSFORMER SEQUENCES 🎯 READY FOR TRAINING!
  
  Purpose: Aggregate dates into asterisk-separated sequences (one row per patient)
  
  Output Columns:
  1. individual_id, index_dt: Patient identifiers
  2. gender_cd: "1*1*1" (gender sequence)
  3. age_in_months: "540*541*542" (age sequence)
  4. cd: "123,456*789,101" (INPUT codes: ~84k vocab, comma within date, asterisk between dates)
  5. target: "45,67*89" (TARGET codes: ~3.5k vocab, next-day shifted) 🎯 NEW!
  6. dt_cnt: Number of dates (≤200)
  
  Format: cd[day N] predicts target[day N+1]
  
  Output: member_o3_train_ending → Ready for transformer training!
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_o3_train_ending`;
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_o3_train_ending`
CLUSTER BY individual_id
OPTIONS (labels=[("owner", "pritha_ghosh_cvshealth_com"),("costcenter","13070")],
expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)) AS
select 
    individual_id,
    index_dt,
    -- Aggregate demographics and codes into asterisk-separated sequences
    -- ORDER BY seqno2 ensures chronological order (oldest → newest) for transformer model
    ARRAY_TO_STRING(ARRAY_AGG(cast(gender_cd as string) ORDER BY seqno2), '*') as gender_cd,
    ARRAY_TO_STRING(ARRAY_AGG(cast(age_in_months as string) ORDER BY seqno2), '*') as age_in_months,
    ARRAY_TO_STRING(ARRAY_AGG(cast(cd as string) ORDER BY seqno2), '*') as cd,  -- INPUT: Nested commas within dates, asterisks between dates
    ARRAY_TO_STRING(ARRAY_AGG(cast(target as string) ORDER BY seqno2), '*') as target,  -- 🎯 TARGET: Training labels (next-day codes)
    count(*) as dt_cnt  -- Number of dates in sequence
from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_o3_train_ending_tmp_ordered`
group by individual_id,index_dt
;

/*==============================================================================
  ✅ COMMERCIAL.SQL - READY FOR TRANSFORMER TRAINING!
  
  Pipeline: 13 steps from base membership to final sequences
  Final Output: a834793_Commercial_member_o3_train_ending
  
  Format: One row per patient with asterisk-separated temporal sequences
  - cd: Input codes (~84k vocab)
  - target: Target codes (~3.5k vocab, next-day shifted)
  - dt_cnt: Number of dates (≤200)
  
  Next: Feed to transformer → Generate 256-dim embeddings
  
==============================================================================*/