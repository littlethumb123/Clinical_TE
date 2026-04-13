/*
Medicare IP Prediction: Create Raw Codes Feature Table
======================================================

Purpose: Create a feature table with top 500 diagnosis codes as binary indicators
         to compare against transformer embeddings

Output Table: a834793_Medicare_raw_codes_features_top500
Rows: One per (member, index_dt)
Columns: 500 binary indicators for top diagnosis codes

Author: Pritha Ghosh
Date: January 2026
*/

-- =============================================================================
-- Step 1: Identify Top 500 Most Common Diagnosis Codes
-- =============================================================================

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_top_500_dx_codes` AS

WITH code_stats AS (
    SELECT 
        dx_code,
        COUNT(DISTINCT member_id) as member_count,
        COUNT(*) as claim_count,
        COUNT(DISTINCT member_id) / (SELECT COUNT(DISTINCT member_id) FROM your_claims_table) as prevalence
    FROM `your_claims_table`  -- ← REPLACE with actual claims table
    WHERE dx_code IS NOT NULL
        AND dx_code != ''
    GROUP BY dx_code
)

SELECT 
    dx_code,
    member_count,
    claim_count,
    prevalence,
    ROW_NUMBER() OVER (ORDER BY member_count DESC) as code_rank
FROM code_stats
WHERE member_count >= 100  -- Filter very rare codes
ORDER BY member_count DESC
LIMIT 500;

-- Review results
SELECT * FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_top_500_dx_codes`
ORDER BY code_rank
LIMIT 20;


-- =============================================================================
-- Step 2: Create Member-Level Features (Binary Indicators)
-- =============================================================================

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_raw_codes_features_top500` AS

WITH base_population AS (
    -- Start with the same population used for embeddings
    SELECT DISTINCT
        individual_id,
        index_dt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_final_dataset_4_te_experiment`
    WHERE mon_6_include = 1
),

member_claims_12mo AS (
    -- Get all claims in 12-month lookback window
    SELECT 
        bp.individual_id,
        bp.index_dt,
        c.dx_code
    FROM base_population bp
    INNER JOIN `your_claims_table` c  -- ← REPLACE with actual claims table
        ON bp.individual_id = c.member_id
        AND c.claim_dt BETWEEN DATE_SUB(bp.index_dt, INTERVAL 12 MONTH) AND bp.index_dt
    INNER JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_top_500_dx_codes` top
        ON c.dx_code = top.dx_code
    WHERE c.dx_code IS NOT NULL
),

member_codes_distinct AS (
    -- Get distinct codes per member
    SELECT DISTINCT
        individual_id,
        index_dt,
        dx_code
    FROM member_claims_12mo
)

-- Pivot to wide format (one column per code)
-- NOTE: You'll need to generate 500 CASE statements (one per top code)
-- This is a template - use a script to generate the full query

SELECT
    individual_id,
    index_dt,
    
    -- Binary indicators (1 if code present, 0 if not)
    MAX(CASE WHEN dx_code = 'E11.9' THEN 1 ELSE 0 END) AS dx_001_E11_9,  -- Diabetes
    MAX(CASE WHEN dx_code = 'I10' THEN 1 ELSE 0 END) AS dx_002_I10,      -- Hypertension
    MAX(CASE WHEN dx_code = 'E78.5' THEN 1 ELSE 0 END) AS dx_003_E78_5,  -- Hyperlipidemia
    MAX(CASE WHEN dx_code = 'I50.9' THEN 1 ELSE 0 END) AS dx_004_I50_9,  -- CHF
    MAX(CASE WHEN dx_code = 'J44.9' THEN 1 ELSE 0 END) AS dx_005_J44_9,  -- COPD
    -- ... repeat for all 500 codes ...
    -- TODO: Generate this programmatically
    
FROM member_codes_distinct
GROUP BY individual_id, index_dt;


-- =============================================================================
-- Step 3: Validate Results
-- =============================================================================

-- Check row count matches baseline
SELECT 
    'Raw codes table' as source,
    COUNT(*) as row_count,
    COUNT(DISTINCT individual_id) as unique_members
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_raw_codes_features_top500`

UNION ALL

SELECT 
    'Baseline table' as source,
    COUNT(*) as row_count,
    COUNT(DISTINCT individual_id) as unique_members
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_final_dataset_4_te_experiment`
WHERE mon_6_include = 1;


-- Check sparsity (% of codes present per member)
SELECT
    AVG(code_count) as avg_codes_per_member,
    MIN(code_count) as min_codes,
    MAX(code_count) as max_codes,
    APPROX_QUANTILES(code_count, 100)[OFFSET(50)] as median_codes,
    AVG(code_count) / 500.0 as avg_density  -- Should be ~1-5%
FROM (
    SELECT 
        individual_id,
        index_dt,
        -- Count non-zero codes (assuming 500 code columns)
        (dx_001_E11_9 + dx_002_I10 + dx_003_E78_5 + 
         /* ... sum all 500 code columns ... */) as code_count
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_raw_codes_features_top500`
);


-- =============================================================================
-- ALTERNATIVE APPROACH: Dynamic Pivot using Python/Script
-- =============================================================================

/*
The manual CASE statement approach above is tedious for 500 codes.
Better approach: Generate the SQL programmatically.

Python pseudocode:

top_codes = client.query("SELECT dx_code FROM top_500_dx_codes ORDER BY code_rank").to_dataframe()

case_statements = []
for i, code in enumerate(top_codes['dx_code'], 1):
    safe_name = code.replace('.', '_').replace('-', '_')
    case_statements.append(
        f"MAX(CASE WHEN dx_code = '{code}' THEN 1 ELSE 0 END) AS dx_{i:03d}_{safe_name}"
    )

pivot_sql = f\"\"\"
SELECT
    individual_id,
    index_dt,
    {',\n    '.join(case_statements)}
FROM member_codes_distinct
GROUP BY individual_id, index_dt
\"\"\"

client.query(pivot_sql).result()
*/


-- =============================================================================
-- SIMPLER ALTERNATIVE: Use PIVOT (if your SQL dialect supports it)
-- =============================================================================

-- BigQuery doesn't support PIVOT, but you can use STRING_AGG as workaround:

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_raw_codes_features_top500_simple` AS

SELECT
    individual_id,
    index_dt,
    STRING_AGG(DISTINCT dx_code, '|') as codes_present,
    COUNT(DISTINCT dx_code) as code_count
FROM member_claims_12mo
GROUP BY individual_id, index_dt;

-- Then in Python, one-hot encode the codes_present string
-- This is actually EASIER than SQL pivot!

