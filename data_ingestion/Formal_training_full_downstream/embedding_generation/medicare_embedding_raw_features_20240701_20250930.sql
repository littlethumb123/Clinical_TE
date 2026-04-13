/*==============================================================================
  MEDICARE EMBEDDING RAW FEATURES (CORRECTED)

  Why this script exists:
  - The source training pipeline in data_ingestion/TE_pretraining_data_ingestion/
    medicare_for_training.sql currently builds cohort membership with:
      eff_dt BETWEEN DATE('2023-01-01') AND DATE('2023-12-31')
  - Therefore, simply filtering a834793_Medicare_member_o3_train_ending for
    2024-2025 can return empty/incorrect output unless upstream was rebuilt.

  Correct approach implemented here:
  - Rebuild raw embedding features from source membership/claims/pharmacy data
    using the same transformation logic used by the Medicare pipeline for
    input codes (cd), but with the requested index_dt window.
  - Keep only embedding-input columns: individual_id, index_dt, gender_cd,
    age_in_months, cd, dt_cnt.

  Requested index_dt window:
  - 2024-07-01 to 2025-09-30 (inclusive)

  Output table:
  - edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicare_embedding_raw_features_20240701_20250930
==============================================================================*/

DECLARE start_index_dt DATE DEFAULT DATE('2024-07-01');
DECLARE end_index_dt DATE DEFAULT DATE('2025-09-30');

-- Step 0: one random index_dt per Medicare member in requested window
CREATE TEMP TABLE base_memberid AS
WITH current_individual_id AS (
  SELECT
    member_id,
    ARRAY_AGG(eff_dt ORDER BY RAND() LIMIT 1)[SAFE_OFFSET(0)] AS index_dt
  FROM `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_MEMBERSHIP`
  WHERE eff_dt BETWEEN start_index_dt AND end_index_dt
    AND business_ln_cd = 'ME'
    AND file_id <> 'C2'
  GROUP BY member_id
)
SELECT DISTINCT
  m.member_id,
  x.individual_id,
  m.index_dt
FROM current_individual_id m
JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` x
  ON m.member_id = x.member_id;

ASSERT (SELECT COUNT(*) FROM base_memberid) > 0
AS 'No Medicare members found in requested index_dt window.';

-- Demographic anchor (same ranking rule as source pipeline)
CREATE TEMP TABLE b_dt AS
WITH base AS (
  SELECT
    rm.individual_id,
    rm.member_id,
    rm.index_dt,
    m.gender_cd,
    m.birth_dt,
    ROW_NUMBER() OVER (
      PARTITION BY rm.individual_id
      ORDER BY orig_covg_eff_dt DESC, hmo_to_trad_conv_dt DESC, rm.member_id DESC
    ) AS ord
  FROM base_memberid rm
  JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEMBER` m
    ON rm.member_id = m.member_id
)
SELECT *
FROM base
WHERE ord = 1;

-- Medical claims stream (input-code fields only)
CREATE TEMP TABLE monthly_claims AS
SELECT
  base.individual_id,
  base.member_id,
  base.index_dt,
  clm.claim_line_id,
  clm.srv_start_dt AS dt,
  CASE
    WHEN clm.days_cnt IS NULL OR clm.days_cnt < 0 THEN 99
    WHEN clm.days_cnt > 10 THEN 11
    ELSE clm.days_cnt
  END AS days_cnt,
  CASE
    WHEN TRIM(b_dt.gender_cd) = 'M' THEN 1
    WHEN TRIM(b_dt.gender_cd) = 'F' THEN 0
    ELSE 2
  END AS gender_cd,
  DATE_DIFF(CAST(clm.srv_start_dt AS DATE), CAST(b_dt.birth_dt AS DATE), MONTH) AS age_in_months,
  CASE
    WHEN TRIM(clm.revenue_cd) = '' OR clm.revenue_cd IS NULL THEN NULL
    WHEN REGEXP_CONTAINS(CAST(clm.revenue_cd AS STRING), r'^[0-9]{3,4}$') THEN UPPER(TRIM(clm.revenue_cd))
    ELSE NULL
  END AS revenue_cd,
  CASE
    WHEN TRIM(clm.hcfa_plc_srv_cd) IS NULL OR TRIM(clm.hcfa_plc_srv_cd) = '' THEN NULL
    WHEN REGEXP_CONTAINS(CAST(clm.hcfa_plc_srv_cd AS STRING), r'^[0-9]{1,2}$') THEN UPPER(TRIM(clm.hcfa_plc_srv_cd))
    ELSE NULL
  END AS hcfa_plc_srv_cd,
  CASE
    WHEN TRIM(nppes.healthcare_provider_taxonomy_code_1) = '' OR nppes.healthcare_provider_taxonomy_code_1 IS NULL THEN NULL
    WHEN REGEXP_CONTAINS(TRIM(nppes.healthcare_provider_taxonomy_code_1), r'^[A-Z0-9]{10}$')
      THEN UPPER(TRIM(nppes.healthcare_provider_taxonomy_code_1))
    ELSE NULL
  END AS provider_taxonomy_cd,
  CASE
    WHEN TRIM(clm.prcdr_cd) = '' THEN NULL
    WHEN LENGTH(TRIM(clm.prcdr_cd)) < 4 THEN NULL
    ELSE UPPER(TRIM(clm.prcdr_cd))
  END AS prcdr_cd,
  CASE
    WHEN TRIM(icd_prc.icd9_prcdr_cd) = '' THEN NULL
    WHEN LENGTH(TRIM(icd_prc.icd9_prcdr_cd)) < 4 THEN NULL
    ELSE UPPER(TRIM(icd_prc.icd9_prcdr_cd))
  END AS icd9_prcdr_cd,
  CASE
    WHEN TRIM(drg.drg_cd) = '' OR drg.drg_cd IS NULL THEN NULL
    WHEN REGEXP_CONTAINS(CAST(drg.drg_cd AS STRING), r'^[0-9]+$') THEN CAST(CAST(drg.drg_cd AS INT64) AS STRING)
    ELSE NULL
  END AS drg_cd
FROM base_memberid base
JOIN b_dt
  ON base.individual_id = b_dt.individual_id
JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_CLAIM_LINE` clm
  ON base.member_id = clm.member_id
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_PRCD` icd_prc
  ON clm.claim_line_id = icd_prc.claim_line_id
 AND clm.member_id = icd_prc.member_id
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_DRG_TYPE` drg
  ON clm.claim_line_id = drg.claim_line_id
 AND clm.member_id = drg.member_id
LEFT JOIN (
  SELECT npi, healthcare_provider_taxonomy_code_1
  FROM `bigquery-public-data.nppes.npi_raw`
) nppes
  ON CAST(clm.srv_prvdr_npi_nbr AS STRING) = nppes.npi
WHERE clm.srv_start_dt > DATE_SUB(base.index_dt, INTERVAL 36 MONTH)
  AND clm.srv_start_dt < DATE_SUB(base.index_dt, INTERVAL 90 DAY)
  AND IF(clm.paid_dt > DATE('1900-01-01'), clm.paid_dt, clm.adjn_dt) < DATE_SUB(base.index_dt, INTERVAL 90 DAY)
  AND clm.duplicate_ind = 'N'
  AND clm.summarized_srv_ind = 'Y'
  AND clm.file_id <> 'C4'
  AND clm.reversal_cd <> 'R';

-- Prescription stream
CREATE TEMP TABLE monthly_rx AS
WITH rx_combined AS (
  SELECT member_id, disp_dt, process_dt, adjudicated_gpi_cd, file_id
  FROM `edp-prod-hcbstorage.edp_hcb_core_cnsv.RX_CLAIM_DTL`
  UNION ALL
  SELECT member_id, disp_dt, process_dt, adjudicated_gpi_cd, file_id
  FROM `edp-prod-hcbstorage.edp_hcb_core_cnsv.XTRNL_RX_CLAIM`
  UNION ALL
  SELECT member_id, disp_dt, process_dt, adjudicated_gpi_cd, file_id
  FROM `edp-prod-hcbstorage.edp_hcb_hcbdw_rxclmrestrict_v.RX_CLAIM_DTL_UNFLTRD`
)
SELECT
  base.individual_id,
  base.index_dt,
  base.member_id,
  CASE
    WHEN TRIM(b_dt.gender_cd) = 'M' THEN 1
    WHEN TRIM(b_dt.gender_cd) = 'F' THEN 0
    ELSE 2
  END AS gender_cd,
  DATE_DIFF(CAST(rx.disp_dt AS DATE), CAST(b_dt.birth_dt AS DATE), MONTH) AS age_in_months,
  rx.disp_dt AS dt,
  CASE
    WHEN TRIM(rx.adjudicated_gpi_cd) IS NULL OR TRIM(rx.adjudicated_gpi_cd) = '' THEN NULL
    WHEN LENGTH(TRIM(rx.adjudicated_gpi_cd)) >= 4
      AND REGEXP_CONTAINS(SUBSTR(TRIM(rx.adjudicated_gpi_cd), 1, 4), r'^[0-9]{4}$')
      THEN CONCAT('gpi', SUBSTR(TRIM(rx.adjudicated_gpi_cd), 1, 4))
    ELSE NULL
  END AS gpi4
FROM base_memberid base
JOIN b_dt
  ON base.individual_id = b_dt.individual_id
JOIN rx_combined rx
  ON base.member_id = rx.member_id
WHERE rx.disp_dt > DATE_SUB(base.index_dt, INTERVAL 36 MONTH)
  AND rx.disp_dt <= base.index_dt
  AND rx.process_dt <= base.index_dt
  AND rx.file_id <> 'C5';

-- Diagnosis-expanded claims stream
CREATE TEMP TABLE monthly_clm_ln AS
SELECT DISTINCT
  base.individual_id,
  base.index_dt,
  base.member_id,
  base.claim_line_id,
  base.dt,
  base.days_cnt,
  base.gender_cd,
  base.age_in_months,
  CASE
    WHEN ARRAY_TO_STRING([dx_parts.l, dx_parts.r], '.') = '' OR ARRAY_TO_STRING([dx_parts.l, dx_parts.r], '.') IS NULL THEN NULL
    WHEN REGEXP_CONTAINS(UPPER(ARRAY_TO_STRING([dx_parts.l, dx_parts.r], '.')), r'^[A-Z][0-9A-Z]{2}[\.\w]*$')
      THEN UPPER(ARRAY_TO_STRING([dx_parts.l, dx_parts.r], '.'))
    ELSE NULL
  END AS icd9_dx_cd
FROM monthly_claims base
JOIN (
  SELECT
    member_id,
    claim_line_id,
    ARRAY[
      STRUCT(
        SPLIT(TRIM(icd9_dx_cd), '.')[SAFE_OFFSET(0)] AS l,
        SUBSTR(SPLIT(TRIM(icd9_dx_cd), '.')[SAFE_OFFSET(1)], 1, 2) AS r
      )
    ] AS icd_array
  FROM `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_DX`
  WHERE CAST(sequence_id AS INT64) < 4
    AND icd9_dx_cd IS NOT NULL
    AND TRIM(icd9_dx_cd) != ''
) dx
  ON base.member_id = dx.member_id
 AND base.claim_line_id = dx.claim_line_id,
UNNEST(dx.icd_array) AS dx_parts;

-- Root patient-date table
CREATE TEMP TABLE member_root AS
WITH root0 AS (
  SELECT
    individual_id,
    index_dt,
    member_id,
    dt,
    gender_cd,
    CASE
      WHEN age_in_months < 0 THEN 0
      WHEN age_in_months > 1439 THEN 1439
      ELSE age_in_months
    END AS age_in_months
  FROM monthly_claims
  UNION ALL
  SELECT
    individual_id,
    index_dt,
    member_id,
    dt,
    gender_cd,
    CASE
      WHEN age_in_months < 0 THEN 0
      WHEN age_in_months > 1439 THEN 1439
      ELSE age_in_months
    END AS age_in_months
  FROM monthly_rx
),
root1 AS (
  SELECT
    r0.*,
    ROW_NUMBER() OVER (
      PARTITION BY r0.individual_id, r0.index_dt, r0.dt
      ORDER BY r0.member_id, r0.gender_cd
    ) AS seqno
  FROM root0 r0
)
SELECT DISTINCT
  individual_id,
  index_dt,
  member_id,
  dt,
  gender_cd,
  age_in_months
FROM root1
WHERE seqno = 1;

-- Shared vocabulary lookup
CREATE TEMP TABLE w2ind AS
SELECT ind, cd
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_member_w2ind`;

-- Step 8 equivalent: map all input code types to indices
CREATE TEMP TABLE member_get_cd AS
SELECT DISTINCT
  base.individual_id,
  base.index_dt,
  base.member_id,
  base.dt,
  w2.cd,
  CASE WHEN w2.ind IS NULL THEN 0 ELSE w2.ind END AS ind
FROM monthly_claims base
LEFT JOIN w2ind w2
  ON CONCAT('days_cnt', CAST(base.days_cnt AS STRING)) = w2.cd
WHERE base.days_cnt IS NOT NULL

UNION ALL

SELECT DISTINCT
  base.individual_id,
  base.index_dt,
  base.member_id,
  base.dt,
  w2.cd,
  CASE WHEN w2.ind IS NULL THEN 0 ELSE w2.ind END AS ind
FROM monthly_claims base
LEFT JOIN w2ind w2
  ON CONCAT('hcfa_plc_srv_cd', CAST(base.hcfa_plc_srv_cd AS STRING)) = w2.cd
WHERE base.hcfa_plc_srv_cd IS NOT NULL

UNION ALL

SELECT DISTINCT
  base.individual_id,
  base.index_dt,
  base.member_id,
  base.dt,
  w2.cd,
  CASE WHEN w2.ind IS NULL THEN 0 ELSE w2.ind END AS ind
FROM monthly_claims base
LEFT JOIN w2ind w2
  ON CONCAT('provider_taxonomy_cd', CAST(base.provider_taxonomy_cd AS STRING)) = w2.cd
WHERE base.provider_taxonomy_cd IS NOT NULL

UNION ALL

SELECT DISTINCT
  base.individual_id,
  base.index_dt,
  base.member_id,
  base.dt,
  w2.cd,
  CASE WHEN w2.ind IS NULL THEN 0 ELSE w2.ind END AS ind
FROM monthly_clm_ln base
LEFT JOIN w2ind w2
  ON CONCAT('icd9_dx_cd', CAST(base.icd9_dx_cd AS STRING)) = w2.cd
WHERE base.icd9_dx_cd IS NOT NULL

UNION ALL

SELECT DISTINCT
  base.individual_id,
  base.index_dt,
  base.member_id,
  base.dt,
  w2.cd,
  CASE WHEN w2.ind IS NULL THEN 0 ELSE w2.ind END AS ind
FROM monthly_claims base
LEFT JOIN w2ind w2
  ON CONCAT('revenue_cd', CAST(base.revenue_cd AS STRING)) = w2.cd
WHERE base.revenue_cd IS NOT NULL

UNION ALL

(
  SELECT DISTINCT
    base.individual_id,
    base.index_dt,
    base.member_id,
    base.dt,
    w2.cd,
    CASE WHEN w2.ind IS NULL THEN 0 ELSE w2.ind END AS ind
  FROM monthly_claims base
  LEFT JOIN w2ind w2
    ON CONCAT('prcdr_cd', CAST(base.prcdr_cd AS STRING)) = w2.cd
  WHERE base.prcdr_cd IS NOT NULL

  UNION DISTINCT

  SELECT DISTINCT
    base.individual_id,
    base.index_dt,
    base.member_id,
    base.dt,
    w2.cd,
    CASE WHEN w2.ind IS NULL THEN 0 ELSE w2.ind END AS ind
  FROM monthly_claims base
  LEFT JOIN w2ind w2
    ON CONCAT('prcdr_cd', CAST(base.icd9_prcdr_cd AS STRING)) = w2.cd
  WHERE base.icd9_prcdr_cd IS NOT NULL
)

UNION ALL

SELECT DISTINCT
  base.individual_id,
  base.index_dt,
  base.member_id,
  base.dt,
  w2.cd,
  CASE WHEN w2.ind IS NULL THEN 0 ELSE w2.ind END AS ind
FROM monthly_claims base
LEFT JOIN w2ind w2
  ON CONCAT('drg_cd', CAST(base.drg_cd AS STRING)) = w2.cd
WHERE base.drg_cd IS NOT NULL

UNION ALL

SELECT DISTINCT
  base.individual_id,
  base.index_dt,
  base.member_id,
  base.dt,
  w2.cd,
  CASE WHEN w2.ind IS NULL THEN 0 ELSE w2.ind END AS ind
FROM monthly_rx base
LEFT JOIN w2ind w2
  ON CAST(base.gpi4 AS STRING) = w2.cd
WHERE base.gpi4 IS NOT NULL;

-- Aggregate day-level input codes (same 80-code cap used in source pipeline)
CREATE TEMP TABLE o1_train AS
WITH x1 AS (
  SELECT
    individual_id,
    index_dt,
    dt,
    ind
  FROM member_get_cd
  GROUP BY individual_id, index_dt, dt, ind
),
x2 AS (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY individual_id, index_dt, dt ORDER BY ind) AS seqno
  FROM x1
),
x3 AS (
  SELECT
    individual_id,
    index_dt,
    dt,
    ARRAY_AGG(CAST(ind AS STRING) ORDER BY ind) AS cd_arr
  FROM x2
  WHERE seqno <= 80
  GROUP BY individual_id, index_dt, dt
)
SELECT
  root2.individual_id,
  root2.index_dt,
  root2.dt,
  root2.gender_cd,
  root2.age_in_months,
  ARRAY_TO_STRING(x3.cd_arr, ',') AS cd
FROM member_root root2
JOIN x3
  ON root2.individual_id = x3.individual_id
 AND root2.index_dt = x3.index_dt
 AND root2.dt = x3.dt;

-- Keep only 200 most recent dates per patient/index_dt
CREATE TEMP TABLE o3_recent AS
WITH ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY individual_id, index_dt ORDER BY dt DESC) AS seqno
  FROM o1_train
)
SELECT *
FROM ranked
WHERE seqno <= 200;

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicare_embedding_raw_features_20240701_20250930`;

CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicare_embedding_raw_features_20240701_20250930`
CLUSTER BY individual_id, index_dt
OPTIONS (
  labels=[("owner", "pritha_ghosh_cvshealth_com"), ("costcenter", "13070")],
  description="Medicare raw TE embedding features regenerated for index_dt 2024-07-01 to 2025-09-30",
  expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 365 DAY)
) AS
SELECT
  individual_id,
  index_dt,
  ARRAY_TO_STRING(ARRAY_AGG(CAST(gender_cd AS STRING) ORDER BY dt), '*') AS gender_cd,
  ARRAY_TO_STRING(ARRAY_AGG(CAST(age_in_months AS STRING) ORDER BY dt), '*') AS age_in_months,
  ARRAY_TO_STRING(ARRAY_AGG(CAST(cd AS STRING) ORDER BY dt), '*') AS cd,
  COUNT(*) AS dt_cnt
FROM o3_recent
WHERE index_dt BETWEEN start_index_dt AND end_index_dt
GROUP BY individual_id, index_dt;

/*
Validation:
SELECT
  COUNT(*) AS row_count,
  COUNT(DISTINCT individual_id) AS member_count,
  MIN(index_dt) AS min_index_dt,
  MAX(index_dt) AS max_index_dt,
  AVG(dt_cnt) AS avg_dt_cnt
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicare_embedding_raw_features_20240701_20250930`;
*/
