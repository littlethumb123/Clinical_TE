/*==============================================================================
  COMMERCIAL NEW-TE EMBEDDING POPULATION DIAGNOSTICS

  Purpose:
  Diagnose why the new TE embedding join keeps only ~83k members while the
  production RAP embedding join keeps ~5.5M members.

  Hypotheses tested:
  1. The new TE embedding table itself is incomplete relative to the raw-feature
     inference input.
  2. The new TE table has poor individual overlap with the tabular outcome table.
  3. The new TE table has acceptable individual overlap but poor date coverage.
  4. The Medicare-style as-of logic is valid for production-style history tables,
     but does not solve a missing-population problem in the new TE table.

  Inputs:
  - Raw inference input:
      edp-prod-storage.edp_ent_sdoheir_cns.a964286_commercial_embedding_raw_features_20241120_20250930
  - New TE embeddings:
      edp-prod-storage.edp_ent_sdoheir_cns.a964286_exp_round10_exp2b_commercial_embeddings_20241120_20250930
  - Production RAP embeddings:
      edp-prod-storage.edp_ent_sdoheir_cns.enhanced_rap_cp_emb_history_wide_4_te_fromal_eval
  - Tabular features + outcome:
      edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_final_dataset_4_te_formal_evaluation_20241120_20250930

==============================================================================*/

DECLARE min_index_dt DATE DEFAULT DATE '2024-11-20';
DECLARE max_index_dt DATE DEFAULT DATE '2025-09-30';
DECLARE buffer_days INT64 DEFAULT 90;


/*==============================================================================
  1. TABLE CENSUS

  Read this first.
  If new_te row_count / unique_individuals is already tiny relative to raw_input,
  the main issue is upstream export coverage, not downstream join logic.
==============================================================================*/

WITH census AS (
  SELECT
    'raw_input' AS source,
    COUNT(*) AS row_count,
    COUNT(DISTINCT CAST(individual_id AS STRING)) AS unique_individuals,
    COUNT(DISTINCT FORMAT('%s|%s', CAST(individual_id AS STRING), CAST(DATE(index_dt) AS STRING))) AS unique_pairs,
    MIN(DATE(index_dt)) AS min_index_dt,
    MAX(DATE(index_dt)) AS max_index_dt
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_commercial_embedding_raw_features_20241120_20250930`
  WHERE DATE(index_dt) BETWEEN min_index_dt AND max_index_dt

  UNION ALL

  SELECT
    'new_te_embeddings' AS source,
    COUNT(*) AS row_count,
    COUNT(DISTINCT CAST(individual_id AS STRING)) AS unique_individuals,
    COUNT(DISTINCT FORMAT('%s|%s', CAST(individual_id AS STRING), CAST(DATE(index_dt) AS STRING))) AS unique_pairs,
    MIN(DATE(index_dt)) AS min_index_dt,
    MAX(DATE(index_dt)) AS max_index_dt
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_exp_round10_exp2b_commercial_embeddings_20241120_20250930`
  WHERE DATE(index_dt) BETWEEN min_index_dt AND max_index_dt

  UNION ALL

  SELECT
    'prod_rap_embeddings' AS source,
    COUNT(*) AS row_count,
    COUNT(DISTINCT CAST(individual_id AS STRING)) AS unique_individuals,
    COUNT(DISTINCT FORMAT('%s|%s', CAST(individual_id AS STRING), CAST(DATE(index_dt) AS STRING))) AS unique_pairs,
    MIN(DATE(index_dt)) AS min_index_dt,
    MAX(DATE(index_dt)) AS max_index_dt
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.enhanced_rap_cp_emb_history_wide_4_te_fromal_eval`
  WHERE DATE(index_dt) BETWEEN min_index_dt AND max_index_dt

  UNION ALL

  SELECT
    'tabular_outcome' AS source,
    COUNT(*) AS row_count,
    COUNT(DISTINCT CAST(individual_id AS STRING)) AS unique_individuals,
    COUNT(DISTINCT FORMAT('%s|%s', CAST(individual_id AS STRING), CAST(DATE(index_dt) AS STRING))) AS unique_pairs,
    MIN(DATE(index_dt)) AS min_index_dt,
    MAX(DATE(index_dt)) AS max_index_dt
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_final_dataset_4_te_formal_evaluation_20241120_20250930`
  WHERE DATE(index_dt) BETWEEN min_index_dt AND max_index_dt
)
SELECT *
FROM census
ORDER BY source
;


-- [{
--   "source": "new_te_embeddings",
--   "row_count": "12936113",
--   "unique_individuals": "12792126",
--   "unique_pairs": "12936113",
--   "min_index_dt": "2024-12-16",
--   "max_index_dt": "2025-09-16"
-- }, {
--   "source": "prod_rap_embeddings",
--   "row_count": "1019708803",
--   "unique_individuals": "18529191",
--   "unique_pairs": "1019708803",
--   "min_index_dt": "2024-11-20",
--   "max_index_dt": "2025-09-30"
-- }, {
--   "source": "raw_input",
--   "row_count": "12936113",
--   "unique_individuals": "12792126",
--   "unique_pairs": "12936113",
--   "min_index_dt": "2024-12-16",
--   "max_index_dt": "2025-09-16"
-- }, {
--   "source": "tabular_outcome",
--   "row_count": "10285921",
--   "unique_individuals": "10186682",
--   "unique_pairs": "10285921",
--   "min_index_dt": "2024-12-16",
--   "max_index_dt": "2025-09-16"
-- }]


/*==============================================================================
  2. RAW INPUT VS NEW-TE EXPORT COVERAGE

  If overlap_pairs and overlap_individuals are very small, the new TE table was
  not exported for most of the inference input population.
==============================================================================*/

WITH raw_input AS (
  SELECT DISTINCT
    CAST(individual_id AS STRING) AS individual_id,
    DATE(index_dt) AS index_dt
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_commercial_embedding_raw_features_20241120_20250930`
  WHERE DATE(index_dt) BETWEEN min_index_dt AND max_index_dt
),
new_te AS (
  SELECT DISTINCT
    CAST(individual_id AS STRING) AS individual_id,
    DATE(index_dt) AS index_dt
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_exp_round10_exp2b_commercial_embeddings_20241120_20250930`
  WHERE DATE(index_dt) BETWEEN min_index_dt AND max_index_dt
),
raw_ids AS (
  SELECT DISTINCT individual_id FROM raw_input
),
new_te_ids AS (
  SELECT DISTINCT individual_id FROM new_te
)
SELECT
  (SELECT COUNT(*) FROM raw_input) AS raw_pairs,
  (SELECT COUNT(*) FROM new_te) AS new_te_pairs,
  (SELECT COUNT(*) FROM raw_ids) AS raw_individuals,
  (SELECT COUNT(*) FROM new_te_ids) AS new_te_individuals,
  (SELECT COUNT(*) FROM raw_input r INNER JOIN new_te n USING (individual_id, index_dt)) AS overlap_pairs,
  (SELECT COUNT(*) FROM raw_ids r INNER JOIN new_te_ids n USING (individual_id)) AS overlap_individuals,
  ROUND(SAFE_DIVIDE((SELECT COUNT(*) FROM raw_input r INNER JOIN new_te n USING (individual_id, index_dt)), (SELECT COUNT(*) FROM raw_input)), 6) AS pair_coverage_from_raw,
  ROUND(SAFE_DIVIDE((SELECT COUNT(*) FROM raw_ids r INNER JOIN new_te_ids n USING (individual_id)), (SELECT COUNT(*) FROM raw_ids)), 6) AS individual_coverage_from_raw
;


-- [{
--   "raw_pairs": "12936113",
--   "new_te_pairs": "12936113",
--   "raw_individuals": "12792126",
--   "new_te_individuals": "12792126",
--   "overlap_pairs": "12936113",
--   "overlap_individuals": "12792126",
--   "pair_coverage_from_raw": "1.0",
--   "individual_coverage_from_raw": "1.0"
-- }]

/*==============================================================================
  3. OUTCOME TABLE VS EMBEDDING TABLES

  This separates individual-universe mismatch from date-alignment mismatch.
  If new_te outcome individual overlap is already tiny, the as-of logic is not
  the main bottleneck.
==============================================================================*/

WITH outcome_pairs AS (
  SELECT DISTINCT
    CAST(individual_id AS STRING) AS individual_id,
    DATE(index_dt) AS index_dt
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_final_dataset_4_te_formal_evaluation_20241120_20250930`
  WHERE DATE(index_dt) BETWEEN min_index_dt AND max_index_dt
),
outcome_ids AS (
  SELECT DISTINCT individual_id FROM outcome_pairs
),
new_te_pairs AS (
  SELECT DISTINCT
    CAST(individual_id AS STRING) AS individual_id,
    DATE(index_dt) AS index_dt
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_exp_round10_exp2b_commercial_embeddings_20241120_20250930`
  WHERE DATE(index_dt) BETWEEN min_index_dt AND max_index_dt
),
new_te_ids AS (
  SELECT DISTINCT individual_id FROM new_te_pairs
),
prod_pairs AS (
  SELECT DISTINCT
    CAST(individual_id AS STRING) AS individual_id,
    DATE(index_dt) AS index_dt
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.enhanced_rap_cp_emb_history_wide_4_te_fromal_eval`
  WHERE DATE(index_dt) BETWEEN min_index_dt AND max_index_dt
),
prod_ids AS (
  SELECT DISTINCT individual_id FROM prod_pairs
),
new_te_asof_90d AS (
  SELECT
    outcome.individual_id,
    outcome.index_dt,
    MAX(new_te.index_dt) AS matched_embedding_index_dt
  FROM outcome_pairs AS outcome
  INNER JOIN new_te_pairs AS new_te
    ON new_te.individual_id = outcome.individual_id
   AND new_te.index_dt <= DATE_SUB(outcome.index_dt, INTERVAL buffer_days DAY)
  GROUP BY outcome.individual_id, outcome.index_dt
),
prod_asof_90d AS (
  SELECT
    outcome.individual_id,
    outcome.index_dt,
    MAX(prod.index_dt) AS matched_embedding_index_dt
  FROM outcome_pairs AS outcome
  INNER JOIN prod_pairs AS prod
    ON prod.individual_id = outcome.individual_id
   AND prod.index_dt <= DATE_SUB(outcome.index_dt, INTERVAL buffer_days DAY)
  GROUP BY outcome.individual_id, outcome.index_dt
)
SELECT
  'new_te' AS source,
  (SELECT COUNT(*) FROM outcome_pairs) AS outcome_pairs,
  (SELECT COUNT(*) FROM outcome_ids) AS outcome_individuals,
  (SELECT COUNT(*) FROM new_te_pairs) AS embedding_pairs,
  (SELECT COUNT(*) FROM new_te_ids) AS embedding_individuals,
  (SELECT COUNT(*) FROM outcome_ids o INNER JOIN new_te_ids e USING (individual_id)) AS overlapping_individuals,
  (SELECT COUNT(*) FROM outcome_pairs o INNER JOIN new_te_pairs e USING (individual_id, index_dt)) AS exact_pair_overlap,
  (SELECT COUNT(*) FROM new_te_asof_90d) AS asof_90d_pair_overlap,
  ROUND(SAFE_DIVIDE((SELECT COUNT(*) FROM outcome_ids o INNER JOIN new_te_ids e USING (individual_id)), (SELECT COUNT(*) FROM outcome_ids)), 6) AS pct_outcome_individuals_with_embedding,
  ROUND(SAFE_DIVIDE((SELECT COUNT(*) FROM new_te_asof_90d), (SELECT COUNT(*) FROM outcome_pairs)), 6) AS pct_outcome_pairs_with_asof_match

UNION ALL

SELECT
  'prod_rap' AS source,
  (SELECT COUNT(*) FROM outcome_pairs) AS outcome_pairs,
  (SELECT COUNT(*) FROM outcome_ids) AS outcome_individuals,
  (SELECT COUNT(*) FROM prod_pairs) AS embedding_pairs,
  (SELECT COUNT(*) FROM prod_ids) AS embedding_individuals,
  (SELECT COUNT(*) FROM outcome_ids o INNER JOIN prod_ids e USING (individual_id)) AS overlapping_individuals,
  (SELECT COUNT(*) FROM outcome_pairs o INNER JOIN prod_pairs e USING (individual_id, index_dt)) AS exact_pair_overlap,
  (SELECT COUNT(*) FROM prod_asof_90d) AS asof_90d_pair_overlap,
  ROUND(SAFE_DIVIDE((SELECT COUNT(*) FROM outcome_ids o INNER JOIN prod_ids e USING (individual_id)), (SELECT COUNT(*) FROM outcome_ids)), 6) AS pct_outcome_individuals_with_embedding,
  ROUND(SAFE_DIVIDE((SELECT COUNT(*) FROM prod_asof_90d), (SELECT COUNT(*) FROM outcome_pairs)), 6) AS pct_outcome_pairs_with_asof_match
;


-- [{
--   "source": "prod_rap",
--   "outcome_pairs": "10285921",
--   "outcome_individuals": "10186682",
--   "embedding_pairs": "1019708803",
--   "embedding_individuals": "18529191",
--   "overlapping_individuals": "10003666",
--   "exact_pair_overlap": "1946431",
--   "asof_90d_pair_overlap": "5558507",
--   "pct_outcome_individuals_with_embedding": "0.982034",
--   "pct_outcome_pairs_with_asof_match": "0.5404"
-- }, {
--   "source": "new_te",
--   "outcome_pairs": "10285921",
--   "outcome_individuals": "10186682",
--   "embedding_pairs": "12936113",
--   "embedding_individuals": "12792126",
--   "overlapping_individuals": "10186682",
--   "exact_pair_overlap": "10285921",
--   "asof_90d_pair_overlap": "83279",
--   "pct_outcome_individuals_with_embedding": "1.0",
--   "pct_outcome_pairs_with_asof_match": "0.008096"
-- }]

/*==============================================================================
  4. DATE DENSITY PER INDIVIDUAL

  This shows whether Medicare-style history-table logic is structurally relevant.
  Production RAP should look like a history table with many dates per person.
  If new TE is near one date per person, the as-of logic can only help dates;
  it cannot recover missing members.
==============================================================================*/

WITH per_person_date_counts AS (
  SELECT
    'new_te' AS source,
    CAST(individual_id AS STRING) AS individual_id,
    COUNT(DISTINCT DATE(index_dt)) AS n_dates
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_exp_round10_exp2b_commercial_embeddings_20241120_20250930`
  WHERE DATE(index_dt) BETWEEN min_index_dt AND max_index_dt
  GROUP BY CAST(individual_id AS STRING)

  UNION ALL

  SELECT
    'prod_rap' AS source,
    CAST(individual_id AS STRING) AS individual_id,
    COUNT(DISTINCT DATE(index_dt)) AS n_dates
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.enhanced_rap_cp_emb_history_wide_4_te_fromal_eval`
  WHERE DATE(index_dt) BETWEEN min_index_dt AND max_index_dt
  GROUP BY CAST(individual_id AS STRING)
)
SELECT
  source,
  COUNT(*) AS individuals,
  MIN(n_dates) AS min_dates_per_individual,
  APPROX_QUANTILES(n_dates, 100)[OFFSET(25)] AS p25_dates_per_individual,
  APPROX_QUANTILES(n_dates, 100)[OFFSET(50)] AS median_dates_per_individual,
  APPROX_QUANTILES(n_dates, 100)[OFFSET(75)] AS p75_dates_per_individual,
  MAX(n_dates) AS max_dates_per_individual,
  ROUND(AVG(n_dates), 2) AS avg_dates_per_individual
FROM per_person_date_counts
GROUP BY source
ORDER BY source
;

-- [{
--   "source": "new_te",
--   "individuals": "12792126",
--   "min_dates_per_individual": "1",
--   "p25_dates_per_individual": "1",
--   "median_dates_per_individual": "1",
--   "p75_dates_per_individual": "1",
--   "max_dates_per_individual": "4",
--   "avg_dates_per_individual": "1.01"
-- }, {
--   "source": "prod_rap",
--   "individuals": "18529191",
--   "min_dates_per_individual": "1",
--   "p25_dates_per_individual": "31",
--   "median_dates_per_individual": "47",
--   "p75_dates_per_individual": "71",
--   "max_dates_per_individual": "305",
--   "avg_dates_per_individual": "55.03"
-- }]
/*==============================================================================
  5. MONTHLY COVERAGE SHAPE

  Use this to see whether the new TE export is missing large month blocks or only
  contains a narrow band of dates.
==============================================================================*/

WITH monthly_counts AS (
  SELECT
    'raw_input' AS source,
    FORMAT_DATE('%Y-%m', DATE(index_dt)) AS year_month,
    COUNT(*) AS row_count,
    COUNT(DISTINCT CAST(individual_id AS STRING)) AS unique_individuals
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_commercial_embedding_raw_features_20241120_20250930`
  WHERE DATE(index_dt) BETWEEN min_index_dt AND max_index_dt
  GROUP BY year_month

  UNION ALL

  SELECT
    'new_te' AS source,
    FORMAT_DATE('%Y-%m', DATE(index_dt)) AS year_month,
    COUNT(*) AS row_count,
    COUNT(DISTINCT CAST(individual_id AS STRING)) AS unique_individuals
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_exp_round10_exp2b_commercial_embeddings_20241120_20250930`
  WHERE DATE(index_dt) BETWEEN min_index_dt AND max_index_dt
  GROUP BY year_month

  UNION ALL

  SELECT
    'prod_rap' AS source,
    FORMAT_DATE('%Y-%m', DATE(index_dt)) AS year_month,
    COUNT(*) AS row_count,
    COUNT(DISTINCT CAST(individual_id AS STRING)) AS unique_individuals
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.enhanced_rap_cp_emb_history_wide_4_te_fromal_eval`
  WHERE DATE(index_dt) BETWEEN min_index_dt AND max_index_dt
  GROUP BY year_month
)
SELECT *
FROM monthly_counts
ORDER BY year_month, source
;

-- [{
--   "source": "prod_rap",
--   "year_month": "2024-11",
--   "row_count": "33253304",
--   "unique_individuals": "14158829"
-- }, {
--   "source": "new_te",
--   "year_month": "2024-12",
--   "row_count": "2636573",
--   "unique_individuals": "2636573"
-- }, {
--   "source": "prod_rap",
--   "year_month": "2024-12",
--   "row_count": "87159847",
--   "unique_individuals": "14589336"
-- }, {
--   "source": "raw_input",
--   "year_month": "2024-12",
--   "row_count": "2636573",
--   "unique_individuals": "2636573"
-- }, {
--   "source": "new_te",
--   "year_month": "2025-01",
--   "row_count": "1263111",
--   "unique_individuals": "1263111"
-- }, {
--   "source": "prod_rap",
--   "year_month": "2025-01",
--   "row_count": "116525272",
--   "unique_individuals": "15283996"
-- }, {
--   "source": "raw_input",
--   "year_month": "2025-01",
--   "row_count": "1263111",
--   "unique_individuals": "1263111"
-- }, {
--   "source": "new_te",
--   "year_month": "2025-02",
--   "row_count": "1138146",
--   "unique_individuals": "1138146"
-- }, {
--   "source": "prod_rap",
--   "year_month": "2025-02",
--   "row_count": "86250123",
--   "unique_individuals": "14353206"
-- }, {
--   "source": "raw_input",
--   "year_month": "2025-02",
--   "row_count": "1138146",
--   "unique_individuals": "1138146"
-- }, {
--   "source": "new_te",
--   "year_month": "2025-03",
--   "row_count": "1118821",
--   "unique_individuals": "1118821"
-- }, {
--   "source": "prod_rap",
--   "year_month": "2025-03",
--   "row_count": "110577193",
--   "unique_individuals": "14450610"
-- }, {
--   "source": "raw_input",
--   "year_month": "2025-03",
--   "row_count": "1118821",
--   "unique_individuals": "1118821"
-- }, {
--   "source": "new_te",
--   "year_month": "2025-04",
--   "row_count": "1104410",
--   "unique_individuals": "1104410"
-- }, {
--   "source": "prod_rap",
--   "year_month": "2025-04",
--   "row_count": "86617762",
--   "unique_individuals": "14559191"
-- }, {
--   "source": "raw_input",
--   "year_month": "2025-04",
--   "row_count": "1104410",
--   "unique_individuals": "1104410"
-- }, {
--   "source": "new_te",
--   "year_month": "2025-05",
--   "row_count": "1110508",
--   "unique_individuals": "1110508"
-- }, {
--   "source": "prod_rap",
--   "year_month": "2025-05",
--   "row_count": "91779118",
--   "unique_individuals": "14591264"
-- }, {
--   "source": "raw_input",
--   "year_month": "2025-05",
--   "row_count": "1110508",
--   "unique_individuals": "1110508"
-- }, {
--   "source": "new_te",
--   "year_month": "2025-06",
--   "row_count": "1118448",
--   "unique_individuals": "1118448"
-- }, {
--   "source": "prod_rap",
--   "year_month": "2025-06",
--   "row_count": "105195603",
--   "unique_individuals": "14538931"
-- }, {
--   "source": "raw_input",
--   "year_month": "2025-06",
--   "row_count": "1118448",
--   "unique_individuals": "1118448"
-- }, {
--   "source": "new_te",
--   "year_month": "2025-07",
--   "row_count": "1120832",
--   "unique_individuals": "1120832"
-- }, {
--   "source": "prod_rap",
--   "year_month": "2025-07",
--   "row_count": "102374566",
--   "unique_individuals": "14567035"
-- }, {
--   "source": "raw_input",
--   "year_month": "2025-07",
--   "row_count": "1120832",
--   "unique_individuals": "1120832"
-- }, {
--   "source": "new_te",
--   "year_month": "2025-08",
--   "row_count": "1139614",
--   "unique_individuals": "1139614"
-- }, {
--   "source": "prod_rap",
--   "year_month": "2025-08",
--   "row_count": "109137237",
--   "unique_individuals": "14603849"
-- }, {
--   "source": "raw_input",
--   "year_month": "2025-08",
--   "row_count": "1139614",
--   "unique_individuals": "1139614"
-- }, {
--   "source": "new_te",
--   "year_month": "2025-09",
--   "row_count": "1185650",
--   "unique_individuals": "1185650"
-- }, {
--   "source": "prod_rap",
--   "year_month": "2025-09",
--   "row_count": "90838778",
--   "unique_individuals": "14617418"
-- }, {
--   "source": "raw_input",
--   "year_month": "2025-09",
--   "row_count": "1185650",
--   "unique_individuals": "1185650"
-- }]