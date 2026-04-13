/*==============================================================================
  COMMERCIAL FORMAL TRAINING FULL DOWNSTREAM
  PREJOINED TABULAR + EMBEDDINGS MATERIALIZATION

  Purpose:
    Materialize the two modeling tables consumed directly by the Commercial
    downstream evaluation notebook. Each output keeps the tabular feature/outcome
    row as the anchor, but the attachment logic differs by embedding source:

            New TE   : exact join on (individual_id, index_dt)
            Prod RAP : latest embedding snapshot where
                                 embedding.index_dt <= outcome.index_dt - 90 days

  Input Tables:
  - Tabular features + outcomes:
      edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_final_dataset_4_te_formal_evaluation_20241120_20250930
  - New TE embeddings:
      edp-prod-storage.edp_ent_sdoheir_cns.a964286_exp_round10_exp2b_commercial_embeddings_20241120_20250930
  - Production RAP embeddings (wide format):
      edp-prod-storage.edp_ent_sdoheir_cns.enhanced_rap_cp_emb_history_wide_4_te_fromal_eval

  Output Tables:
  - edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_formal_training_full_downstream_new_te_features_outcomes_exp_round10_exp2b
  - edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_formal_training_full_downstream_prod_rap_features_outcomes_exp_round10_exp2b

  Notes:
  - Output is restricted to the notebook evaluation window:
      2024-11-20 through 2025-09-30
  - Output index_dt remains the tabular outcome business date.
    - matched_embedding_index_dt records the actual embedding snapshot chosen by
        the join and should be treated as an audit column, not a model feature.
    - New TE embeddings are generated on the same business anchor date as the
        formal downstream tabular cohort, so they should be joined exactly.
    - Production RAP embeddings behave like a dense history table and therefore
        retain the 90-day as-of lookup.
  - Both outputs preserve all tabular columns and all embedding columns.

  Team: Clinical & Social Determinants Intelligence (CSDI)
  Last Updated: April 2026

==============================================================================*/

DECLARE min_index_dt DATE DEFAULT DATE '2024-11-20';
DECLARE max_index_dt DATE DEFAULT DATE '2025-09-30';
DECLARE buffer_days INT64 DEFAULT 90;


/*==============================================================================
    1. NEW TE EMBEDDINGS + TABULAR FEATURES/OUTCOMES
         EXACT JOIN ON (individual_id, index_dt)
==============================================================================*/

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_formal_training_full_downstream_new_te_features_outcomes_exp_round10_exp2b`
AS
WITH outcome_filtered AS (
    SELECT
        t.* REPLACE(
            CAST(t.individual_id AS STRING) AS individual_id,
            CAST(t.index_dt AS DATE) AS index_dt
        )
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_final_dataset_4_te_formal_evaluation_20241120_20250930` AS t
    WHERE CAST(t.index_dt AS DATE) BETWEEN min_index_dt AND max_index_dt
),
new_te_deduped AS (
    SELECT * EXCEPT (row_num)
    FROM (
        SELECT
            emb.* REPLACE(
                CAST(emb.individual_id AS STRING) AS individual_id,
                CAST(emb.index_dt AS DATE) AS index_dt
            ),
            ROW_NUMBER() OVER (
                PARTITION BY CAST(emb.individual_id AS STRING), CAST(emb.index_dt AS DATE)
                ORDER BY CAST(emb.index_dt AS DATE) DESC
            ) AS row_num
        FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_exp_round10_exp2b_commercial_embeddings_20241120_20250930` AS emb
        WHERE CAST(emb.index_dt AS DATE) BETWEEN min_index_dt AND max_index_dt
    )
    WHERE row_num = 1
)
SELECT
    outcome.*,
    emb.index_dt AS matched_embedding_index_dt,
    emb.* EXCEPT (individual_id, index_dt)
FROM outcome_filtered AS outcome
INNER JOIN new_te_deduped AS emb
    ON emb.individual_id = outcome.individual_id
   AND emb.index_dt = outcome.index_dt
;


/*==============================================================================
    2. PRODUCTION RAP EMBEDDINGS + TABULAR FEATURES/OUTCOMES
         90-DAY AS-OF JOIN
==============================================================================*/

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_formal_training_full_downstream_prod_rap_features_outcomes_exp_round10_exp2b`
AS
WITH outcome_filtered AS (
    SELECT
        t.* REPLACE(
            CAST(t.individual_id AS STRING) AS individual_id,
            CAST(t.index_dt AS DATE) AS index_dt
        )
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_final_dataset_4_te_formal_evaluation_20241120_20250930` AS t
    WHERE CAST(t.index_dt AS DATE) BETWEEN min_index_dt AND max_index_dt
),
prod_rap_deduped AS (
    SELECT * EXCEPT (row_num)
    FROM (
        SELECT
            emb.* REPLACE(
                CAST(emb.individual_id AS STRING) AS individual_id,
                CAST(emb.index_dt AS DATE) AS index_dt
            ),
            ROW_NUMBER() OVER (
                PARTITION BY CAST(emb.individual_id AS STRING), CAST(emb.index_dt AS DATE)
                ORDER BY CAST(emb.index_dt AS DATE) DESC
            ) AS row_num
        FROM `edp-prod-storage.edp_ent_sdoheir_cns.enhanced_rap_cp_emb_history_wide_4_te_fromal_eval` AS emb
        WHERE CAST(emb.index_dt AS DATE) <= DATE_SUB(max_index_dt, INTERVAL buffer_days DAY)
    )
    WHERE row_num = 1
),
outcome_matched_snapshot AS (
    SELECT
        outcome.individual_id,
        outcome.index_dt,
        MAX(emb.index_dt) AS matched_embedding_index_dt
    FROM outcome_filtered AS outcome
    INNER JOIN prod_rap_deduped AS emb
        ON emb.individual_id = outcome.individual_id
       AND emb.index_dt <= DATE_SUB(outcome.index_dt, INTERVAL buffer_days DAY)
    GROUP BY outcome.individual_id, outcome.index_dt
)
SELECT
    outcome.*,
    matched.matched_embedding_index_dt,
    emb.* EXCEPT (individual_id, index_dt)
FROM outcome_filtered AS outcome
INNER JOIN outcome_matched_snapshot AS matched
    ON outcome.individual_id = matched.individual_id
   AND outcome.index_dt = matched.index_dt
INNER JOIN prod_rap_deduped AS emb
    ON emb.individual_id = matched.individual_id
   AND emb.index_dt = matched.matched_embedding_index_dt
;


/*==============================================================================
  3. VALIDATION QUERIES
==============================================================================*/

-- New TE joined table: row count, member count, date range, target rate.
-- Expected after fix: new_te row_count should closely match the tabular outcome table.
SELECT
    'new_te' AS table_name,
    COUNT(*) AS row_count,
    COUNT(DISTINCT individual_id) AS unique_individuals,
    MIN(index_dt) AS min_index_dt,
    MAX(index_dt) AS max_index_dt,
    MIN(matched_embedding_index_dt) AS min_matched_embedding_index_dt,
    MAX(matched_embedding_index_dt) AS max_matched_embedding_index_dt,
    AVG(CAST(ip6 AS FLOAT64)) AS positive_rate
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_formal_training_full_downstream_new_te_features_outcomes_exp_round10_exp2b`

UNION ALL

SELECT
    'prod_rap' AS table_name,
    COUNT(*) AS row_count,
    COUNT(DISTINCT individual_id) AS unique_individuals,
    MIN(index_dt) AS min_index_dt,
    MAX(index_dt) AS max_index_dt,
    MIN(matched_embedding_index_dt) AS min_matched_embedding_index_dt,
    MAX(matched_embedding_index_dt) AS max_matched_embedding_index_dt,
    AVG(CAST(ip6 AS FLOAT64)) AS positive_rate
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_formal_training_full_downstream_prod_rap_features_outcomes_exp_round10_exp2b`
;

-- Confirm one row per (individual_id, index_dt) in each joined table.
SELECT
    table_name,
    COUNT(*) AS duplicate_key_rows
FROM (
    SELECT
        'new_te' AS table_name,
        individual_id,
        index_dt,
        COUNT(*) AS row_cnt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_formal_training_full_downstream_new_te_features_outcomes_exp_round10_exp2b`
    GROUP BY individual_id, index_dt
    HAVING COUNT(*) > 1

    UNION ALL

    SELECT
        'prod_rap' AS table_name,
        individual_id,
        index_dt,
        COUNT(*) AS row_cnt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_formal_training_full_downstream_prod_rap_features_outcomes_exp_round10_exp2b`
    GROUP BY individual_id, index_dt
    HAVING COUNT(*) > 1
)
GROUP BY table_name
;

-- Preview first 5 rows from each output.
SELECT 'new_te' AS table_name, t.*
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_formal_training_full_downstream_new_te_features_outcomes_exp_round10_exp2b` AS t
LIMIT 5
;

SELECT 'prod_rap' AS table_name, t.*
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_formal_training_full_downstream_prod_rap_features_outcomes_exp_round10_exp2b` AS t
LIMIT 5
;