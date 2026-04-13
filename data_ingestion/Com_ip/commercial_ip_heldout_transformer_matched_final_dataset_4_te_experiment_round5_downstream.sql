-- ============================================================================
-- 1. SUMMARY STATISTICS: Overall discrepancy overview
-- ============================================================================
-- WITH joined_data AS (
--     SELECT 
--         t1.individual_id,
--         t1.index_dt AS features_index_dt,
--         t2.index_dt AS heldout_index_dt,
--         DATE_DIFF(t2.index_dt, t1.index_dt, DAY) AS index_dt_diff_days
--     FROM 
--         `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_final_dataset_4_te_experiment` AS t1
--     INNER JOIN 
--         `edp-prod-storage.edp_ent_sdoheir_cns.a964286_commercial_heldout_transformer_input_4_te_experiment_round_5` AS t2
--         ON t1.individual_id = t2.individual_id
-- )
-- SELECT 
--     COUNT(*) AS total_members,
--     COUNTIF(features_index_dt = heldout_index_dt) AS matching_index_dt,
--     COUNTIF(features_index_dt != heldout_index_dt) AS mismatched_index_dt,
--     ROUND(100.0 * COUNTIF(features_index_dt = heldout_index_dt) / COUNT(*), 2) AS pct_matching,
--     ROUND(100.0 * COUNTIF(features_index_dt != heldout_index_dt) / COUNT(*), 2) AS pct_mismatched,
--     AVG(ABS(index_dt_diff_days)) AS avg_abs_diff_days,
--     MAX(ABS(index_dt_diff_days)) AS max_abs_diff_days,
--     MIN(index_dt_diff_days) AS min_diff_days,
--     MAX(index_dt_diff_days) AS max_diff_days
-- FROM joined_data;

-- WITH joined_data AS (
--     SELECT 
--         t1.individual_id,
--         t1.index_dt AS features_index_dt,
--         t2.index_dt AS heldout_index_dt,
--         DATE_DIFF(t2.index_dt, t1.index_dt, DAY) AS index_dt_diff_days
--     FROM 
--         `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_final_dataset_4_te_experiment` AS t1
--     INNER JOIN 
--         `edp-prod-storage.edp_ent_sdoheir_cns.a964286_commercial_heldout_transformer_input_4_te_experiment_round_5` AS t2
--         ON t1.individual_id = t2.individual_id
-- )
-- SELECT 
--     CASE 
--         WHEN index_dt_diff_days = 0 THEN '0: Exact match'
--         WHEN ABS(index_dt_diff_days) BETWEEN 1 AND 7 THEN '1: Within 1 week'
--         WHEN ABS(index_dt_diff_days) BETWEEN 8 AND 30 THEN '2: 1 week - 1 month'
--         WHEN ABS(index_dt_diff_days) BETWEEN 31 AND 90 THEN '3: 1-3 months'
--         WHEN ABS(index_dt_diff_days) BETWEEN 91 AND 180 THEN '4: 3-6 months'
--         WHEN ABS(index_dt_diff_days) BETWEEN 181 AND 365 THEN '5: 6-12 months'
--         ELSE '6: More than 1 year'
--     END AS discrepancy_bucket,
--     COUNT(*) AS member_count,
--     ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct_of_total,
--     AVG(index_dt_diff_days) AS avg_diff_days,
--     MIN(index_dt_diff_days) AS min_diff_days,
--     MAX(index_dt_diff_days) AS max_diff_days
-- FROM joined_data
-- GROUP BY discrepancy_bucket
-- ORDER BY discrepancy_bucket;



CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_ccommercial_ip_heldout_transformer_matched_final_dataset_4_te_experiment_round5_downstream` 
OPTIONS (
    labels=[("owner", "daniel_xing_cvshealth_com"), ("costcenter", "13070")],
    description="Combined features and heldout table (not in transformer training dataset) joined by individual_id and index_dt for commercial IP downstream tasks - round 5"
) AS
SELECT 
b.*
from 
`edp-prod-storage.edp_ent_sdoheir_cns.a964286_commercial_heldout_transformer_input_4_te_experiment_round_5` a
left join
`edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_final_dataset_4_te_experiment` b
on a.individual_id = b.individual_id and a.index_dt = b.index_dt
where b.mon_6_include = 1