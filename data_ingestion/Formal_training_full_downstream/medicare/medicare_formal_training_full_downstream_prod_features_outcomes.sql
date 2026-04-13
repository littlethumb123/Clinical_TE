/*==============================================================================
  MEDICARE FORMAL TRAINING FULL DOWNSTREAM
  PRODUCTION FEATURES + OUTCOMES MATERIALIZATION

  Purpose:
  Materialize the exact production feature columns used by the downstream
  evaluation notebook, joined to the formal-training Medicare outcome table by
  matching each outcome row to the most recent deduplicated production feature
  snapshot where `max_date <= index_dt - 90 days`.

  Input Tables:
  - Production features: anbc-hcb-prod.clin_analytics_hcb_prod.inpatient_me_features_history
  - Outcomes: edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_final_exp_round10_exp2b

  Output Table:
  - edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_formal_training_full_downstream_prod_features_outcomes_exp_round10_exp2b

  Notes:
  - Output is restricted to the date window used by the notebook:
      2024-07-01 through 2025-09-30
  - Production history is deduplicated to one row per `(individual_id,
    max_date)` using the latest `run_dt` before performing the as-of join.
  - Output `index_dt` remains the outcome business date; it is not replaced
    with the matched production snapshot date.
  - Only the baseline features, production embedding features, and outcome
    columns needed by the notebook are selected.
  - If you change the destination table name here, update the matching notebook
    constant in the downstream evaluation notebook.

  Team: Clinical & Social Determinants Intelligence (CSDI)
  Last Updated: April 2026

==============================================================================*/

CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_formal_training_full_downstream_prod_features_outcomes_exp_round10_exp2b`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"), ("cost_center", "13070")]
)
AS
WITH outcome_filtered AS (
    SELECT DISTINCT
        CAST(individual_id AS STRING) AS individual_id,
        index_dt,
        CAST(ip6 AS INT64) AS ip6,
        CAST(mon_6_include AS INT64) AS mon_6_include
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_outcome_6mo_final_exp_round10_exp2b`
    WHERE mon_6_include = 1
      AND index_dt BETWEEN DATE '2024-07-01' AND DATE '2025-09-30'
),
prod_deduped AS (
  SELECT * EXCEPT (row_num)
  FROM (
    SELECT
      prod.*,
      ROW_NUMBER() OVER (
        PARTITION BY prod.individual_id, prod.max_date
        ORDER BY prod.run_dt DESC
      ) AS row_num
    FROM `anbc-hcb-prod.clin_analytics_hcb_prod.inpatient_me_features_history` AS prod
    WHERE prod.max_date <= DATE_SUB(DATE '2025-09-30', INTERVAL 90 DAY)
  )
  WHERE row_num = 1
),
outcome_matched_snapshot AS (
  SELECT
    outcome.individual_id,
    outcome.index_dt,
    MAX(prod.max_date) AS matched_max_date
  FROM outcome_filtered AS outcome
  INNER JOIN prod_deduped AS prod
    ON CAST(prod.individual_id AS STRING) = outcome.individual_id
     AND prod.max_date <= DATE_SUB(outcome.index_dt, INTERVAL 90 DAY)
  GROUP BY outcome.individual_id, outcome.index_dt
)
SELECT
    CAST(prod.individual_id AS STRING) AS individual_id,
  outcome.index_dt AS index_dt,
    prod.camemhpd_aff,
    prod.camemhpd_alc,
    prod.camemhpd_cbd,
    prod.camemhpd_chf,
    prod.camemhpd_chr_flag,
    prod.camemhpd_cop,
    prod.camemhpd_crf,
    prod.camemhpd_cv_cond,
    prod.camemhpd_dia,
    prod.camemhpd_hyp,
    prod.camemhpd_ngd,
    prod.camemmedutilization_clm_ln_cnt,
    prod.camemmedutilization_er_clm_cnt,
    prod.camemmedutilization_uniq_dx_cd_cnt,
    prod.camemmedutilization_uniq_rev_cd_cnt,
    prod.camemmedcasedc1_ip_cnt_dc1,
    prod.camemmedcasedc1_ip_days_dc1,
    prod.camemmedcasedc2_ip_cnt_dc2,
    prod.camemmedcasedc2_ip_days_dc2,
    prod.camemmedcasedc3_ip_cnt_dc3,
    prod.camemeipdxdc1_dxc1085_cnt_dc1,
    prod.camemeipprcdc1_prc141_cnt_dc1,
    prod.camemeipprcdc1_prc155_cnt_dc1,
    prod.camemeipprcdc1_prc219_cnt_dc1,
    prod.camemeipprcdc1_prcc1102_cnt_dc1,
    prod.camemeipprcdc1_prcc1115_cnt_dc1,
    prod.camemrevenuedc3_rev730_cnt_dc3,
    prod.camemrevenuedc4_rev430_cnt_dc4,
    prod.camemerucdc1_erclm_cnt_dc1,
    prod.camemerucdc2_erclm_cnt_dc2,
    prod.camemrxclassutilizationdc5_anticonvulsants_misc_flag_dc5,
    prod.camemrxclassutilizationdc5_loop_diuretics_flag_dc5,
    prod.camemrxgrouputilizationdc1_antidepressants_days_dc1,
    prod.camemrxgrouputilizationdc1_corticosteroids_days_dc1,
    prod.camemrxgrouputilizationdc2_anticonvulsants_days_dc2,
    prod.camemrxgrouputilizationdc2_corticosteroids_days_dc2,
    prod.camemrxgrouputilizationdc3_anticonvulsants_flag_dc3,
    prod.camemspcclmdc1_spcclmwhos_cnt_dc1,
    prod.camemspcclmdc2_spcclmwhos_cnt_dc2,
    prod.camemspcofcdc3_spcd_cnt_dc3,
    prod.camemtgtptpdc2_cm_soe,
    prod.camemtgtptpdc5_cm_soe,
    prod.camemtxtnotesdc5_txt_end_dc5,
    prod.camemtxtnotesdc5_txt_short_dc5,
    prod.camemylm_ylm_homeagesourcer,
    prod.camemylm_ylm_orent,
    prod.camemylm_ylm_tw_hvalsecinv,
    prod.camemylm_ylm_tw_hvyinvtrad,
    prod.camemmbrshp_age65_74,
    prod.camemmbrshp_agenbr,
    prod.e_caperetdem21220,
    prod.e_caperetdem444,
    prod.emb6,
    prod.emb7,
    prod.emb14,
    prod.emb20,
    prod.emb23,
    prod.emb30,
    prod.emb31,
    prod.emb36,
    prod.emb43,
    prod.emb47,
    prod.emb49,
    prod.emb57,
    prod.emb64,
    prod.emb76,
    prod.emb81,
    prod.emb89,
    prod.emb94,
    prod.emb95,
    prod.emb98,
    prod.emb104,
    prod.emb110,
    prod.emb111,
    prod.emb124,
    prod.emb126,
    prod.emb127,
    prod.emb130,
    prod.emb131,
    prod.emb138,
    prod.emb173,
    prod.emb174,
    prod.emb177,
    prod.emb178,
    prod.emb188,
    prod.emb192,
    prod.emb195,
    prod.emb203,
    prod.emb205,
    prod.emb207,
    prod.emb212,
    prod.emb219,
    prod.emb224,
    prod.emb229,
    prod.emb230,
    prod.emb233,
    prod.emb238,
    prod.emb244,
    prod.emb253,
    prod.emb254,
    outcome.ip6,
    outcome.mon_6_include
  FROM outcome_filtered AS outcome
  INNER JOIN outcome_matched_snapshot AS matched
    ON outcome.individual_id = matched.individual_id
     AND outcome.index_dt = matched.index_dt
  INNER JOIN prod_deduped AS prod
    ON CAST(prod.individual_id AS STRING) = matched.individual_id
     AND prod.max_date = matched.matched_max_date
;


/*==============================================================================
  VALIDATION QUERIES
==============================================================================*/

-- 1. Row count, member count, date range, and target rate.
SELECT
    COUNT(*) AS row_count,
    COUNT(DISTINCT individual_id) AS unique_individuals,
    MIN(index_dt) AS min_index_dt,
    MAX(index_dt) AS max_index_dt,
    AVG(CAST(ip6 AS FLOAT64)) AS positive_rate
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_formal_training_full_downstream_prod_features_outcomes_exp_round10_exp2b`;

-- 2. Confirm one row per (individual_id, index_dt).
SELECT
    COUNT(*) AS duplicate_key_rows
FROM (
    SELECT
        individual_id,
        index_dt,
        COUNT(*) AS row_cnt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_formal_training_full_downstream_prod_features_outcomes_exp_round10_exp2b`
    GROUP BY individual_id, index_dt
    HAVING COUNT(*) > 1
);

-- 3. Preview first 5 rows.
SELECT *
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicare_formal_training_full_downstream_prod_features_outcomes_exp_round10_exp2b`
LIMIT 5;