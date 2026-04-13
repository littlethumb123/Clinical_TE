-- Variables must be declared at the start
DECLARE max_dims INT64;
DECLARE sql_query STRING;

-- Get maximum embedding dimensions from records in the date range
SET max_dims = (
  SELECT MAX(ARRAY_LENGTH(embs)) 
  FROM `anbc-hcb-prod.clin_analytics_hcb_prod.enhanced_rap_cp_emb_history`
  WHERE index_dt BETWEEN '2023-01-01' AND '2023-12-31'
);

-- Build dynamic SQL string with column list
SET sql_query = (
  SELECT CONCAT(
    'CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_commercial_legacy_embs_2023_4_te_experiment_round_5` AS SELECT individual_id, index_dt, ',
    STRING_AGG(
      FORMAT('embs[OFFSET(%d)] AS embs_%d_legacy', pos, pos + 1), 
      ', ' ORDER BY pos
    ),
    " FROM `anbc-hcb-prod.clin_analytics_hcb_prod.enhanced_rap_cp_emb_history` WHERE index_dt BETWEEN '2023-01-01' AND '2023-12-31' ORDER BY individual_id, index_dt"
  )
  FROM UNNEST(GENERATE_ARRAY(0, max_dims - 1)) AS pos
);

-- Execute the dynamic SQL
EXECUTE IMMEDIATE sql_query;