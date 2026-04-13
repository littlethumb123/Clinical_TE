#!/bin/bash
#---- ACS Data
bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_acs`'

bq query \
--use_legacy_sql=false \
'
CREATE OR REPLACE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_acs`
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
WITH yr AS (
    SELECT MAX(published_year) AS max_year
    FROM `anbc-hcb-prod.insights_share_hcb_prod.v_enriched_sd_acs_summary_by_block_code`
    WHERE LENGTH(block_code)=11
)

SELECT
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
    , COALESCE(MAX(b.social_risk_score), 0) AS social_risk_score
    , COALESCE(MAX(b.sdi_score), 0) AS sdi_score
    , COALESCE(MAX(b.svi_score), 0) AS svi_score
    , COALESCE(MAX(b.adi_score), 0) AS adi_score
FROM (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `'$ST'`) AS st
LEFT JOIN 
    `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_geoid` AS id
    ON st.asdb_member_key=id.asdb_member_key
       AND st.asdb_plan_key=id.asdb_plan_key
LEFT JOIN 
    (SELECT
         *
     FROM `edp-prod-storage.edp_ent_sdoheir_srcv.srs_acs_block_group_allscores_historical_data` AS a
     WHERE effective_year = 2022) AS b
    ON id.ctfips = b.ctfips
GROUP BY
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
'
