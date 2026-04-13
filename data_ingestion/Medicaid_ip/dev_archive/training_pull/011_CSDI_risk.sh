
#!/bin/bash
#---- BRFSS Date
bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_csdi`'

bq query \
--use_legacy_sql=false \
'
CREATE OR REPLACE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_csdi`
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
WITH yr AS (SELECT MAX(published_year) AS max_year FROM `anbc-hcb-prod.insights_share_hcb_prod.v_enriched_sd_cdc_brfss_ziplevel`)

SELECT
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
    , COALESCE(MAX(c.citizenship_index), 0) AS citizenship_index
    , COALESCE(MAX(c.education_index), 0) AS education_index
    , COALESCE(MAX(c.food_access), 0) AS food_access
    , COALESCE(MAX(c.health_access), 0) AS health_access
    , COALESCE(MAX(c.health_habits), 0) AS health_habits
    , COALESCE(MAX(c.housing_desert), 0) AS housing_desert
    , COALESCE(MAX(c.housing_ownership), 0) AS housing_ownership     
    , COALESCE(MAX(c.housing_quality), 0) AS housing_quality     
    , COALESCE(MAX(c.income_index), 0) AS income_index    
    , COALESCE(MAX(c.income_inequality), 0) AS income_inequality    
    , COALESCE(MAX(c.language_score), 0) AS language_score    
    , COALESCE(MAX(c.natural_disaster), 0) AS natural_disaster 
    , COALESCE(MAX(c.poverty_score), 0) AS poverty_score 
    , COALESCE(MAX(c.proactive_health), 0) AS proactive_health
    , COALESCE(MAX(c.racial_diversity), 0) AS racial_diversity
    , COALESCE(MAX(c.social_isolation), 0) AS social_isolation    
    , COALESCE(MAX(c.technology_access), 0) AS technology_access    
    , COALESCE(MAX(c.transport_access), 0) AS transport_access     
    , COALESCE(MAX(c.unemployment_index), 0) AS unemployment_index    
    , COALESCE(MAX(c.water_quality), 0) AS water_quality     
    , COALESCE(MAX(c.disability_score), 0) AS disability_score    
    , COALESCE(MAX(c.health_infra), 0) AS health_infra    
    , COALESCE(MAX(c.social_risk_score), 0) AS social_risk_score    
FROM (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt, bgfips FROM `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_geoid`) AS st
LEFT JOIN 
    (
     SELECT * FROM `edp-prod-storage.edp_ent_sdoheir_srcv.risk_index_block_group_historical_data`
         WHERE effective_year = CAST('$SDOH_YEAR' AS INT)
    ) AS c
    ON TRIM(st.bgfips)=TRIM(c.bgfips)
GROUP BY st.asdb_member_key, st.asdb_plan_key, st.index_dt
'
