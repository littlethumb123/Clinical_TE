DECLARE v_index_dt DATE;
SET v_index_dt = (
    SELECT CAST(creation_time AS DATE)
    FROM `anbc-hcb-prod.clin_analytics_hcb_prod.INFORMATION_SCHEMA.TABLES`
    WHERE table_name = 'dly_clm_ip_transformer_me_mpp_gpu_embed_scores'
);


IF "{{DEC_DATASET}}" = "clin_analytics_dec_hcb_prod" THEN

    CREATE TABLE IF NOT EXISTS `{{FINAL_DB}}.dly_clm_ip_transformer_me_emb_history`
    PARTITION BY index_dt
    OPTIONS (labels=[("owner", "{{OWNER}}"), ("costcenter", "{{COSTCENTER}}")]) AS
    SELECT
        *,
        v_index_dt AS index_dt
    FROM `anbc-hcb-prod.clin_analytics_hcb_prod.dly_clm_ip_transformer_me_mpp_gpu_embed_scores`;

    DELETE FROM `{{FINAL_DB}}.dly_clm_ip_transformer_me_emb_history`
    WHERE index_dt = v_index_dt;

    INSERT INTO `{{FINAL_DB}}.dly_clm_ip_transformer_me_emb_history`
    SELECT
        *,
        v_index_dt AS index_dt
    FROM `anbc-hcb-prod.clin_analytics_hcb_prod.dly_clm_ip_transformer_me_mpp_gpu_embed_scores`;
    
    -- create or replace view `anbc-hcb-prod.clin_analytics_share_hcb_prod.v_dly_clm_ip_transformer_me_emb_history`
    -- OPTIONS (labels=[("owner", "{{OWNER}}"), ("costcenter", "{{COSTCENTER}}")]) AS
    -- select * from `{{FINAL_DB}}.dly_clm_ip_transformer_me_emb_history`
    -- ;
END IF;