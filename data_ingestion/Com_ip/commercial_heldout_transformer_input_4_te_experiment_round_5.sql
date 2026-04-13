DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_commercial_heldout_transformer_input_4_te_experiment_round_5`;

CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_commercial_heldout_transformer_input_4_te_experiment_round_5`
CLUSTER BY individual_id
OPTIONS (
    labels=[("owner", "daniel_xing_cvshealth_com"), ("costcenter", "13070")],
    description="Commercial members NOT used in pretraining - transformer input features for downstream evaluation embedding generation"
) AS
SELECT 
    full_data.individual_id,
    full_data.lob,
    full_data.index_dt,
    full_data.gender_cd,
    full_data.age_in_months,
    full_data.cd,
    full_data.target,
    full_data.dt_cnt
FROM 
    `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_ending` AS full_data
-- Only include members that also exist in the features table
INNER JOIN 
    `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_final_dataset_4_te_experiment` AS features
    ON full_data.individual_id = features.individual_id
WHERE 
    -- Only Commercial LOB members
    full_data.lob = 'Commercial'
    -- Exclude ALL members who were in pretraining (any LOB)
    AND NOT EXISTS (
        SELECT 1 
        FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_10pct_sample` AS pretrain
        WHERE pretrain.individual_id = full_data.individual_id
    )
;