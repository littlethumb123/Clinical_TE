---------------------------------------------------
--                create views                   --
---------------------------------------------------
IF "{{DEC_DATASET}}" = "clin_analytics_dec_hcb_prod" THEN
    CREATE VIEW IF NOT EXISTS `anbc-hcb-prod.clin_analytics_share_hcb_prod.v_{{PREFIX}}_o3_score_ending_tmp`
    OPTIONS (labels=[('owner','{{OWNER}}'),('costcenter','{{COSTCENTER}}')]) AS
    select *
    from `{{FINAL_DB}}.{{PREFIX}}_o3_score_ending_tmp`
    ;

    CREATE VIEW IF NOT EXISTS `anbc-hcb-prod.clin_analytics_share_hcb_prod.v_inpatient_me_scores`
    OPTIONS (labels=[('owner','{{OWNER}}'),('costcenter','{{COSTCENTER}}')]) AS
    select *
    from `anbc-hcb-prod.clin_analytics_hcb_prod.inpatient_me_scores`
    ;



END IF;