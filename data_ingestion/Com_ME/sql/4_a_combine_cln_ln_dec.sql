create or replace table `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_d1b_score_ending_tmp`
    CLUSTER BY individual_id
    OPTIONS ( labels=[("owner", "{{ params.owner }}"),("costcenter", "{{ params.costcenter }}")],expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
as
 select * 
 from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_archive_clm_ln_bk1`
 union distinct
  select * 
 from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_archive_clm_ln_bk2`
  union distinct
 select * 
 from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_daily_clm_ln`
 union distinct
 select * 
 from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_monthly_clm_ln`
;