create or replace table `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_d1c_score_ending_tmp`
    CLUSTER BY individual_id
    OPTIONS ( labels=[("owner", "{{ params.owner }}"),("costcenter", "{{ params.costcenter }}")],expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
as
select * 
from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_monthly_rx`
union distinct
select * 
from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_archive_rx`
;

ALTER TABLE `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_d1c_score_ending_tmp`
    SET OPTIONS (expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 12 DAY))
;