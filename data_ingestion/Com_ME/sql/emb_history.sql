DECLARE is_init INT64 DEFAULT @isinit;
IF  is_init = 1 THEN

CREATE OR REPLACE TABLE {{ dag_run.conf["FINAL_DATASET"] }}.{{ dag_run.conf["prefix"] }}_embed_scores_history
PARTITION BY index_dt
CLUSTER BY individual_id
OPTIONS ( labels=[("owner", "{{ params.owner }}"),("costcenter", "{{ params.costcenter }}")]) AS
select individual_id,date(index_dt) as index_dt,embs
from {{ dag_run.conf["FINAL_DATASET"] }}.{{ dag_run.conf["prefix"] }}_embed_scores;

ELSE 

MERGE INTO `{{ dag_run.conf["FINAL_DATASET"] }}.{{ dag_run.conf["prefix"] }}_embed_scores_history` AS target
USING `{{ dag_run.conf["FINAL_DATASET"] }}.{{ dag_run.conf["prefix"] }}_embed_scores` AS source
ON target.individual_id=source.individual_id AND target.index_dt=date(source.index_dt)
WHEN MATCHED THEN
    UPDATE SET
    target.embs = source.embs
WHEN NOT MATCHED THEN
    INSERT (individual_id, index_dt, embs)
    VALUES (source.individual_id, date(source.index_dt), source.embs);

END IF;