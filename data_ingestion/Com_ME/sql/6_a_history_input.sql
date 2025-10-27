DECLARE history_len INT64 DEFAULT @historydays;
DECLARE embinit INT64 DEFAULT @emb_init;
IF  embinit = 1 AND  history_len !=0 THEN
    drop table if exists `{{ dag_run.conf["FINAL_DATASET"] }}.{{ dag_run.conf["prefix"] }}_o3_score_ending_history`;
    create table if not exists `{{ dag_run.conf["FINAL_DATASET"] }}.{{ dag_run.conf["prefix"] }}_o3_score_ending_history`
    partition by index_dt
    CLUSTER BY individual_id
    OPTIONS ( labels=[("owner", "{{ params.owner }}"),("costcenter", "{{ params.costcenter }}")],partition_expiration_days=(history_len)) as
    select *
    from `{{ dag_run.conf["FINAL_DATASET"] }}.{{ dag_run.conf["prefix"] }}_o3_score_ending`
    ;
END IF;

IF  embinit != 1 AND  history_len !=0 THEN
    alter table `{{ dag_run.conf["FINAL_DATASET"] }}.{{ dag_run.conf["prefix"] }}_o3_score_ending_history`
    SET OPTIONS (partition_expiration_days=(history_len));

    MERGE INTO `{{ dag_run.conf["FINAL_DATASET"] }}.{{ dag_run.conf["prefix"] }}_o3_score_ending_history` AS target
    USING `{{ dag_run.conf["FINAL_DATASET"] }}.{{ dag_run.conf["prefix"] }}_o3_score_ending` AS source
    ON target.individual_id=source.individual_id AND target.index_dt=source.index_dt
    WHEN MATCHED THEN
        UPDATE SET
        target.gender_cd = source.gender_cd,
        target.age_in_months = source.age_in_months,
        target.cd = source.cd,
        target.dt_cnt = source.dt_cnt
    WHEN NOT MATCHED THEN
        INSERT (individual_id, index_dt, gender_cd,age_in_months,cd,dt_cnt)
        VALUES (source.individual_id, source.index_dt, source.gender_cd,source.age_in_months,source.cd,source.dt_cnt);
END IF;