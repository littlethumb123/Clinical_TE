create table if not exists `{{FINAL_DB}}.{{PREFIX}}_o3_score_ending_history`
partition by run_dt
 OPTIONS ( labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")])
as
    select *, date('{current_dt}') as run_dt
    from `{{DEC_TARGET_DB}}.{{PREFIX}}_o3_score_ending_tmps`
;

DELETE FROM `{{FINAL_DB}}.{{PREFIX}}_o3_score_ending_history`
WHERE run_dt=date('{current_dt}');
INSERT INTO `{{FINAL_DB}}.{{PREFIX}}_o3_score_ending_history`
SELECT * FROM `{{FINAL_DB}}.{{PREFIX}}_o3_score_ending`;

create table if not exists `{{FINAL_DB}}.{{PREFIX}}_o3_score_ending_history`
partition by run_dt
 OPTIONS ( labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")])
as
    select *, date('{current_dt}') as run_dt
    from `{{DEC_TARGET_DB}}.{{PREFIX}}_o3_score_ending_tmps`
;

DELETE FROM `{{FINAL_DB}}.{{PREFIX}}_o3_score_ending_history`
WHERE run_dt=date('{current_dt}');
INSERT INTO `{{FINAL_DB}}.{{PREFIX}}_o3_score_ending_history`
SELECT * FROM `{{FINAL_DB}}.{{PREFIX}}_o3_score_ending`;