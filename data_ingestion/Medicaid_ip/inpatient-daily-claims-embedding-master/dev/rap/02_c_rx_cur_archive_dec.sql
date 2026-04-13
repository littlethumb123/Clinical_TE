CREATE TEMPORARY FUNCTION months_between_hive(date1 DATE, date2 DATE) AS (
  FLOOR((
    EXTRACT(YEAR FROM date1) - EXTRACT(YEAR FROM date2)
  ) * 12 + (
    EXTRACT(MONTH FROM date1) - EXTRACT(MONTH FROM date2)
  ) + (
    EXTRACT(DAY FROM date1) - EXTRACT(DAY FROM date2)
  ) / 30.436875)
);



create or replace table `{{DEC_TARGET_DB}}.{{PREFIX}}_rx_current`
    OPTIONS ( labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")])
as
with birth_dt as (
    select a.individual_id,
            a.index_dt,
           a.member_id,
           a.gender_cd,
           b.birth_dt
    from `{{TARGET_DB}}.{{PREFIX}}_member_base` a
    left join `edp-prod-hcbstorage.edp_hcb_core_cnsv.UNMASK_MEMBER` b
            on a.member_id=b.member_id
)
    select
        base.individual_id,
        base.index_dt,
        base.member_id,
        case
         when trim(base.gender_cd) = 'M' then 1
         when trim(base.gender_cd) = 'F' then 0
         else 2 end                                           as gender_cd,
        months_between_hive(date(rx.disp_dt), date(base.birth_dt)) as age_in_months,
         rx.disp_dt                                               as dt,
         concat('gpi', SUBSTR(trim(rx.adjudicated_gpi_cd), 1, 4)) as gpi4
    from birth_dt base
    inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.UNMSK_RX_CLAIM_DTL` rx
        on base.member_id = rx.member_id
    where base.index_dt > rx.disp_dt
        and DATE_ADD(rx.disp_dt, INTERVAL 36 MONTH) > base.index_dt
        and rx.disp_dt >= date('2020-01-01') and rx.disp_dt <= (select date(index_dt) from `{{TARGET_DB}}.{{PREFIX}}_member_base` limit 1)
;

create or replace table `{{DEC_TARGET_DB}}.{{PREFIX}}_rx_y2020`
    OPTIONS ( labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")])
as
with birth_dt as (
    select a.individual_id,
            a.index_dt,
           a.member_id,
           a.gender_cd,
           b.birth_dt
    from `{{TARGET_DB}}.{{PREFIX}}_member_base` a
    left join `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEMBER_ARC` b
            on a.member_id=b.member_id
)
    select
        base.individual_id,
        base.index_dt,
        base.member_id,
        case
            when trim(base.gender_cd) = 'M' then 1
            when trim(base.gender_cd) = 'F' then 0
            else 2 end                                                           as gender_cd,
        months_between_hive(date(rx.disp_dt), date(base.birth_dt)) as age_in_months,
        rx.disp_dt                                                               as dt,
        concat('gpi', SUBSTR(trim(rx.adjudicated_gpi_cd), 1, 4))                 as gpi4
    from birth_dt base
    inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.UNMSK_RX_CLAIM_DTL_Y2020` rx
        on base.member_id = rx.member_id
    where base.index_dt > rx.disp_dt
        and DATE_ADD(rx.disp_dt, INTERVAL 36 MONTH) > base.index_dt
        and rx.disp_dt >= date('2018-01-01')
        and rx.disp_dt <= date('2019-12-31')
;


create or replace table `{{DEC_TARGET_DB}}.{{PREFIX}}_rx_y2018`
    OPTIONS ( labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")])
as
with birth_dt as (
    select a.individual_id,
           a.index_dt,
           a.member_id,
           a.gender_cd,
           b.birth_dt
    from `{{TARGET_DB}}.{{PREFIX}}_member_base` a
    left join `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEMBER_ARC` b
            on a.member_id=b.member_id
)
    select
        base.individual_id,
        base.index_dt,
        base.member_id,
        case
            when trim(base.gender_cd) = 'M' then 1
            when trim(base.gender_cd) = 'F' then 0
            else 2 end                                                           as gender_cd,
        months_between_hive(date(rx.disp_dt), date(base.birth_dt)) as age_in_months,
        rx.disp_dt                                                               as dt,
        concat('gpi', SUBSTR(trim(rx.adjudicated_gpi_cd), 1, 4))                 as gpi4
    from birth_dt base
    inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.UNMSK_RX_CLAIM_DTL_ARCHIVE` rx
            on base.member_id = rx.member_id
    where base.index_dt > rx.disp_dt
      and DATE_ADD(rx.disp_dt, INTERVAL 36 MONTH) > base.index_dt
      and rx.disp_dt >= date('2016-01-01')
      and rx.disp_dt <= date('2017-12-31')
;


create or replace table `{{DEC_TARGET_DB}}.{{PREFIX}}_d1c_score_ending_tmp`
    OPTIONS ( labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")])
as
select * from `{{DEC_TARGET_DB}}.{{PREFIX}}_rx_current`
union distinct
select * from `{{DEC_TARGET_DB}}.{{PREFIX}}_rx_y2020`
union distinct
select * from `{{DEC_TARGET_DB}}.{{PREFIX}}_rx_y2018`

;

ALTER TABLE `{{DEC_TARGET_DB}}.{{PREFIX}}_rx_current`
    SET OPTIONS (expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
;

ALTER TABLE `{{DEC_TARGET_DB}}.{{PREFIX}}_rx_y2020`
    SET OPTIONS (expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
;

ALTER TABLE `{{DEC_TARGET_DB}}.{{PREFIX}}_rx_y2018`
    SET OPTIONS (expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
;

-- ALTER TABLE `{{TARGET_DB}}.{{PREFIX}}_d1c_score_ending_tmp`
--     SET OPTIONS (expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
-- ;