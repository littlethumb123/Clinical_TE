CREATE TEMPORARY FUNCTION months_between_hive(date1 DATE, date2 DATE) AS (
  FLOOR((
    EXTRACT(YEAR FROM date1) - EXTRACT(YEAR FROM date2)
  ) * 12 + (
    EXTRACT(MONTH FROM date1) - EXTRACT(MONTH FROM date2)
  ) + (
    EXTRACT(DAY FROM date1) - EXTRACT(DAY FROM date2)
  ) / 30.436875)
);

-- drop table if exists `{{DEC_TARGET_DB}}.{{PREFIX}}_archive_edw_clms_y2021`;
-- create table `{{DEC_TARGET_DB}}.{{PREFIX}}_archive_edw_clms_y2021`
-- OPTIONS (labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")]) as
-- with birth_dt as (
--     select a.individual_id,
--            a.member_id,
--            a.index_dt,
--            a.gender_cd,
--            b.birth_dt
--     from `{{TARGET_DB}}.{{PREFIX}}_member_base` a
--     left join `edp-prod-hcbstorage.edp_hcb_core_cnsv.UNMASK_MEMBER` b
--             on a.member_id=b.member_id
-- )
--     select
--         base.individual_id,
--         base.member_id,
--         clm.claim_line_id,
--         clm.srv_start_dt,
--         case when (clm.days_cnt is null or clm.days_cnt <0) then 99 when clm.days_cnt > 10 then 11 else clm.days_cnt end as days_cnt,
--         case when trim(base.gender_cd)='M' then 1 when trim(base.gender_cd)='F' then 0 else 2 end as gender_cd,
--         months_between_hive(date(clm.srv_start_dt), date(base.birth_dt)) as age_in_months,
--          case when (clm.revenue_cd is null or trim(clm.revenue_cd)='') then null else trim(clm.revenue_cd) end as revenue_cd,
--         case when (clm.hcfa_plc_srv_cd is null or trim(clm.hcfa_plc_srv_cd)='') then null else trim(clm.hcfa_plc_srv_cd) end as hcfa_plc_srv_cd,
--         case when (clm.src_specialty_cd is null or trim(clm.src_specialty_cd)='') then null else trim(clm.src_specialty_cd) end as src_specialty_cd,
--         case when (clm.prcdr_cd is null or trim(clm.prcdr_cd)='') then null else trim(clm.prcdr_cd) end as prcdr_cd,
--         case when (icd_prc.icd9_prcdr_cd is null or trim(icd_prc.icd9_prcdr_cd)='') then null else upper(trim(icd_prc.icd9_prcdr_cd)) end as icd9_prcdr_cd
--     from birth_dt base
--     inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_CLAIM_LINE_Y2021` clm
--             on base.member_id=clm.member_id
--     left join `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_PRCD_ARC` icd_prc
--            on clm.claim_line_id=icd_prc.claim_line_id
--     where base.index_dt > clm.received_dt and base.index_dt > clm.srv_start_dt
--         and DATE_ADD(clm.srv_start_dt, INTERVAL 36 MONTH) > base.index_dt
--         and clm.duplicate_ind='N' and clm.summarized_srv_ind='Y'
--         and clm.srv_start_dt >= '2019-01-01' and clm.srv_start_dt <= '2021-12-31'
-- ;

drop table if exists `{{DEC_TARGET_DB}}.{{PREFIX}}_archive_edw_clms_y2020`;
create table `{{DEC_TARGET_DB}}.{{PREFIX}}_archive_edw_clms_y2020`
OPTIONS (labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")]) as
with birth_dt as (
    select a.individual_id,
           a.member_id,
           a.index_dt,
           a.gender_cd,
           b.birth_dt
    from `{{TARGET_DB}}.{{PREFIX}}_member_base` a
    left join `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEMBER_ARC` b
            on a.member_id=b.member_id
)
     select
         base.individual_id,
         base.member_id,
         clm.claim_line_id,
         clm.srv_start_dt,
         case
             when (clm.days_cnt is null or clm.days_cnt <0) then 99
             when clm.days_cnt > 10 then 11
             else clm.days_cnt
             end as days_cnt,
         case
             when trim(base.gender_cd)='M' then 1
             when trim(base.gender_cd)='F' then 0
             else 2
             end as gender_cd,
         months_between_hive(date(clm.srv_start_dt), date(base.birth_dt)) as age_in_months,
          case when (clm.revenue_cd is null or trim(clm.revenue_cd)='') then null else trim(clm.revenue_cd) end as revenue_cd,
        case when (clm.hcfa_plc_srv_cd is null or trim(clm.hcfa_plc_srv_cd)='') then null else trim(clm.hcfa_plc_srv_cd) end as hcfa_plc_srv_cd,
        case when (clm.src_specialty_cd is null or trim(clm.src_specialty_cd)='') then null else trim(clm.src_specialty_cd) end as src_specialty_cd,
        case when (clm.prcdr_cd is null or trim(clm.prcdr_cd)='') then null else trim(clm.prcdr_cd) end as prcdr_cd,
        case when (icd_prc.icd9_prcdr_cd is null or trim(icd_prc.icd9_prcdr_cd)='') then null else upper(trim(icd_prc.icd9_prcdr_cd)) end as icd9_prcdr_cd
     from birth_dt base
     inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_CLAIM_LINE_Y2020` clm
             on base.member_id=clm.member_id
     left join `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_PRCD_ARC` icd_prc
            on clm.claim_line_id=icd_prc.claim_line_id
     where base.index_dt > clm.received_dt and base.index_dt > clm.srv_start_dt
       and DATE_ADD(clm.srv_start_dt, INTERVAL 36 MONTH) > base.index_dt
       and clm.duplicate_ind='N' and clm.summarized_srv_ind='Y'
       and clm.srv_start_dt >= '2018-01-01' and clm.srv_start_dt <= '2019-12-31'
;

-- drop table if exists `{{DEC_TARGET_DB}}.{{PREFIX}}_archive_edw_clms_y2019`;
-- create table `{{DEC_TARGET_DB}}.{{PREFIX}}_archive_edw_clms_y2019`
-- OPTIONS (labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")]) as
-- with birth_dt as (
--     select a.individual_id,
--            a.member_id,
--            a.index_dt,
--            a.gender_cd,
--            b.birth_dt
--     from `{{TARGET_DB}}.{{PREFIX}}_member_base` a
--     left join `edp-prod-hcbstorage.edp_hcb_core_cnsv.UNMASK_MEMBER` b
--             on a.member_id=b.member_id
-- )
--      select
--          base.individual_id,
--          base.member_id,
--          clm.claim_line_id,
--          clm.srv_start_dt,
--          case
--              when (clm.days_cnt is null or clm.days_cnt <0) then 99
--              when clm.days_cnt > 10 then 11
--              else clm.days_cnt
--              end as days_cnt,
--          case
--              when trim(base.gender_cd)='M' then 1
--              when trim(base.gender_cd)='F' then 0
--              else 2
--              end as gender_cd,
--          months_between_hive(date(clm.srv_start_dt), date(base.birth_dt)) as age_in_months,
--          case when (clm.revenue_cd is null or trim(clm.revenue_cd)='') then null else trim(clm.revenue_cd) end as revenue_cd,
--         case when (clm.hcfa_plc_srv_cd is null or trim(clm.hcfa_plc_srv_cd)='') then null else trim(clm.hcfa_plc_srv_cd) end as hcfa_plc_srv_cd,
--         case when (clm.src_specialty_cd is null or trim(clm.src_specialty_cd)='') then null else trim(clm.src_specialty_cd) end as src_specialty_cd,
--         case when (clm.prcdr_cd is null or trim(clm.prcdr_cd)='') then null else trim(clm.prcdr_cd) end as prcdr_cd,
--         case when (icd_prc.icd9_prcdr_cd is null or trim(icd_prc.icd9_prcdr_cd)='') then null else upper(trim(icd_prc.icd9_prcdr_cd)) end as icd9_prcdr_cd
--      from birth_dt base
--      inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_CLAIM_LINE_Y2019` clm
--              on base.member_id=clm.member_id
--      left join `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_PRCD_ARC` icd_prc
--             on clm.claim_line_id=icd_prc.claim_line_id
--      where base.index_dt > clm.received_dt and base.index_dt > clm.srv_start_dt
--        and DATE_ADD(clm.srv_start_dt, INTERVAL 36 MONTH) > base.index_dt
--        and clm.duplicate_ind='N' and clm.summarized_srv_ind='Y'
--        and clm.srv_start_dt >= '2017-01-01' and clm.srv_start_dt <= '2019-12-31'
-- ;

drop table if exists `{{DEC_TARGET_DB}}.{{PREFIX}}_archive_edw_clms_y2018`;
create table `{{DEC_TARGET_DB}}.{{PREFIX}}_archive_edw_clms_y2018`
OPTIONS (labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")]) as
with birth_dt as (
    select a.individual_id,
           a.member_id,
           a.index_dt,
           a.gender_cd,
           b.birth_dt
    from `{{TARGET_DB}}.{{PREFIX}}_member_base` a
    left join `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEMBER_ARC` b
            on a.member_id=b.member_id
)
  select
        base.individual_id,
        base.member_id,
        clm.claim_line_id,
        clm.srv_start_dt,
        case
            when (clm.days_cnt is null or clm.days_cnt <0) then 99
            when clm.days_cnt > 10 then 11
            else clm.days_cnt
        end as days_cnt,
        case
            when trim(base.gender_cd)='M' then 1
            when trim(base.gender_cd)='F' then 0
            else 2
        end as gender_cd,
        months_between_hive(date(clm.srv_start_dt), date(base.birth_dt)) as age_in_months,
         case when (clm.revenue_cd is null or trim(clm.revenue_cd)='') then null else trim(clm.revenue_cd) end as revenue_cd,
        case when (clm.hcfa_plc_srv_cd is null or trim(clm.hcfa_plc_srv_cd)='') then null else trim(clm.hcfa_plc_srv_cd) end as hcfa_plc_srv_cd,
        case when (clm.src_specialty_cd is null or trim(clm.src_specialty_cd)='') then null else trim(clm.src_specialty_cd) end as src_specialty_cd,
        case when (clm.prcdr_cd is null or trim(clm.prcdr_cd)='') then null else trim(clm.prcdr_cd) end as prcdr_cd,
        case when (icd_prc.icd9_prcdr_cd is null or trim(icd_prc.icd9_prcdr_cd)='') then null else upper(trim(icd_prc.icd9_prcdr_cd)) end as icd9_prcdr_cd
  from birth_dt base
  inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_CLAIM_LINE_Y2018` clm
          on base.member_id=clm.member_id
  left join `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_PRCD_ARC` icd_prc
         on clm.claim_line_id=icd_prc.claim_line_id
   where base.index_dt > clm.received_dt and base.index_dt > clm.srv_start_dt
        and DATE_ADD(clm.srv_start_dt, INTERVAL 36 MONTH) > base.index_dt
        and clm.duplicate_ind='N' and clm.summarized_srv_ind='Y'
        and clm.srv_start_dt >= '2016-01-01' and clm.srv_start_dt <= '2017-12-31'
;

drop table if exists `{{DEC_TARGET_DB}}.{{PREFIX}}_archive_edw_clms`;
create table `{{DEC_TARGET_DB}}.{{PREFIX}}_archive_edw_clms`
OPTIONS (labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")]) as 
-- select * from `{{DEC_TARGET_DB}}.{{PREFIX}}_archive_edw_clms_y2021`
-- union distinct
select * from `{{DEC_TARGET_DB}}.{{PREFIX}}_archive_edw_clms_y2020`
union distinct
-- select * from `{{DEC_TARGET_DB}}.{{PREFIX}}_archive_edw_clms_y2019`
-- union distinct
select * from `{{DEC_TARGET_DB}}.{{PREFIX}}_archive_edw_clms_y2018`
;

ALTER TABLE `{{DEC_TARGET_DB}}.{{PREFIX}}_archive_edw_clms_y2020`
    SET OPTIONS (expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
;

ALTER TABLE `{{DEC_TARGET_DB}}.{{PREFIX}}_archive_edw_clms_y2018`
    SET OPTIONS (expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
;

-- ALTER TABLE `{{DEC_TARGET_DB}}.{{PREFIX}}_archive_edw_clms`
--     SET OPTIONS (expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
-- ;