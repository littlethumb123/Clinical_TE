CREATE TEMPORARY FUNCTION months_between_hive(date1 DATE, date2 DATE) AS (
  FLOOR((
    EXTRACT(YEAR FROM date1) - EXTRACT(YEAR FROM date2)
  ) * 12 + (
    EXTRACT(MONTH FROM date1) - EXTRACT(MONTH FROM date2)
  ) + (
    EXTRACT(DAY FROM date1) - EXTRACT(DAY FROM date2)
  ) / 30.436875)
);


drop table if exists `{{DEC_TARGET_DB}}.{{PREFIX}}_daily_edw_clm_combined`;
create table `{{DEC_TARGET_DB}}.{{PREFIX}}_daily_edw_clm_combined`
OPTIONS (labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")])
as
with birth_dt as (
    select a.individual_id,
           a.member_id,
           a.index_dt,
           a.gender_cd,
           b.birth_dt
    from `{{TARGET_DB}}.{{PREFIX}}_member_base` a
    left join `edp-prod-hcbstorage.edp_hcb_core_cnsv.UNMASK_MEMBER` b
            on a.member_id=b.member_id
),
x0a as (
    select distinct
        individual_id,
        member_id,
        srv_start_dt,
        hcfa_plc_srv_cd,
        prcdr_cd
    from `{{TARGET_DB}}.{{PREFIX}}_edw_claims` as a
    union distinct
    select distinct
        a.individual_id,
        a.member_id,
        srv_start_dt,
        hcfa_plc_srv_cd,
        prcdr_cd
    from `{{TARGET_DB}}.{{PREFIX}}_hdr_dtl_combined` as b
    inner join `{{TARGET_DB}}.{{PREFIX}}_member_base` as a
            on a.individual_id = b.individual_id
    where
        date(a.index_dt) > date(b.srv_start_dt)
)
    select
        base.individual_id,
        base.member_id,
         coalesce(safe_cast(clm.claim_line_id as string), dly.claim_line_id) as claim_line_id,
         coalesce(clm.srv_start_dt, dly.srv_start_dt) as srv_start_dt,
        case
             when (coalesce(clm.days_cnt, dly.days_cnt) is null or coalesce(clm.days_cnt, dly.days_cnt) <0) then 99
             when coalesce(clm.days_cnt, dly.days_cnt) > 10 then 11
             else coalesce(clm.days_cnt, dly.days_cnt) end as days_cnt,
        case
             when trim(member.gender_cd)='M' then 1
             when trim(member.gender_cd)='F' then 0
             else 2 end as gender_cd,
         months_between_hive(COALESCE(CAST(clm.srv_start_dt as DATE), CAST(dly.srv_start_dt as DATE)), CAST(member.birth_dt as DATE)) as age_in_months,
        case
             when trim(coalesce(clm.revenue_cd, dly.revenue_cd))='' then null
             else trim(coalesce(clm.revenue_cd, dly.revenue_cd)) end as revenue_cd,
        case
             when trim(coalesce(clm.hcfa_plc_srv_cd, dly.hcfa_plc_srv_cd))='' then null
             else trim(coalesce(clm.hcfa_plc_srv_cd, dly.hcfa_plc_srv_cd)) end as hcfa_plc_srv_cd,
        case
             when trim(coalesce(clm.src_specialty_cd, dly.src_specialty_cd))='' then null
             else trim(coalesce(clm.src_specialty_cd, dly.src_specialty_cd)) end as src_specialty_cd,
        case
             when trim(coalesce(clm.prcdr_cd, dly.prcdr_cd))='' then null
             else trim(coalesce(clm.prcdr_cd, dly.prcdr_cd)) end as prcdr_cd,
        case
             when trim(coalesce(clm.icd9_prcdr_cd, dly.icd9_prcdr_cd))='' then null
             else trim(coalesce(clm.icd9_prcdr_cd, dly.icd9_prcdr_cd)) end as icd9_prcdr_cd
    from x0a as base
    left join birth_dt as member
    on base.member_id=member.member_id
    left join `{{TARGET_DB}}.{{PREFIX}}_edw_claims` as clm
           on base.individual_id = clm.individual_id
           and base.srv_start_dt = clm.srv_start_dt
           and base.hcfa_plc_srv_cd = clm.hcfa_plc_srv_cd
           and base.prcdr_cd = clm.prcdr_cd
    left join (select b.*,a.member_id
               from `{{TARGET_DB}}.{{PREFIX}}_hdr_dtl_combined` b
               inner join `{{TARGET_DB}}.{{PREFIX}}_member_base` as a
                       on a.individual_id = b.individual_id
               where date(a.index_dt) > date(b.srv_start_dt)
              ) as dly
        on base.individual_id = dly.individual_id
        and base.srv_start_dt = dly.srv_start_dt
        and base.hcfa_plc_srv_cd = dly.hcfa_plc_srv_cd
        and base.prcdr_cd = dly.prcdr_cd
    ;
