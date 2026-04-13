drop table if exists `{{DEC_TARGET_DB}}.{{PREFIX}}_root`;
create table `{{DEC_TARGET_DB}}.{{PREFIX}}_root`
    OPTIONS ( labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")])
as
with root0 as (
    select
        individual_id,
        member_id,
        srv_start_dt as dt,
        gender_cd,
        case
            when age_in_months < 0 then 0
            when age_in_months > 1440 then 1440
            else age_in_months end as age_in_months
     from `{{DEC_TARGET_DB}}.{{PREFIX}}_d1a_score_ending_tmp` clm
    where age_in_months is not null
    union distinct
     select
        individual_id,
        member_id,
        dt,
        gender_cd,
        case
            when age_in_months < 0 then 0
            when age_in_months > 1440 then 1440
            else age_in_months end as age_in_months
     from `{{DEC_TARGET_DB}}.{{PREFIX}}_d1c_score_ending_tmp` rx
    where age_in_months is not null
),

root1 as (
    select
        r0.*,
        row_number() over (partition by r0.individual_id, r0.dt) as seqno
     from root0 r0
)
    select
        individual_id,
        member_id,
        dt,
        gender_cd,
        age_in_months
     from root1
     where seqno = 1
;

drop table if exists `{{DEC_TARGET_DB}}.{{PREFIX}}_get_cd`;
create table `{{DEC_TARGET_DB}}.{{PREFIX}}_get_cd`
    OPTIONS ( labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")])
as
    select
        base.individual_id,
        base.member_id,
        base.srv_start_dt as dt,
        w2ind.cd,
        case when w2ind.ind is null then 0 else w2ind.ind end as ind
    from `{{DEC_TARGET_DB}}.{{PREFIX}}_d1a_score_ending_tmp` base
    left join (select ind, cd 
               from `anbc-hcb-prod.clin_analytics_hcb_prod.HPT_CP_IP_V9_W2IND`) w2ind
           on concat('days_cnt', cast(days_cnt as string)) = w2ind.cd
    where days_cnt is not null
union distinct
    select
        base.individual_id,
        base.member_id,
        base.srv_start_dt as dt,
        w2ind.cd,
        case when w2ind.ind is null then 0 else w2ind.ind end as ind
    from `{{DEC_TARGET_DB}}.{{PREFIX}}_d1a_score_ending_tmp` base
    left join (select ind, cd 
               from `anbc-hcb-prod.clin_analytics_hcb_prod.HPT_CP_IP_V9_W2IND`) w2ind
           on concat('hcfa_plc_srv_cd', cast (hcfa_plc_srv_cd as string)) = w2ind.cd
    where hcfa_plc_srv_cd is not null
union distinct
    select
        base.individual_id,
        base.member_id,
        base.srv_start_dt as dt,
        w2ind.cd,
        case when w2ind.ind is null then 0 else w2ind.ind end as ind
    from `{{DEC_TARGET_DB}}.{{PREFIX}}_d1a_score_ending_tmp` base
    left join (select ind, cd from `anbc-hcb-prod.clin_analytics_hcb_prod.HPT_CP_IP_V9_W2IND`) w2ind
           on concat('src_specialty_cd', cast (src_specialty_cd as string)) = w2ind.cd
    where src_specialty_cd is not null
union distinct
    select
        base.individual_id,
        base.member_id,
        base.dt,
        w2ind.cd,
        case when w2ind.ind is null then 0 else w2ind.ind end as ind
    from `{{DEC_TARGET_DB}}.{{PREFIX}}_d1b_score_ending_tmp` base
    left join (select ind, cd from `anbc-hcb-prod.clin_analytics_hcb_prod.HPT_CP_IP_V9_W2IND`) w2ind
           on concat('icd9_dx_cd', cast (icd9_dx_cd as string)) = w2ind.cd
    where icd9_dx_cd is not null
union distinct
    select
        base.individual_id,
        base.member_id,
        base.srv_start_dt as dt,
        w2ind.cd,
        case when w2ind.ind is null then 0 else w2ind.ind end as ind
    from `{{DEC_TARGET_DB}}.{{PREFIX}}_d1a_score_ending_tmp` base
    left join (select ind, cd from `anbc-hcb-prod.clin_analytics_hcb_prod.HPT_CP_IP_V9_W2IND`) w2ind
           on concat('revenue_cd', cast (revenue_cd as string)) = w2ind.cd
    where revenue_cd is not null
union distinct
    select
        base.individual_id,
        base.member_id,
        base.srv_start_dt as dt,
        w2ind.cd,
        case when w2ind.ind is null then 0 else w2ind.ind end as ind
    from `{{DEC_TARGET_DB}}.{{PREFIX}}_d1a_score_ending_tmp` base
    left join (select ind, cd from `anbc-hcb-prod.clin_analytics_hcb_prod.HPT_CP_IP_V9_W2IND`) w2ind
           on concat('prcdr_cd', cast (prcdr_cd as string)) = w2ind.cd
    where prcdr_cd is not null
union distinct
    select
        base.individual_id,
        base.member_id,
        base.srv_start_dt as dt,
        w2ind.cd,
        case when w2ind.ind is null then 0 else w2ind.ind end as ind
    from `{{DEC_TARGET_DB}}.{{PREFIX}}_d1a_score_ending_tmp` base
    left join (select ind, cd from `anbc-hcb-prod.clin_analytics_hcb_prod.HPT_CP_IP_V9_W2IND`) w2ind
           on concat('prcdr_cd', cast (icd9_prcdr_cd as string)) = w2ind.cd
    where icd9_prcdr_cd is not null
union distinct
    select
        base.individual_id,
        base.member_id,
        base.dt,
        w2ind.cd,
        case when w2ind.ind is null then 0 else w2ind.ind end as ind
    from `{{DEC_TARGET_DB}}.{{PREFIX}}_d1c_score_ending_tmp` base
    left join (select ind, cd from `anbc-hcb-prod.clin_analytics_hcb_prod.HPT_CP_IP_V9_W2IND`) w2ind
           on concat('gpi', cast (gpi4 as string)) = w2ind.cd
    where gpi4 is not null
;

drop table if exists `{{DEC_TARGET_DB}}.{{PREFIX}}_o1_score_ending_tmp`;
create table `{{DEC_TARGET_DB}}.{{PREFIX}}_o1_score_ending_tmp`
    OPTIONS ( labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")])
as
with x1 as (
    select
        individual_id,
        dt,
        ind
    from `{{DEC_TARGET_DB}}.{{PREFIX}}_get_cd`
    group by individual_id, dt, ind
    order by dt desc
),

x2 as (
    select
        *,
        row_number() over (partition by individual_id, dt order by ind) as seqno
    from x1
),

x3 as (
    select
        individual_id ,
        dt,
        ARRAY_AGG(cast(ind as string) order by ind) as cd_arr
    from x2
    where seqno<=80
    group by individual_id, dt
)
select
  root2.individual_id,
  root2.dt,
  root2.gender_cd,
  root2.age_in_months,
  ARRAY_TO_STRING(x3.cd_arr, ',') as cd
from `{{DEC_TARGET_DB}}.{{PREFIX}}_root` root2
inner join x3 x3
         on root2.individual_id = x3.individual_id and root2.dt = x3.dt

;


ALTER TABLE `{{DEC_TARGET_DB}}.{{PREFIX}}_root`
    SET OPTIONS (expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
;

drop table if exists `{{DEC_TARGET_DB}}.{{PREFIX}}_o3_score_ending_tmp_1`;
create table `{{DEC_TARGET_DB}}.{{PREFIX}}_o3_score_ending_tmp_1`
    OPTIONS ( labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")])
as
with x1 as (
    select *,
        row_number() over (partition by individual_id order by dt desc) as seqno
    from `{{DEC_TARGET_DB}}.{{PREFIX}}_o1_score_ending_tmp`
)
    select * from x1 where seqno<=200
;

drop table if exists `{{DEC_TARGET_DB}}.{{PREFIX}}_o3_score_ending_tmp_ordered`;
create table `{{DEC_TARGET_DB}}.{{PREFIX}}_o3_score_ending_tmp_ordered`
 OPTIONS ( labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")])
as
    select *
        ,row_number() over (partition by individual_id order by dt) as seqno2
    from `{{DEC_TARGET_DB}}.{{PREFIX}}_o3_score_ending_tmp_1`
    order by individual_id,seqno2
;

drop table if exists `{{FINAL_DB}}.{{PREFIX}}_o3_score_ending_tmp`;
create table `{{FINAL_DB}}.{{PREFIX}}_o3_score_ending_tmp`
 OPTIONS ( labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")])
as
    select individual_id,
        ARRAY_TO_STRING(ARRAY_AGG(safe_cast(gender_cd as string)), '*') as gender_cd,

        ARRAY_TO_STRING(ARRAY_AGG(safe_cast(age_in_months as string)), '*') as age_in_months,

        ARRAY_TO_STRING(ARRAY_AGG(cast(cd as string)), '*') as cd,

        count(*) as dt_cnt
    from `{{DEC_TARGET_DB}}.{{PREFIX}}_o3_score_ending_tmp_ordered`
    group by individual_id
;

ALTER TABLE `{{DEC_TARGET_DB}}.{{PREFIX}}_o3_score_ending_tmp_ordered`
    SET OPTIONS (expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
;

ALTER TABLE `{{DEC_TARGET_DB}}.{{PREFIX}}_o3_score_ending_tmp_1`
    SET OPTIONS (expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
;

