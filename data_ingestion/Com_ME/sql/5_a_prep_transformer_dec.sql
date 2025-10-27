create or replace table `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_root`
    CLUSTER BY individual_id
    OPTIONS ( labels=[("owner", "{{ params.owner }}"),("costcenter", "{{ params.costcenter }}")],expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
as
with root0 as (
    select
        individual_id,
        index_dt,
        member_id,
        dt,
        gender_cd,
        case
            when age_in_months < 0 then 0
            when age_in_months > 1440 then 1439
            else age_in_months end as age_in_months
     from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_d1a_score_ending_tmp`
    union distinct
    select
        individual_id,
        index_dt,
        member_id,
        dt,
        gender_cd,
        case
            when age_in_months < 0 then 0
            when age_in_months > 1440 then 1440
            else age_in_months end as age_in_months
    from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_d1c_score_ending_tmp`
),

root1 as (
    select
        r0.*,
        row_number() over (partition by r0.individual_id, r0.index_dt,r0.dt) as seqno
     from root0 r0
)
    select DISTINCT
        individual_id,
        index_dt,
        member_id,
        dt,
        gender_cd,
        age_in_months
     from root1
     where seqno = 1
;


create or replace table `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_get_cd`
    OPTIONS ( labels=[("owner", "{{ params.owner }}"),("costcenter", "{{ params.costcenter }}")],expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
as
      select distinct
            base.individual_id,
            base.index_dt,
            base.member_id,
            base.dt,
            w2ind.cd,
            case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from
          `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_d1a_score_ending_tmp` base
            left join (select ind, cd from `anbc-hcb-prod.clin_analytics_share_hcb_prod.HPT_CP_IP_V9_W2IND`) w2ind
            on concat('days_cnt', cast (days_cnt as string)) = w2ind.cd
      where days_cnt is not null
    union all
      select distinct
          base.individual_id,
          base.index_dt,
          base.member_id,
          base.dt,
          w2ind.cd,
          case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from
          `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_d1a_score_ending_tmp` base
          left join (select ind, cd from `anbc-hcb-prod.clin_analytics_share_hcb_prod.HPT_CP_IP_V9_W2IND`) w2ind
            on concat('hcfa_plc_srv_cd', cast (base.hcfa_plc_srv_cd as string)) = w2ind.cd
      where base.hcfa_plc_srv_cd is not null
    union all
      select distinct
          base.individual_id,
          base.index_dt,
          base.member_id,
          base.dt,
          w2ind.cd,
          case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_d1a_score_ending_tmp` base
               left join (select ind, cd from `anbc-hcb-prod.clin_analytics_share_hcb_prod.HPT_CP_IP_V9_W2IND`) w2ind
            on concat('src_specialty_cd', cast (base.src_specialty_cd as string)) = w2ind.cd
      where base.src_specialty_cd is not null
    union all
      select distinct
          base.individual_id,
          base.index_dt,
          base.member_id,
          base.dt,
          w2ind.cd,
          case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from
          `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_d1b_score_ending_tmp` base
              left join (select ind, cd from `anbc-hcb-prod.clin_analytics_share_hcb_prod.HPT_CP_IP_V9_W2IND`) w2ind
            on concat('icd9_dx_cd', cast (base.icd9_dx_cd as string)) = w2ind.cd
      where base.icd9_dx_cd is not null
    union all
      select distinct
          base.individual_id,
          base.index_dt,
          base.member_id,
          base.dt,
          w2ind.cd,
          case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from
          `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_d1a_score_ending_tmp` base
              left join (select ind, cd from `anbc-hcb-prod.clin_analytics_share_hcb_prod.HPT_CP_IP_V9_W2IND`) w2ind
            on concat('revenue_cd', cast (base.revenue_cd as string)) = w2ind.cd
      where base.revenue_cd is not null
    union all
        (
          select distinct
              base.individual_id,
              base.index_dt,
              base.member_id,
              base.dt,
              w2ind.cd,
              case when w2ind.ind is null then 0 else w2ind.ind end as ind
          from
              `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_d1a_score_ending_tmp` base
                  left join (select ind, cd from `anbc-hcb-prod.clin_analytics_share_hcb_prod.HPT_CP_IP_V9_W2IND`) w2ind
                on concat('prcdr_cd', cast (base.prcdr_cd as string)) = w2ind.cd
          where base.prcdr_cd is not null
        union distinct
          select distinct
              base.individual_id,
              base.index_dt,
              base.member_id,
              base.dt,
              w2ind.cd,
              case when w2ind.ind is null then 0 else w2ind.ind end as ind
          from
              `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_d1a_score_ending_tmp` base
                  left join (select ind, cd from `anbc-hcb-prod.clin_analytics_share_hcb_prod.HPT_CP_IP_V9_W2IND`) w2ind
                on concat('prcdr_cd', cast (base.icd9_prcdr_cd as string)) = w2ind.cd
          where base.icd9_prcdr_cd is not null
        )
    union all
      select distinct
          base.individual_id,
          base.index_dt,
          base.member_id,
          base.dt,
          w2ind.cd,
          case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from
          `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_d1c_score_ending_tmp` base
              left join (select ind, cd from `anbc-hcb-prod.clin_analytics_share_hcb_prod.HPT_CP_IP_V9_W2IND`) w2ind
            on concat('gpi', cast (base.gpi4 as string)) = w2ind.cd
      where base.gpi4 is not null
;

create or replace table  `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_o1_score_ending_tmp`
OPTIONS ( labels=[("owner", "{{ params.owner }}"),("costcenter", "{{ params.costcenter }}")],expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
AS
with x1 as (
    select
        individual_id,
        index_dt,
        dt,
        ind
    from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_get_cd` x0
    group by individual_id,index_dt, dt, ind
    order by dt, ind
),

x2 as (
    select
        *,
        row_number() over (partition by individual_id,index_dt,dt order by ind) as seqno
    from x1
),

x3 as (
    select
        individual_id ,
        index_dt,
        dt,
        ARRAY_AGG(cast(ind as string) order by ind) as cd_arr
    from x2
    where seqno<=80
    group by individual_id, index_dt,dt
)

select
  root2.individual_id,
  root2.index_dt,
  root2.dt,
  root2.gender_cd,
  root2.age_in_months,
  ARRAY_TO_STRING(x3.cd_arr, ',') as cd
from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_root` root2
inner join x3
    on root2.individual_id = x3.individual_id and root2.dt = x3.dt and root2.index_dt=x3.index_dt

;

ALTER TABLE `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_o1_score_ending_tmp`
    SET OPTIONS (expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 12 DAY))
;

CREATE OR REPLACE TABLE `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_o3_score_ending_tmp_1`
CLUSTER BY individual_id
OPTIONS ( labels=[("owner", "{{ params.owner }}"),("costcenter", "{{ params.costcenter }}")],expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
as
with x1 as (
    select *,
        row_number() over (partition by individual_id,index_dt order by dt desc) as seqno
    from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_o1_score_ending_tmp`
)
    select * from x1 where seqno<=200
;


create or replace table `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_o3_score_ending_tmp_ordered`
 OPTIONS ( labels=[("owner", "{{ params.owner }}"),("costcenter", "{{ params.costcenter }}")],expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
as
    select *
        ,row_number() over (partition by individual_id,index_dt order by dt) as seqno2
    from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_o3_score_ending_tmp_1`
    order by individual_id,index_dt,seqno2
;

create or replace table  `{{ dag_run.conf["FINAL_DATASET"] }}.{{ dag_run.conf["prefix"] }}_o3_score_ending`
CLUSTER BY individual_id
OPTIONS ( labels=[("owner", "{{ params.owner }}"),("costcenter", "{{ params.costcenter }}")],expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
AS
select individual_id,index_dt,
    ARRAY_TO_STRING(ARRAY_AGG(cast(gender_cd as string)), '*') as gender_cd,
    --concat('*',array_agg(cast(gender_cd as string))) as gender_cd,
    ARRAY_TO_STRING(ARRAY_AGG(cast(age_in_months as string)), '*') as age_in_months,
    --concat('*',array_agg(cast(age_in_months as string))) as age_in_months,
    ARRAY_TO_STRING(ARRAY_AGG(cast(cd as string)), '*') as cd,
    --concat('*',array_agg(cast(cd as string))) as cd,
    count(*) as dt_cnt
from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_o3_score_ending_tmp_ordered`
group by individual_id,index_dt

;

-- select * from `{{ dag_run.conf["FINAL_DATASET"] }}.{{ dag_run.conf["prefix"] }}_o3_score_ending_tmp_ordered`
--     where individual_id=31592282;

-- create or replace table  `{{ dag_run.conf["FINAL_DATASET"] }}.{{ dag_run.conf["prefix"] }}_tensor`
-- CLUSTER BY individual_id
-- OPTIONS ( labels=[("owner", "{{ params.owner }}"),("costcenter", "{{ params.costcenter }}")],expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
-- as
-- with t as (select individual_id,index_dt,dt,age_in_months,gender_cd,
--                  replace(cd,",","*") as cd
--           from `{{ dag_run.conf["FINAL_DATASET"] }}.{{ dag_run.conf["prefix"] }}_cp_o3_score_ending_tmp_ordered`
-- )
-- , c as (select individual_id,index_dt,dt,age_in_months,gender_cd,
--                  CONCAT(cast(age_in_months as string),"*",cast(gender_cd as string),"*",cd) as cd
--           from t
-- )
-- select individual_id,index_dt,
--       ARRAY_AGG(cd) as tensor
--       from c
-- group by individual_id,index_dt
-- ;