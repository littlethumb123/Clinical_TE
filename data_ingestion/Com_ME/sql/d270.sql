---------------------------------------------
-- START OF create table {{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1a_score_ending
---------------------------------------------
drop table if exists `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1a_score_ending_tmp`;
create table `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1a_score_ending_tmp`
    CLUSTER BY member_id
    OPTIONS ( labels=[("owner", "{{ params.owner }}"),("costcenter", "{{ params.costcenter }}")])
as
with birth_dt as (
    SELECT rm.individual_id,rm.member_id,rm.index_dt,MAX(m.gender_cd) AS gender_cd,MAX(birth_dt) AS birth_dt
    FROM ``{{DATASET}}.{{ dag_run.conf["prefix"] }}_base_memberid`` rm
    JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEMBER` m
    ON rm.member_id = m.member_id
GROUP BY rm.individual_id,rm.member_id,rm.index_dt
),
x0d as (
    select
        base.individual_id,
        base.member_id,
        base.index_dt,
        clm.claim_line_id,
        clm.srv_start_dt as dt,
        case when (clm.days_cnt is null or clm.days_cnt < 0) then 99 when clm.days_cnt > 10 then 11 else clm.days_cnt end as days_cnt,
        case when trim(base.gender_cd)='M' then 1 when trim(base.gender_cd)='F' then 0 else 2 end as gender_cd,
        `{{SHARE_BQDB}}.months_between_floor`(clm.srv_start_dt, base.birth_dt) as age_in_months,
        case when trim(clm.revenue_cd)='' then null else upper(trim(clm.revenue_cd)) end as revenue_cd,
        case when trim(clm.hcfa_plc_srv_cd)='' then null else upper(trim(clm.hcfa_plc_srv_cd)) end as hcfa_plc_srv_cd,
        case when trim(clm.src_specialty_cd)='' then null else upper(trim(clm.src_specialty_cd)) end as src_specialty_cd,
        case when trim(clm.prcdr_cd)='' then null else upper(trim(clm.prcdr_cd)) end as prcdr_cd,
        case when trim(icd_prc.icd9_prcdr_cd)='' then null else upper(trim(icd_prc.icd9_prcdr_cd)) end as icd9_prcdr_cd
    from birth_dt base
    inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.D_EMIS_CLAIM_LINE` clm
        on base.member_id=clm.member_id
    left join `edp-prod-hcbstorage.edp_hcb_core_cnsv.D_CLM_LN_X_ICD9_PRCD` icd_prc
        on clm.claim_line_id=icd_prc.claim_line_id and clm.member_id=icd_prc.member_id
    where 
        clm.srv_start_dt >= date("2023-01-01")   ---Daily claims only has 2023+, need update when expand to 2020
        and clm.srv_start_dt > DATE_SUB(base.index_dt, interval 36 month)
	and clm.srv_start_dt <= base.index_dt
        and IF(clm.paid_dt > DATE("1900-01-01"), clm.paid_dt, clm.adjn_dt) <= base.index_dt 
        and IF(clm.paid_dt > DATE("1900-01-01"), clm.paid_dt, clm.adjn_dt) >= DATE_SUB(base.index_dt, INTERVAL 91 DAY)
        and clm.duplicate_ind = 'N' 
	and clm.summarized_srv_ind = 'Y'
        and clm.days_cnt >= 0
	and clm.file_id <> 'C4'
),

x0 as (
    select
        base.individual_id,
        base.member_id,
        base.index_dt,
        clm.claim_line_id,
        clm.srv_start_dt as dt,
        case when (clm.days_cnt is null or clm.days_cnt < 0) then 99 when clm.days_cnt > 10 then 11 else clm.days_cnt end as days_cnt,
        case when trim(base.gender_cd)='M' then 1 when trim(base.gender_cd)='F' then 0 else 2 end as gender_cd,
        `{{SHARE_BQDB}}.months_between_floor`(clm.srv_start_dt, base.birth_dt) as age_in_months,
        case when trim(clm.revenue_cd)='' then null else upper(trim(clm.revenue_cd)) end as revenue_cd,
        case when trim(clm.hcfa_plc_srv_cd)='' then null else upper(trim(clm.hcfa_plc_srv_cd)) end as hcfa_plc_srv_cd,
        case when trim(clm.src_specialty_cd)='' then null else upper(trim(clm.src_specialty_cd)) end as src_specialty_cd,
        case when trim(clm.prcdr_cd)='' then null else upper(trim(clm.prcdr_cd)) end as prcdr_cd,
        case when trim(icd_prc.icd9_prcdr_cd)='' then null else upper(trim(icd_prc.icd9_prcdr_cd)) end as icd9_prcdr_cd
    from birth_dt base
    inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_CLAIM_LINE` clm
    on base.member_id=clm.member_id
    left join `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_PRCD` icd_prc
    on clm.claim_line_id=icd_prc.claim_line_id and clm.member_id=icd_prc.member_id
    where 
    	clm.srv_start_dt >= date("2022-01-01")  -- need to change at year end when changing archive year
	and clm.srv_start_dt > DATE_SUB(base.index_dt, interval 36 month)
        and clm.srv_start_dt <= base.index_dt 
	and IF(clm.paid_dt > DATE("1900-01-01"), clm.paid_dt, clm.adjn_dt) <= base.index_dt 
        and clm.duplicate_ind = 'N' 
	and clm.summarized_srv_ind = 'Y'
        and clm.days_cnt >= 0
	and clm.file_id <> 'C4'
),

x1 as (
    select
        base.individual_id,
        base.member_id,
        base.index_dt,
        clm.claim_line_id,
        clm.srv_start_dt as dt,
        case when (clm.days_cnt is null or clm.days_cnt < 0) then 99 when clm.days_cnt > 10 then 11 else clm.days_cnt end as days_cnt,
        case when trim(base.gender_cd)='M' then 1 when trim(base.gender_cd)='F' then 0 else 2 end as gender_cd,
        `{{SHARE_BQDB}}.months_between_floor`(clm.srv_start_dt, base.birth_dt) as age_in_months,
        case when (trim(clm.revenue_cd) = '') then null else upper(trim(clm.revenue_cd)) end as revenue_cd,
        case when (trim(clm.hcfa_plc_srv_cd)='') then null else upper(trim(clm.hcfa_plc_srv_cd)) end as hcfa_plc_srv_cd,
        case when (trim(clm.src_specialty_cd)='') then null else upper(trim(clm.src_specialty_cd)) end as src_specialty_cd,
        case when (trim(clm.prcdr_cd)='') then null else upper(trim(clm.prcdr_cd)) end as prcdr_cd,
        case when (trim(icd_prc.icd9_prcdr_cd)='') then null else upper(trim(icd_prc.icd9_prcdr_cd)) end as icd9_prcdr_cd
    from birth_dt base
        inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_CLAIM_LINE_ARC` clm
        on base.member_id=clm.member_id
        left join `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_PRCD_ARC` icd_prc
            on clm.claim_line_id=icd_prc.claim_line_id and clm.member_id=icd_prc.member_id
    where 
        clm.srv_start_dt between date("2020-01-01") and date("2021-12-31")
	and clm.srv_start_dt > DATE_SUB(base.index_dt, interval 36 month)
        and clm.srv_start_dt <= base.index_dt 
	and IF(clm.paid_dt > DATE("1900-01-01"), clm.paid_dt, clm.adjn_dt) <= base.index_dt
        and clm.duplicate_ind = 'N' 
	and clm.summarized_srv_ind = 'Y'
        and clm.days_cnt >= 0
),

x2 as (
     select
         base.individual_id,
         base.member_id,
         base.index_dt,
         clm.claim_line_id,
         clm.srv_start_dt as dt,
         case when (clm.days_cnt is null or clm.days_cnt < 0) then 99 when clm.days_cnt > 10 then 11 else clm.days_cnt end as days_cnt,
         case when trim(base.gender_cd)='M' then 1 when trim(base.gender_cd)='F' then 0 else 2 end as gender_cd,
         `{{SHARE_BQDB}}.months_between_floor`(clm.srv_start_dt, base.birth_dt) as age_in_months,
         case when trim(clm.revenue_cd)='' then null else upper(trim(clm.revenue_cd)) end as revenue_cd,
         case when trim(clm.hcfa_plc_srv_cd)='' then null else upper(trim(clm.hcfa_plc_srv_cd)) end as hcfa_plc_srv_cd,
         case when trim(clm.src_specialty_cd)='' then null else upper(trim(clm.src_specialty_cd)) end as src_specialty_cd,
         case when trim(clm.prcdr_cd)='' then null else upper(trim(clm.prcdr_cd)) end as prcdr_cd,
         case when trim(icd_prc.icd9_prcdr_cd)='' then null else upper(trim(icd_prc.icd9_prcdr_cd)) end as icd9_prcdr_cd
     from birth_dt base
          inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_CLAIM_LINE_Y2021` clm
                     on base.member_id=clm.member_id
          left join `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_PRCD_ARC` icd_prc
                    on clm.claim_line_id=icd_prc.claim_line_id and clm.member_id=icd_prc.member_id
     where 
         clm.srv_start_dt between date("2018-01-01") and date("2019-12-31")
        and clm.srv_start_dt > DATE_SUB(base.index_dt, INTERVAL 36 MONTH)
        and clm.srv_start_dt <= base.index_dt
	and IF(clm.paid_dt > DATE("1900-01-01"), clm.paid_dt, clm.adjn_dt) <= base.index_dt 
        and clm.duplicate_ind = 'N' 
	and clm.summarized_srv_ind = 'Y'
        and clm.days_cnt >= 0
),

x3 as (
    select
        base.individual_id,
        base.member_id,
        base.index_dt,
        clm.claim_line_id,
        clm.srv_start_dt as dt,
        case when (clm.days_cnt is null or clm.days_cnt < 0) then 99 when clm.days_cnt > 10 then 11 else clm.days_cnt end as days_cnt,
        case when trim(base.gender_cd)='M' then 1 when trim(base.gender_cd)='F' then 0 else 2 end as gender_cd,
        `{{SHARE_BQDB}}.months_between_floor`(clm.srv_start_dt, base.birth_dt) as age_in_months,
        case when trim(clm.revenue_cd)='' then null else upper(trim(clm.revenue_cd)) end as revenue_cd,
        case when trim(clm.hcfa_plc_srv_cd)='' then null else upper(trim(clm.hcfa_plc_srv_cd)) end as hcfa_plc_srv_cd,
        case when trim(clm.src_specialty_cd)='' then null else upper(trim(clm.src_specialty_cd)) end as src_specialty_cd,
        case when trim(clm.prcdr_cd)='' then null else upper(trim(clm.prcdr_cd)) end as prcdr_cd,
        case when trim(icd_prc.icd9_prcdr_cd)='' then null else upper(trim(icd_prc.icd9_prcdr_cd)) end as icd9_prcdr_cd
     from birth_dt base
     inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_CLAIM_LINE_Y2020` clm
                on base.member_id=clm.member_id
     left join `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_PRCD_ARC` icd_prc
            on clm.claim_line_id=icd_prc.claim_line_id and clm.member_id=icd_prc.member_id
     where 
         clm.srv_start_dt between date("2017-01-01") and date("2017-12-31")
        and clm.srv_start_dt > DATE_SUB(base.index_dt, INTERVAL 36 MONTH)
        and clm.srv_start_dt <= base.index_dt
	and IF(clm.paid_dt > DATE("1900-01-01"), clm.paid_dt, clm.adjn_dt) <= base.index_dt 
        and clm.duplicate_ind = 'N' 
	and clm.summarized_srv_ind = 'Y'
        and clm.days_cnt >= 0
),

x5 as (

   select individual_id,
          member_id,
          index_dt,
          claim_line_id,
          dt,
          days_cnt,
          gender_cd,
          age_in_months,
          revenue_cd,
          hcfa_plc_srv_cd,
          src_specialty_cd,
          prcdr_cd,
          icd9_prcdr_cd
   from x0d
   -- where COALESCE(revenue_cd,hcfa_plc_srv_cd,prcdr_cd,icd9_prcdr_cd) is not null
  union distinct
   select individual_id,
          member_id,
          index_dt,
          claim_line_id,
          dt,
          days_cnt,
          gender_cd,
          age_in_months,
          revenue_cd,
          hcfa_plc_srv_cd,
          src_specialty_cd,
          prcdr_cd,
          icd9_prcdr_cd
   from x0
   -- where COALESCE(revenue_cd,hcfa_plc_srv_cd,prcdr_cd,icd9_prcdr_cd) is not null
  union distinct
   select individual_id,
          member_id,
          index_dt,
          claim_line_id,
          dt,
          days_cnt,
          gender_cd,
          age_in_months,
          revenue_cd,
          hcfa_plc_srv_cd,
          src_specialty_cd,
          prcdr_cd,
          icd9_prcdr_cd
   from x1
   -- where COALESCE(revenue_cd,hcfa_plc_srv_cd,prcdr_cd,icd9_prcdr_cd) is not null
  union distinct
   select individual_id,
          member_id,
          index_dt,
          claim_line_id,
          dt,
          days_cnt,
          gender_cd,
          age_in_months,
          revenue_cd,
          hcfa_plc_srv_cd,
          src_specialty_cd,
          prcdr_cd,
          icd9_prcdr_cd
   from x2
   -- where COALESCE(revenue_cd,hcfa_plc_srv_cd,prcdr_cd,icd9_prcdr_cd) is not null
 union distinct
   select individual_id,
          member_id,
          index_dt,
          claim_line_id,
          dt,
          days_cnt,
          gender_cd,
          age_in_months,
          revenue_cd,
          hcfa_plc_srv_cd,
          src_specialty_cd,
          prcdr_cd,
          icd9_prcdr_cd
   from x3
   -- where COALESCE(revenue_cd,hcfa_plc_srv_cd,prcdr_cd,icd9_prcdr_cd) is not null
)

select a.individual_id,
       a.member_id,
       x.index_dt,
       x.claim_line_id,
       x.dt,
       x.days_cnt,
       x.gender_cd,
       x.age_in_months,
       x.revenue_cd,
       x.hcfa_plc_srv_cd,
       x.src_specialty_cd,
       x.prcdr_cd,
       x.icd9_prcdr_cd
    from ``{{DATASET}}.{{ dag_run.conf["prefix"] }}_base_memberid`` as a
        inner join x5 as x
        on a.individual_id = x.individual_id and a.member_id = x.member_id and a.index_dt=x.index_dt
    -- where COALESCE(x.revenue_cd,x.hcfa_plc_srv_cd,x.prcdr_cd,x.icd9_prcdr_cd) is not null
;

ALTER TABLE `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1a_score_ending_tmp`
    SET OPTIONS (expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 12 DAY))
;
---------------------------------------------
-- END OF create table {{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1a_score_ending
---------------------------------------------

-- select 'da' as step, count(1) as cnt
--         , count(distinct individual_id) as cnt_ind
-- from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1a_score_ending_tmp`
-- ;
---------------------------------------------

---------------------------------------------
-- START OF create table {{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1b_score_ending
---------------------------------------------
drop table if exists `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1b_score_ending_tmp`;
create table `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1b_score_ending_tmp`
    CLUSTER BY individual_id
    OPTIONS ( labels=[("owner", "{{ params.owner }}"),("costcenter", "{{ params.costcenter }}")])
as
with cohort as (
    select
        SELECT rm.individual_id,rm.member_id,rm.index_dt,MAX(m.gender_cd) AS gender_cd,MAX(birth_dt) AS birth_dt
    from ``{{DATASET}}.{{ dag_run.conf["prefix"] }}_base_memberid`` rm
    JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEMBER` m
    ON rm.member_id = m.member_id
GROUP BY rm.individual_id,rm.member_id,rm.index_dt
),
base as (
    select mem.individual_id, mem.member_id, d1a.index_dt, d1a.claim_line_id, d1a.dt,
           d1a.days_cnt, mem.gender_cd, d1a.age_in_months
    from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1a_score_ending_tmp` as d1a
    inner join cohort as mem
            on mem.member_id = d1a.member_id
    where
        DATE_ADD(d1a.dt, INTERVAL 36 MONTH) > d1a.index_dt and d1a.dt <= d1a.index_dt
),

xad as (
    select distinct
        base.individual_id,
        base.index_dt,
        base.member_id,
        base.claim_line_id,
        -- b.sequence_id,
        base.dt,
        base.days_cnt,
        base.gender_cd,
        base.age_in_months,
        -- b.xw as icd9_dx_cd2,
        b.x as icd9_dx_cd
    from
        base
        left join
        ( select
              member_id,
              claim_line_id,
              xw,
              case
                  when xw is null then null
                  when (dot_pos < 2) then xw
                  when (dot_pos = 2) then substr(xw, 1, 4)
                  when (dot_pos = 3) then substr(xw, 1, 5)
                  when (dot_pos = 4) then substr(xw, 1, 6)
                  when (dot_pos = 5) then substr(xw, 1, 7)
                  else  substr(xw, 1, 8)
                end as x
              -- , sequence_id
          from
              ( select
                    member_id,
                    claim_line_id,
                    case
                        when (icd9_dx_cd is null or icd9_dx_cd = '') then null
                        else upper(trim(icd9_dx_cd))
                      end as xw,
                    case
                        when (icd9_dx_cd is null or icd9_dx_cd = '') then 0
                        else strpos(icd9_dx_cd, '.')
                      end as dot_pos
                    -- , cast(sequence_id as int) as sequence_id
                from `edp-prod-hcbstorage.edp_hcb_core_cnsv.D_CLM_LN_X_ICD9_DX`
                where cast(sequence_id as int) < 4
              )
        ) b
        on base.member_id = b.member_id and base.claim_line_id=b.claim_line_id
    where b.xw is not null
),

xa as (
    select
        base.individual_id,
        base.index_dt,
        base.member_id,
        base.claim_line_id,
        -- b.sequence_id,
        base.dt,
        base.days_cnt,
        base.gender_cd,
        base.age_in_months,
        -- b.xw as icd9_dx_cd2,
        b.x as icd9_dx_cd
    from
        base
        left join
        ( select
              member_id,
              claim_line_id,
              xw,
              case
                  when xw is null then null
                  when (dot_pos < 2) then xw
                  when (dot_pos = 2) then substr(xw, 1, 4)
                  when (dot_pos = 3) then substr(xw, 1, 5)
                  when (dot_pos = 4) then substr(xw, 1, 6)
                  when (dot_pos = 5) then substr(xw, 1, 7)
                  else  substr(xw, 1, 8)
                end as x
              -- , sequence_id
          from
              ( select
                    member_id,
                    claim_line_id,
                    case
                        when (icd9_dx_cd is null or trim(icd9_dx_cd) = '') then null
                        else upper(trim(icd9_dx_cd))
                      end as xw,
                    case
                        when (icd9_dx_cd is null or trim(icd9_dx_cd) = '') then 0
                        else strpos(icd9_dx_cd, '.')
                      end as dot_pos
                    -- , cast(sequence_id as int) as sequence_id
                from `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_DX`
                where cast(sequence_id as int) < 4
              )
        ) b
        on base.member_id = b.member_id and base.claim_line_id=b.claim_line_id
    where b.xw is not null
),

xb as (
    select
        base.individual_id,
        base.index_dt,
        base.member_id,
        base.claim_line_id,
        -- b.sequence_id,
        base.dt,
        base.days_cnt,
        base.gender_cd,
        base.age_in_months,
       -- b.xw as icd9_dx_cd2,
        b.x as icd9_dx_cd
    from
        base
        left join
        ( select
              member_id,
              claim_line_id,
              xw,
              case
                  when xw is null then null
                  when (dot_pos < 2) then xw
                  when (dot_pos = 2) then substr(xw, 1, 4)
                  when (dot_pos = 3) then substr(xw, 1, 5)
                  when (dot_pos = 4) then substr(xw, 1, 6)
                  when (dot_pos = 5) then substr(xw, 1, 7)
                  else  substr(xw, 1, 8)
                end as x
             -- , sequence_id
          from
              ( select
                    member_id,
                    claim_line_id,
                    case
                        when (icd9_dx_cd is null or trim(icd9_dx_cd) = '') then null
                        else upper(trim(icd9_dx_cd))
                        end as xw,
                    case
                        when (icd9_dx_cd is null or trim(icd9_dx_cd)= '') then 0
                        else strpos(icd9_dx_cd, '.')
                        end as dot_pos
                    -- , cast(sequence_id as int) as sequence_id
                from `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_DX_ARC`
                where cast(sequence_id as int) < 4
              )
        ) b
        on base.member_id = b.member_id and base.claim_line_id=b.claim_line_id
    where b.xw is not null
),

xc as (
     select
         base.individual_id,
         base.index_dt,
         base.member_id,
         base.claim_line_id,
         -- b.sequence_id,
         base.dt,
         base.days_cnt,
         base.gender_cd,
         base.age_in_months,
         -- b.xw as icd9_dx_cd2,
         b.x as icd9_dx_cd
     from
        base
        left join
          ( select
                member_id,
                claim_line_id,
                xw,
                case
                    when xw is null then null
                    when (dot_pos < 2) then xw
                    when (dot_pos = 2) then substr(xw, 1, 4)
                    when (dot_pos = 3) then substr(xw, 1, 5)
                    when (dot_pos = 4) then substr(xw, 1, 6)
                    when (dot_pos = 5) then substr(xw, 1, 7)
                    else  substr(xw, 1, 8)
                  end as x
                -- , sequence_id
            from
                ( select
                      member_id,
                      claim_line_id,
                      case
                          when (icd9_dx_cd is null or trim(icd9_dx_cd) = '') then null
                          else upper(trim(icd9_dx_cd))
                          end as xw,
                      case
                          when (icd9_dx_cd is null or trim(icd9_dx_cd) = '') then 0
                          else strpos(icd9_dx_cd, '.')
                          end as dot_pos
                      -- , cast(sequence_id as int) as sequence_id
                  from `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_DX_Y2021`
                  where cast(sequence_id as int) < 4
                )
          ) b
          on base.member_id = b.member_id and base.claim_line_id=b.claim_line_id
     where b.xw is not null
),
 xd as (
    select
        base.individual_id,
        base.index_dt,
        base.member_id,
        base.claim_line_id,
        -- b.sequence_id,
        base.dt,
        base.days_cnt,
        base.gender_cd,
        base.age_in_months,
        -- b.xw as icd9_dx_cd2,
        b.x as icd9_dx_cd
    from
        base
        left join
        ( select
              member_id,
              claim_line_id,
              xw,
              case
                  when xw is null then null
                  when (dot_pos < 2) then xw
                  when (dot_pos = 2) then substr(xw, 1, 4)
                  when (dot_pos = 3) then substr(xw, 1, 5)
                  when (dot_pos = 4) then substr(xw, 1, 6)
                  when (dot_pos = 5) then substr(xw, 1, 7)
                  else  substr(xw, 1, 8)
                end as x
              -- , sequence_id
          from
              ( select
                    member_id,
                    claim_line_id,
                    case
                        when (icd9_dx_cd is null or trim(icd9_dx_cd) = '') then null
                        else upper(trim(icd9_dx_cd))
                        end as xw,
                    case
                        when (icd9_dx_cd is null or trim(icd9_dx_cd) = '') then 0
                        else strpos(icd9_dx_cd, '.')
                        end as dot_pos
                    -- , cast(sequence_id as int) as sequence_id
                from `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_DX_Y2020`
                where cast(sequence_id as int) < 4
              )
        ) b
        on base.member_id = b.member_id and base.claim_line_id=b.claim_line_id
    where b.xw is not null
),

x1 as (
    select * from xad
  union distinct
    select * from xa
  union distinct
    select * from xb
  union distinct
    select * from xc
  union distinct
    select * from xd
--     union distinct
--         select * from xe
)

select * from x1 where icd9_dx_cd is not null
;

ALTER TABLE `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1b_score_ending_tmp`
    SET OPTIONS (expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 12 DAY))
;
---------------------------------------------
-- END OF create table {{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1b_score_ending
---------------------------------------------

--197 unique icd group

select 'd1b' as step,
       count(1) as cnt,
    count(distinct d1be.individual_id) as cnt_ind
from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1b_score_ending_tmp` d1be
;
---------------------------------------------

---------------------------------------------
-- START OF create table {{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1c_score_ending
---------------------------------------------
drop table if exists `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1c_score_ending_tmp`;
create table `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1c_score_ending_tmp`
    CLUSTER BY individual_id
    OPTIONS ( labels=[("owner", "{{ params.owner }}"),("costcenter", "{{ params.costcenter }}")])
as
with cohort as (

    SELECT rm.individual_id,rm.member_id,rm.index_dt,MAX(m.gender_cd) AS gender_cd,MAX(birth_dt) AS birth_dt
    FROM `anbc-hcb-dev.clin_analytics_hcb_dev.a092446_temb2_base_memberid` rm
    JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEMBER` m
    ON rm.member_id = m.member_id
GROUP BY rm.individual_id,rm.member_id,rm.index_dt
), 

x0 as (
    select
        base.individual_id,
        base.index_dt,
        base.member_id,
        case
         when trim(base.gender_cd) = 'M' then 1
         when trim(base.gender_cd) = 'F' then 0
         else 2 end                                           as gender_cd,
        `{{SHARE_BQDB}}.months_between_floor`(rx.disp_dt, base.birth_dt) as age_in_months,
         rx.disp_dt                                               as dt,
         concat('gpi', SUBSTR(trim(rx.adjudicated_gpi_cd), 1, 4)) as gpi4
    from cohort base
        inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.D_RX_CLAIM_DTL` rx
            on base.member_id = rx.member_id
    where 
        rx.disp_dt >= date("2019-01-01")
	    and rx.disp_dt <= base.index_dt
        and rx.disp_dt >= DATE_SUB(base.index_dt, INTERVAL 36 MONTH) 
        and rx.process_dt <= base.index_dt
	and rx.file_id <> 'C5'
),

x1 as (
    select
        base.individual_id,
        base.index_dt,
        base.member_id,
        case
            when trim(base.gender_cd) = 'M' then 1
            when trim(base.gender_cd) = 'F' then 0
            else 2 end                                                           as gender_cd,
        `{{SHARE_BQDB}}.months_between_floor`(rx.disp_dt, base.birth_dt) as age_in_months,
        rx.disp_dt                                                               as dt,
        concat('gpi', SUBSTR(trim(rx.adjudicated_gpi_cd), 1, 4))                 as gpi4
    from cohort base
        inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.RX_CLAIM_DTL_Y2020` rx
          on base.member_id = rx.member_id
    where 
        rx.disp_dt between date("2017-01-01") and date("2018-12-31")
	and rx.disp_dt <= base.index_dt
        and rx.disp_dt >= DATE_SUB(base.index_dt, INTERVAL 36 MONTH) 
        and rx.process_dt <= base.index_dt
)

select * from x0
union distinct
select * from x1


;

ALTER TABLE `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1c_score_ending_tmp`
    SET OPTIONS (expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 12 DAY))
;
---------------------------------------------
-- END OF create table {{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1c_score_ending
---------------------------------------------
select 'd1c' as step, count(1) as cnt
        , count(distinct individual_id) as cnt_ind
from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1c_score_ending_tmp`
;
---------------------------------------------

---------------------------------------------
-- START OF create table {{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_o1_score_ending_tmp
---------------------------------------------
drop table if exists `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_o1_score_ending_tmp`;
create table `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_o1_score_ending_tmp`
    CLUSTER BY individual_id
    OPTIONS ( labels=[("owner", "{{ params.owner }}"),("costcenter", "{{ params.costcenter }}")])
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
            when age_in_months > 1440 then 1440
            else age_in_months end as age_in_months
     from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1a_score_ending_tmp`
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
    from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1c_score_ending_tmp`
),

root1 as (
    select
        r0.*,
        row_number() over (partition by r0.individual_id, r0.index_dt,r0.dt) as seqno
     from root0 r0
),

root2 as (
    select DISTINCT
        individual_id,
        index_dt,
        member_id,
        dt,
        gender_cd,
        age_in_months
     from root1
     where seqno = 1
),

x0 as (
      select distinct
            base.individual_id,
            base.index_dt,
            base.member_id,
            base.dt,
            w2ind.cd,
            case when w2ind.ind is null then 0 else w2ind.ind end as ind
      from
          `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1a_score_ending_tmp` base
            left join (select ind, cd from `{{BQDB}}.HPT_CP_IP_V9_W2IND`) w2ind
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
          `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1a_score_ending_tmp` base
          left join (select ind, cd from `{{BQDB}}.HPT_CP_IP_V9_W2IND`) w2ind
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
      from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1a_score_ending_tmp` base
               left join (select ind, cd from `{{BQDB}}.HPT_CP_IP_V9_W2IND`) w2ind
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
          `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1b_score_ending_tmp` base
              left join (select ind, cd from `{{BQDB}}.HPT_CP_IP_V9_W2IND`) w2ind
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
          `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1a_score_ending_tmp` base
              left join (select ind, cd from `{{BQDB}}.HPT_CP_IP_V9_W2IND`) w2ind
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
              `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1a_score_ending_tmp` base
                  left join (select ind, cd from `{{BQDB}}.HPT_CP_IP_V9_W2IND`) w2ind
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
              `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1a_score_ending_tmp` base
                  left join (select ind, cd from `{{BQDB}}.HPT_CP_IP_V9_W2IND`) w2ind
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
          `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_d1c_score_ending_tmp` base
              left join (select ind, cd from `{{BQDB}}.HPT_CP_IP_V9_W2IND`) w2ind
            on concat('gpi', cast (base.gpi4 as string)) = w2ind.cd
      where base.gpi4 is not null
),

x1 as (
    select
        individual_id,
        index_dt,
        dt,
        ind
    from x0
    group by individual_id,index_dt, dt, ind
    order by dt, ind
),

x2 as (
    select
        *,
        row_number() over (partition by individual_id,index_dt,dt) as seqno
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
from
  root2
  inner join x3
    on root2.individual_id = x3.individual_id and root2.dt = x3.dt and root2.index_dt=x3.index_dt

;

ALTER TABLE `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_o1_score_ending_tmp`
    SET OPTIONS (expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 12 DAY))
;

-- ALTER TABLE `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_o1_score_ending_tmp`
--     ADD PRIMARY KEY (individual_id, dt) NOT ENFORCED
-- ;

---------------------------------------------
-- END OF create table {{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_o1_score_ending
---------------------------------------------

select 'o1' as step, count(1) as cnt,
        count(distinct individual_id) as cnt_ind
from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_o1_score_ending_tmp`
;
---------------------------------------------

---------------------------------------------
-- START OF create table {{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_o3_score_ending
---------------------------------------------
drop table if exists `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_o3_score_ending{{SCOREDATE}}`;
create table `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_o3_score_ending{{SCOREDATE}}`
    CLUSTER BY individual_id
    OPTIONS ( labels=[("owner", "{{ params.owner }}"),("costcenter", "{{ params.costcenter }}")])
as
with x1 as (
    select   individual_id,
             index_dt,
             dt,
             gender_cd,
             age_in_months,
             cd,
             row_number() over (partition by individual_id,index_dt order by dt desc) as seqno
    from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_o1_score_ending_tmp`
),

x2 as (
    select * from x1 where seqno<=200
),

x4 as (
    select *,
        row_number() over (partition by individual_id,index_dt order by dt) as seqno2
    from x2
),

x5 as (
    select individual_id,index_dt,
        ARRAY_TO_STRING(ARRAY_REVERSE(ARRAY_AGG(cast(gender_cd as string))), '*') as gender_cd,
        --concat('*',array_agg(cast(gender_cd as string))) as gender_cd,
        ARRAY_TO_STRING(ARRAY_REVERSE(ARRAY_AGG(cast(age_in_months as string))), '*') as age_in_months,
        --concat('*',array_agg(cast(age_in_months as string))) as age_in_months,
        ARRAY_TO_STRING(ARRAY_REVERSE(ARRAY_AGG(cast(cd as string))), '*') as cd,
        --concat('*',array_agg(cast(cd as string))) as cd,
        count(*) as dt_cnt
    from x4
    group by individual_id,index_dt
)

select * from x5
;
---------------------------------------------
-- END OF create table {{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_o3_score_ending
---------------------------------------------

select
    'o3' as step, '{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_o3_score_ending{{SCOREDATE}}' as table_name,
    count(1) as cnt, count(distinct individual_id) as cnt_ind
from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_o3_score_ending{{SCOREDATE}}`
;

ALTER TABLE `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_o3_score_ending{{SCOREDATE}}`
    ADD PRIMARY KEY (individual_id) NOT ENFORCED
;
---------------------------------------------
---------------------------------------------

---------------------------------------------
-- START OF create view {{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_transformer_input
---------------------------------------------
drop view if exists `{{TABLE_SHARE_PREFIX}}_input{{SCOREDATE}}`;
create view `{{TABLE_SHARE_PREFIX}}_input{{SCOREDATE}}`
    OPTIONS ( labels=[("owner", "{{ params.owner }}"),("costcenter", "{{ params.costcenter }}")])
as
select *
from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf[emb_prefix] }}_o3_score_ending{{SCOREDATE}}`
;