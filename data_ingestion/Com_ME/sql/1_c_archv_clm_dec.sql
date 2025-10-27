create or replace table `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_archive_claims`
    CLUSTER BY member_id
    OPTIONS ( labels=[("owner", "{{ params.owner }}"),("costcenter", "{{ params.costcenter }}")],expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
as
with base as (
    SELECT rm.individual_id,rm.member_id,rm.index_dt,m.gender_cd, m.birth_dt
    		, row_number() over (partition by rm.individual_id order by orig_covg_eff_dt desc, hmo_to_trad_conv_dt desc, rm.member_id desc) as ord
    FROM `{{DATASET}}.{{ dag_run.conf["prefix"] }}_base_memberid` rm
    JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEMBER` m
    ON rm.member_id = m.member_id
)
, b_dt as (
	-- 1 individual_id will have only 1 gender_cd and 1 birth_dt
	select *
	from base
	where ord = 1
)
, x0 as (
    select
        base.individual_id,
        base.member_id,
        base.index_dt,
        clm.claim_line_id,
        clm.srv_start_dt as dt,
        case when (clm.days_cnt is null or clm.days_cnt < 0) then 99 when clm.days_cnt > 10 then 11 else clm.days_cnt end as days_cnt,
        case when trim(b_dt.gender_cd)='M' then 1 when trim(b_dt.gender_cd)='F' then 0 else 2 end as gender_cd,
        `{{SHARE_BQDB}}.months_between_floor`(clm.srv_start_dt, b_dt.birth_dt) as age_in_months,
        case when (trim(clm.revenue_cd) = '') then null else upper(trim(clm.revenue_cd)) end as revenue_cd,
        case when (trim(clm.hcfa_plc_srv_cd)='') then null else upper(trim(clm.hcfa_plc_srv_cd)) end as hcfa_plc_srv_cd,
        case when (trim(clm.src_specialty_cd)='') then null else upper(trim(clm.src_specialty_cd)) end as src_specialty_cd,
        case when (trim(clm.prcdr_cd)='') then null else upper(trim(clm.prcdr_cd)) end as prcdr_cd,
        case when (trim(icd_prc.icd9_prcdr_cd)='') then null else upper(trim(icd_prc.icd9_prcdr_cd)) end as icd9_prcdr_cd
    from base
    join b_dt
    	on base.individual_id = b_dt.individual_id
        inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_CLAIM_LINE_ARC` clm    -- need to change at year end when changing archive year
        on base.member_id=clm.member_id
        left join `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_PRCD_ARC` icd_prc
            on clm.claim_line_id=icd_prc.claim_line_id and clm.member_id=icd_prc.member_id
    where 
	clm.srv_start_dt between date("2020-01-01") and date("2021-12-31")   -- need to change at year end when adding archive year
        and clm.srv_start_dt > DATE_SUB(base.index_dt, INTERVAL 36 MONTH)  
        and clm.srv_start_dt <= base.index_dt
	and IF(clm.paid_dt > DATE("1900-01-01"), clm.paid_dt, clm.adjn_dt) <= base.index_dt
        and clm.duplicate_ind = 'N' 
	and clm.summarized_srv_ind = 'Y'
	and clm.reversal_cd <> 'R'
)
, x1 as (
    select
        base.individual_id,
        base.member_id,
        base.index_dt,
        clm.claim_line_id,
        clm.srv_start_dt as dt,
        case when (clm.days_cnt is null or clm.days_cnt < 0) then 99 when clm.days_cnt > 10 then 11 else clm.days_cnt end as days_cnt,
        case when trim(b_dt.gender_cd)='M' then 1 when trim(b_dt.gender_cd)='F' then 0 else 2 end as gender_cd,
        `{{SHARE_BQDB}}.months_between_floor`(clm.srv_start_dt, b_dt.birth_dt) as age_in_months,
        case when (trim(clm.revenue_cd) = '') then null else upper(trim(clm.revenue_cd)) end as revenue_cd,
        case when (trim(clm.hcfa_plc_srv_cd)='') then null else upper(trim(clm.hcfa_plc_srv_cd)) end as hcfa_plc_srv_cd,
        case when (trim(clm.src_specialty_cd)='') then null else upper(trim(clm.src_specialty_cd)) end as src_specialty_cd,
        case when (trim(clm.prcdr_cd)='') then null else upper(trim(clm.prcdr_cd)) end as prcdr_cd,
        case when (trim(icd_prc.icd9_prcdr_cd)='') then null else upper(trim(icd_prc.icd9_prcdr_cd)) end as icd9_prcdr_cd
    from base
    join b_dt
    	on base.individual_id = b_dt.individual_id
        inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_CLAIM_LINE_Y2021` clm    -- need to change at year end when changing archive year
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
	and clm.reversal_cd <> 'R'
),

x2 as (
     select
         base.individual_id,
         base.member_id,
         base.index_dt,
         clm.claim_line_id,
         clm.srv_start_dt as dt,
         case when (clm.days_cnt is null or clm.days_cnt < 0) then 99 when clm.days_cnt > 10 then 11 else clm.days_cnt end as days_cnt,
         case when trim(b_dt.gender_cd)='M' then 1 when trim(b_dt.gender_cd)='F' then 0 else 2 end as gender_cd,
         `{{SHARE_BQDB}}.months_between_floor`(clm.srv_start_dt, b_dt.birth_dt) as age_in_months,
         case when trim(clm.revenue_cd)='' then null else upper(trim(clm.revenue_cd)) end as revenue_cd,
         case when trim(clm.hcfa_plc_srv_cd)='' then null else upper(trim(clm.hcfa_plc_srv_cd)) end as hcfa_plc_srv_cd,
         case when trim(clm.src_specialty_cd)='' then null else upper(trim(clm.src_specialty_cd)) end as src_specialty_cd,
         case when trim(clm.prcdr_cd)='' then null else upper(trim(clm.prcdr_cd)) end as prcdr_cd,
         case when trim(icd_prc.icd9_prcdr_cd)='' then null else upper(trim(icd_prc.icd9_prcdr_cd)) end as icd9_prcdr_cd
     from base
     join b_dt
     	on base.individual_id = b_dt.individual_id
          inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_CLAIM_LINE_Y2020` clm   -- need to change at year end when changing archive year
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
	and clm.reversal_cd <> 'R'
)

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
   ;