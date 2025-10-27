create or replace table `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_monthly_clm_ln`
    CLUSTER BY individual_id
    OPTIONS ( labels=[("owner", "{{ params.owner }}"),("costcenter", "{{ params.costcenter }}")],expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
as
    select distinct
        base.individual_id,
        base.index_dt,
        base.member_id,
        base.claim_line_id,
        base.dt,
        base.days_cnt,
        base.gender_cd,
        base.age_in_months,
        UPPER(case when ARRAY_TO_STRING([var_st.l,var_st.r],'.')='' THEN NULL 
             else ARRAY_TO_STRING([var_st.l,var_st.r],'.') END) AS icd9_dx_cd
    from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_monthly_claims` base
    inner join
        ( select
              member_id,
              claim_line_id,
              ARRAY[STRUCT(split(trim(icd9_dx_cd),'.')[safe_offset(0)] as l,
                    substr(split(trim(icd9_dx_cd),'.')[safe_offset(1)],1,2) as r)] AS icd_array
          from `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_DX`
          where cast(sequence_id as int) < 4 and icd9_dx_cd is not NULL and trim(icd9_dx_cd)!=""
        ) b
        on base.member_id = b.member_id and base.claim_line_id=b.claim_line_id,
    UNNEST(icd_array) as var_st
;