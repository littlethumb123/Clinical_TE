create or replace table `{{TARGET_DB}}.{{PREFIX}}_edw_claims`
OPTIONS (labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")])
as
select a.individual_id,
        a.member_id,
        a.gender_cd,
        b.claim_line_id,
        b.srv_start_dt,
        b.days_cnt,
        b.revenue_cd,
        b.hcfa_plc_srv_cd,
        b.src_specialty_cd,
        b.plc_srv_ctg_cd,
        b.prcdr_cd,
        c.icd9_prcdr_cd
    from `{{TARGET_DB}}.{{PREFIX}}_member_base` as a
    inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` as i
        on a.individual_id = i.individual_id
    inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_CLAIM_LINE` as b
            on i.member_id = b.member_id
    left join `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_PRCD` as c
           on b.claim_line_id = c.claim_line_id
        --  and b.srv_start_dt = c.srv_start_dt
        --  and b.hcfa_plc_srv_cd = c.hcfa_plc_srv_cd
        --  and b.prcdr_cd = c.prcdr_cd
    where
        b.summarized_srv_ind = 'Y'
        and b.duplicate_ind = 'N'
        and b.srv_start_dt >= date('2020-01-01')
        and b.srv_start_dt <= date(a.index_dt)
        and DATE_ADD(b.srv_start_dt, INTERVAL 36 MONTH) > a.index_dt
        and a.index_dt > b.received_dt
        and a.index_dt > b.srv_start_dt
;