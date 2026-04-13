drop table if exists `{{TARGET_DB}}.{{PREFIX}}_hdr_dtl_combined`;
create table `{{TARGET_DB}}.{{PREFIX}}_hdr_dtl_combined`
OPTIONS ( labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")]) as
with z0 as (
    select
        a.*,
        b.clm_dtl_id,
        b.prcdr_cd,
        b.prcdr_group_nbr,
        b.revenue_cd,
        b.hcfa_plc_srv_cd,
        b.srvc_to_dt,
        b.srvc_from_dt,
        b.admsn_days_cnt,
        b.claim_line_status_flag,
        b.claim_line_status,
        b.min_clm_detail_prcs_dt
    from `{{TARGET_DB}}.{{PREFIX}}_daily_claim_header_raw` as a
    left join `{{TARGET_DB}}.{{PREFIX}}_daily_claim_detail_raw` as b
           on upper(trim(a.source)) = upper(trim(b.source))
           and a.clm_prcs_dt = b.clm_prcs_dt
            and upper(trim(a.clm_hdr_id)) = upper(trim(b.clm_hdr_id))
            and upper(trim(a.clm_status_cd_std)) = upper(trim(b.clm_status_cd_std))
    where
        a.icd_diag1_cd is not null
),

z1 as (
    select
        a.individual_id,
        a.clm_hdr_id as claim_line_id,
        a.srvc_from_dt as srv_start_dt,
        a.admsn_days_cnt as days_cnt,
        regexp_extract(a.revenue_cd, r'.{3}$') as revenue_cd,
        a.hcfa_plc_srv_cd,
        a.src_specialty_cd,
        a.icd_diag1_cd,
        a.icd_diag2_cd,
        a.icd_diag3_cd,
        a.icd_prcdr1_cd,
        a.icd_prcdr2_cd,
        a.icd_prcdr3_cd,
        max(case when b.clm_dtl_id = '1' then b.prcdr_cd else null end) as prcdr_cd
    from z0 as a
    inner join z0 as b
            on a.individual_id = b.individual_id
           and a.clm_hdr_id = b.clm_hdr_id
    group by
        a.individual_id,
        a.clm_hdr_id,
        a.srvc_from_dt,
        a.admsn_days_cnt,
        a.revenue_cd,
        a.hcfa_plc_srv_cd,
        a.src_specialty_cd,
        a.icd_diag1_cd,
        a.icd_diag2_cd,
        a.icd_diag3_cd,
        a.icd_prcdr1_cd,
        a.icd_prcdr2_cd,
        a.icd_prcdr3_cd
)

    select distinct
    individual_id,
    claim_line_id,
    srv_start_dt,
    CAST(ROUND(CAST(days_cnt AS FLOAT64), 0) as INT64) as days_cnt,
    revenue_cd,
    hcfa_plc_srv_cd,
    src_specialty_cd,
    icd_diag1_cd,
    icd_diag2_cd,
    icd_diag3_cd,
    prcdr_cd,
    icd9_prcdr_cd
    from z1 b,
    unnest([icd_prcdr1_cd, icd_prcdr2_cd, icd_prcdr3_cd]) as icd9_prcdr_cd
    ;

