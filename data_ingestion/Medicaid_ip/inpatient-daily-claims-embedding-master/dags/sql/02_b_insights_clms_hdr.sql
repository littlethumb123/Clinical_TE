drop table if exists `{{TARGET_DB}}.{{PREFIX}}_daily_claim_header_raw`;
create table `{{TARGET_DB}}.{{PREFIX}}_daily_claim_header_raw`
OPTIONS (labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")]) as
with x0 as (
    select distinct
        a.individual_id,
        b.clm_id,
        b.clm_hdr_id,
        b.clm_prcs_dt,
        coalesce(b.indiv_anlytcs_id,
                 if ((trim(b.clm_hdr_id_src_cd) = 'HRP' and trim(b.hrp_lob_cd) = 'M') or
                     (trim(b.clm_hdr_id_src_cd) = 'ACAS' and trim(b.pat_rltnshp_desc) in ('INSURED', 'MEMBER')) or
                     (trim(b.clm_hdr_id_src_cd) = 'GIAS' and trim(b.pat_rltnshp_desc) = 'MEMBER') or
                     (trim(b.clm_hdr_id_src_cd) = 'HMOC' and trim(b.pat_rltnshp_desc) = 'SELF'),
                     b.indiv_anlytcs_sbscrbr_id, b.indiv_anlytcs_id)) as indiv_anlytcs_id,
        b.indiv_anlytcs_sbscrbr_id,
        b.clm_hdr_id_src_cd as source,
        b.clm_rcvd_dt,
        b.clm_status_cd_std,
        b.srvcng_prvdr_spclty_cd as src_specialty_cd,
        b.icd_diag1_cd_std as icd_diag1_cd,
        b.icd_diag2_cd_std as icd_diag2_cd,
        b.icd_diag3_cd_std as icd_diag3_cd,
        trim(b.icd_prcdr1_cd) AS icd_prcdr1_cd,
        trim(b.icd_prcdr2_cd) AS icd_prcdr2_cd,
        trim(b.icd_prcdr3_cd) AS icd_prcdr3_cd,
        coalesce(trim(b.sbmtd_drg_cd), trim(b.hosp_drg_cd), trim(b.prov_sbmtd_drg_cd)) AS drg_cd,
        if (trim(b.hrp_lob_cd)='M', 1, 0) AS hrp_medicare,
        case
            when (b.clm_status_cd_std in ('02', 'C', '') or b.clm_status_cd_std is null) then 1
            when b.clm_status_cd_std = 'P' then 0 else -1 end as claim_status
    from `{{TARGET_DB}}.{{PREFIX}}_member_base` as a
    inner join `anbc-hcb-prod.insights_share_hcb_prod.v_insights_iai_proxy_xwalk` as i
            on CAST(a.individual_id as string) = trim(i.indiv_id)
    inner join `anbc-hcb-prod.insights_share_hcb_prod.v_enriched_resolved_claim_header` as b
            on trim(i.indiv_anlytcs_id) = trim(b.indiv_anlytcs_id)
    where 
        -- i.bus_eff_dt <= DATE(a.index_dt) and 
        DATE(b.clm_prcs_dt) >= DATE_SUB(DATE(a.index_dt), INTERVAL 90 DAY)
      and DATE(b.clm_prcs_dt) <= DATE(a.index_dt)
      and (trim(b.clm_status_cd_std) in ('02', 'C', '', 'P')
        or b.clm_status_cd_std is null)
      and b.clm_type_cd_src != 'D'

    -- union distinct

    -- select distinct
    --     a.individual_id,
    --     b.clm_id,
    --     b.clm_hdr_id,
    --     b.clm_prcs_dt,
    --     coalesce(b.indiv_anlytcs_id,
    --              if ((trim(b.clm_hdr_id_src_cd) = 'HRP' and trim(b.hrp_lob_cd) = 'M') or
    --                  (trim(b.clm_hdr_id_src_cd) = 'ACAS' and trim(b.pat_rltnshp_desc) in ('INSURED', 'MEMBER')) or
    --                  (trim(b.clm_hdr_id_src_cd) = 'GIAS' and trim(b.pat_rltnshp_desc) = 'MEMBER') or
    --                  (trim(b.clm_hdr_id_src_cd) = 'HMOC' and trim(b.pat_rltnshp_desc) = 'SELF'),
    --                  b.indiv_anlytcs_sbscrbr_id, b.indiv_anlytcs_id)) as indiv_anlytcs_id,
    --     b.indiv_anlytcs_sbscrbr_id,
    --     b.clm_hdr_id_src_cd as source,
    --     b.clm_rcvd_dt,
    --     b.clm_status_cd_std,
    --     b.srvcng_prvdr_spclty_cd as src_specialty_cd,
    --     b.icd_diag1_cd_std as icd_diag1_cd,
    --     b.icd_diag2_cd_std as icd_diag2_cd,
    --     b.icd_diag3_cd_std as icd_diag3_cd,
    --     trim(b.icd_prcdr1_cd) AS icd_prcdr1_cd,
    --     trim(b.icd_prcdr2_cd) AS icd_prcdr2_cd,
    --     trim(b.icd_prcdr3_cd) AS icd_prcdr3_cd,
    --     coalesce(trim(b.sbmtd_drg_cd), trim(b.hosp_drg_cd), trim(b.prov_sbmtd_drg_cd)) AS drg_cd,
    --     if (trim(b.hrp_lob_cd)='M', 1, 0) AS hrp_medicare,
    --     case
    --         when (b.clm_status_cd_std in ('02', 'C', '') or b.clm_status_cd_std is null) then 1
    --         when b.clm_status_cd_std = 'P' then 0 else -1 end as claim_status
    -- from `{{TARGET_DB}}.{{PREFIX}}_member_base` as a
    -- inner join `anbc-hcb-prod.insights_share_hcb_prod.v_insights_iai_proxy_xwalk` as i
    --         on CAST(a.individual_id as string) = i.indiv_id
    -- inner join `anbc-hcb-prod.insights_share_hcb_prod.v_enriched_resolved_claim_header_hist` as b
    --         on trim(i.indiv_anlytcs_id) = trim(b.indiv_anlytcs_id)
    -- where
    --     i.bus_eff_dt <= DATE(a.index_dt)
    --   and DATE(b.clm_prcs_dt) >= DATE_SUB(DATE(a.index_dt), INTERVAL 90 DAY)
    --   and DATE(b.clm_prcs_dt) <= DATE(a.index_dt)
    --   and (b.clm_status_cd_std in ('02', 'C', '', 'P')
    --     or b.clm_status_cd_std is null)
    --   and b.clm_type_cd_src != 'D'
),

     x1 as (
         select
             clm_hdr_id,
             max(claim_status) as max_claim_status,
             max(clm_prcs_dt) as max_clm_prcs_dt,
             min(clm_prcs_dt) as min_clm_prcs_dt
         from x0
         group by
             clm_hdr_id
     )
         select
             a.*,
             case
                 when claim_status = 1 then 'Complete'
                 when claim_status = 0 then 'Pending'
                 else Null end as claim_status_desc,
             b.min_clm_prcs_dt as min_clm_header_prcs_dt
         from x0 as a
         inner join x1 as b
                 on a.clm_hdr_id = b.clm_hdr_id
                 and a.claim_status = b.max_claim_status
                 and a.clm_prcs_dt = b.max_clm_prcs_dt
;

