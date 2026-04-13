create or replace table `{{TARGET_DB}}.{{PREFIX}}_daily_claim_header_raw`
OPTIONS (labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")]) as
with x0 as (
    select distinct
        a.individual_id,
        a.index_dt,
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
            on CAST(a.individual_id as string) = i.indiv_id
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


create or replace table `{{TARGET_DB}}.{{PREFIX}}_daily_claim_detail_raw`
OPTIONS ( labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")]) as
with y0 as (
         select distinct
             b.indiv_anlytcs_id,
             b.indiv_anlytcs_sbscrbr_id,
             b.clm_hdr_id_src_cd as source,
             b.clm_hdr_id,
             b.clm_dtl_id,
             b.clm_id,
             trim(b.prcdr_cd) as prcdr_cd,
             cast(b.prcdr_grp_no as int) as prcdr_group_nbr,
             b.srvc_to_dt,
             b.srvc_from_dt,
             b.clm_prcs_dt,
             b.clm_status_cd_std,
             lpad(regexp_replace(trim(b.revnu_cd), '[^0-9]', ''), 4, '0') as revenue_cd,
             trim(b.pos_cd) as hcfa_plc_srv_cd,
             b.admsn_days_cnt,
             case
                 when (b.clm_status_cd_std in ('02', 'C', '') or b.clm_status_cd_std is null) then 1
                 when b.clm_status_cd_std = 'P' then 0 else -1 end as claim_line_status_flag
         from `{{TARGET_DB}}.{{PREFIX}}_member_base` as a
         inner join `anbc-hcb-prod.insights_share_hcb_prod.v_insights_iai_proxy_xwalk` as i
                 on CAST(a.individual_id as string) = i.indiv_id
         inner join `anbc-hcb-prod.insights_share_hcb_prod.v_enriched_resolved_claim_detail` as b
                 on trim(i.indiv_anlytcs_id) = trim(b.indiv_anlytcs_id)
         where
            --  i.bus_eff_dt <= DATE(a.index_dt) and 
             DATE(b.clm_prcs_dt) >= DATE_SUB(DATE(a.index_dt), INTERVAL 90 DAY)
           and DATE(b.clm_prcs_dt) <= DATE(a.index_dt)
           and (trim(b.clm_status_cd_std) in ('02', 'C', '', 'P')
             or b.clm_status_cd_std is null)

        --  UNION DISTINCT

        -- select distinct
        --      b.indiv_anlytcs_id,
        --      b.indiv_anlytcs_sbscrbr_id,
        --      b.clm_hdr_id_src_cd as source,
        --      b.clm_hdr_id,
        --      b.clm_dtl_id,
        --      b.clm_id,
        --      trim(b.prcdr_cd) as prcdr_cd,
        --      cast(b.prcdr_grp_no as int) as prcdr_group_nbr,
        --      b.srvc_to_dt,
        --      b.srvc_from_dt,
        --      b.clm_prcs_dt,
        --      b.clm_status_cd_std,
        --      lpad(regexp_replace(trim(b.revnu_cd), '[^0-9]', ''), 4, '0') as revenue_cd,
        --      trim(b.pos_cd) as hcfa_plc_srv_cd,
        --      b.admsn_days_cnt,
        --      case
        --          when (b.clm_status_cd_std in ('02', 'C', '') or b.clm_status_cd_std is null) then 1
        --          when b.clm_status_cd_std = 'P' then 0 else -1 end as claim_line_status_flag
        -- from `{{TARGET_DB}}.{{PREFIX}}_member_base` as a
        -- inner join `anbc-hcb-prod.insights_share_hcb_prod.v_insights_iai_proxy_xwalk` as i
        --             on CAST(a.individual_id as string) = i.indiv_id
        -- inner join `anbc-hcb-prod.insights_share_hcb_prod.v_enriched_resolved_claim_detail_hist` as b
        --             on trim(i.indiv_anlytcs_id) = trim(b.indiv_anlytcs_id)
        --  where
        --      i.bus_eff_dt <= DATE(a.index_dt)
        --    and DATE(b.clm_prcs_dt) >= DATE_SUB(DATE(a.index_dt), INTERVAL 90 DAY)
        --    and DATE(b.clm_prcs_dt) <= DATE(a.index_dt)
        --    and (b.clm_status_cd_std in ('02', 'C', '', 'P')
        --      or b.clm_status_cd_std is null)
     ),

     y1 as (
         select
             clm_hdr_id,
             clm_dtl_id,
             max(claim_line_status_flag) as max_claim_line_status_flag,
             max(clm_prcs_dt) as max_clm_prcs_dt, min(clm_prcs_dt) as min_clm_prcs_dt
         from y0
         group by
             clm_hdr_id,
             clm_dtl_id
     )
    select
        a.*,
        case
            when claim_line_status_flag = 1 then 'Complete'
            when claim_line_status_flag = 0 then 'Pending'
            else Null end as claim_line_status,
        b.min_clm_prcs_dt as min_clm_detail_prcs_dt
    from y0 as a
    inner join y1 as b
            on a.clm_hdr_id = b.clm_hdr_id
                and a.clm_dtl_id = b.clm_dtl_id
                and a.claim_line_status_flag = b.max_claim_line_status_flag
                and a.clm_prcs_dt = b.max_clm_prcs_dt
    ;


create or replace  table `{{TARGET_DB}}.{{PREFIX}}_hdr_dtl_combined`
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
        a.index_dt,
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
            and a.index_dt = b.index_dt
           and a.clm_hdr_id = b.clm_hdr_id
    group by
        a.individual_id,
        a.index_dt,
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
    index_dt,
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

