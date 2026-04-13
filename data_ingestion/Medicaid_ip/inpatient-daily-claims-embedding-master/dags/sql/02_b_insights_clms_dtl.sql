drop table if exists `{{TARGET_DB}}.{{PREFIX}}_daily_claim_detail_raw`;
create table `{{TARGET_DB}}.{{PREFIX}}_daily_claim_detail_raw`
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
             case when (b.clm_status_cd_std in ('02', 'C', '') or b.clm_status_cd_std is null) then 1
                 when b.clm_status_cd_std = 'P' then 0 else -1 end as claim_line_status_flag
         from `{{TARGET_DB}}.{{PREFIX}}_member_base` as a
         inner join `anbc-hcb-prod.insights_share_hcb_prod.v_insights_iai_proxy_xwalk` as i
                 on CAST(a.individual_id as string) = trim(i.indiv_id)
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

