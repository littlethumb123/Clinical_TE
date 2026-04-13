drop table if exists `{{DEC_TARGET_DB}}.{{PREFIX}}_icd9_current`;
create table `{{DEC_TARGET_DB}}.{{PREFIX}}_icd9_current`
    OPTIONS ( labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")])
as
    select base.individual_id,
           base.member_id,
           base.claim_line_id,
           base.srv_start_dt,
           case when b.sequence_id = 1 then b.icd9_dx_cd
                when b.sequence_id is null and c.icd_diag1_cd is not null then c.icd_diag1_cd
                else null end as icd1,
           case when b.sequence_id = 2 then b.icd9_dx_cd
                 when b.sequence_id is null and c.icd_diag2_cd is not null then c.icd_diag2_cd
                else null end as icd2,
           case when b.sequence_id = 3 then b.icd9_dx_cd
                when b.sequence_id is null and c.icd_diag3_cd is not null then c.icd_diag3_cd
                else null end as icd3
    from
       (select individual_id, member_id, claim_line_id, srv_start_dt
        from `{{DEC_TARGET_DB}}.{{PREFIX}}_d1a_score_ending_tmp`
        where srv_start_dt >= date('2020-01-01') and srv_start_dt <= (select date(index_dt) 
                                                                from `{{TARGET_DB}}.{{PREFIX}}_member_base`
                                                                limit 1)
       ) base
    left join (
         select
            claim_line_id,
            split(trim(icd9_dx_cd),'.') as icd9_dx_cd,
            -- case when regexp_contains(trim(icd9_dx_cd), r'\.\w{2}') 
            -- then concat(regexp_extract(trim(icd9_dx_cd), r'^(.*)\.'),regexp_extract(trim(icd9_dx_cd), r'\.\w{2}'))
            -- else trim(icd9_dx_cd) end as icd9_dx_cd,
            cast(sequence_id as int) as sequence_id
        from `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_DX`
        where cast(sequence_id as int) < 4
        ) b
        on base.claim_line_id=safe_cast(b.claim_line_id as string)
    left join (select claim_line_id, 
               split(trim(icd_diag1_cd),'.')  as icd_diag1_cd,
               split(trim(icd_diag2_cd),'.')  as icd_diag2_cd,
               split(trim(icd_diag3_cd),'.')  as icd_diag3_cd,
               from `{{TARGET_DB}}.{{PREFIX}}_hdr_dtl_combined`) as c
           on base.claim_line_id = c.claim_line_id
;


