
create or replace table `{{DEC_TARGET_DB}}.{{PREFIX}}_d1a_score_ending_tmp`
OPTIONS (labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")])
as
select individual_id,
       index_dt,
        member_id,
        claim_line_id,
        srv_start_dt,
        days_cnt,
        gender_cd,
        age_in_months,
        revenue_cd,
        hcfa_plc_srv_cd,
        src_specialty_cd,
        prcdr_cd,
        icd9_prcdr_cd,
        "current" as source
from `{{DEC_TARGET_DB}}.{{PREFIX}}_daily_edw_clm_combined`
union distinct
select individual_id,
        index_dt,
        member_id,
        safe_cast(claim_line_id as string) as claim_line_id,
        srv_start_dt,
        days_cnt,
        gender_cd,
        age_in_months,
        revenue_cd,
        hcfa_plc_srv_cd,
        src_specialty_cd,
        prcdr_cd,
        icd9_prcdr_cd,
        "archive" as source
 from `{{DEC_TARGET_DB}}.{{PREFIX}}_archive_edw_clms`
;