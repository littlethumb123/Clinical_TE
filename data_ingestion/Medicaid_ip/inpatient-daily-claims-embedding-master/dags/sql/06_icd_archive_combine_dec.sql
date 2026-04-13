drop table if exists `{{DEC_TARGET_DB}}.{{PREFIX}}_d1b_score_ending_tmp`;
create table `{{DEC_TARGET_DB}}.{{PREFIX}}_d1b_score_ending_tmp`
    OPTIONS ( labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")])
as
with x1 as (
        select * from `{{DEC_TARGET_DB}}.{{PREFIX}}_icd9_current`
    -- union distinct
    --     select * from `{{DEC_TARGET_DB}}.{{PREFIX}}_icd9_y2021`
    union all
        select * from `{{DEC_TARGET_DB}}.{{PREFIX}}_icd9_y2020`
    -- union distinct
    --     select * from `{{DEC_TARGET_DB}}.{{PREFIX}}_icd9_y2019`
    union all
        select * from `{{DEC_TARGET_DB}}.{{PREFIX}}_icd9_y2018`
)
,
x2 as (
    select  
         individual_id,
         member_id,
         claim_line_id,
         srv_start_dt,
         ARRAY[STRUCT(icd1[safe_offset(0)] as l,substr(icd1[safe_offset(1)],1,2) as r),
               STRUCT(icd2[safe_offset(0)] as l,substr(icd2[safe_offset(1)],1,2) as r),
               STRUCT(icd3[safe_offset(0)] as l,substr(icd3[safe_offset(1)],1,2) as r)] AS icd_array
     from x1
)
     select distinct
        individual_id,
        member_id,
        claim_line_id,
        srv_start_dt as dt,
        var_st.l as icd9_dx_cd2,
        case when ARRAY_TO_STRING([var_st.l,var_st.r],'.')='' THEN NULL else        ARRAY_TO_STRING([var_st.l,var_st.r],'.') END AS icd9_dx_cd
    from x2,
         unnest(icd_array) as var_st
;