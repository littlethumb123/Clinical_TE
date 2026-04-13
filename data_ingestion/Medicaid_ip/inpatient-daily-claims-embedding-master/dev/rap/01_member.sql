create or replace table `{{TARGET_DB}}.{{PREFIX}}_member_base` 
OPTIONS ( labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")])
as
select distinct 
    a.member_id,
    a.individual_id,
    b.gender_cd,
    date(a.index_dt) as index_dt
-- from `anbc-hcb-dev.clin_analytics_hcb_dev.a831544_SNF_development_cohort_uniq_pme_ref_windv` as a
from `clin_analytics_hcb_dev.a538985_rap_enhancement_cohort_uniq_pme_ref_windv` as a
inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.PRSPCTV_EMS_MBRSHP` as b 
    on a.member_id = b.member_id
;