-- DECLARE eff_dt DATE;
-- SET eff_dt=date_add(DATE("{{current_dt}}"), INTERVAL -1 MONTH);

drop table if exists `{{TARGET_DB}}.{{PREFIX}}_member_base`;
create table `{{TARGET_DB}}.{{PREFIX}}_member_base` 
OPTIONS ( labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")])
as
select distinct 
    a.member_id,
    i.individual_id,
    a.business_ln_cd,
    a.gender_cd,
    date("{{current_dt}}") as index_dt,
    date("{{current_dt}}") as eff_dt
from `edp-prod-hcbstorage.edp_hcb_core_cnsv.PRSPCTV_MEMBERSHIP`as a
inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.PRODUCT_LINE` as b 
    on a.product_ln_cd = b.product_ln_cd
inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` as i
        on a.member_id = i.member_id
where
    EXTRACT(YEAR FROM a.eff_dt)=EXTRACT(YEAR FROM date("{{current_dt}}")) 
    AND EXTRACT(MONTH FROM a.eff_dt)=EXTRACT(MONTH FROM date("{{current_dt}}")) 
    and ((a.business_ln_cd like 'ME%' 
        --    and substr(cast(i.individual_id as string), length(SAFE_CAST(i.individual_id AS STRING))-1, 2) = '43'
           )
    --  OR  (a.business_ln_cd like 'CP%' 
        --   and substr(cast(i.individual_id as string), length(SAFE_CAST(i.individual_id AS STRING))-2, 3) = '654'
            --   )
        )
    and trim(b.product_type_cd) = 'M'
    and trim(b.prod_ctg_cd) <> '02'
;