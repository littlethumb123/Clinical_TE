-- attach member_id for the cohort by the effective index date
create or replace table `{{DATASET}}.{{ dag_run.conf["prefix"] }}_base_memberid`
    CLUSTER BY member_id
    OPTIONS (labels=[("owner", "{{ params.owner }}"),("costcenter","{{ params.costcenter }}")],expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
as

WITH mbrship AS (

    SELECT member_id, eff_dt
    FROM `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_MEMBERSHIP`
    WHERE eff_dt >= DATE("2022-01-16")    ---- Need to change at year end when doing archive 
     AND file_id <> 'C2'
	
    UNION ALL
    SELECT member_id, eff_dt
    FROM `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_MEMBERSHIP_Y2025`
    WHERE eff_dt BETWEEN DATE("2021-01-16") AND DATE("2021-12-16")
	
    UNION ALL
    SELECT member_id, eff_dt
    FROM `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_MEMBERSHIP_Y2021`
    WHERE eff_dt BETWEEN DATE("2018-01-16") AND DATE("2020-12-16")
	
    UNION ALL
    SELECT member_id, eff_dt
    FROM `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_MEMBERSHIP_ARC`
    WHERE eff_dt BETWEEN DATE("2017-01-16") AND DATE("2017-12-16")
)

select distinct m.individual_id, x.member_id, m.index_dt
	from {{ dag_run.conf["te_base"] }} m
	join `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` x
	on m.individual_id = x.individual_id
	
	join mbrship AS em
	on x.member_id =em.member_id         
    WHERE DATE_SUB(em.eff_dt, INTERVAL 1 MONTH) <= m.index_dt  -- member_id effective before index date (after 16th enrollment go to next month)
;