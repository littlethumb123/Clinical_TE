create or replace table `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_archive_rx`
    CLUSTER BY member_id
    OPTIONS ( labels=[("owner", "{{ params.owner }}"),("costcenter", "{{ params.costcenter }}")],expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
as
with base as (
    SELECT rm.individual_id,rm.member_id,rm.index_dt,m.gender_cd, m.birth_dt
    		, row_number() over (partition by rm.individual_id order by orig_covg_eff_dt desc, hmo_to_trad_conv_dt desc, rm.member_id desc) as ord
    FROM `{{DATASET}}.{{ dag_run.conf["prefix"] }}_base_memberid` rm
    JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEMBER` m
    ON rm.member_id = m.member_id
)
, b_dt as (
	-- 1 individual_id will have only 1 gender_cd and 1 birth_dt
	select *
	from base
	where ord = 1
),
x1 as (
    select
        base.individual_id,
        base.index_dt,
        base.member_id,
        case
            when trim(b_dt.gender_cd) = 'M' then 1
            when trim(b_dt.gender_cd) = 'F' then 0
            else 2 end                                                           as gender_cd,
        `{{SHARE_BQDB}}.months_between_floor`(rx.disp_dt, b_dt.birth_dt) as age_in_months,
        rx.disp_dt                                                               as dt,
        concat('gpi', SUBSTR(trim(rx.adjudicated_gpi_cd), 1, 4))                 as gpi4
    from base
    join b_dt
    	on base.individual_id = b_dt.individual_id
        inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.RX_CLAIM_DTL_Y2020` rx   -- need to change at year end when changing archive year
          on base.member_id = rx.member_id
    where 
	rx.disp_dt between date("2017-01-01") and date("2018-12-31")
	and rx.disp_dt > DATE_SUB(base.index_dt, INTERVAL 36 MONTH)
	and rx.disp_dt <= base.index_dt 
        and rx.process_dt <= base.index_dt
)
-- ,

-- x2 as (
--     select
--         base.individual_id,
--         base.index_dt,
--         base.member_id,
--         case
--             when trim(b_dt.gender_cd) = 'M' then 1
--             when trim(b_dt.gender_cd) = 'F' then 0
--             else 2 end                                                           as gender_cd,
--         `{{SHARE_BQDB}}.months_between_floor`(rx.disp_dt, b_dt.birth_dt) as age_in_months,
--         rx.disp_dt                                                               as dt,
--         concat('gpi', SUBSTR(trim(rx.adjudicated_gpi_cd), 1, 4))                 as gpi4
--     from base
--     join b_dt
--     	on base.individual_id = b_dt.individual_id
--              inner join `edp-prod-hcbstorage.edp_hcb_core_cnsv.UNMSK_RX_CLAIM_DTL_ARCHIVE` rx
--                         on base.member_id = rx.member_id
--     where rx.disp_dt <= base.index_dt 
--     	and base.index_dt <= DATE_ADD(rx.disp_dt, INTERVAL 36 MONTH) 
--     	and rx.disp_dt between date('201-01-01') and '2017-12-31'   -- need to change at year end when changing archive year
-- )
select * from x1
-- union distinct
-- select * from x2
;