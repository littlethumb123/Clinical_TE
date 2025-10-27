
drop table if exists dev_bpum_enc.a321276_o1_${trunk};
create table dev_bpum_enc.a321276_o1_${trunk} as

with root0 as (
select individual_id,
dt,
gender_cd,
case when age_in_months <0 then 0 when age_in_months >1440 then 1440 else age_in_months end as age_in_months
from dev_bpum_enc.a321276_d1a_${trunk}
union
select individual_id,
dt,
gender_cd,
case when age_in_months <0 then 0 when age_in_months >1440 then 1440 else age_in_months end as age_in_months
from dev_bpum_enc.a321276_d1c_${trunk}
),

root1 as (
select *,row_number() over (partition by individual_id,dt) as seqno
from root0
),
root2 as (
select individual_id,dt,gender_cd,age_in_months from root1 where seqno=1
),

x0 as (
select base.individual_id,base.dt,case when w2ind.ind is null then 0 else w2ind.ind end as ind
from dev_bpum_enc.a321276_d1a_${trunk} base 
left join dev_bpum_enc.a321276_v9_w2ind w2ind
on concat('days_cnt',cast(days_cnt as string)) = w2ind.cd
where days_cnt is not null
union
select base.individual_id,base.dt,case when w2ind.ind is null then 0 else w2ind.ind end as ind
from dev_bpum_enc.a321276_d1a_${trunk} base 
left join dev_bpum_enc.a321276_v9_w2ind w2ind
on concat('hcfa_plc_srv_cd',cast(hcfa_plc_srv_cd as string)) = w2ind.cd
where hcfa_plc_srv_cd is not null
union
select base.individual_id,base.dt,case when w2ind.ind is null then 0 else w2ind.ind end as ind
from dev_bpum_enc.a321276_d1a_${trunk} base 
left join dev_bpum_enc.a321276_v9_w2ind w2ind
on concat('src_specialty_cd',cast(src_specialty_cd as string)) = w2ind.cd
where src_specialty_cd is not null
union
select base.individual_id,base.dt,case when w2ind.ind is null then 0 else w2ind.ind end as ind
from dev_bpum_enc.a321276_d1b_${trunk} base 
left join dev_bpum_enc.a321276_v9_w2ind w2ind
on concat('icd9_dx_cd',cast(icd9_dx_cd as string)) = w2ind.cd
where icd9_dx_cd is not null
union
select base.individual_id,base.dt,case when w2ind.ind is null then 0 else w2ind.ind end as ind
from dev_bpum_enc.a321276_d1a_${trunk} base 
left join dev_bpum_enc.a321276_v9_w2ind w2ind
on concat('revenue_cd',cast(revenue_cd as string)) = w2ind.cd
where revenue_cd is not null
union
select base.individual_id,base.dt,case when w2ind.ind is null then 0 else w2ind.ind end as ind
from dev_bpum_enc.a321276_d1a_${trunk} base 
left join dev_bpum_enc.a321276_v9_w2ind w2ind
on concat('prcdr_cd',cast(prcdr_cd as string)) = w2ind.cd
where prcdr_cd is not null
union
select base.individual_id,base.dt,case when w2ind.ind is null then 0 else w2ind.ind end as ind
from dev_bpum_enc.a321276_d1a_${trunk} base 
left join dev_bpum_enc.a321276_v9_w2ind w2ind
on concat('prcdr_cd',cast(icd9_prcdr_cd as string)) = w2ind.cd
where icd9_prcdr_cd is not null
union
select base.individual_id,base.dt,case when w2ind.ind is null then 0 else w2ind.ind end as ind
from dev_bpum_enc.a321276_d1c_${trunk} base 
left join dev_bpum_enc.a321276_v9_w2ind w2ind
on concat('gpi',cast(gpi4 as string)) = w2ind.cd
where gpi4 is not null
),

x1 as (
select individual_id,dt,ind
from x0 
group by individual_id,dt,ind 
),

x2 as (
select *,row_number() over (partition by individual_id,dt) as seqno
from x1
),

x3 as (
select individual_id,dt,
concat_ws(',',collect_list(cast(ind as string))) as cd
from x2 
where seqno<=25
group by individual_id,dt
),

y0 as (
select base.individual_id,base.dt,case when w2ind.ind is null then 0 else w2ind.ind end as ind
from dev_bpum_enc.a321276_d1a_${trunk} base 
left join dev_bpum_enc.a321276_v9_w2ind_target w2ind
on concat('plc_srv_ctg_cd',cast(plc_srv_ctg_cd as string)) = w2ind.cd
union
select base.individual_id,base.dt,case when w2ind.ind is null then 0 else w2ind.ind end as ind
from dev_bpum_enc.a321276_d1a_${trunk} base 
left join dev_bpum_enc.a321276_v9_w2ind_target w2ind
on concat('srv_spclty_ctg_cd',cast(srv_spclty_ctg_cd as string))  = w2ind.cd
union
select base.individual_id,base.dt,case when w2ind.ind is null then 0 else w2ind.ind end as ind
from dev_bpum_enc.a321276_d1a_${trunk} base 
left join dev_bpum_enc.a321276_v9_w2ind_target w2ind
on concat('prcdr_group_nbr',cast(cpt_group_nbr as string))  = w2ind.cd
union
select base.individual_id,base.dt,case when w2ind.ind is null then 0 else w2ind.ind end as ind
from dev_bpum_enc.a321276_d1a_${trunk} base 
left join dev_bpum_enc.a321276_v9_w2ind_target w2ind
on concat('prcdr_group_nbr',cast(icd9_prcdr_group_nbr as string))  = w2ind.cd
where icd9_prcdr_group_nbr is not null
union
select base.individual_id,base.dt,case when w2ind.ind is null then 0 else w2ind.ind end as ind
from dev_bpum_enc.a321276_d1b_${trunk} base 
left join dev_bpum_enc.a321276_v9_w2ind_target w2ind
on concat('icd9_dx_cd',cast(icd9_dx_cd2 as string))  = w2ind.cd
union
select base.individual_id,base.dt,case when w2ind.ind is null then 0 else w2ind.ind end as ind
from dev_bpum_enc.a321276_d1c_${trunk} base 
left join dev_bpum_enc.a321276_v9_w2ind_target w2ind
on concat('gpi',cast(gpi2 as string))  = w2ind.cd
),

y1 as (
select individual_id,dt,ind from y0 group by individual_id,dt,ind
),

y2 as (
select individual_id,dt,
concat_ws(',',collect_list(cast(ind as string))) as target
from y1
group by individual_id,dt
)

select root2.individual_id,
root2.dt,
root2.gender_cd,
root2.age_in_months,
x3.cd,
y2.target
from root2 
inner join x3 
on root2.individual_id=x3.individual_id and root2.dt=x3.dt
inner join y2
on root2.individual_id=y2.individual_id and root2.dt=y2.dt
;


drop table if exists dev_bpum_enc.a321276_o2_${trunk};


drop table if exists dev_bpum_enc.a321276_finetuneIP_${trunk};
create table dev_bpum_enc.a321276_finetuneIP_${trunk} as

with x1 as (
select x0.individual_id,x0.dt,base.member_id
from dev_bpum_enc.a321276_o1_${trunk} x0 
inner join dev_bpum_enc.a321276_member base 
on x0.individual_id=base.individual_id
group by x0.individual_id,x0.dt,base.member_id
),

x2 as (
select individual_id,dt from x1 group by individual_id,dt
),

y0 as (
select x1.individual_id,x1.dt
from x1
inner join edw_archive_2020_enc.edw_medical_case AS medical_case
on x1.member_id=medical_case.member_id
where med_case_start_dt>x1.dt and add_months(x1.dt,6)>=med_case_start_dt
AND TRIM(med_cs_ps_ctg_cd) = 'I' 
AND TRIM(birth_outcome_cd) = 'N' 
AND icd9_dx_cd NOT LIKE 'O%' 
AND icd9_dx_cd NOT LIKE 'P%' 
AND icd9_dx_cd NOT LIKE 'Q%' 
AND icd9_dx_cd NOT LIKE 'S%' 
AND icd9_dx_cd NOT LIKE 'T%' 
AND icd9_dx_cd NOT LIKE 'V%' 
AND icd9_dx_cd NOT LIKE 'W%' 
AND icd9_dx_cd NOT LIKE 'X%' 
AND icd9_dx_cd NOT LIKE 'Y%' 
union
select x1.individual_id,x1.dt
from x1
inner join edw_archive_2018_enc.edw_medical_case AS medical_case
on x1.member_id=medical_case.member_id
where med_case_start_dt>x1.dt and add_months(x1.dt,6)>=med_case_start_dt
AND TRIM(med_cs_ps_ctg_cd) = 'I' 
AND TRIM(birth_outcome_cd) = 'N' 
AND icd9_dx_cd NOT LIKE 'O%' 
AND icd9_dx_cd NOT LIKE 'P%' 
AND icd9_dx_cd NOT LIKE 'Q%' 
AND icd9_dx_cd NOT LIKE 'S%' 
AND icd9_dx_cd NOT LIKE 'T%' 
AND icd9_dx_cd NOT LIKE 'V%' 
AND icd9_dx_cd NOT LIKE 'W%' 
AND icd9_dx_cd NOT LIKE 'X%' 
AND icd9_dx_cd NOT LIKE 'Y%' 
),
y1 as (
select individual_id,dt from y0 group by individual_id,dt
)

select x2.*,case when y1.individual_id is null then 0 else 1 end as ip_6m
from x2 left join y1 
on x2.individual_id=y1.individual_id and x2.dt=y1.dt;


drop table if exists dev_bpum_enc.a321276_o3_${trunk};
create table dev_bpum_enc.a321276_o3_${trunk} as

with x1 as (
select *,
row_number() over (partition by individual_id order by dt desc) as seqno
from dev_bpum_enc.a321276_o1_${trunk}
),

x2 as (
select * from x1 where seqno<=70
),

x3 as (
select x2.individual_id,x2.dt,x2.gender_cd,x2.age_in_months,x2.cd,x1.target,finetuneIP.ip_6m
from x2 
inner join x1 
on x2.individual_id=x1.individual_id and x2.seqno = x1.seqno-1
inner join dev_bpum_enc.a321276_finetuneIP_${trunk} finetuneIP
on x2.individual_id=finetuneIP.individual_id and x2.dt=finetuneIP.dt
),

x4 as (
select *,
row_number() over (partition by individual_id order by dt) as seqno
from x3
),

x5 as (
select individual_id,
concat_ws('*',collect_list(cast(gender_cd as string))) as gender_cd,
concat_ws('*',collect_list(cast(age_in_months as string))) as age_in_months,
concat_ws('*',collect_list(cast(cd as string))) as cd,
concat_ws('*',collect_list(cast(target as string))) as target,
concat_ws('*',collect_list(cast(ip_6m as string))) as ip_6m,
count(*) as dt_cnt
from x4 group by individual_id
)

select * from x5;














