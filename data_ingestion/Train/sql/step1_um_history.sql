-- build um history
-- sometimes um is not allowed in embedding.


drop table if exists dev_bpum_enc.a321276_um1_${trunk};
create table dev_bpum_enc.a321276_um1_${trunk} as

with x0 as (
select base.individual_id,
case when trim(member.gender_cd)='M' then 1 when trim(member.gender_cd)='F' then 0 else 2 end as gender_cd,
int(months_between(cast(decisions.src_posted_dts as date),member.birth_dt)*10) as age_in_months,
cast(decisions.src_posted_dts as date) as dt,
decisions.pme_reference_no,
trim(decisions.decision_cd) as decision_cd,
trim(decisions.bed_type_cd) as bed_type_cd,
trim(decisions.lnitm_type_cd) as lnitm_type_cd,
decisions.lnitm_sequence_no,
decisions.dcsn_sequence_no,
decisions.dcsnrsn_sequenc_no,
case when trim(tum_event.admit_class_cd)='' then null else trim(tum_event.admit_class_cd) end as admit_class_cd,
case when trim(tum_event.pme_event_type_cd)='' then null else trim(tum_event.pme_event_type_cd) end as pme_event_type_cd
from dev_bpum_enc.a321276_member base 
inner join edw_archive_2020_enc.edw_tum_decisions decisions
on base.member_id=decisions.member_id 
inner join edw_archive_2020_enc.edw_masked_member member
on base.member_id=member.member_id
left join edw_archive_2020_enc.edw_tum_event tum_event
on decisions.pme_reference_no = tum_event.pme_reference_no
where TRUE
and substr(cast(base.individual_id as string),-1,2)=cast(${trunk} as string)
and decisions.src_posted_dts >= '2018-01-01' and decisions.src_posted_dts <='2019-12-31'
),
x1 as (
select base.individual_id,
case when trim(member.gender_cd)='M' then 1 when trim(member.gender_cd)='F' then 0 else 2 end as gender_cd,
int(months_between(cast(decisions.src_posted_dts as date),member.birth_dt)*10) as age_in_months,
cast(decisions.src_posted_dts as date) as dt,
decisions.pme_reference_no,
trim(decisions.decision_cd) as decision_cd,
trim(decisions.bed_type_cd) as bed_type_cd,
trim(decisions.lnitm_type_cd) as lnitm_type_cd,
decisions.lnitm_sequence_no,
decisions.dcsn_sequence_no,
decisions.dcsnrsn_sequenc_no,
case when trim(tum_event.admit_class_cd)='' then null else trim(tum_event.admit_class_cd) end as admit_class_cd,
case when trim(tum_event.src_stay_srv_ty_cd)='' then null else trim(tum_event.src_stay_srv_ty_cd) end as src_stay_srv_ty_cd,
case when trim(tum_event.admission_type_cd)='' then null else trim(tum_event.admission_type_cd) end as admission_type_cd,
case when trim(tum_event.pme_event_type_cd)='' then null else trim(tum_event.pme_event_type_cd) end as pme_event_type_cd
from dev_bpum_enc.a321276_member base 
inner join edw_archive_2018_enc.edw_tum_decisions decisions
on base.member_id=decisions.member_id 
inner join edw_archive_2018_enc.edw_masked_member member
on base.member_id=member.member_id
left join edw_archive_2018_enc.edw_tum_event tum_event
on decisions.pme_reference_no = tum_event.pme_reference_no
where TRUE
and substr(cast(base.individual_id as string),-1,2)=cast(${trunk} as string)
and decisions.src_posted_dts >= '2016-01-01' and decisions.src_posted_dts <='2017-12-31'
),
x2 as (
select * from x0 union select * from x1
),
x3 as (
select *,
row_number() over (partition by individual_id,pme_reference_no,lnitm_sequence_no 
order by dcsn_sequence_no desc,dcsnrsn_sequenc_no desc) as seqno
from x2
),
x4 as (
select * from x3 where seqno=1
)

select * from x4;

drop table if exists dev_bpum_enc.a321276_um2_${trunk};
create table dev_bpum_enc.a321276_um2_${trunk} as
with u1b as (
select individual_id,pme_reference_no,max(dt) as dt
from dev_bpum_enc.a321276_um1_${trunk} 
group by individual_id,pme_reference_no
),

x1 as (
select 
u1b.individual_id,
u1b.dt,
case when trim(services.prcdr_cd)='' then null else trim(services.prcdr_cd) end as prcdr_cd
from u1b 
inner join edw_archive_2020_enc.edw_tum_services services 
on u1b.pme_reference_no=services.pme_reference_no
where u1b.dt>= '2018-01-01' and u1b.dt <='2019-12-31'
group by u1b.individual_id,
u1b.dt,
case when trim(services.prcdr_cd)='' then null else trim(services.prcdr_cd) end
),

x2 as (
select 
u1b.individual_id,
u1b.dt,
case when trim(services.prcdr_cd)='' then null else trim(services.prcdr_cd) end as prcdr_cd
from u1b 
inner join edw_archive_2018_enc.edw_tum_services services 
on u1b.pme_reference_no=services.pme_reference_no
where u1b.dt>= '2016-01-01' and u1b.dt <='2017-12-31'
group by u1b.individual_id,
u1b.dt,
case when trim(services.prcdr_cd)='' then null else trim(services.prcdr_cd) end
)

select * from x1 union select * from x2
;




drop table if exists dev_bpum_enc.a321276_um3_${trunk};
create table dev_bpum_enc.a321276_um3_${trunk} as
with u1b as (
select individual_id,pme_reference_no,max(dt) as dt
from dev_bpum_enc.a321276_um1_${trunk} 
group by individual_id,pme_reference_no
),

u2b1 as (
select 
u1b.individual_id,
u1b.dt,
case when trim(umdx.icd9_dx_cd)='' then null else split(trim(umdx.icd9_dx_cd),'\\.') end as x
from u1b 
inner join edw_archive_2020_enc.edw_tum_diagnosis umdx 
on u1b.pme_reference_no=umdx.pme_reference_no 
where u1b.dt>= '2018-01-01' and u1b.dt <='2019-12-31'
group by u1b.individual_id,
u1b.dt,
case when trim(umdx.icd9_dx_cd)='' then null else split(trim(umdx.icd9_dx_cd),'\\.') end
),
u2b2 as (
select 
u1b.individual_id,
u1b.dt,
case when trim(umdx.icd9_dx_cd)='' then null else split(trim(umdx.icd9_dx_cd),'\\.') end as x
from u1b 
inner join edw_archive_2018_enc.edw_tum_diagnosis umdx 
on u1b.pme_reference_no=umdx.pme_reference_no 
where u1b.dt>= '2016-01-01' and u1b.dt <='2017-12-31'
group by u1b.individual_id,
u1b.dt,
case when trim(umdx.icd9_dx_cd)='' then null else split(trim(umdx.icd9_dx_cd),'\\.') end
),

u2b as (
select * from u2b1 union select * from u2b1
),

u3b as (
select individual_id,
dt,
concat_ws('.',x[0],substr(x[1],1,2)) as icd9_dx_cd
from u2b
)

select * from u3b;

