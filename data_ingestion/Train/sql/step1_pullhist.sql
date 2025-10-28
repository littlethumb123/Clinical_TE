
-- for embedding purpose, it's ok that rceiv dt > '2019-12-31'
-- for prediction, need to make sure BOTH receive and srv dt < index dt
-- in the end of this code, build tables for w2ind

SET hivevar:trunk=;

drop table if exists dev_bpum_enc.a321276_d1a_${trunk};
create table dev_bpum_enc.a321276_d1a_${trunk} as
with x1 as (
select
base.individual_id, 
base.member_id,
clm.claim_line_id,
clm.srv_start_dt  as dt,
case when (clm.days_cnt is null or clm.days_cnt <0) then 99 when clm.days_cnt > 10 then 11 else clm.days_cnt end as days_cnt,
case when trim(member.gender_cd)='M' then 1 when trim(member.gender_cd)='F' then 0 else 2 end as gender_cd,
int(months_between(clm.srv_start_dt ,member.birth_dt)) as age_in_months,
case when trim(clm.revenue_cd)='' then null else trim(clm.revenue_cd) end as revenue_cd,
case when trim(clm.hcfa_plc_srv_cd)='' then null else trim(clm.hcfa_plc_srv_cd) end as hcfa_plc_srv_cd,
case when trim(clm.src_specialty_cd)='' then null else trim(clm.src_specialty_cd) end as src_specialty_cd,
case when trim(clm.prcdr_cd)='' then null else trim(clm.prcdr_cd) end as prcdr_cd,
case when trim(icd_prc.icd9_prcdr_cd)='' then null else trim(icd_prc.icd9_prcdr_cd) end as icd9_prcdr_cd,
--
case when trim(clm.srv_spclty_ctg_cd)='' then null else trim(clm.srv_spclty_ctg_cd) end as srv_spclty_ctg_cd,
case when trim(clm.plc_srv_ctg_cd)='' then null else trim(clm.plc_srv_ctg_cd) end as plc_srv_ctg_cd,
prcdr1.prcdr_group_nbr as cpt_group_nbr,
prcdr2.prcdr_group_nbr as icd9_prcdr_group_nbr
from dev_bpum_enc.a321276_member base 
inner join edw_archive_2020_enc.edw_emis_claim_line clm 
on base.member_id=clm.member_id
inner join edw_archive_2020_enc.edw_masked_member member
on base.member_id=member.member_id
left join edw_archive_2020_enc.edw_CLM_LN_X_ICD9_PRCD icd_prc
on clm.claim_line_id=icd_prc.claim_line_id
left join edw_enc.edw_procedure prcdr1
on trim(clm.prcdr_cd)=trim(prcdr1.prcdr_cd)
left join edw_enc.edw_procedure prcdr2
on trim(icd_prc.icd9_prcdr_cd)=trim(prcdr2.prcdr_cd)
where TRUE
and substr(cast(base.individual_id as string),-1,2)=cast(${trunk} as string)
and clm.duplicate_ind='N' and clm.summarized_srv_ind='Y'
and clm.srv_start_dt  >= '2018-01-01' and clm.srv_start_dt  <='2019-12-31'
),

x2 as (
select
base.individual_id, 
base.member_id,
clm.claim_line_id,
clm.srv_start_dt  as dt,
case when (clm.days_cnt is null or clm.days_cnt <0) then 99 when clm.days_cnt > 10 then 11 else clm.days_cnt end as days_cnt,
case when trim(member.gender_cd)='M' then 1 when trim(member.gender_cd)='F' then 0 else 2 end as gender_cd,
int(months_between(clm.srv_start_dt ,member.birth_dt)) as age_in_months,
case when trim(clm.revenue_cd)='' then null else trim(clm.revenue_cd) end as revenue_cd,
case when trim(clm.hcfa_plc_srv_cd)='' then null else trim(clm.hcfa_plc_srv_cd) end as hcfa_plc_srv_cd,
case when trim(clm.src_specialty_cd)='' then null else trim(clm.src_specialty_cd) end as src_specialty_cd,
case when trim(clm.prcdr_cd)='' then null else trim(clm.prcdr_cd) end as prcdr_cd,
case when trim(icd_prc.icd9_prcdr_cd)='' then null else trim(icd_prc.icd9_prcdr_cd) end as icd9_prcdr_cd,
--
case when trim(clm.srv_spclty_ctg_cd)='' then null else trim(clm.srv_spclty_ctg_cd) end as srv_spclty_ctg_cd,
case when trim(clm.plc_srv_ctg_cd)='' then null else trim(clm.plc_srv_ctg_cd) end as plc_srv_ctg_cd,
prcdr1.prcdr_group_nbr as cpt_group_nbr,
prcdr2.prcdr_group_nbr as icd9_prcdr_group_nbr
from dev_bpum_enc.a321276_member base 
inner join edw_archive_2018_enc.edw_emis_claim_line clm 
on base.member_id=clm.member_id
inner join edw_archive_2018_enc.edw_masked_member member
on base.member_id=member.member_id
left join edw_archive_2018_enc.edw_CLM_LN_X_ICD9_PRCD icd_prc
on clm.claim_line_id=icd_prc.claim_line_id
left join edw_enc.edw_procedure prcdr1
on trim(clm.prcdr_cd)=trim(prcdr1.prcdr_cd)
left join edw_enc.edw_procedure prcdr2
on trim(icd_prc.icd9_prcdr_cd)=trim(prcdr2.prcdr_cd)
where TRUE
and substr(cast(base.individual_id as string),-1,2)=cast(${trunk} as string)
and clm.duplicate_ind='N' and clm.summarized_srv_ind='Y'
and clm.srv_start_dt  >= '2016-01-01' and clm.srv_start_dt  <='2017-12-31'
)

select * from x1 union select * from x2;



;
-- most of icd9_prc can find prc group match.
--643 unique revenue_cd
--56 unique hcfa_plc_srv_cd
--592 unique specialty cd
--174 uniqu prcd group
-- unique icd group

-- this is derived table from claim table - diagnosis
drop table if exists dev_bpum_enc.a321276_d1b_${trunk};
create table dev_bpum_enc.a321276_d1b_${trunk} as
with x1 as (
select 
base.individual_id, 
base.member_id,
base.claim_line_id,
base.dt,
split(trim(b.icd9_dx_cd),'\\.') as x
from
dev_bpum_enc.a321276_d1a_${trunk} base 
inner join edw_archive_2020_enc.edw_CLM_LN_X_ICD9_DX b 
on base.claim_line_id=b.claim_line_id
left join edw_enc.edw_icd9_diagnosis dx
on trim(b.icd9_dx_cd)=trim(dx.icd9_dx_cd)
where base.dt >= '2018-01-01' and base.dt <='2019-12-31'
and cast(b.sequence_id as int) <4
),
x2 as (
select 
base.individual_id, 
base.member_id,
base.claim_line_id,
base.dt,
split(trim(b.icd9_dx_cd),'\\.') as x
from
dev_bpum_enc.a321276_d1a_${trunk} base 
inner join edw_archive_2018_enc.edw_CLM_LN_X_ICD9_DX b 
on base.claim_line_id=b.claim_line_id
left join edw_enc.edw_icd9_diagnosis dx
on trim(b.icd9_dx_cd)=trim(dx.icd9_dx_cd)
where base.dt >= '2016-01-01' and base.dt <='2017-12-31'
and cast(b.sequence_id as int) <4
),
x3 as (
select * from x1 union select * from x2
),
x4 as (
select individual_id,
member_id,
claim_line_id,
dt,
x[0] as icd9_dx_cd2,
concat_ws('.',x[0],substr(x[1],1,2)) as icd9_dx_cd
from x3
)
select * from x4
;
--197 unique icd group


drop table if exists dev_bpum_enc.a321276_d1c_${trunk};
create table dev_bpum_enc.a321276_d1c_${trunk} as

with x1 as (
select base.individual_id,
base.member_id,
case when trim(member.gender_cd)='M' then 1 when trim(member.gender_cd)='F' then 0 else 2 end as gender_cd,
int(months_between(rx.disp_dt,member.birth_dt)) as age_in_months,
rx.disp_dt as dt,
concat('gpi', SUBSTR(trim(rx.adjudicated_gpi_cd), 1, 4)) as gpi4,
concat('gpi', SUBSTR(trim(rx.adjudicated_gpi_cd), 1, 2)) as gpi2
from dev_bpum_enc.a321276_member base 
inner join edw_archive_2020_enc.edw_unmsk_rx_claim_dtl rx 
on base.member_id=rx.member_id 
inner join edw_archive_2020_enc.edw_masked_member member
on base.member_id=member.member_id
where TRUE
and substr(cast(base.individual_id as string),-1,2)=cast(${trunk} as string)
and rx.disp_dt >= '2018-01-01' and rx.disp_dt <='2019-12-31'
),
x2 as (
select base.individual_id,
base.member_id,
case when trim(member.gender_cd)='M' then 1 when trim(member.gender_cd)='F' then 0 else 2 end as gender_cd,
int(months_between(rx.disp_dt,member.birth_dt)) as age_in_months,
rx.disp_dt as dt,
concat('gpi', SUBSTR(trim(rx.adjudicated_gpi_cd), 1, 4)) as gpi4,
concat('gpi', SUBSTR(trim(rx.adjudicated_gpi_cd), 1, 2)) as gpi2
from dev_bpum_enc.a321276_member base 
inner join edw_archive_2018_enc.edw_unmsk_rx_claim_dtl rx 
on base.member_id=rx.member_id 
inner join edw_archive_2018_enc.edw_masked_member member
on base.member_id=member.member_id
where TRUE
and substr(cast(base.individual_id as string),-1,2)=cast(${trunk} as string)
and rx.disp_dt >= '2016-01-01' and rx.disp_dt <='2017-12-31'
)
select * from x1 union select * from x2;







drop table if exists dev_bpum_enc.a321276_d1d_${trunk};
create table dev_bpum_enc.a321276_d1d_${trunk} as

with x1 as (
select 
concat('gender_cd',cast(gender_cd as string)) as cd
from dev_bpum_enc.a321276_d1a_${trunk}
where gender_cd is not null
union
select 
concat('days_cnt',cast(days_cnt as string)) as cd
from dev_bpum_enc.a321276_d1a_${trunk}
where days_cnt is not null
union
select 
concat('revenue_cd',cast(revenue_cd as string)) as cd
from dev_bpum_enc.a321276_d1a_${trunk}
where revenue_cd is not null
union
select 
concat('hcfa_plc_srv_cd',cast(hcfa_plc_srv_cd as string)) as cd
from dev_bpum_enc.a321276_d1a_${trunk}
where hcfa_plc_srv_cd is not null
union
select 
concat('src_specialty_cd',cast(src_specialty_cd as string)) as cd
from dev_bpum_enc.a321276_d1a_${trunk}
where src_specialty_cd is not null
union
select 
concat('prcdr_cd',cast(prcdr_cd as string)) as cd
from dev_bpum_enc.a321276_d1a_${trunk}
where prcdr_cd is not null
union
select 
concat('prcdr_cd',cast(icd9_prcdr_cd as string)) as cd
from dev_bpum_enc.a321276_d1a_${trunk}
where icd9_prcdr_cd is not null
union
select 
concat('icd9_dx_cd',cast(icd9_dx_cd as string)) as cd
from dev_bpum_enc.a321276_d1b_${trunk}
where icd9_dx_cd is not null
)

select cd from x1 group by cd;



drop table if exists dev_bpum_enc.a321276_d1e_${trunk};
create table dev_bpum_enc.a321276_d1e_${trunk} as

with x1 as (
select 
concat('plc_srv_ctg_cd',cast(plc_srv_ctg_cd as string)) as cd
from dev_bpum_enc.a321276_d1a_${trunk}
where plc_srv_ctg_cd is not null
union
select 
concat('srv_spclty_ctg_cd',cast(srv_spclty_ctg_cd as string)) as cd
from dev_bpum_enc.a321276_d1a_${trunk}
where srv_spclty_ctg_cd is not null
union
select 
concat('prcdr_group_nbr',cast(cpt_group_nbr as string)) as cd
from dev_bpum_enc.a321276_d1a_${trunk}
where cpt_group_nbr is not null
union
select 
concat('prcdr_group_nbr',cast(icd9_prcdr_group_nbr as string)) as cd
from dev_bpum_enc.a321276_d1a_${trunk}
where icd9_prcdr_group_nbr is not null
union
select 
concat('icd9_dx_cd',cast(icd9_dx_cd2 as string)) as cd
from dev_bpum_enc.a321276_d1b_${trunk}
where icd9_dx_cd2 is not null
union
select 
concat('gpi',cast(gpi2 as string)) as cd
from dev_bpum_enc.a321276_d1c_${trunk}
where gpi2 is not null
)

select cd from x1 group by cd;

