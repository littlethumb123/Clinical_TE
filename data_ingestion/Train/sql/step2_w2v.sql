-- build data for w2v, pretrained code emb for transformer
-- in this repo, used cold start.

drop table if exists dev_bpum_enc.a321276_v9_w2v_${trunk};
create table dev_bpum_enc.a321276_v9_w2v as

with a1 as (
select individual_id,concat('revenue_cd',cast(revenue_cd as string)) as cd,dt
from dev_bpum_enc.a321276_d1a_${trunk}
where trim(revenue_cd) <>'' and revenue_cd is not null
group by individual_id,concat('revenue_cd',cast(revenue_cd as string)),dt
),

a2 as (
select *,
row_number() over (partition by individual_id order by dt) as seqno
from a1
),

a3 as (
select individual_id,
concat_ws(',',collect_list(cast(cd as string))) as cd,
'revenue_cd' as tp,
count(*) as cnt
from a2 group by individual_id having cnt>=10
),

----

b1 as (
select individual_id,concat('hcfa_plc_srv_cd',cast(hcfa_plc_srv_cd as string)) as cd,dt
from dev_bpum_enc.a321276_d1a_${trunk}
where trim(hcfa_plc_srv_cd) <>'' and hcfa_plc_srv_cd is not null
group by individual_id,concat('hcfa_plc_srv_cd',cast(hcfa_plc_srv_cd as string)),dt
),

b2 as (
select *,
row_number() over (partition by individual_id order by dt) as seqno
from b1
),

b3 as (
select individual_id,
concat_ws(',',collect_list(cast(cd as string))) as cd,
'hcfa_plc_srv_cd' as tp,
count(*) as cnt
from b2 group by individual_id having cnt>=10
),

----

c1 as (
select individual_id,concat('src_specialty_cd',cast(src_specialty_cd as string)) as cd,dt
from dev_bpum_enc.a321276_d1a_${trunk}
where trim(src_specialty_cd) <>'' and src_specialty_cd is not null
group by individual_id,concat('src_specialty_cd',cast(src_specialty_cd as string)),dt
),

c2 as (
select *,
row_number() over (partition by individual_id order by dt) as seqno
from c1
),

c3 as (
select individual_id,
concat_ws(',',collect_list(cast(cd as string))) as cd,
'src_specialty_cd' as tp,
count(*) as cnt
from c2 group by individual_id having cnt>=10
),

----

d1 as (
select individual_id,concat('prcdr_cd',cast(prcdr_cd as string)) as cd,dt
from dev_bpum_enc.a321276_d1a_${trunk}
where trim(prcdr_cd) <>'' and prcdr_cd is not null
group by individual_id,concat('prcdr_cd',cast(prcdr_cd as string)),dt
union
select individual_id,concat('prcdr_cd',cast(icd9_prcdr_cd as string)) as cd,dt
from dev_bpum_enc.a321276_d1a_${trunk}
where trim(icd9_prcdr_cd) <>'' and icd9_prcdr_cd is not null
group by individual_id,concat('prcdr_cd',cast(icd9_prcdr_cd as string)),dt
),

d2 as (
select *,
row_number() over (partition by individual_id order by dt) as seqno
from d1
),

d3 as (
select individual_id,
concat_ws(',',collect_list(cast(cd as string))) as cd,
'prcdr_cd' as tp,
count(*) as cnt
from d2 group by individual_id having cnt>=10
),

----

e1 as (
select individual_id,concat('icd9_dx_cd',cast(icd9_dx_cd as string)) as cd,dt
from dev_bpum_enc.a321276_d1b_${trunk}
where trim(icd9_dx_cd) <>'' and icd9_dx_cd is not null
group by individual_id,concat('icd9_dx_cd',cast(icd9_dx_cd as string)),dt
),

e2 as (
select *,
row_number() over (partition by individual_id order by dt) as seqno
from e1
),

e3 as (
select individual_id,
concat_ws(',',collect_list(cast(cd as string))) as cd,
'icd9_dx_cd' as tp,
count(*) as cnt
from e2 group by individual_id having cnt>=10
),

----

f1 as (
select individual_id,concat('gpi',cast(gpi4 as string)) as cd,dt
from dev_bpum_enc.a321276_d1c_${trunk}
where trim(gpi4) <>'' and gpi4 is not null
group by individual_id,concat('gpi',cast(gpi4 as string)),dt
),

f2 as (
select *,
row_number() over (partition by individual_id order by dt) as seqno
from f1
),

f3 as (
select individual_id,
concat_ws(',',collect_list(cast(cd as string))) as cd,
'gpi' as tp,
count(*) as cnt
from f2 group by individual_id having cnt>=10
),

mg as (
select * from a3 
union select * from b3
union select * from c3
union select * from d3
union select * from e3
union select * from f3
)

select * from mg;





