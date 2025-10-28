
-- this is all member who were active as of 2019 December,
-- including both medicare and commericial
-- make sure individual - membre has 1:1 relation to exclud overefit at any scenario
drop table if exists dev_bpum_enc.a321276_member;
create table dev_bpum_enc.a321276_member as

with mbr1 as (
select edw_indvdl_cust_dist.individual_id,
membership.member_id,
membership.business_ln_cd
from edw_enc.edw_emis_membership membership
inner join edw_enc.edw_indvdl_cust_dist edw_indvdl_cust_dist
on membership.member_id=edw_indvdl_cust_dist.member_id
where membership.eff_dt = '2019-12-16'
),
mbr2 as (
select edw_indvdl_cust_dist.individual_id,
membership.member_id,
membership.business_ln_cd
from edw_archive_2020_enc.edw_emis_membership membership
inner join edw_archive_2020_enc.edw_indvdl_cust_dist edw_indvdl_cust_dist
on membership.member_id=edw_indvdl_cust_dist.member_id
where membership.eff_dt = '2019-12-16'
),
mbr3 as (
select * from mbr1 
union
select * from mbr2 
),
mbr4 as 
(select individual_id,
member_id,
business_ln_cd from 
mbr3
group by individual_id,
member_id,
business_ln_cd
),
mbr5 as ( ---here mak sure 1 member only map to 1 individual_id, otherwise to split by individual 
--may cause overlap between train/val 
select member_id,
count(*) as cnt
from mbr4
group by member_id
having cnt=1
),
mbr6 as ( ---here mak sure 1 individual only map to 1 membere, so one patient has continous medical history, this is to ensure transformer on complete med history
select individual_id,
count(*) as cnt
from mbr4
group by individual_id
having cnt=1
),
mbr7 as (
select mbr4.*
from mbr4 
inner join mbr5 
on mbr4.member_id=mbr5.member_id
inner join mbr6
on mbr4.individual_id=mbr6.individual_id
)
select * from mbr7;
