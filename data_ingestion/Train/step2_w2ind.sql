-- one for x 
-- one for y



drop table if exists dev_bpum_enc.a321276_v9_w2ind;
create table dev_bpum_enc.a321276_v9_w2ind as

with x1 as (
select cd from dev_bpum_enc.a321276_d1d_0
union select cd from dev_bpum_enc.a321276_d1d_1
union select cd from dev_bpum_enc.a321276_d1d_2
union select cd from dev_bpum_enc.a321276_d1d_3
union select cd from dev_bpum_enc.a321276_d1d_4
union select cd from dev_bpum_enc.a321276_d1d_5
union select cd from dev_bpum_enc.a321276_d1d_6
union select cd from dev_bpum_enc.a321276_d1d_7
union select cd from dev_bpum_enc.a321276_d1d_8
union select cd from dev_bpum_enc.a321276_d1d_9
),

x2 as (
select cd 
from x1 
group by cd
),

x3 as (
select cd,
row_number() over() as ind
from x2
),

x4 as (
select '' as cd, 0 as ind 
union select * from x3
)

select * from x4;




drop table if exists dev_bpum_enc.a321276_v9_w2ind_target;
create table dev_bpum_enc.a321276_v9_w2ind_target as

with x1 as (
select cd from dev_bpum_enc.a321276_d1e_0
union select cd from dev_bpum_enc.a321276_d1e_1
union select cd from dev_bpum_enc.a321276_d1e_2
union select cd from dev_bpum_enc.a321276_d1e_3
union select cd from dev_bpum_enc.a321276_d1e_4
union select cd from dev_bpum_enc.a321276_d1e_5
union select cd from dev_bpum_enc.a321276_d1e_6
union select cd from dev_bpum_enc.a321276_d1e_7
union select cd from dev_bpum_enc.a321276_d1e_8
union select cd from dev_bpum_enc.a321276_d1e_9
),

x2 as (
select cd 
from x1 
group by cd
),

x3 as (
select cd,
row_number() over() as ind
from x2
),

x4 as (
select '' as cd, 0 as ind 
union select * from x3
)

select * from x4;


