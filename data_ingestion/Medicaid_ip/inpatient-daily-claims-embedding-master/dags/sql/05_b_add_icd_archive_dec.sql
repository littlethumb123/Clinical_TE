-- drop table if exists `{{DEC_TARGET_DB}}.{{PREFIX}}_icd9_y2021`;
-- create table `{{DEC_TARGET_DB}}.{{PREFIX}}_icd9_y2021`
--     OPTIONS ( labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")])
-- as
-- select
--         base.individual_id,
--         base.member_id,
--         base.claim_line_id,
--         base.srv_start_dt,
--         case
--             when b.sequence_id = 1  and b.icd9_dx_cd is not null then b.icd9_dx_cd
--             else null end as icd1,
--         case
--             when b.sequence_id = 2  and b.icd9_dx_cd is not null then b.icd9_dx_cd
--             else null end as icd2,
--         case
--             when b.sequence_id = 3  and b.icd9_dx_cd is not null then b.icd9_dx_cd
--             else null end as icd3
--     from
--             ( select individual_id, member_id, claim_line_id, srv_start_dt
--               from `{{DEC_TARGET_DB}}.{{PREFIX}}_d1a_score_ending_tmp`
--               where srv_start_dt >= '2019-01-01' and srv_start_dt <= '2020-12-31'
--             ) base
--         inner join
--             ( select
--                 b1.claim_line_id,
--                 b1.sequence_id,
--                 case
--                     when (array_length(b1.xsplit) = 1) then b1.xw
--                     else concat(xsplit[offset(0)], '.', substr(b1.xsplit[offset(1)], 1, 2))
--                 end as icd9_dx_cd
--             from
--                 ( select
--                   claim_line_id,
--                   split(trim(icd9_dx_cd),'.') as xsplit,
--                   case when regexp_contains(trim(icd9_dx_cd), r'\.\w{2}') 
--                     then concat(regexp_extract(trim(icd9_dx_cd), r'^(.*)\.'),regexp_extract(trim(icd9_dx_cd), r'\.\w{2}'))
--                     else trim(icd9_dx_cd) end as icd9_dx_cd,
--                   cast(sequence_id as int) as sequence_id
--               from `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_DX_Y2021`
--               where cast(sequence_id as int) < 4
--             ) b1
--         ) as b
--         on base.claim_line_id=b.claim_line_id
-- ;

drop table if exists `{{DEC_TARGET_DB}}.{{PREFIX}}_icd9_y2020`;
create table `{{DEC_TARGET_DB}}.{{PREFIX}}_icd9_y2020`
    OPTIONS ( labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")])
as
 select
         base.individual_id,
         base.member_id,
         base.claim_line_id,
         base.srv_start_dt,
         case
             when b.sequence_id = 1  and b.icd9_dx_cd is not null then b.icd9_dx_cd
             else null
         end as icd1,
         case
             when b.sequence_id = 2  and b.icd9_dx_cd is not null then b.icd9_dx_cd
             else null
         end as icd2,
         case
             when b.sequence_id = 3  and b.icd9_dx_cd is not null then b.icd9_dx_cd
             else null
         end as icd3
     from
         ( select individual_id, member_id, claim_line_id, srv_start_dt
           from `{{DEC_TARGET_DB}}.{{PREFIX}}_d1a_score_ending_tmp`
           where srv_start_dt >= date('2018-01-01') and srv_start_dt <= date('2019-12-31')
         ) base
        inner join
         ( select claim_line_id,
                  split(trim(icd9_dx_cd),'.') as icd9_dx_cd,
                  cast(sequence_id as int) as sequence_id
                from `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_DX_Y2020`
                where cast(sequence_id as int) < 4
        ) as b
        on base.claim_line_id=safe_cast(b.claim_line_id as string)
;


-- drop table if exists `{{DEC_TARGET_DB}}.{{PREFIX}}_icd9_y2019`;
-- create table `{{DEC_TARGET_DB}}.{{PREFIX}}_icd9_y2019`
--     OPTIONS ( labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")]) as
--     select
--         base.individual_id,
--         base.member_id,
--         base.claim_line_id,
--         base.srv_start_dt,
--         case
--             when b.sequence_id = 1  and b.icd9_dx_cd is not null then b.icd9_dx_cd
--             else null
--         end as icd1,
--         case
--             when b.sequence_id = 2  and b.icd9_dx_cd is not null then b.icd9_dx_cd
--             else null
--         end as icd2,
--         case
--             when b.sequence_id = 3  and b.icd9_dx_cd is not null then b.icd9_dx_cd
--             else null
--         end as icd3
--     from
--          (   select individual_id, member_id, claim_line_id, srv_start_dt
--              from `{{DEC_TARGET_DB}}.{{PREFIX}}_d1a_score_ending_tmp`
--              where srv_start_dt >= '2017-01-01' and srv_start_dt <= '2018-12-31'
--          ) base
--         inner join
--          (
--              select
--                  b1.claim_line_id,
--                  b1.sequence_id,
--                  case
--                      when (array_length(b1.xsplit) = 1) then b1.xw
--                      else concat(xsplit[offset(0)], '.', substr(b1.xsplit[offset(1)], 1, 2))
--                  end as icd9_dx_cd
--              from
--             (   select
--                     claim_line_id,
--                     split(trim(icd9_dx_cd),'.') as xsplit,
--                     case when regexp_contains(trim(icd9_dx_cd), r'\.\w{2}') 
--                         then concat(regexp_extract(trim(icd9_dx_cd), r'^(.*)\.'),regexp_extract(trim(icd9_dx_cd), r'\.\w{2}'))
--                         else trim(icd9_dx_cd) end as icd9_dx_cd,
--                     cast(sequence_id as int) as sequence_id
--                 from `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_DX_Y2019`
--                 where cast(sequence_id as int) < 4
--             ) b1
--         ) as b
--         on base.claim_line_id=b.claim_line_id
-- ;

drop table if exists `{{DEC_TARGET_DB}}.{{PREFIX}}_icd9_y2018`;
create table `{{DEC_TARGET_DB}}.{{PREFIX}}_icd9_y2018`
    OPTIONS ( labels=[("owner", "{{OWNER}}"),("costcenter", "{{COSTCENTER}}")]) as
     select
        base.individual_id,
        base.member_id,
        base.claim_line_id,
        base.srv_start_dt,
        case
            when b.sequence_id = 1  and b.icd9_dx_cd is not null then b.icd9_dx_cd
            else null
        end as icd1,
        case
            when b.sequence_id = 2  and b.icd9_dx_cd is not null then b.icd9_dx_cd
            else null
        end as icd2,
        case
            when b.sequence_id = 3  and b.icd9_dx_cd is not null then b.icd9_dx_cd
            else null
        end as icd3
     from
            ( select individual_id, member_id, claim_line_id, srv_start_dt
              from `{{DEC_TARGET_DB}}.{{PREFIX}}_d1a_score_ending_tmp`
              where srv_start_dt >= date('2016-01-01') and srv_start_dt <= date('2017-12-31')
            ) base
        inner join
            ( select  claim_line_id,
                  split(trim(icd9_dx_cd),'.') as icd9_dx_cd,
                  cast(sequence_id as int) as sequence_id
                from `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_ICD9_DX_Y2018`
                where cast(sequence_id as int) < 4
        ) as b
        on base.claim_line_id=safe_cast(b.claim_line_id as string)
;
