create or replace table `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_d1a_score_ending_tmp`
    CLUSTER BY member_id
    OPTIONS ( labels=[("owner", "{{ params.owner }}"),("costcenter", "{{ params.costcenter }}")],expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 5 DAY))
as
   select individual_id,
          member_id,
          index_dt,
          claim_line_id,
          dt,
          days_cnt,
          gender_cd,
          age_in_months,
          revenue_cd,
          hcfa_plc_srv_cd,
          src_specialty_cd,
          prcdr_cd,
          icd9_prcdr_cd
   from`{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_daily_claims`
  union distinct
   select individual_id,
          member_id,
          index_dt,
          claim_line_id,
          dt,
          days_cnt,
          gender_cd,
          age_in_months,
          revenue_cd,
          hcfa_plc_srv_cd,
          src_specialty_cd,
          prcdr_cd,
          icd9_prcdr_cd
   from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_monthly_claims`
  union distinct
   select individual_id,
          member_id,
          index_dt,
          claim_line_id,
          dt,
          days_cnt,
          gender_cd,
          age_in_months,
          revenue_cd,
          hcfa_plc_srv_cd,
          src_specialty_cd,
          prcdr_cd,
          icd9_prcdr_cd
   from `{{ dag_run.conf["DEC_DATASET"] }}.{{ dag_run.conf["prefix"] }}_archive_claims`
;