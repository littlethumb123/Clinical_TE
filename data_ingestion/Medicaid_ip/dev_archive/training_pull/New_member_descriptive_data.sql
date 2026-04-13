CREATE OR REPLACE TABLE `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_pre_descriptives`
OPTIONS (labels = [("owner", "palmere1_aetna_com"),("cost_center", "13070")])
AS
SELECT
  base.asdb_member_key
  , base.index_dt
  , base.coa_population_category
  , CASE WHEN TRIM(base.coa_population_category) = 'Foster' THEN 'Foster'
          WHEN TRIM(base.coa_population_category) IN ('CHIP', 'TANF') AND demo.agenbr BETWEEN 0 AND 18 AND demo.tenure_yr1 > 8 THEN 'Medicaid KID'
          WHEN TRIM(base.coa_population_category) IN ('CHIP', 'TANF') AND demo.agenbr BETWEEN 0 AND 18 AND demo.tenure_yr1 BETWEEN 4 AND 8 THEN 'Medicaid Kid new 48'
          WHEN TRIM(base.coa_population_category) IN ('CHIP', 'TANF') AND demo.agenbr BETWEEN 0 AND 18 AND demo.tenure_yr1 BETWEEN 0 AND 3 THEN 'Medicaid Kid new 03'
          WHEN TRIM(base.coa_population_category) IN ('ABD Non Dual LTSS', 'LTSS Only', 'Dual Int LTSS', 'Dual Elig LTSS') THEN 'LTSS'
          WHEN TRIM(base.coa_population_category) IN ('ABD Non Dual Non LTSS', 'DD') AND demo.tenure_yr1 > 8 THEN 'ABD'
          WHEN TRIM(base.coa_population_category) IN ('ABD Non Dual Non LTSS', 'DD') AND demo.tenure_yr1 <= 8 THEN 'ABD new'
          WHEN TRIM(base.coa_population_category) IN ('BH Int SMI', 'BH Only') THEN 'BH'
          WHEN TRIM(base.coa_population_category) IN ('DSNP Medicare Only', 'Dual Elig NonLTSS', 'Dual Int DD', 'Dual Int NonLTSS') THEN 'Dual'
          WHEN TRIM(base.coa_population_category) IN ('CHIP', 'TANF', 'Expansion') AND demo.agenbr > 18 AND demo.tenure_yr1 > 8 THEN 'Medicaid Adult'
          WHEN TRIM(base.coa_population_category) IN ('CHIP', 'TANF', 'Expansion') AND demo.agenbr > 18 AND demo.tenure_yr1 BETWEEN 4 AND 8 THEN 'Medicaid Adult new 48'
          WHEN TRIM(base.coa_population_category) IN ('CHIP', 'TANF', 'Expansion') AND demo.agenbr > 18 AND demo.tenure_yr1 BETWEEN 0 AND 3 THEN 'Medicaid Adult new 03'
          ELSE 'Medicaid Adult' END AS ss_cohort
  , demo.agenbr --used to pick ss_cohort
  , demo.tenure_yr1 --used to calculate PMPM and PTPY
  , cond.major_chronic_cnt --average
  , cost.sum_paid_amt --PMPM by ss_cohort
  , cost.inpatient_cost --PMPM by ss_cohort
  , cost.emergency_cost --PMPM by ss_cohort
  , cost.outpatient_cost --PMPM by ss_cohort
  , ed.ed_flag --% w/flag = 1 by ss_cohort
  , ed.sum_ed_visits --PTPY by ss_cohort
  , ed.sum_avoidable --PTPY by ss_cohort
  , ed.sum_preventable --PTPY by ss_cohort
  , ed.sum_unnecessary --PTPY by ss_cohort
  , ip.acute_ip_flag --% w/flag = 1 by ss_cohort
  , ip.sum_acute_ip_admits --PTPY by ss_cohort
FROM
  `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_member_index` AS base
LEFT JOIN
  (SELECT asdb_member_key, agenbr, tenure_yr1 - 1 AS tenure_yr1, post_mnths FROM `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_demographics`) AS demo
    ON base.asdb_member_key = demo.asdb_member_key
LEFT JOIN
  `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_conditions` AS cond
    ON base.asdb_member_key = cond.asdb_member_key
LEFT JOIN
  `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_other_cost_utilization_yr1` AS cost
    ON base.asdb_member_key = cost.asdb_member_key
LEFT JOIN
  `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_ed_yr1` AS ed
    ON base.asdb_member_key = ed.asdb_member_key
LEFT JOIN
  `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_ip_yr1` AS ip
    ON base.asdb_member_key = ip.asdb_member_key
WHERE 1=1
  AND demo.post_mnths >= 6
  AND NOT base.asdb_plan_key IN (33, 54)
;

SELECT COUNT(ss_cohort), ss_cohort FROM `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_pre_descriptives` GROUP BY ss_cohort;

SELECT COUNT(tenure_yr1), tenure_yr1 FROM `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_pre_descriptives` GROUP BY tenure_yr1;