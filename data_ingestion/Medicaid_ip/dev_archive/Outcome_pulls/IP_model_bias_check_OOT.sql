CREATE OR REPLACE TABLE anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_bias_check
AS
SELECT
  indx.asdb_member_key
  , CASE WHEN TRIM(UPPER(demo.gender)) = "M" THEN 1 ELSE 0 END AS male
  , CASE WHEN TRIM(UPPER(demo.gender)) = "F" THEN 1 ELSE 0 END AS female
  , CASE WHEN TRIM(UPPER(demo.gender)) NOT IN ("M", "F") THEN 1 ELSE 0 END AS other_gender
  , CASE WHEN TRIM(UPPER(demo.ethnicity_desc)) LIKE "%WHITE%" THEN 1
      WHEN TRIM(UPPER(demo.ethnicity_desc)) LIKE "%CAUCAS%" THEN 1
      ELSE 0 END AS white
  , CASE WHEN TRIM(UPPER(demo.ethnicity_desc)) LIKE "%BLACK%" THEN 1
      WHEN TRIM(UPPER(demo.ethnicity_desc)) LIKE "%AFRICAN%" THEN 1
      ELSE 0 END AS black
  , CASE WHEN TRIM(UPPER(demo.ethnicity_desc)) LIKE "%HISPANIC%" AND TRIM(UPPER(demo.ethnicity_desc)) NOT LIKE "%NON%" THEN 1 
      WHEN TRIM(UPPER(demo.ethnicity_desc)) LIKE "%MEXICAN%" THEN 1
      WHEN TRIM(UPPER(demo.ethnicity_desc)) LIKE "%PUERTO%" THEN 1
      WHEN TRIM(UPPER(demo.ethnicity_desc)) LIKE "%CUBAN%" THEN 1
      ELSE 0 END AS hispanic
  , CASE WHEN TRIM(UPPER(demo.ethnicity_desc)) LIKE "%ASIAN%" AND TRIM(UPPER(demo.ethnicity_desc)) NOT LIKE "%CAUCASIAN%" THEN 1 
      WHEN TRIM(UPPER(demo.ethnicity_desc)) LIKE "%PACIFIC%" THEN 1
      WHEN TRIM(UPPER(demo.ethnicity_desc)) LIKE "%CAMBODIAN%" THEN 1
      WHEN TRIM(UPPER(demo.ethnicity_desc)) LIKE "%CHINESE%" THEN 1
      WHEN TRIM(UPPER(demo.ethnicity_desc)) LIKE "%FILIPINO%" THEN 1
      WHEN TRIM(UPPER(demo.ethnicity_desc)) LIKE "%GUAMANIAN%" THEN 1
      WHEN TRIM(UPPER(demo.ethnicity_desc)) LIKE "%HAWAIIAN%" THEN 1
      WHEN TRIM(UPPER(demo.ethnicity_desc)) LIKE "%JAPANESE%" THEN 1
      WHEN TRIM(UPPER(demo.ethnicity_desc)) LIKE "%KOREAN%" THEN 1
      WHEN TRIM(UPPER(demo.ethnicity_desc)) LIKE "%LAOTIAN%" THEN 1
      WHEN TRIM(UPPER(demo.ethnicity_desc)) LIKE "%SAMOAN%" THEN 1
      WHEN TRIM(UPPER(demo.ethnicity_desc)) LIKE "%VIETNAMESE%" THEN 1
      ELSE 0 END AS asian
  , CASE WHEN prediction.y_pred_test >= 0.5 THEN 1 
      WHEN prediction.y_pred_test < 0.5 THEN 0 
      ELSE NULL END AS predicted_ip
  , outcome.acute_ip_flag AS ground_truth
FROM
  `anbc-hcb-dev.cm_medicaid_hcb_dev.a091749_all_cohort` AS indx
LEFT JOIN
  `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_medicaid_IP_predict` AS prediction
    ON indx.asdb_member_key = prediction.asdb_member_key
LEFT JOIN
  `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_OOT_outcome_ip` AS outcome
    ON indx.asdb_member_key = outcome.asdb_member_key
LEFT JOIN
  `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_MEMBER` AS demo
    ON indx.asdb_member_key = demo.asdb_member_key
;


SELECT COUNT(black), black FROM `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_bias_check` GROUP BY black;
--384,744 (17.6%) out of 2,191,619
SELECT COUNT(white), white FROM `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_bias_check` GROUP BY white;
--1,275,402 (58.2%) out of 2,191,619
SELECT COUNT(hispanic), hispanic FROM `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_bias_check` GROUP BY hispanic;
--262,385 (12%) out of 2,191,619
SELECT COUNT(asian), asian FROM `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_bias_check` GROUP BY asian;
-- 49,321 (2.3%) out of 2,191,619

--219,767 (9.9%) unknown or other race

SELECT COUNT(male), male FROM `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_bias_check` GROUP BY male;
--1,023,816 out of 2,191,619
SELECT COUNT(female), female FROM `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_bias_check` GROUP BY female;
--1,167,509  out of 2,191,619

--294 missing gender
