CREATE OR REPLACE TABLE `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_komod0_jan_2024_member`
OPTIONS (labels = [("owner", "palmere1_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
SELECT DISTINCT 
     a.asdb_member_key
     , a.asdb_plan_key
     , SPLIT(a.asdb_plan_code_version, "_")[offset(0)] AS asdb_plan_nm
     , CAST(a.asdb_elig_dt AS date) AS asdb_elig_dt
     , a.coa_population_category
     , CASE WHEN TRIM(a.coa_population_category) = "ABD Non Dual LTSS" OR
          TRIM(a.coa_population_category) = "LTSS Only" OR 
               TRIM(a.coa_population_category) = "Dual Elig LTSS" OR 
               TRIM(a.coa_population_category) = "Dual Int LTSS"
               THEN "LTSS"
          WHEN TRIM(a.coa_population_category) = "ABD Non Dual Non LTSS" OR
               TRIM(a.coa_population_category) = "DD"
               THEN "ABD"
          WHEN TRIM(a.coa_population_category) = "BH Int SMI" OR
               TRIM(a.coa_population_category) = "BH Only"
               THEN "BH"
          WHEN TRIM(a.coa_population_category) = "DSNP Medicare Only" OR
               TRIM(a.coa_population_category) = "Dual Elig NonLTSS" 
               THEN "Dual Elig"
          WHEN TRIM(a.coa_population_category) = "Dual Int DD" OR
               TRIM(a.coa_population_category) = "Dual Int NonLTSS"
               THEN "Dual Int"  -- Dual Elig and Dual Int could be combined depending on analysis need
          WHEN TRIM(a.coa_population_category) = "CHIP" OR
               TRIM(a.coa_population_category) = "TANF" 
               THEN "TANF/CHIP"
          WHEN TRIM(a.coa_population_category) = "Expansion" 
               THEN "Expansion"
          ELSE "Other"
          END AS coa_population_group
    , b.memid
    , b.age_in_mths_no
    , b.gender
    , b.ethnicity_desc
    , b.primarylanguage_desc
    , b.headofhouse
FROM 
     `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ELIG_DATA_MBR_PER_MTH` AS a -- STABLE GCP VIEW
LEFT JOIN
    `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_MEMBER` AS b
    ON a.asdb_member_key = b.asdb_member_key

WHERE 
     CAST(asdb_elig_dt AS DATE) BETWEEN PARSE_DATE("%Y%m%d", CAST("20240101" AS STRING)) AND PARSE_DATE("%Y%m%d", CAST("20240131" AS STRING));




CREATE OR REPLACE TABLE `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_komod0_jan_2024_member_map`
OPTIONS (labels = [("owner", "palmere1_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
SELECT 
    a.asdb_member_key
    , a.asdb_plan_key
    , a.asdb_plan_nm
    , a.asdb_elig_dt
    , a.coa_population_category
    , a.coa_population_group
    , a.memid
    , a.age_in_mths_no
    , a.gender
    , a.ethnicity_desc
    , a.primarylanguage_desc
    , a.headofhouse
    , b.patient_id
    , b.individual_id
    , b.individual_analytics_identifier
    , b.medicaid_id
    , b.alternate_id_cumb_hmo
    , b.ssn_hash
    , b.yob
    , b.gender AS k_gender
    , b.zip3
    , b.cov_start_date
    , b. cov_end_date
    , b.plan_type
    , b.upk_token_1
    , b.upk_token_2
    , b.upk_token_5
FROM
    `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_komod0_jan_2024_member` AS a
LEFT JOIN
    `anbc-hcb-prod.veds_srcapp_komododatavantmdcd_hcb_prod.v_aetna_mdcd_beneficiary_komodo_trace_2024_01` AS b
        ON a.memid = b.medicaid_id
;
