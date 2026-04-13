--`anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_OOT_member_index`
--2023-11-01 index dt
--
CREATE OR REPLACE TABLE `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_OOT_PQI_flag`
AS
WITH inclusion AS (--6,735 out of 87,567
    SELECT DISTINCT
        asdb_member_key
        , prindiag
        , PQI
    FROM
        `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_OOT_outcome_ip_cases` AS a
    LEFT JOIN
        `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_PQI_inclusion` AS b
          ON CONCAT(SUBSTR(TRIM(a.prindiag), 1, 3), SUBSTR(TRIM(a.prindiag), 5, 9)) = b.ICD_code
    WHERE PQI IS NOT NULL
)
, exclusion AS (--20,717 diagnoses that are exclusions
    SELECT DISTINCT
        asdb_member_key
        , prindiag
        , PQI_exclusion
    FROM
        `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_OOT_med_claims_yr1` AS a
    LEFT JOIN
        `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_PQI_exclusion` AS b
          ON CONCAT(SUBSTR(TRIM(a.prindiag), 1, 3), SUBSTR(TRIM(a.prindiag), 5, 9)) = b.ICD_code
    WHERE PQI_exclusion IS NOT NULL
)
, seperate1 AS (
    SELECT DISTINCT
        asdb_member_key
        , CASE WHEN PQI = "PQI1" THEN 1 ELSE 0 END AS pqi1
        , CASE WHEN PQI = "PQI3" THEN 1 ELSE 0 END AS pqi3
        , CASE WHEN PQI = "PQI5" THEN 1 ELSE 0 END AS pqi5
        , CASE WHEN PQI = "PQI7" THEN 1 ELSE 0 END AS pqi7
        , CASE WHEN PQI = "PQI8" THEN 1 ELSE 0 END AS pqi8
        , CASE WHEN PQI = "PQI11" THEN 1 ELSE 0 END AS pqi11
        , CASE WHEN PQI = "PQI12" THEN 1 ELSE 0 END AS pqi12
        , CASE WHEN PQI = "PQI14" THEN 1 ELSE 0 END AS pqi14
        , CASE WHEN PQI = "PQI15" THEN 1 ELSE 0 END AS pqi15
    FROM
        inclusion
)
, seperate2 AS (
    SELECT DISTINCT
        asdb_member_key
        , CASE WHEN PQI_exclusion = "PQI5" THEN 1 ELSE 0 END AS pqi5_ex
        , CASE WHEN PQI_exclusion = "PQI7" THEN 1 ELSE 0 END AS pqi7_ex        
        , CASE WHEN PQI_exclusion = "PQI11" THEN 1 ELSE 0 END AS pqi11_ex 
        , CASE WHEN PQI_exclusion = "PQI12" THEN 1 ELSE 0 END AS pqi12_ex 
        , CASE WHEN PQI_exclusion = "PQI15" THEN 1 ELSE 0 END AS pqi15_ex 
    FROM
        exclusion
)
SELECT
    base.asdb_member_key
    , base.asdb_plan_key
    , base.index_dt
    , COALESCE(MAX(pqi1), 0) AS pqi1
    , COALESCE(MAX(pqi3), 0) AS pqi3
    , COALESCE(MAX(pqi5), 0) AS pqi5
    , COALESCE(MAX(pqi7), 0) AS pqi7
    , COALESCE(MAX(pqi8), 0) AS pqi8
    , COALESCE(MAX(pqi11), 0) AS pqi11
    , COALESCE(MAX(pqi12), 0) AS pqi12
    , COALESCE(MAX(pqi14), 0) AS pqi14
    , COALESCE(MAX(pqi15), 0) AS pqi15
    , COALESCE(MAX(pqi5_ex), 0) AS pqi5_ex
    , COALESCE(MAX(pqi7_ex), 0) AS pqi7_ex
    , COALESCE(MAX(pqi11_ex), 0) AS pqi11_ex
    , COALESCE(MAX(pqi12_ex), 0) AS pqi12_ex
    , COALESCE(MAX(pqi15_ex), 0) AS pqi15_ex
FROM
    `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_OOT_member_index` AS base
LEFT JOIN
    seperate1
        ON base.asdb_member_key = seperate1.asdb_member_key
LEFT JOIN
    seperate2
        ON base.asdb_member_key = seperate2.asdb_member_key
GROUP BY
    asdb_member_key
    , asdb_plan_key
    , index_dt    
;


