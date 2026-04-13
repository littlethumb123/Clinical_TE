#!/bin/bash

bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_member_score_ending`'

bq query \
--use_legacy_sql=false \
'
CREATE OR REPLACE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_member_score_ending`
--PARTITION BY RANGE_BUCKET(asdb_plan_key, GENERATE_ARRAY(0,100,1))
--CLUSTER BY index_dt, coa_population_group
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
SELECT 
    st.asdb_member_key AS individual_id
    , st.index_dt AS index_dt 
    , st.asdb_member_key AS member_id
FROM 
    (SELECT DISTINCT asdb_member_key, index_dt FROM `'$ST'`) AS st
GROUP BY
    st.asdb_member_key
    , st.index_dt
'
#-- CPT procedure codes for transformer
bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_d1a_score_ending`'

bq query \
--use_legacy_sql=false \
'
CREATE OR REPLACE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_d1a_score_ending`
--PARTITION BY RANGE_BUCKET(asdb_plan_key, GENERATE_ARRAY(0,100,1))
--CLUSTER BY index_dt, coa_population_group
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
WITH clm AS (
    SELECT
        base.individual_id
        , base.member_id
        , clm.claimid
        , clm.asdb_incurred_dt
        , clm.ip_paid_days_ct
        , clm.revcode
        , clm.location
        , clm.servcode
        , clm.asdb_svc_prov_key
        , clm.asdb_coe_id_dev
        , clm.final_claim
        , clm.status_header
        , clm.status_detail

    FROM 
        `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_member_score_ending` AS base 
    LEFT JOIN
        `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_CLM_DATA_STAGE` AS clm
            ON base.member_id = clm.asdb_member_key
    WHERE 1=1
        AND clm.final_claim = 1
        AND TRIM(UPPER(clm.status_header)) = "PAID"
        AND TRIM(UPPER(clm.status_detail)) NOT IN ("DENY", "DENIED")
        AND CAST(base.index_dt AS DATE) > CAST(clm.asdb_paid_dt AS DATE)
        AND CAST(base.index_dt AS DATE) > CAST(clm.asdb_incurred_dt AS DATE) 
        AND DATE_ADD(CAST(clm.asdb_incurred_dt AS DATE), INTERVAL 24 MONTH) > CAST(base.index_dt AS DATE)
)
SELECT
    clm.individual_id
    , clm.member_id
    , clm.claimid AS claim_line_id
    , CAST(clm.asdb_incurred_dt AS DATE) AS dt
    , CASE WHEN (clm.ip_paid_days_ct IS NULL OR clm.ip_paid_days_ct < 0) THEN 99 
        WHEN clm.ip_paid_days_ct > 10 THEN 11 
        ELSE clm.ip_paid_days_ct END AS days_cnt
    , CASE WHEN TRIM(member.gender) = "M" THEN 1
        WHEN TRIM(member.gender) = "F" THEN 0 
        ELSE 2 END AS gender_cd
    , DATE_DIFF(CAST(clm.asdb_incurred_dt AS DATE), CAST(member.dob AS DATE), MONTH) AS age_in_months
    , CASE WHEN TRIM(clm.revcode) = "" THEN NULL 
        ELSE CAST(TRIM(clm.revcode) AS NUMERIC) END AS revenue_cd
    , CASE WHEN TRIM(clm.location) IS NOT NULL THEN CAST(clm.location AS STRING)
        WHEN coe.asdb_coe_general_type = "Outpatient" 
            AND coe.asdb_coe_sub_cat = "Non Hospital" 
            AND coe.emis_cat = "Selected Ambulatory Facility"
            THEN CAST(11 AS STRING)
        WHEN coe.asdb_coe_general_type = "Outpatient" 
            AND coe.asdb_coe_sub_cat = "Hospital" 
            AND coe.emis_cat IN ("Laboratory", "Medical Pharmacy", "Mental Health", "Radiology", "Selected Ambulatory Facility") 
            THEN CAST(22 AS STRING)
        WHEN coe.asdb_coe_general_type = "Inpatient" 
            AND coe.asdb_coe_sub_cat = "Hospital" 
            AND coe.emis_cat IN ("Inpatient Facility", "Inpatient Facility (or Institutional Services)") 
            THEN CAST(21 AS STRING)
        WHEN coe.asdb_coe_general_type = "Outpatient" 
            AND coe.asdb_coe_sub_cat = "Hospital" 
            AND coe.emis_cat = "Emergency" 
            THEN CAST(23 AS STRING)   
        WHEN coe.asdb_coe_general_type = "Laboratory" 
            AND coe.asdb_coe_sub_cat = "Professional" 
            AND coe.emis_cat = "Laboratory"
            THEN CAST(81 AS STRING)
        WHEN coe.asdb_coe_general_type = "Long Term Care, Other, Outpatient" 
            AND coe.asdb_coe_sub_cat = "Home Based Services, Professional, Non Hospital" 
            AND coe.emis_cat IN ("Home-Based Services", "Home Health", "Home Health") 
            THEN CAST(12 AS STRING)
        WHEN coe.asdb_coe_general_type = "Inpatient" 
            AND coe.asdb_coe_sub_cat = "Non Hospital" 
            AND coe.emis_cat = "Inpatient Facility"
            THEN CAST(19 AS STRING)   
        WHEN coe.asdb_coe_general_type = "Long Term Care" 
            AND coe.asdb_coe_sub_cat = "Institution" 
            AND coe.emis_cat = "Institutional Services"
            THEN CAST(16 AS STRING)   
        WHEN coe.asdb_coe_general_type = "Inpatient" 
            AND coe.asdb_coe_sub_cat = "Hospital" 
            AND coe.emis_cat = "Mental Health"
            THEN CAST(51 AS STRING)   
        ELSE NULL END AS hcfa_plc_srv_cd
    , map.src_specialty_cd
    , CASE WHEN TRIM(clm.servcode) = "" THEN NULL
        ELSE TRIM(clm.servcode) END AS prcdr_cd
    , CASE WHEN TRIM(icd.icdpx1) = "" THEN NULL
        ELSE TRIM(icdpx1) END AS icd9_prcdr_cd
FROM clm
LEFT JOIN 
    (SELECT asdb_member_key, gender, dob FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_MEMBER`) AS member
        ON clm.member_id = member.asdb_member_key
LEFT JOIN
    (SELECT asdb_svc_prov_key, prov_specialty FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_SVC_PROV`) AS prv
        ON clm.asdb_svc_prov_key = prv.asdb_svc_prov_key
LEFT JOIN 
    (SELECT asdb_coe_id, asdb_coe_general_type, asdb_coe_sub_cat, emis_cat FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_TYPE_OF_SERVICE`) AS coe
        ON CAST(clm.asdb_coe_id_dev AS INT) = CAST(coe.asdb_coe_id AS INT)
LEFT JOIN
    (SELECT src_specialty_cd, asdb_svc_prov_key FROM `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_provider_db_x_walk_20240416`) AS map
        ON prv.asdb_svc_prov_key = map.asdb_svc_prov_key
LEFT JOIN
    (SELECT claimid, icdpx1 FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_src.ASDB_CLAIMICDPROCSUMMARY`) AS icd
        ON clm.claimid = icd.claimid
'
#-- ICD codes for transformer ---
bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_d1b_score_ending`'

bq query \
--use_legacy_sql=false \
'
CREATE OR REPLACE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_d1b_score_ending`
--PARTITION BY RANGE_BUCKET(asdb_plan_key, GENERATE_ARRAY(0,100,1))
--CLUSTER BY index_dt, coa_population_group
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
WITH base AS (
    SELECT 
        base.individual_id
        , base.member_id
        , base.claim_line_id
        , base.dt
        , indx.index_dt
    FROM
        `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_d1a_score_ending` AS base
   LEFT JOIN
        `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_member_score_ending` AS indx
            ON base.individual_id = indx.individual_id
)
, p AS (
    SELECT 
        base.individual_id
        , base.member_id
        , base.claim_line_id
        , base.dt
        , SPLIT(TRIM(b.icddxpri), ".")[offset(0)] AS x_0
        , SPLIT(TRIM(b.icddxpri), ".")[safe_offset(1)] AS x_1
    FROM
        base 
    INNER JOIN
        (SELECT claimid, icddxpri FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_CLAIMDIAGSUMMARY`) AS b 
            ON base.claim_line_id = b.claimid
    WHERE 
        base.dt >= DATE_SUB(base.index_dt, INTERVAL 24 MONTH)
        AND base.dt < base.index_dt
)
, s1 AS (
    SELECT 
        base.individual_id
        , base.member_id
        , base.claim_line_id
        , base.dt
        , SPLIT(TRIM(b.icddxsec1), ".")[offset(0)] AS x_0
        , SPLIT(TRIM(b.icddxsec1), ".")[safe_offset(1)] AS x_1
    FROM
        base 
    INNER JOIN
        (SELECT claimid, icddxsec1 FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_CLAIMDIAGSUMMARY`) AS b 
            ON base.claim_line_id = b.claimid
    WHERE 
        base.dt >= DATE_SUB(base.index_dt, INTERVAL 24 MONTH)
        AND base.dt < base.index_dt
)
, s2 AS (
    SELECT 
        base.individual_id
        , base.member_id
        , base.claim_line_id
        , base.dt
        , SPLIT(TRIM(b.icddxsec2), ".")[offset(0)] AS x_0
        , SPLIT(TRIM(b.icddxsec2), ".")[safe_offset(1)] AS x_1
    FROM
        base 
    INNER JOIN
        (SELECT claimid, icddxsec2 FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_CLAIMDIAGSUMMARY`) AS b 
            ON base.claim_line_id = b.claimid
    WHERE 
        base.dt >= DATE_SUB(base.index_dt, INTERVAL 24 MONTH)
        AND base.dt < base.index_dt
)
, s3 AS (
    SELECT 
        base.individual_id
        , base.member_id
        , base.claim_line_id
        , base.dt
        , SPLIT(TRIM(b.icddxsec3), ".")[offset(0)] AS x_0
        , SPLIT(TRIM(b.icddxsec3), ".")[safe_offset(1)] AS x_1
    FROM
        base 
    INNER JOIN
        (SELECT claimid, icddxsec3 FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_CLAIMDIAGSUMMARY`) AS b 
            ON base.claim_line_id = b.claimid
    WHERE 
        base.dt >= DATE_SUB(base.index_dt, INTERVAL 24 MONTH)
        AND base.dt < base.index_dt
)
, x1 AS (
    SELECT 
        * 
    FROM 
        p 
    UNION DISTINCT
        SELECT * FROM s1 
    UNION DISTINCT
        SELECT * FROM s2 
    UNION DISTINCT
        SELECT * FROM s3 
)
SELECT 
    individual_id
    , member_id
    , claim_line_id
    , dt
    , CASE WHEN x_1 IS NULL THEN x_0
        ELSE CONCAT(x_0, ".", SUBSTR(x_1, 1, 2)) END AS icd9_dx_cd
    FROM 
        x1
'


#-- MERGE ALL FOR TRANSFORMER PRE-TABLE - one row per day with claims per member ---
bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_o1_score_ending`'

bq query \
--use_legacy_sql=false \
'
CREATE OR REPLACE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_o1_score_ending`
--PARTITION BY RANGE_BUCKET(asdb_plan_key, GENERATE_ARRAY(0,100,1))
--CLUSTER BY index_dt, coa_population_group
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
WITH root0 AS (
    SELECT
        individual_id
        , member_id
        , dt
        , gender_cd
        , CASE WHEN age_in_months < 0 THEN 0 
            WHEN age_in_months > 1440 THEN 1440 
            ELSE age_in_months end AS age_in_months
    FROM 
        `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_d1a_score_ending`
)
, root1 AS (
    SELECT 
        *
        , ROW_NUMBER() OVER (PARTITION BY individual_id, dt) AS seqno
    from 
        root0
)
, root2 AS (
    SELECT 
        individual_id
        , member_id
        , dt
        , gender_cd
        , age_in_months 
    FROM 
        root1 
    WHERE 
        seqno = 1
)
, x0 as (
    SELECT 
        base.individual_id
        , member_id
        , base.dt
        , CASE WHEN w2ind.ind IS NULL THEN 0 
            ELSE w2ind.ind END AS ind
    FROM 
        `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_d1a_score_ending` AS base 
    LEFT JOIN 
        `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_v9_w2ind` AS w2ind --replace with production table if possible
            ON CONCAT("days_cnt", CAST(days_cnt AS STRING)) = w2ind.cd
    WHERE 
        days_cnt IS NOT NULL
    UNION DISTINCT
        SELECT 
            base.individual_id
            , member_id
            , base.dt
            , CASE WHEN w2ind.ind IS NULL THEN 0 
                ELSE w2ind.ind END AS ind
        FROM
            `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_d1a_score_ending` AS base 
        LEFT JOIN
            `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_v9_w2ind` AS w2ind --replace with production table if possible
                ON concat("hcfa_plc_srv_cd", cast(hcfa_plc_srv_cd AS string)) = w2ind.cd
        WHERE hcfa_plc_srv_cd IS NOT NULL
    UNION DISTINCT
        SELECT 
            base.individual_id
            , member_id
            , base.dt
            , CASE WHEN w2ind.ind IS NULL THEN 0 
                ELSE w2ind.ind END AS ind
        FROM 
            `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_d1a_score_ending` AS base 
        LEFT JOIN
            `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_v9_w2ind` AS w2ind --replace with production table if possible
                ON CONCAT("src_specialty_cd", CAST(src_specialty_cd AS STRING)) = w2ind.cd
        WHERE
            src_specialty_cd IS NOT NULL
    UNION DISTINCT
        SELECT 
            base.individual_id
            , member_id
            , base.dt
            , CASE WHEN w2ind.ind IS NULL THEN 0 
                ELSE w2ind.ind END AS ind
        FROM 
            `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_d1b_score_ending` AS base 
        LEFT JOIN
            `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_v9_w2ind` AS w2ind --replace with production table if possible
                ON CONCAT("icd9_dx_cd", CAST(icd9_dx_cd AS STRING)) = w2ind.cd
        WHERE 
            icd9_dx_cd IS NOT NULL
    UNION DISTINCT
        SELECT 
            base.individual_id
            , member_id
            , base.dt
            , CASE WHEN w2ind.ind IS NULL THEN 0 
                ELSE w2ind.ind END AS ind
        FROM 
            `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_d1a_score_ending` AS base 
        LEFT JOIN
            `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_v9_w2ind` AS w2ind --replace with production table if possible
                ON CONCAT("revenue_cd", CAST(revenue_cd AS STRING)) = w2ind.cd
        WHERE 
            revenue_cd IS NOT NULL
    UNION DISTINCT
        SELECT
            base.individual_id
            , member_id
            , base.dt
            , case when w2ind.ind IS NULL THEN 0 
                else w2ind.ind END AS ind
        FROM
            `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_d1a_score_ending` AS base 
        LEFT JOIN
            `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_v9_w2ind` AS w2ind --replace with production table if possible
                ON CONCAT("prcdr_cd", CAST(prcdr_cd AS STRING)) = w2ind.cd
        WHERE 
            prcdr_cd IS NOT NULL
)
, x1 AS (
    SELECT 
        individual_id
        , dt
        , ind
    FROM 
        x0 
    GROUP BY
        individual_id
        , dt
        , ind 
    )
    , x2 AS (
    SELECT 
        *
        , ROW_NUMBER() OVER (PARTITION BY individual_id,dt) AS seqno
    FROM 
        x1
)
, x3 AS (
    SELECT 
        individual_id
        , dt
        , STRING_AGG(CAST(ind AS STRING), ",") AS cd
    FROM 
        x2 
    WHERE 
        seqno<=80
    GROUP BY 
        individual_id
        , dt
)
SELECT 
    root2.individual_id
    , root2.dt
    , root2.gender_cd
    , root2.age_in_months
    , x3.cd
FROM 
    root2 
INNER JOIN
    x3 
        ON root2.individual_id = x3.individual_id 
        AND root2.dt = x3.dt
'

#--create one row per member ---
bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_o3_score_ending`'

bq query \
--use_legacy_sql=false \
'
CREATE OR REPLACE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_o3_score_ending`
--PARTITION BY RANGE_BUCKET(asdb_plan_key, GENERATE_ARRAY(0,100,1))
--CLUSTER BY index_dt, coa_population_group
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
WITH x1 AS (
    SELECT 
        *
        , ROW_NUMBER() OVER (PARTITION BY individual_id ORDER BY dt ASC) AS seqno
    FROM 
        `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_o1_score_ending`
)
, x2 AS (
    SELECT 
        * 
    FROM 
        x1 
    WHERE 
        seqno <= 200
)
, x4 AS (
    SELECT 
        *
        , ROW_NUMBER() OVER (PARTITION BY individual_id ORDER BY dt ASC) AS seqno2
    FROM 
        x2
) 
, x5 AS (
    SELECT 
        individual_id
        , STRING_AGG(CAST(gender_cd AS STRING), "*") AS gender_cd
        , STRING_AGG(CAST(age_in_months AS STRING), "*") AS age_in_months
        , STRING_AGG(CAST(cd AS STRING), "*") AS cd
        , COUNT(*) AS dt_cnt
    FROM
        x4 
    GROUP BY 
        individual_id
)
SELECT 
    * 
FROM 
    x5
'
