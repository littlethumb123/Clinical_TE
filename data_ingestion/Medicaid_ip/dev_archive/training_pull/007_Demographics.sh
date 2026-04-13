
# !/bin/bash
bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_demographics_months`'


'
CREATE OR REPLACE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_demographics_months`
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
SELECT
    mnth.asdb_member_key
    , st.index_dt
    , CAST(mnth.asdb_elig_dt AS DATE) AS asdb_elig_dt
    FROM
        (SELECT asdb_member_key, asdb_elig_dt FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ELIG_DATA_MBR_PER_MTH`) AS mnth
    LEFT JOIN
        (SELECT asdb_member_key, index_dt FROM `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_member_index`) AS st
            ON st.asdb_member_key = mnth.asdb_member_key
    WHERE CAST(mnth.asdb_elig_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 24 MONTH) AND DATE_ADD(st.index_dt, INTERVAL 6 MONTH) 
'


#---- Demographics
bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_demographics`'

bq query \
--use_legacy_sql=false \
'
CREATE OR REPLACE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_demographics`
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
WITH mth AS (
    SELECT
    mnth.asdb_member_key
    , st.index_dt
    , CAST(mnth.asdb_elig_dt AS DATE) AS asdb_elig_dt
    , ROW_NUMBER() OVER(PARTITION BY(mnth.asdb_member_key) ORDER BY mnth.asdb_elig_dt) AS mnths
    FROM
        (SELECT asdb_member_key, asdb_elig_dt FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ELIG_DATA_MBR_PER_MTH`) AS mnth
    LEFT JOIN
        (SELECT asdb_member_key, index_dt FROM `'$ST'`) AS st
            ON st.asdb_member_key = mnth.asdb_member_key
    WHERE CAST(mnth.asdb_elig_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 12 MONTH) AND index_dt
),
mth2 AS (
    SELECT
    mnth.asdb_member_key
    , st.index_dt
    , CAST(mnth.asdb_elig_dt AS DATE) AS asdb_elig_dt
    , ROW_NUMBER() OVER(PARTITION BY(mnth.asdb_member_key) ORDER BY mnth.asdb_elig_dt) AS mnths
    FROM
        (SELECT asdb_member_key, asdb_elig_dt FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ELIG_DATA_MBR_PER_MTH`) AS mnth
    LEFT JOIN
        (SELECT asdb_member_key, index_dt FROM `'$ST'`) AS st
            ON st.asdb_member_key = mnth.asdb_member_key
    WHERE CAST(mnth.asdb_elig_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 24 MONTH) AND DATE_SUB(st.index_dt, INTERVAL 13 MONTH) 
),
post AS (
    SELECT
    mnth.asdb_member_key
    , st.index_dt
    , CAST(mnth.asdb_elig_dt AS DATE) AS asdb_elig_dt
    , ROW_NUMBER() OVER(PARTITION BY(mnth.asdb_member_key) ORDER BY mnth.asdb_elig_dt) AS mnths
    FROM
        (SELECT asdb_member_key, asdb_elig_dt FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ELIG_DATA_MBR_PER_MTH`) AS mnth
    LEFT JOIN
        (SELECT asdb_member_key, index_dt FROM `'$ST'`) AS st
            ON st.asdb_member_key = mnth.asdb_member_key
    WHERE CAST(mnth.asdb_elig_dt AS DATE) BETWEEN DATE_ADD(st.index_dt, INTERVAL 1 MONTH)  AND DATE_ADD(st.index_dt, INTERVAL 6 MONTH) 
)
SELECT 
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
    , FLOOR(DATE_DIFF(DATE(st.index_dt), DATE(mb.dob), YEAR)) AS agenbr
    , mb.gender
    , mb.ethnicity_code
    , mb.primarylanguage_desc
    , COALESCE(mth.tenure, 0) AS tenure_yr1
    , COALESCE(mth2.tenure, 0) AS tenure_yr2
    , COALESCE(post.tenure, 0) AS post_mnths
    , zcuu.urbsubr
    , zcuu.zip_weight_avg_medinc
FROM 
  (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_member_index`) AS st
LEFT JOIN 
    `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_MEMBER` AS mb
        ON st.asdb_member_key = mb.asdb_member_key
LEFT JOIN 
    (SELECT asdb_member_key, MAX(mnths) AS tenure FROM mth GROUP BY asdb_member_key) AS mth
        ON st.asdb_member_key = mth.asdb_member_key
LEFT JOIN 
    (SELECT asdb_member_key, MAX(mnths) AS tenure FROM mth2 GROUP BY asdb_member_key) AS mth2
        ON st.asdb_member_key = mth2.asdb_member_key
LEFT JOIN 
    (SELECT asdb_member_key, MAX(mnths) AS tenure FROM post GROUP BY asdb_member_key) AS post
        ON st.asdb_member_key = post.asdb_member_key
LEFT JOIN 
    `edp-prod-hcbstorage.edp_hcb_tra_ckd_phm_srcv.ZIP_CENSUS_USPS_URBRUR` AS zcuu
        ON TRIM(mb.member_zip) = TRIM(zcuu.zip_cd)
'
