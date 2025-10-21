---------------------------------------------------------------------------------------------------

-------          Project: Medicaid Transformer Embeddings Model                           ---------

-------          Original Author: Packiaraj N                                             ---------

-------          Modified by:                                                             ---------

-------          Date: 2024-09-22                                                         ---------

-------          All Medicaid Members                                                     ---------

---------------------------------------------------------------------------------------------------
-- History run logic -- will be commented out for regular runs
-- Update History table with proper index dt
--UPDATE {{CM_MD_SCHEMA}}.medicaid_transformer_embed_scores_hist SET index_dt = CAST(RUN_DT AS DATE) WHERE index_dt = DATE_TRUNC(CURRENT_DATE(), MONTH);

-- Update RUN_DT for the next run
--UPDATE {{CM_MD_SCHEMA}}.MD_MODEL_RUN_DATE_CONFIG SET run_dt = CAST(DATE_ADD(DATE_TRUNC(DATE(RUN_DT), MONTH), INTERVAL 1 MONTH) AS STRING) WHERE model_id = 'Medicaid_Transformer_Embeddings';

--  -------------------------------------------------------------------------------------------  --
--                                  Provider ID Mapping                                          --
--  -------------------------------------------------------------------------------------------  --

CREATE OR REPLACE TABLE  `{{CM_MD_SCHEMA}}.MD_TE_PROVIDER_DB_XWALK`
{{LABELS}}
AS

WITH
--edw_subtable
edw AS
(
    SELECT
        CAST(b.best_npi_no_pin AS INT) AS npi
        , b.provider_id
        , a.sort_nm
        --, MIN(a.eff_dt) AS eff_dt
        , MAX(b.exp_dt) AS exp_dt
        , a.provider_class_cd
        , a.specialty_cd AS src_specialty_cd
    FROM
        (SELECT provider_id, sort_nm, provider_class_cd, specialty_cd FROM `edp-prod-hcbstorage.edp_hcb_core_cnsv.PROVIDER_DM`) AS a
    LEFT JOIN
        (SELECT best_npi_no_pin, provider_id, exp_dt FROM `edp-prod-hcbstorage.edp_hcb_core_cnsv.PRVDR_X_NPI_PRIMARY`) AS b
            ON b.provider_id = a.provider_id
    WHERE
        b.best_npi_no_pin > 999999999
    GROUP BY
        b.best_npi_no_pin
        , b.provider_id
        , a.sort_nm
        , a.provider_class_cd
        , a.specialty_cd
)
--asdb subtable
, asdb AS
(
    SELECT DISTINCT
        SAFE_CAST(a.npi AS INT64) AS npi
        , a.svc_prov_id
        , a.asdb_svc_prov_key
        , a.prov_fullname
        , b.asdb_pcp_prov_key
    FROM
        (SELECT npi, svc_prov_id, asdb_svc_prov_key, prov_fullname FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_SVC_PROV`) AS a
    LEFT JOIN
        (SELECT asdb_pcp_prov_key, npi FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ASDB_PCP_PROV`) AS b
            ON SAFE_CAST(a.npi AS INT64) = SAFE_CAST(b.npi AS INT)
    WHERE
        SAFE_CAST(a.npi AS INT64) > 999999999
)--komodo subtable
, kmdo AS
(
    SELECT DISTINCT
        CAST(provider_id AS INT64) AS npi
        , CONCAT(last_name, ", ", first_name, " ", middle_name) AS provider_nm
    FROM
        `anbc-hcb-prod.eds_srcapp_komodombr_share_hcb_prod.provider_summaries`
)

SELECT DISTINCT
      COALESCE(edw.npi, asdb.npi, kmdo.npi) AS npi
    , COALESCE(edw.sort_nm, asdb.prov_fullname, kmdo.provider_nm) AS provider_nm
    , edw.provider_class_cd
    , edw.provider_id
    , edw.src_specialty_cd
    , asdb.svc_prov_id AS asdb_svc_prov_id
    , asdb.asdb_svc_prov_key AS asdb_svc_prov_key
    , asdb.asdb_pcp_prov_key AS asdb_pcp_prov_key
    , CASE WHEN edw.exp_dt = '9999-12-31' THEN 'Active'
        WHEN edw.exp_dt IS NULL THEN 'Unknown'
        ELSE 'Inactive' END AS edw_active
FROM edw
FULL JOIN asdb
    ON edw.npi = asdb.npi
FULL JOIN kmdo
    ON COALESCE(edw.npi, asdb.npi) = kmdo.npi 
ORDER BY
    npi
;
--  -------------------------------------------------------------------------------------------  --
