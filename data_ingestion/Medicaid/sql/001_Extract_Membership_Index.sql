---------------------------------------------------------------------------------------------------

-------          Project: Medicaid Transformer Embeddings Model                           ---------

-------          Original Author: Packiaraj N                                             ---------

-------          Modified by:                                                             ---------

-------          Date: 2024-07-22                                                         ---------

-------          All Medicaid Members                                                     ---------

---------------------------------------------------------------------------------------------------

--  -------------------------------------------------------------------------------------------  --
--                                    Member Index Data                                          --
--  -------------------------------------------------------------------------------------------  --

--  Get Model Run Date from Config Table
DECLARE RUN_DT STRING;
EXECUTE IMMEDIATE("SELECT run_dt FROM {{CM_MD_SCHEMA}}.MD_MODEL_RUN_DATE_CONFIG WHERE model_id = 'Medicaid_Transformer_Embeddings'") INTO RUN_DT;

--  -------------------------------------------------------------------------------------------  --

CREATE OR REPLACE TABLE  `{{CM_MD_SCHEMA}}.{{PREFIX}}_MEMBER_INDEX`
{{LABELS}}
AS

SELECT DISTINCT 
       asdb_member_key
     , asdb_plan_key
     , SPLIT(asdb_plan_code_version, "_")[offset(0)] as asdb_plan_nm
     , CAST(asdb_elig_dt AS date) AS index_dt
     , coa_population_category
     , CASE WHEN TRIM(coa_population_category) = "ABD Non Dual LTSS" OR
          TRIM(coa_population_category) = "LTSS Only" OR 
               TRIM(coa_population_category) = "Dual Elig LTSS" OR 
               TRIM(coa_population_category) = "Dual Int LTSS"
               THEN "LTSS"
          WHEN TRIM(coa_population_category) = "ABD Non Dual Non LTSS" OR
               TRIM(coa_population_category) = "DD"
               THEN "ABD"
          WHEN TRIM(coa_population_category) = "BH Int SMI" OR
               TRIM(coa_population_category) = "BH Only"
               THEN "BH"
          WHEN TRIM(coa_population_category) = "DSNP Medicare Only" OR
               TRIM(coa_population_category) = "Dual Elig NonLTSS" 
               THEN "Dual Elig"
          WHEN TRIM(coa_population_category) = "Dual Int DD" OR
               TRIM(coa_population_category) = "Dual Int NonLTSS"
               THEN "Dual Int"  -- Dual Elig and Dual Int could be combined depending on analysis need
          WHEN TRIM(coa_population_category) = "CHIP" OR
               TRIM(coa_population_category) = "TANF" 
               THEN "TANF/CHIP"
          WHEN TRIM(coa_population_category) = "Expansion" 
               THEN "Expansion"
          ELSE "Other"
          END AS coa_population_group
  FROM 
     `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ELIG_DATA_MBR_PER_MTH`
 WHERE 
     CAST(asdb_elig_dt AS DATE) BETWEEN DATE_SUB(DATE_TRUNC(DATE(RUN_DT), MONTH), INTERVAL 1 MONTH) AND DATE_TRUNC(DATE(RUN_DT), MONTH)

--  -------------------------------------------------------------------------------------------  --
--                                       End of Script                                           --
--  -------------------------------------------------------------------------------------------  --