------------------------------------------------------------------------------------------
------- Project: Medicaid IP model 2024                                          ---------
------- Original Author: Elle Palmer                                             ---------
------- Date: 2024-08-28                                                         ---------
------- Last modified by:                                       ---------
------- On:                                                            ---------
------- Population: All Medicaid                                                 ---------
------------------------------------------------------------------------------------------

--------------------------------
-- Source table dependencies ---
--------------------------------
--`anbc-hcb-prod.cm_medicaid_hcb_prod.ICD10_X_ER_TYPE`
--
--`anbc-hcb-prod.insights_share_hcb_prod.v_enriched_address_medicaid`
--
--`edp-prod-storage.edp_ent_sdoheir_srcv.risk_index_block_group_historical_data`
--
--`edp-prod-hcbstorage.edp_hcb_anbor_enrsrcv.EDW_DRUG`
--
--`edp-prod-hcbstorage.edp_hcb_tra_ckd_phm_srcv.ZIP_CENSUS_USPS_URBRUR`
--
--`edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ASDB_ICE_OP`
--`edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ICE_IP`
--`edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_CLM_DATA_STAGE`
--`edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_TYPE_OF_SERVICE`
--`edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_SVC_PROV`
--`edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_PPM_MEMBER_CONDITION_HISTORY`
--`edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_RX_DATA_STAGE`
--`edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ELIG_DATA_MBR_PER_MTH`
--`edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_MEMBER`

-------------------
--- system vars ---
-------------------
--CM_MD_SCHEMA = project.db
--PREFIX       = MDCD_IP_2024
--EMB          = table name for current embeddings
--ST           = `{{CM_MD_SCHEMA}}.{{PREFIX}}_member_index`
--INDEX_DT     = job run date
--SDOH_YR      = max year available (right now 2023)

----------------------------------------
--- Get membership for current month ---
----------------------------------------
CREATE OR REPLACE TABLE `{{CM_MD_SCHEMA}}.{{PREFIX}}_member_index` 
{{LABELS}} 
AS
SELECT DISTINCT
    asdb_member_key
    , asdb_plan_key
    , CAST(asdb_elig_dt AS DATE) AS asdb_elig_dt
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
    (SELECT asdb_member_key, asdb_plan_key, asdb_elig_dt, coa_population_category FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ELIG_DATA_MBR_PER_MTH`)
WHERE 1 = 1
    AND DATE_TRUNC(CAST(asdb_elig_dt AS DATE), MONTH) = DATE_TRUNC(DATE {{INDEX_DT}}, MONTH)
;

----------------------------
--- ED 0-12 months prior ---
----------------------------
CREATE OR REPLACE TABLE `{{CM_MD_SCHEMA}}.{{PREFIX}}_ed_cases` 
{{LABELS}} 
AS
SELECT
    st.asdb_member_key
    , mc.asdb_incurred_dt AS ed_vis_dt
    , mc.event_ct
    , CASE WHEN TRIM(nyu.er_type) = "PREVENTABLE" THEN 1 
        ELSE 0 
        END AS preventable_er_visits    
FROM
    (SELECT DISTINCT asdb_member_key FROM `{{ST}}`) AS st
INNER JOIN
     (SELECT asdb_member_key, asdb_incurred_dt, event_ct, prindiag, asdb_coe_id FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ASDB_ICE_OP`) AS mc
        ON st.asdb_member_key=mc.asdb_member_key
LEFT JOIN 
    (SELECT er_type, dx_cd FROM `anbc-hcb-prod.cm_medicaid_hcb_prod.ICD10_X_ER_TYPE`) AS nyu
        ON TRIM(mc.prindiag) = TRIM(nyu.dx_cd)
WHERE 1 = 1 
    AND CAST(mc.asdb_incurred_dt AS DATE) BETWEEN DATE_SUB({{INDEX_DT}}, INTERVAL 12 MONTH) AND DATE_SUB({{INDEX_DT}}, INTERVAL 1 DAY)
    AND CAST(mc.asdb_coe_id AS INT64) = 20100
    AND event_ct=1
;

CREATE OR REPLACE TABLE `{{CM_MD_SCHEMA}}.{{PREFIX}}_ed` 
{{LABELS}} 
AS
SELECT
    st.asdb_member_key
    , COALESCE(SUM(pre.event_ct), 0) AS sum_ed_visits_yr1
    , COALESCE(SUM(pre.preventable_er_visits), 0) AS sum_preventable_yr1
FROM
    (SELECT DISTINCT asdb_member_key FROM `{{ST}}`) AS st
LEFT JOIN
    `{{CM_MD_SCHEMA}}.{{PREFIX}}_ed_cases` AS pre
        ON st.asdb_member_key = pre.asdb_member_key
GROUP BY 
    st.asdb_member_key
;

----------------------------
--- IP 0-12 months prior ---
----------------------------
CREATE OR REPLACE TABLE `{{CM_MD_SCHEMA}}.{{PREFIX}}_ip_cases` 
{{LABELS}} 
AS
SELECT 
    mc.asdb_member_key
    , CASE WHEN mc.asdb_coe_id IN (10200,10700,10800) THEN "Acute"
        ELSE "Non-Acute" END AS ip_type
    , DATE_DIFF(mc.final_discharge_dt, mc.asdb_event_start_dt, DAY) AS calc_los
    , mc.event_ct
FROM
    (SELECT DISTINCT asdb_member_key FROM `{{ST}}`) AS st
INNER JOIN 
    (SELECT 
        asdb_member_key
        , asdb_coe_id
        , final_discharge_dt
        , asdb_event_start_dt
        , event_ct 
    FROM 
        `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ICE_IP`
    )  AS mc
        ON st.asdb_member_key=mc.asdb_member_key
WHERE 1 = 1
    AND CAST(mc.asdb_event_start_dt AS DATE) BETWEEN DATE_SUB({{INDEX_DT}}, INTERVAL 12 MONTH) AND DATE_SUB({{INDEX_DT}}, INTERVAL 1 DAY)
    AND mc.event_ct = 1
;

CREATE OR REPLACE TABLE `{{CM_MD_SCHEMA}}.{{PREFIX}}_ip` 
{{LABELS}} 
AS
WITH acute AS (
    SELECT
        asdb_member_key
        , CASE WHEN SUM(event_ct) > 0 THEN 1 
            ELSE 0 END AS acute_ip_flag
        , SUM(event_ct) AS sum_acute_ip_admits
        , SUM(calc_los) AS sum_acute_calc_los
    FROM  
        `{{CM_MD_SCHEMA}}.{{PREFIX}}_ip_cases` AS pre
    WHERE 1 = 1
        AND ip_type = "Acute"
    GROUP BY 
        asdb_member_key
)
SELECT
    st.asdb_member_key
    , COALESCE(a.acute_ip_flag, 0) AS acute_ip_flag_yr1
    , COALESCE(a.sum_acute_ip_admits, 0) AS sum_acute_ip_admits_yr1
    , COALESCE(a.sum_acute_calc_los, 0) AS sum_acute_calc_los_yr1
FROM 
    (SELECT DISTINCT asdb_member_key FROM `{{ST}}`) AS st
LEFT JOIN 
    acute AS a
        ON st.asdb_member_key = a.asdb_member_key
;

------------------------------------------------------------------
--- pull data needed to create claims and utilization features ---
------------------------------------------------------------------
CREATE OR REPLACE TABLE `{{CM_MD_SCHEMA}}.{{PREFIX}}_med_claims`
{{LABELS}} 
AS
SELECT
       st.asdb_member_key
       , clm.asdb_coe_id
       , coe.asdb_coe_general_type
       , coe.asdb_coe_sub_cat
       , clm.asdb_svc_prov_key
       , CAST(clm.asdb_incurred_dt AS DATE) AS asdb_incurred_dt
       , clm.prindiag
       , clm.emis_cat
FROM 
       (SELECT DISTINCT asdb_member_key FROM `{{ST}}`) AS st
INNER JOIN 
    (SELECT DISTINCT
        asdb_member_key
        , asdb_svc_prov_key
        , asdb_incurred_dt
        , prindiag
        , emis_cat
        , asdb_coe_id
    FROM 
        `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_CLM_DATA_STAGE`
    WHERE 1 = 1
        AND final_claim = 1
        AND TRIM(UPPER(status_header)) = "PAID"
        AND TRIM(UPPER(status_detail)) NOT IN ("DENY", "DENIED")
    ) AS clm
        ON st.asdb_member_key = clm.asdb_member_key
LEFT JOIN 
    (SELECT 
        asdb_coe_id
        , asdb_coe_general_type
        , asdb_coe_sub_cat
    FROM
        `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_TYPE_OF_SERVICE`
    ) AS coe
        ON clm.asdb_coe_id = coe.asdb_coe_id
WHERE 1 = 1
    AND CAST(asdb_incurred_dt AS DATE) BETWEEN DATE_SUB({{INDEX_DT}}, INTERVAL 24 MONTH) AND DATE_SUB({{INDEX_DT}}, INTERVAL 1 DAY)
;

-----------------------------
--- Utilization variables ---
-----------------------------
CREATE OR REPLACE TABLE `{{CM_MD_SCHEMA}}.{{PREFIX}}_utilization` 
{{LABELS}} 
AS
WITH op AS (
    SELECT
        clm.asdb_member_key
        , SUM(clm.op_ct) AS sum_op_visits_yr1
    FROM
        (SELECT 
            clm.asdb_member_key
            , CASE WHEN ROW_NUMBER() OVER(PARTITION BY clm.asdb_member_key, clm.asdb_svc_prov_key, clm.asdb_incurred_dt) = 1 THEN 1 
                ELSE 0 END AS op_ct
        FROM 
            (SELECT 
                asdb_member_key
                , asdb_svc_prov_key
                , asdb_incurred_dt
                , asdb_coe_general_type
                , emis_cat 
            FROM 
                `{{CM_MD_SCHEMA}}.{{PREFIX}}_med_claims`
            ) AS clm
        WHERE 1 = 1
            AND TRIM(clm.asdb_coe_general_type) != "Inpatient"
            AND TRIM(clm.emis_cat) != "Institutional Services"
            AND TRIM(clm.emis_cat) != "Emergency"
            AND CAST(clm.asdb_incurred_dt AS DATE) BETWEEN DATE_SUB({{INDEX_DT}}, INTERVAL 12 MONTH) AND DATE_SUB({{INDEX_DT}}, INTERVAL 1 DAY)
        ) AS clm
    GROUP BY
        clm.asdb_member_key
)
, clm_yr1 AS (
    SELECT 
        asdb_member_key
        , asdb_svc_prov_key
        , emis_cat
        , asdb_coe_general_type
        , asdb_coe_sub_cat
        , asdb_coe_id
        , prindiag
        , CASE WHEN TRIM(asdb_coe_general_type)="Inpatient" OR TRIM(emis_cat)="Institutional Services" THEN "Inpatient"
            WHEN TRIM(emis_cat)="Emergency" THEN "Emergency"
            ELSE "Outpatient" 
            END AS plc_svc_ctg
    FROM 
        `{{CM_MD_SCHEMA}}.{{PREFIX}}_med_claims`
    WHERE 1 = 1
        AND CAST(asdb_incurred_dt AS DATE) BETWEEN DATE_SUB({{INDEX_DT}}, INTERVAL 12 MONTH) AND DATE_SUB({{INDEX_DT}}, INTERVAL 1 DAY)
)
, clm_yr2 AS (
    SELECT 
        asdb_member_key
        , emis_cat
        , asdb_coe_general_type
        , asdb_coe_sub_cat
    FROM 
        `{{CM_MD_SCHEMA}}.{{PREFIX}}_med_claims`
    WHERE 1 = 1 
        AND CAST(asdb_incurred_dt AS DATE) BETWEEN DATE_SUB({{INDEX_DT}}, INTERVAL 24 MONTH) AND DATE_SUB(DATE_SUB({{INDEX_DT}}, INTERVAL 1 DAY), INTERVAL 12 MONTH)
)
, ut_yr1 AS (
    SELECT
        clm.asdb_member_key
        , COALESCE(SUM(clm.emis_ed_clm_yr1), 0) AS emis_ed_clm_yr1
        , COALESCE(SUM(clm.emis_hh_clm_yr1), 0) AS emis_hh_clm_yr1
        , COALESCE(SUM(clm.emis_ip_clm_yr1), 0) AS emis_ip_clm_yr1
        , COALESCE(SUM(clm.emis_misc_clm_yr1), 0) AS emis_misc_clm_yr1
        , COALESCE(SUM(clm.emis_spec_clm_yr1), 0) AS emis_spec_clm_yr1
        , COALESCE(SUM(clm.ltc_clm_yr1), 0) AS ltc_clm_yr1
        , COALESCE(SUM(clm.coe_other_clm_yr1), 0) AS coe_other_clm_yr1
        , COALESCE(SUM(clm.coe_eval_clm_yr1), 0) AS coe_eval_clm_yr1
        , COALESCE(SUM(clm.coe_mrx_clm_yr1), 0) AS coe_mrx_clm_yr1
        , COALESCE(SUM(clm.coe_radio_clm_yr1), 0) AS coe_radio_clm_yr1
        , COALESCE(SUM(clm.spec_op_visit_yr1), 0) AS sum_spec
    FROM
        (SELECT
            clm_yr1.asdb_member_key
            , CASE WHEN TRIM(clm_yr1.emis_cat)="Emergency" THEN 1 ELSE 0 END AS emis_ed_clm_yr1
            , CASE WHEN TRIM(clm_yr1.emis_cat)="Home Health" THEN 1 ELSE 0 END AS emis_hh_clm_yr1
            , CASE WHEN TRIM(clm_yr1.emis_cat)="Inpatient Facility" THEN 1 ELSE 0 END AS emis_ip_clm_yr1
            , CASE WHEN TRIM(clm_yr1.emis_cat)="Misc. Medical" THEN 1 ELSE 0 END AS emis_misc_clm_yr1
            , CASE WHEN TRIM(clm_yr1.emis_cat)="Specialist Physician" THEN 1 ELSE 0 END AS emis_spec_clm_yr1
            , CASE WHEN TRIM(clm_yr1.asdb_coe_general_type)="Long Term Care" THEN 1 ELSE 0 END as ltc_clm_yr1
            , CASE WHEN TRIM(clm_yr1.asdb_coe_general_type)="Other" AND TRIM(asdb_coe_sub_cat)="Professional" THEN 1 ELSE 0 END AS coe_other_clm_yr1
            , CASE WHEN TRIM(clm_yr1.asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Evaluation & Management" THEN 1 ELSE 0 END AS coe_eval_clm_yr1
            , CASE WHEN TRIM(clm_yr1.asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Medicine" THEN 1 ELSE 0 END AS coe_mrx_clm_yr1
            , CASE WHEN TRIM(clm_yr1.asdb_coe_general_type)="Radiology" AND TRIM(asdb_coe_sub_cat)="Professional" THEN 1 ELSE 0 END AS coe_radio_clm_yr1
            , CASE WHEN clm_yr1.plc_svc_ctg = "Outpatient"
                    AND (clm_yr1.asdb_coe_id IN (63000, 63100, 63200, 63300, 63400, 63500, 63600, 63999)
                        OR TRIM(clm_yr1.prindiag) IN ("V20.2","V20.31","V20.32","V70.0","V70.3","V70.5","V70.6","V70.8","V70.9","V72.31","V72.3",
                            "Z00.110","Z00.111","Z00.129", "Z00.8","Z01.411","Z01.419","Z01.42"))
                    AND NOT (TRIM(clm_yr1.emis_cat) = "Primary Physician" OR TRIM(lower(fac.prov_specialty)) LIKE "%gynecol%")
                THEN 1 ELSE 0 END AS spec_op_visit_yr1
        FROM
            clm_yr1
        LEFT JOIN 
            (SELECT 
                 asdb_svc_prov_key
                 , prov_specialty 
             FROM 
                 `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_SVC_PROV`) AS fac
                    ON clm_yr1.asdb_svc_prov_key = fac.asdb_svc_prov_key 
        ) AS clm
    GROUP BY
        clm.asdb_member_key
)
, ut_yr2 AS (
    SELECT
        clm.asdb_member_key
        , SUM(clm.emis_ip_clm_yr2) AS emis_ip_clm_yr2
        , SUM(clm.emis_ambul_clm_yr2) AS emis_ambul_clm_yr2
        , SUM(clm.coe_ip_hos_clm_yr2) AS coe_ip_hos_clm_yr2
    FROM
        (SELECT
            clm_yr2.asdb_member_key
            , CASE WHEN TRIM(clm_yr2.emis_cat)="Inpatient Facility" THEN 1 ELSE 0 END AS emis_ip_clm_yr2
            , CASE WHEN TRIM(clm_yr2.emis_cat)="Selected Ambulatory Facility" THEN 1 ELSE 0 END AS emis_ambul_clm_yr2
            , CASE WHEN TRIM(clm_yr2.asdb_coe_general_type)="Inpatient" AND TRIM(clm_yr2.asdb_coe_sub_cat)="Hospital" THEN 1 ELSE 0 END AS coe_ip_hos_clm_yr2
        FROM 
            clm_yr2
        ) AS clm
    GROUP BY
        clm.asdb_member_key
)
SELECT
    st.asdb_member_key
    , COALESCE(op.sum_op_visits_yr1, 0) AS sum_op_visits_yr1
    , COALESCE(ut_yr1.emis_ed_clm_yr1, 0) AS emis_ed_clm_yr1
    , COALESCE(ut_yr1.emis_hh_clm_yr1, 0) AS emis_hh_clm_yr1
    , COALESCE(ut_yr1.emis_ip_clm_yr1, 0) AS emis_ip_clm_yr1
    , COALESCE(ut_yr1.emis_misc_clm_yr1, 0) AS emis_misc_clm_yr1
    , COALESCE(ut_yr1.emis_spec_clm_yr1, 0) AS emis_spec_clm_yr1
    , COALESCE(ut_yr1.ltc_clm_yr1, 0) AS ltc_clm_yr1
    , COALESCE(ut_yr1.coe_other_clm_yr1, 0) AS coe_other_clm_yr1
    , COALESCE(ut_yr1.coe_eval_clm_yr1, 0) AS coe_eval_clm_yr1
    , COALESCE(ut_yr1.coe_mrx_clm_yr1, 0) AS coe_mrx_clm_yr1
    , COALESCE(ut_yr1.coe_radio_clm_yr1, 0) AS coe_radio_clm_yr1
    , COALESCE(ut_yr1.sum_spec, 0) AS sum_spec
    , COALESCE(ut_yr2.emis_ip_clm_yr2, 0) AS emis_ip_clm_yr2
    , COALESCE(ut_yr2.emis_ambul_clm_yr2, 0) AS emis_ambul_clm_yr2
    , COALESCE(ut_yr2.coe_ip_hos_clm_yr2, 0) AS coe_ip_hos_clm_yr2
FROM
    (SELECT DISTINCT asdb_member_key FROM `{{ST}}`) AS st
LEFT JOIN 
    op
        ON st.asdb_member_key = op.asdb_member_key
LEFT JOIN
    ut_yr1
        ON st.asdb_member_key = ut_yr1.asdb_member_key
LEFT JOIN
    ut_yr2
        ON st.asdb_member_key = ut_yr2.asdb_member_key
;

--------------------------
--- Chronic Conditions ---
--------------------------
CREATE OR REPLACE TABLE `{{CM_MD_SCHEMA}}.{{PREFIX}}_conditions` 
{{LABELS}} 
AS
WITH pre AS (
    SELECT
        st.asdb_member_key
        , MIN(b.rpt_end_dt) AS first_rpt
        , MAX(b.rpt_end_dt) AS last_rpt
        , MAX(CASE WHEN b.cond_rank=52 THEN 1 ELSE 0 END) AS abdominal_pain
        , MAX(CASE WHEN b.cond_rank=34 THEN 1 ELSE 0 END) AS AID
        , MAX(CASE WHEN b.cond_rank=69 THEN 1 ELSE 0 END) AS IDA
        , MAX(CASE WHEN b.cond_rank=41 THEN 1 ELSE 0 END) AS ANX
        , MAX(CASE WHEN b.cond_rank=61 THEN 1 ELSE 0 END) AS OST
        , MAX(CASE WHEN b.cond_rank=33 THEN 1 ELSE 0 END) AS AST
        , MAX(CASE WHEN b.cond_rank=45 THEN 1 ELSE 0 END) AS AUT
        , MAX(CASE WHEN b.cond_rank=51 THEN 1 ELSE 0 END) AS CHO
        , MAX(CASE WHEN b.cond_rank=39 THEN 1 ELSE 0 END) AS burns
        , MAX(CASE WHEN b.cond_rank=16 THEN 1 ELSE 0 END) AS cad
        , MAX(CASE WHEN b.cond_rank=29 THEN 1 ELSE 0 END) AS Cancer
        , MAX(CASE WHEN b.cond_rank=55 THEN 1 ELSE 0 END) AS narc
        , MAX(CASE WHEN b.cond_rank=17 THEN 1 ELSE 0 END) AS CBD
        , MAX(CASE WHEN b.cond_rank=4 THEN 1 ELSE 0 END) AS CHF
        , MAX(CASE WHEN b.cond_rank=3 THEN 1 ELSE 0 END) AS CRF
        , MAX(CASE WHEN b.cond_rank=62 THEN 1 ELSE 0 END) AS VNA
        , MAX(CASE WHEN b.cond_rank=30 THEN 1 ELSE 0 END) AS CHD
        , MAX(CASE WHEN b.cond_rank=44 THEN 1 ELSE 0 END) AS COP
        , MAX(CASE WHEN b.cond_rank=12 THEN 1 ELSE 0 END) AS CYS
        , MAX(CASE WHEN b.cond_rank=37 THEN 1 ELSE 0 END) AS DEP
        , MAX(CASE WHEN b.cond_rank=24 THEN 1 ELSE 0 END) AS DIA
        , MAX(CASE WHEN b.cond_rank=35 THEN 1 ELSE 0 END) AS EDO
        , MAX(CASE WHEN b.cond_rank=1 THEN 1 ELSE 0 END) AS esrd
        , MAX(CASE WHEN b.cond_rank=20 THEN 1 ELSE 0 END) AS EPL
        , MAX(CASE WHEN b.cond_rank=19 OR b.cond_rank=9 THEN 1 ELSE 0 END) AS CRO
        , MAX(CASE WHEN b.cond_rank=27 THEN 1 ELSE 0 END) AS MOH
        , MAX(CASE WHEN b.cond_rank=2 THEN 1 ELSE 0 END) AS HEM
        , MAX(CASE WHEN b.cond_rank=74 THEN 1 ELSE 0 END) AS HepC
        , MAX(CASE WHEN b.cond_rank=46 THEN 1 ELSE 0 END) AS HYP
        , MAX(CASE WHEN b.cond_rank=54 THEN 1 ELSE 0 END) AS HYC
        , MAX(CASE WHEN b.cond_rank=10 THEN 1 ELSE 0 END) AS immune
        , MAX(CASE WHEN b.cond_rank=72 THEN 1 ELSE 0 END) AS intel_dsblty
        , MAX(CASE WHEN b.cond_rank=6 THEN 1 ELSE 0 END) AS meta_cancer
        , MAX(CASE WHEN b.cond_rank=21 THEN 1 ELSE 0 END) AS liver_dis
        , MAX(CASE WHEN b.cond_rank=26 THEN 1 ELSE 0 END) AS MSS
        , MAX(CASE WHEN b.cond_rank=73 THEN 1 ELSE 0 END) AS OBE
        , MAX(CASE WHEN b.cond_rank=99 THEN 1 ELSE 0 END) AS oud
        , MAX(CASE WHEN b.cond_rank=64 THEN 1 ELSE 0 END) AS liver_other
        , MAX(CASE WHEN b.cond_rank=11 THEN 1 ELSE 0 END) AS paralysis
        , MAX(CASE WHEN b.cond_rank=42 THEN 1 ELSE 0 END) AS PAR
        , MAX(CASE WHEN b.cond_rank=57 THEN 1 ELSE 0 END) AS PUD
        , MAX(CASE WHEN b.cond_rank=18 THEN 1 ELSE 0 END) AS hmd
        , MAX(CASE WHEN b.cond_rank=50 THEN 1 ELSE 0 END) AS PVD
        , MAX(CASE WHEN b.cond_rank=43 THEN 1 ELSE 0 END) AS autoimmune
        , MAX(CASE WHEN b.cond_rank=32 THEN 1 ELSE 0 END) AS DEM
        , MAX(CASE WHEN b.cond_rank=7 THEN 1 ELSE 0 END) AS SCA
        , MAX(CASE WHEN b.cond_rank=66 THEN 1 ELSE 0 END) AS sleep_apnea
        , MAX(CASE WHEN b.cond_rank=13 THEN 1 ELSE 0 END) AS spinal_inj
        , MAX(CASE WHEN b.cond_rank=31 THEN 1 ELSE 0 END) AS back
        , MAX(CASE WHEN b.cond_rank=22 THEN 1 ELSE 0 END) AS substance
        , MAX(CASE WHEN b.cond_rank=14 THEN 1 ELSE 0 END) AS ALC
        , MAX(CASE WHEN b.cond_rank=36 THEN 1 ELSE 0 END) AS bipolar 
        , MAX(CASE WHEN b.cond_rank=25 THEN 1 ELSE 0 END) AS psychoses
    FROM 
        (SELECT DISTINCT asdb_member_key FROM `{{ST}}`) AS st
    LEFT JOIN 
        (SELECT 
             ppm_member_key
             , cond_rank
             , rpt_end_dt 
         FROM 
             `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_PPM_MEMBER_CONDITION_HISTORY`) AS b 
                ON st.asdb_member_key = b.ppm_member_key
                AND DATE_TRUNC({{INDEX_DT}}, MONTH) BETWEEN DATE_ADD(LAST_DAY(CAST(b.rpt_end_dt AS DATE), MONTH), INTERVAL 1 DAY) 
                    AND DATE_ADD(LAST_DAY(CAST(b.rpt_end_dt AS DATE), MONTH), INTERVAL 12 MONTH)
    GROUP BY 
        st.asdb_member_key
)
SELECT
    asdb_member_key
    , OST
    , AST
    , CHF
    , CRF
    , CHD
    , DIA
    , esrd
    , EPL
    , MOH
    , HYP
    , oud
    , paralysis
    , PVD
    , SCA
    , substance
    , ALC
    , (abdominal_pain+AID+ANX+OST+AST+AUT+CHO+burns+cad+Cancer+narc+CBD+CHF+CRF+VNA+CHD+
        COP+CYS+DEP+DIA+EDO+esrd+EPL+CRO+MOH+HEM+HepC+immune+intel_dsblty+meta_cancer+
        liver_dis+MSS+OBE+oud+liver_other+paralysis+PAR+hmd+PVD+autoimmune+DEM+SCA+
        sleep_apnea+spinal_inj+back+substance+ALC+bipolar+psychoses) AS major_chronic_cnt
FROM 
    pre
;

---------
--- Rx --
---------
CREATE OR REPLACE  TABLE `{{CM_MD_SCHEMA}}.{{PREFIX}}_rx_claims` 
{{LABELS}} 
AS
SELECT 
    st.asdb_member_key
    , rx.asdb_incurred_dt
    , rx.days_supply
    , rx.gpi
    , rx.pharmacytype
    , rx.drugtype
    , rx.formularyflag 
    , rx.ClaimType
    , rx.ndcnum
FROM 
    (SELECT DISTINCT asdb_member_key FROM `{{ST}}`) AS st
INNER JOIN
    `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_RX_DATA_STAGE` AS rx
        ON st.asdb_member_key = rx.asdb_member_key
WHERE 1 = 1
    AND rx.ClaimType = "P"  --paid claims only
    AND CAST(rx.asdb_incurred_dt AS DATE) BETWEEN DATE_SUB({{INDEX_DT}}, INTERVAL 24 MONTH) AND DATE_SUB({{INDEX_DT}}, INTERVAL 1 DAY)
;

CREATE OR REPLACE  TABLE `{{CM_MD_SCHEMA}}.{{PREFIX}}_rx` 
{{LABELS}} 
AS
WITH rx_yr1 AS (
    SELECT DISTINCT
        rx.asdb_member_key
        , CAST(rx.asdb_incurred_dt AS DATE) AS disp_dt
        , rx.days_supply
        , ROUND(CASE WHEN rx. days_supply >= 0.1 AND rx.days_supply < 30 THEN 30
                   ELSE rx.days_supply END / 30) AS scripts
        , SUBSTR(rx.gpi,1,2) AS gpi2
        , CASE WHEN rx.pharmacytype="R" THEN 1 ELSE 0 END AS retail_flag
        , CASE WHEN rx.drugtype = 3 THEN 1 ELSE 0 END AS generic_fill_flag
        , CASE WHEN rx.drugtype = 2 THEN 1 ELSE 0 END AS branded_generic_fill_flag
        , CASE WHEN rx.formularyflag = "F" or rx.drugtype = 3 THEN 1 ELSE 0 END AS formulary_fill_flag
    FROM 
        (SELECT 
            asdb_member_key
            , asdb_incurred_dt
            , days_supply
            , gpi
            , pharmacytype
            , drugtype
            , formularyflag 
            , ClaimType
        FROM 
            `{{CM_MD_SCHEMA}}.{{PREFIX}}_rx_claims`
        ) AS rx
        WHERE 1 = 1
            AND CAST(rx.asdb_incurred_dt AS DATE) BETWEEN DATE_SUB({{INDEX_DT}}, INTERVAL 12 MONTH) AND DATE_SUB({{INDEX_DT}}, INTERVAL 1 DAY)
)
, rx_yr2 AS (
    SELECT DISTINCT
        rx.asdb_member_key
        , CAST(rx.asdb_incurred_dt AS DATE) AS disp_dt
        , SUBSTR(rx.gpi,1,2) AS gpi2
        , CASE WHEN rx.drugtype = 2 THEN 1 ELSE 0 END AS branded_generic_fill_flag
        , CASE WHEN c.maint_drug_cd="X" THEN 1 ELSE 0 END AS maint_drug_flag 
    FROM 
        (SELECT 
            asdb_member_key
            , asdb_incurred_dt
            , gpi
            , drugtype
            , ClaimType 
            , ndcnum
         FROM 
            `{{CM_MD_SCHEMA}}.{{PREFIX}}_rx_claims`
        ) AS rx
    LEFT JOIN 
        (SELECT ndc_cd, maint_drug_cd FROM `edp-prod-hcbstorage.edp_hcb_anbor_enrsrcv.EDW_DRUG`) AS c
            ON TRIM(rx.ndcnum) = TRIM(c.ndc_cd)
    WHERE 1 = 1
        AND CAST(asdb_incurred_dt AS DATE) BETWEEN DATE_SUB({{INDEX_DT}}, INTERVAL 24 MONTH) AND DATE_SUB(DATE_SUB({{INDEX_DT}}, INTERVAL 1 DAY), INTERVAL 12 MONTH)
)
, tmp_yr1 AS (
    SELECT 
        asdb_member_key
        , SUM(retail_flag) AS retail_fills_yr1
        , SUM(generic_fill_flag) AS generic_fills_yr1
        , SUM(branded_generic_fill_flag) AS branded_generic_fills_yr1
        , SUM(formulary_fill_flag) AS formulary_fills_yr1
        , SUM(CASE WHEN gpi2="72" THEN days_supply ELSE 0 END) AS anticonvulsant_days_supply_yr1
    FROM 
        rx_yr1
    GROUP BY 
        asdb_member_key
)
, tmp_yr2 AS (
    SELECT 
        asdb_member_key
        , COUNT(*) AS rx_claim_cnt_yr2
        , SUM(branded_generic_fill_flag) AS branded_generic_fills_yr2
        , SUM(maint_drug_flag) AS maint_drug_fills_yr2
    FROM 
        rx_yr2
    GROUP BY 
        asdb_member_key
)
SELECT
    st.asdb_member_key
    , COALESCE(retail_fills_yr1, 0) AS retail_fills_yr1
    , COALESCE(generic_fills_yr1, 0) AS generic_fills_yr1
    , COALESCE(branded_generic_fills_yr1, 0) AS branded_generic_fills_yr1
    , COALESCE(formulary_fills_yr1, 0) AS formulary_fills_yr1
    , COALESCE(anticonvulsant_days_supply_yr1, 0) AS anticonvulsant_days_supply_yr1
    , COALESCE(rx_claim_cnt_yr2, 0) AS rx_claim_cnt_yr2
    , COALESCE(branded_generic_fills_yr2, 0) AS branded_generic_fills_yr2
    , COALESCE(maint_drug_fills_yr2, 0) AS maint_drug_fills_yr2
FROM
    (SELECT DISTINCT asdb_member_key FROM `{{ST}}`) AS st
LEFT JOIN
    tmp_yr1 AS yr1
        ON st.asdb_member_key = yr1.asdb_member_key
LEFT JOIN
    tmp_yr2 AS yr2
        ON st.asdb_member_key = yr2.asdb_member_key
;

--------------------
--- Demographics ---
--------------------
CREATE OR REPLACE TABLE `{{CM_MD_SCHEMA}}.{{PREFIX}}_demographics` 
{{LABELS}} 
AS
WITH mth AS (
    SELECT
        st.asdb_member_key
        , CAST(mnth.asdb_elig_dt AS DATE) AS asdb_elig_dt
        , ROW_NUMBER() OVER(PARTITION BY(mnth.asdb_member_key)) AS mnths
    FROM
        (SELECT 
             asdb_member_key
             , asdb_elig_dt 
         FROM 
             `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ELIG_DATA_MBR_PER_MTH`
        ) AS mnth
    LEFT JOIN
        (SELECT asdb_member_key FROM `{{ST}}`) AS st
            ON st.asdb_member_key = mnth.asdb_member_key
    WHERE 1 = 1 
        AND CAST(mnth.asdb_elig_dt AS DATE) BETWEEN DATE_SUB({{INDEX_DT}}, INTERVAL 12 MONTH) AND DATE_SUB({{INDEX_DT}}, INTERVAL 1 DAY)
)
SELECT 
    st.asdb_member_key
    , FLOOR(DATE_DIFF(DATE({{INDEX_DT}}), DATE(mb.dob), YEAR)) AS agenbr
    , CASE WHEN TRIM(UPPER(mb.gender)) = "M" THEN 0
        WHEN TRIM(UPPER(mb.gender)) = "F" THEN 1
        ELSE NULL END AS gender
    , COALESCE(mth.tenure, 0) AS tenure_yr1
    , coa_population_group                         
FROM 
  (SELECT DISTINCT asdb_member_key, coa_population_group FROM `{{ST}}`) AS st
LEFT JOIN 
    (SELECT asdb_member_key, dob, gender FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_MEMBER`) AS mb
        ON st.asdb_member_key = mb.asdb_member_key
LEFT JOIN 
    (SELECT asdb_member_key, MAX(mnths) AS tenure FROM mth GROUP BY asdb_member_key) AS mth
        ON st.asdb_member_key = mth.asdb_member_key
;

------------
--- SDoH ---
------------
CREATE OR REPLACE TABLE `{{CM_MD_SCHEMA}}.{{PREFIX}}_csdi` 
{{LABELS}} 
AS
WITH maxdt AS (
    SELECT 
        iodb_member_key
        , MAX(source_pstd_dts) AS source_pstd_dts
    FROM 
        `anbc-hcb-prod.insights_share_hcb_prod.v_enriched_address_medicaid`
    GROUP BY 
        iodb_member_key
)
, geo AS (
    SELECT 
        st.asdb_member_key
        , mb.iodb_member_key
        , id.ctfips
        , id.bgfips
    FROM
        (SELECT DISTINCT asdb_member_key FROM `{{ST}}`) AS st
    LEFT JOIN 
        (SELECT
            asdb_member_key
            , iodb_member_key
         FROM `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_MEMBER`) AS mb
            ON st.asdb_member_key=mb.asdb_member_key
    LEFT JOIN 
        (SELECT
             block_code AS bgfips --12 digit unique block code identifier
             , CONCAT(fips_state_county_code, census_tract) AS ctfips --11 digit unique census tract identifier
             , iodb_member_key
             , source_pstd_dts
             , geo_accuracy_code
        FROM 
            `anbc-hcb-prod.insights_share_hcb_prod.v_enriched_address_medicaid`
        WHERE 1 = 1
            AND TRIM(geo_accuracy_code) IN ("1", "2", "5", "6")
    ) AS id
            ON mb.iodb_member_key=id.iodb_member_key
    LEFT JOIN maxdt 
        ON id.iodb_member_key=maxdt.iodb_member_key
           AND id.source_pstd_dts=maxdt.source_pstd_dts

)
SELECT
    st.asdb_member_key
    , COALESCE(MAX(c.water_quality), 0) AS water_quality     
FROM 
    (SELECT DISTINCT asdb_member_key,  bgfips FROM geo) AS st
LEFT JOIN 
    (SELECT bgfips, water_quality FROM `edp-prod-storage.edp_ent_sdoheir_srcv.risk_index_block_group_historical_data` WHERE effective_year = CAST({{SDOH_YR}} AS INT)) AS c
        ON TRIM(st.bgfips)=TRIM(c.bgfips)
GROUP BY 
    st.asdb_member_key
;

----------------------------
--- combine all features ---
----------------------------
CREATE OR REPLACE TABLE `{{CM_MD_SCHEMA}}.{{PREFIX}}_features` 
{{LABELS}} 
AS
SELECT
    COALESCE(ed.asdb_member_key, 0) AS asdb_member_key
    , COALESCE(ed.sum_ed_visits_yr1, 0) AS sum_ed_visits_yr1
    , COALESCE(ed.sum_preventable_yr1, 0) AS sum_preventable_yr1
    , COALESCE(ip.acute_ip_flag_yr1, 0) AS acute_ip_flag_yr1
    , COALESCE(ip.sum_acute_ip_admits_yr1, 0) AS sum_acute_ip_admits_yr1
    , COALESCE(ip.sum_acute_calc_los_yr1, 0) AS sum_acute_calc_los_yr1
    , COALESCE(ut.sum_op_visits_yr1, 0) AS sum_op_visits_yr1
    , COALESCE(ut.emis_ed_clm_yr1, 0) AS emis_ed_clm_yr1
    , COALESCE(ut.emis_hh_clm_yr1, 0) AS emis_hh_clm_yr1
    , COALESCE(ut.emis_ip_clm_yr1, 0) AS emis_ip_clm_yr1
    , COALESCE(ut.emis_misc_clm_yr1, 0) AS emis_misc_clm_yr1
    , COALESCE(ut.emis_spec_clm_yr1, 0) AS emis_spec_clm_yr1
    , COALESCE(ut.ltc_clm_yr1, 0) AS ltc_clm_yr1
    , COALESCE(ut.coe_other_clm_yr1, 0) AS coe_other_clm_yr1
    , COALESCE(ut.coe_eval_clm_yr1, 0) AS coe_eval_clm_yr1
    , COALESCE(ut.coe_mrx_clm_yr1, 0) AS coe_mrx_clm_yr1
    , COALESCE(ut.coe_radio_clm_yr1, 0) AS coe_radio_clm_yr1
    , COALESCE(ut.emis_ip_clm_yr2, 0) AS emis_ip_clm_yr2
    , COALESCE(ut.emis_ambul_clm_yr2, 0) AS emis_ambul_clm_yr2
    , COALESCE(ut.coe_ip_hos_clm_yr2, 0) AS coe_ip_hos_clm_yr2
    , COALESCE(cond.OST, 0) AS OST
    , COALESCE(cond.AST, 0) AS AST
    , COALESCE(cond.CHF, 0) AS CHF
    , COALESCE(cond.CRF, 0) AS CRF
    , COALESCE(cond.CHD, 0) AS CHD
    , COALESCE(cond.DIA, 0) AS DIA
    , COALESCE(cond.esrd, 0) AS esrd
    , COALESCE(cond.EPL, 0) AS EPL
    , COALESCE(cond.MOH, 0) AS MOH
    , COALESCE(cond.HYP, 0) AS HYP
    , COALESCE(cond.oud, 0) AS oud
    , COALESCE(cond.paralysis, 0) AS paralysis
    , COALESCE(cond.PVD, 0) AS PVD
    , COALESCE(cond.SCA, 0) AS SCA
    , COALESCE(cond.substance, 0) AS substance
    , COALESCE(cond.ALC, 0) AS ALC
    , COALESCE(cond.major_chronic_cnt, 0) AS major_chronic_cnt
    , COALESCE(rx.retail_fills_yr1, 0) AS retail_fills_yr1
    , COALESCE(rx.generic_fills_yr1, 0) AS generic_fills_yr1
    , COALESCE(rx.branded_generic_fills_yr1, 0) AS branded_generic_fills_yr1
    , COALESCE(rx.formulary_fills_yr1, 0) AS formulary_fills_yr1
    , COALESCE(rx.anticonvulsant_days_supply_yr1, 0) AS anticonvulsant_days_supply_yr1
    , COALESCE(rx.rx_claim_cnt_yr2, 0) AS rx_claim_cnt_yr2
    , COALESCE(rx.branded_generic_fills_yr2, 0) AS branded_generic_fills_yr2
    , COALESCE(rx.maint_drug_fills_yr2, 0) AS maint_drug_fills_yr2
    , COALESCE(demo.agenbr, 0) AS agenbr
    , COALESCE(demo.gender, 0) AS gender
    , COALESCE(demo.tenure_yr1, 0) AS tenure_yr1
    , COALESCE(csdi.water_quality, 0) AS water_quality
    , COALESCE(ut.sum_spec, 0) AS sum_spec
    , COALESCE(emb.emb7, 0) AS emb7
    , COALESCE(emb.emb20, 0) AS emb20
    , COALESCE(emb.emb23, 0) AS emb23
    , COALESCE(emb.emb31, 0) AS emb31
    , COALESCE(emb.emb36, 0) AS emb36
    , COALESCE(emb.emb39, 0) AS emb39
    , COALESCE(emb.emb47, 0) AS emb47
    , COALESCE(emb.emb58, 0) AS emb58
    , COALESCE(emb.emb81, 0) AS emb81
    , COALESCE(emb.emb94, 0) AS emb94
    , COALESCE(emb.emb96, 0) AS emb96
    , COALESCE(emb.emb126, 0) AS emb126
    , COALESCE(emb.emb138, 0) AS emb138
    , COALESCE(emb.emb154, 0) AS emb154
    , COALESCE(emb.emb177, 0) AS emb177
    , COALESCE(emb.emb195, 0) AS emb195
    , COALESCE(emb.emb212, 0) AS emb212
    , COALESCE(emb.emb219, 0) AS emb219
    , COALESCE(emb.emb224, 0) AS emb224
    , COALESCE(emb.emb229, 0) AS emb229
    , COALESCE(emb.emb233, 0) AS emb233
    , COALESCE(emb.emb238, 0) AS emb238
    , COALESCE(emb.emb253, 0) AS emb253
    , CASE WHEN demo.coa_population_group = "ABD" THEN 1 ELSE 0       END AS coa_population_group_ABD
    , CASE WHEN demo.coa_population_group = "BH" THEN 1 ELSE 0        END AS coa_population_group_BH
    , CASE WHEN demo.coa_population_group = "Dual Elig" THEN 1 ELSE 0 END AS coa_population_group_Dual_Elig                                    
    , CASE WHEN demo.coa_population_group = "Expansion" THEN 1 ELSE 0 END AS coa_population_group_Expansion                                                           
    , CASE WHEN demo.coa_population_group = "LTSS" THEN 1 ELSE 0      END AS coa_population_group_LTSS        
    , CASE WHEN demo.coa_population_group = "TANF/CHIP" THEN 1 ELSE 0 END AS coa_population_group_TANF_CHIP  
FROM
    `{{CM_MD_SCHEMA}}.{{PREFIX}}_ed` AS ed
LEFT JOIN
    `{{CM_MD_SCHEMA}}.{{PREFIX}}_ip` AS ip
        ON ed.asdb_member_key = ip.asdb_member_key
LEFT JOIN
    `{{CM_MD_SCHEMA}}.{{PREFIX}}_utilization` AS ut
        ON ed.asdb_member_key = ut.asdb_member_key
LEFT JOIN
    `{{CM_MD_SCHEMA}}.{{PREFIX}}_conditions` AS cond
        ON ed.asdb_member_key = cond.asdb_member_key
LEFT JOIN
    `{{CM_MD_SCHEMA}}.{{PREFIX}}_rx` AS rx
        ON ed.asdb_member_key = rx.asdb_member_key
LEFT JOIN
    `{{CM_MD_SCHEMA}}.{{PREFIX}}_demographics` AS demo
        ON ed.asdb_member_key = demo.asdb_member_key
LEFT JOIN
    `{{CM_MD_SCHEMA}}.{{PREFIX}}_csdi` AS csdi
        ON ed.asdb_member_key = csdi.asdb_member_key
LEFT JOIN
    `{{CM_MD_SCHEMA}}.{{EMB}}` AS emb
        ON ed.asdb_member_key = emb.individual_id
;

--------------------------------
--- Drop intermediate tables ---
--------------------------------
DROP TABLE IF EXISTS `{{CM_MD_SCHEMA}}.{{PREFIX}}_member_index`;
DROP TABLE IF EXISTS `{{CM_MD_SCHEMA}}.{{PREFIX}}_ed`;
DROP TABLE IF EXISTS `{{CM_MD_SCHEMA}}.{{PREFIX}}_ip`;
DROP TABLE IF EXISTS `{{CM_MD_SCHEMA}}.{{PREFIX}}_utilization`;
DROP TABLE IF EXISTS `{{CM_MD_SCHEMA}}.{{PREFIX}}_conditions`;
DROP TABLE IF EXISTS `{{CM_MD_SCHEMA}}.{{PREFIX}}_rx`;
DROP TABLE IF EXISTS `{{CM_MD_SCHEMA}}.{{PREFIX}}_demographics`;
DROP TABLE IF EXISTS `{{CM_MD_SCHEMA}}.{{PREFIX}}_csdi`;