#!/bin/bash
#-- identify ED cases
bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_ed_cases_yr2`'

bq query \
--use_legacy_sql=false \
'
CREATE OR REPLACE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_ed_cases_yr2`
--PARTITION BY RANGE_BUCKET(asdb_plan_key, GENERATE_ARRAY(0,100,1))
--CLUSTER BY ed_vis_dt
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
SELECT 
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
    , mc.asdb_incurred_dt AS ed_vis_dt
    , mc.event_ct
    , mc.prindiag
    , mc.cost
    , mc.op_severitylvl
    , CASE WHEN TRIM(nyu.avoidable_ind) = "Y" THEN 1 
        ELSE 0 
        END AS avoidable_er_visits
    , CASE WHEN TRIM(nyu.er_type) = "UNNECESSARY" THEN 1 
        ELSE 0 
        END AS unnecessary_er_visits
    , CASE WHEN TRIM(nyu.er_type) = "PREVENTABLE" THEN 1 
        ELSE 0 
        END AS preventable_er_visits
FROM 
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `'$ST'`) AS st
INNER JOIN 
    `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ASDB_ICE_OP` AS mc

-- view references table (edp_hcb_mdcd_core_src.T_ASDB_ASDB_ICE_OP) but it is not a SNAP table

        ON st.asdb_member_key=mc.asdb_member_key
        AND st.asdb_plan_key=mc.asdb_plan_key
LEFT JOIN 
    `anbc-hcb-prod.cm_medicaid_hcb_prod.ICD10_X_ER_TYPE` AS nyu

-- no available view for this table

        ON TRIM(mc.prindiag) = TRIM(nyu.dx_cd)
WHERE 
    CAST(mc.asdb_incurred_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 24 MONTH) AND DATE_SUB(DATE_SUB(st.index_dt, INTERVAL 1 DAY), INTERVAL 12 MONTH)
    AND CAST(mc.asdb_coe_id AS INT64) = 20100
    AND event_ct=1
'

#-- Summarize ED
bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_ed_yr2`'

bq query \
--use_legacy_sql=false \
'
CREATE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_ed_yr2`
--PARTITION BY RANGE_BUCKET(asdb_plan_key, GENERATE_ARRAY(0,100,1))
--CLUSTER BY ed_vis_dt
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
SELECT 
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
    , COALESCE(SUM(mc.event_ct), 0) AS sum_ed_visits
    , CASE WHEN COALESCE(SUM(mc.event_ct), 0) > 0 THEN 1 ELSE 0 END AS ed_flag
    , COALESCE(SUM(mc.cost), 0) AS sum_ed_cost
    , COALESCE(SUM(mc.avoidable_er_visits), 0) AS sum_avoidable
    , COALESCE(SUM(mc.unnecessary_er_visits), 0) AS sum_unnecessary
    , COALESCE(SUM(mc.preventable_er_visits), 0) AS sum_preventable
    , MAX(mc.op_severitylvl) AS max_ed_severitylvl
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "1-Low" THEN mc.event_ct ELSE 0 END) AS low_sev_ed_visits
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "2-Low/Med" THEN mc.event_ct ELSE 0 END) AS low_med_sev_ed_visits
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "3-Med" THEN mc.event_ct ELSE 0 END) AS med_sev_ed_visits
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "4-Med/High" THEN mc.event_ct ELSE 0 END) AS med_high_sev_ed_visits
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "5-High" THEN mc.event_ct ELSE 0 END) AS high_sev_ed_visits
    , CASE WHEN SUM(CASE WHEN TRIM(mc.op_severitylvl) = "1-Low" THEN mc.event_ct ELSE 0 END) > 0 THEN 1 
        ELSE 0 
        END AS low_sev_ed_flag
    , CASE WHEN SUM(CASE WHEN TRIM(mc.op_severitylvl) = "2-Low/Med" THEN mc.event_ct ELSE 0 END) > 0 THEN 1 
        ELSE 0 
        END AS low_med_sev_ed_flag
    , CASE WHEN SUM(CASE WHEN TRIM(mc.op_severitylvl) = "3-Med" THEN mc.event_ct ELSE 0 END) > 0 THEN 1 
        ELSE 0 
        END AS med_sev_ed_flag
    , CASE WHEN SUM(CASE WHEN TRIM(mc.op_severitylvl) = "4-Med/High" THEN mc.event_ct ELSE 0 END) > 0 THEN 1 
        ELSE 0 
        END AS med_high_sev_ed_flag
    , CASE WHEN SUM(CASE WHEN TRIM(mc.op_severitylvl) = "5-High" THEN mc.event_ct ELSE 0 END) > 0 THEN 1 
        ELSE 0 
        END AS high_sev_ed_flag
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "1-Low" THEN mc.cost ELSE 0 END) AS low_sev_ed_cost
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "2-Low/Med" THEN mc.cost ELSE 0 END) AS low_med_sev_ed_cost
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "3-Med" THEN mc.cost ELSE 0 END) AS med_sev_ed_cost
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "4-Med/High" THEN mc.cost ELSE 0 END) AS med_high_sev_ed_cost
    , SUM(CASE WHEN TRIM(mc.op_severitylvl) = "5-High" THEN mc.cost ELSE 0 END) AS high_sev_ed_cost
FROM 
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `'$ST'`) AS st
LEFT JOIN 
    `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_ed_cases_yr2` AS mc
        ON st.asdb_member_key = mc.asdb_member_key
        AND st.asdb_plan_key = mc.asdb_plan_key
GROUP BY 
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
'

bq query \
--use_legacy_sql=false \
'
SELECT COUNT (DISTINCT asdb_member_key) AS distinct_count FROM `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_ed_yr2`
'

# ---- Extract IP Admits ----
# ---- 3/24/22: Modify code to pull acute & non-acute IP and create flag

bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_ip_cases_yr2`'

bq query \
--use_legacy_sql=false \
'
CREATE OR REPLACE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_ip_cases_yr2`
--PARTITION BY RANGE_BUCKET(asdb_plan_key, GENERATE_ARRAY(0,100,1))
--CLUSTER BY asdb_event_start_dt
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
SELECT 
    mc.asdb_member_key
    , mc.asdb_plan_key
    , st.index_dt
    , mc.asdb_event_start_dt
    , mc.asdb_event_end_dt
    , mc.final_discharge_dt
    , mc.prindiag
    , CASE WHEN mc.asdb_coe_id IN (10200,10700,10800)
            THEN "Acute"
        WHEN mc.asdb_coe_id IN (10000,10100,10300)
            THEN "Maternity/Infant"
        ELSE "Non-Acute"
        END AS ip_type
    , DATE_DIFF(mc.final_discharge_dt, mc.asdb_event_start_dt, DAY) AS calc_los
    -- MAX(DATE_ADD(mc.final_discharge_dt,mc.close_phadmit_days)) AS readmit_dt,
    -- MAX(CASE WHEN mc.close_phadmit_days BETWEEN 1 AND 30 THEN 1 ELSE 0 END) AS readmit_flag,
    -- MAX(CASE WHEN mc.close_phadmit_days BETWEEN 1 AND 30 THEN mc.close_phadmit_days ELSE 0 END) AS close_phadmit_days,
    , mc.event_ct
    , mc.admit_los
    , mc.paid_los
    , mc.cost AS ip_paid_amt
FROM
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `'$ST'`) AS st
INNER JOIN 
    `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ICE_IP`  AS mc

-- view already references table (edp_hcb_mdcd_core_src.T_ASDB_ICE_IP), but it is not a SNAP table

        ON st.asdb_member_key=mc.asdb_member_key
        AND st.asdb_plan_key=mc.asdb_plan_key
WHERE 
    CAST(mc.asdb_event_start_dt AS DATE) BETWEEN DATE_SUB(st.index_dt, INTERVAL 24 MONTH) AND DATE_SUB(DATE_SUB(st.index_dt, INTERVAL 1 DAY), INTERVAL 12 MONTH)
    AND mc.event_ct=1;
'

#----- summarize IP admits
#----- 3/24/22: Modify code to include acute & non-acute IP
bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_ip_yr2`'

bq query \
--use_legacy_sql=false \
'
CREATE TABLE  `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_ip_yr2`
--PARTITION BY RANGE_BUCKET(asdb_plan_key, GENERATE_ARRAY(0,100,1))
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
WITH acute AS (
    SELECT
        asdb_member_key
        , asdb_plan_key
        , index_dt
        , CASE WHEN SUM(event_ct) > 0 THEN 1 
            ELSE 0 
            END AS acute_ip_flag
        , SUM(event_ct) AS sum_acute_ip_admits
        , SUM(calc_los) AS sum_acute_calc_los
        , SUM(admit_los) AS sum_acute_admit_los
        , SUM(paid_los) AS sum_acute_paid_los
        , SUM(ip_paid_amt) AS sum_acute_ip_cost
    FROM  
        `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_ip_cases_yr2`
    WHERE 
        ip_type = "Acute"
    GROUP BY 
        asdb_member_key
        , asdb_plan_key
        , index_dt
),
nonacute AS (
    SELECT 
        asdb_member_key
        , asdb_plan_key
        , CASE WHEN SUM(event_ct) > 0 THEN 1 
            ELSE 0 
            END AS non_acute_ip_flag
        , SUM(event_ct) AS sum_non_acute_ip_admits
        , SUM(calc_los) AS sum_non_acute_calc_los
        , SUM(admit_los) AS sum_non_acute_admit_los
        , SUM(paid_los) AS sum_non_acute_paid_los
        , SUM(ip_paid_amt) AS sum_non_acute_ip_cost
    FROM 
        `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_ip_cases_yr2`
    WHERE 
        ip_type="Non-Acute"
    GROUP BY 
        asdb_member_key
        , asdb_plan_key
),
maternity AS (
    SELECT asdb_member_key,
        asdb_plan_key,
        CASE WHEN SUM(event_ct)>0 THEN 1 ELSE 0 END AS maternity_ip_flag,
        SUM(event_ct) AS sum_maternity_ip_admits,
        SUM(calc_los) AS sum_maternity_calc_los,
        SUM(admit_los) AS sum_maternity_admit_los,
        SUM(paid_los) AS sum_maternity_paid_los,
        SUM(ip_paid_amt) AS sum_maternity_ip_cost
    FROM 
        `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_ip_cases_yr2`
    WHERE 
        ip_type="Maternity/Infant"
    GROUP BY 
        asdb_member_key
        , asdb_plan_key
)
SELECT 
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
    , COALESCE(a.acute_ip_flag, 0) AS acute_ip_flag
    , COALESCE(a.sum_acute_ip_admits, 0) AS sum_acute_ip_admits
    , COALESCE(a.sum_acute_calc_los, 0) AS sum_acute_calc_los
    , COALESCE(a.sum_acute_admit_los, 0) AS sum_acute_admit_los
    , COALESCE(a.sum_acute_paid_los, 0) AS sum_acute_paid_los
    , COALESCE(a.sum_acute_ip_cost, 0) AS sum_acute_ip_cost
    , COALESCE(b.non_acute_ip_flag, 0) AS non_acute_ip_flag
    , COALESCE(b.sum_non_acute_ip_admits, 0) AS sum_non_acute_ip_admits
    , COALESCE(b.sum_non_acute_calc_los, 0) AS sum_non_acute_calc_los
    , COALESCE(b.sum_non_acute_admit_los, 0) AS sum_non_acute_admit_los
    , COALESCE(b.sum_non_acute_paid_los, 0) AS sum_non_acute_paid_los
    , COALESCE(b.sum_non_acute_ip_cost, 0) AS sum_non_acute_ip_cost
    , COALESCE(c.maternity_ip_flag, 0) AS maternity_ip_flag
    , COALESCE(c.sum_maternity_ip_admits, 0) AS sum_maternity_ip_admits
FROM 
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `'$ST'`) AS st
LEFT JOIN 
    acute AS a
        ON st.asdb_member_key = a.asdb_member_key
        AND st.asdb_plan_key = a.asdb_plan_key
LEFT JOIN 
    nonacute AS b
        ON st.asdb_member_key = b.asdb_member_key
        AND st.asdb_plan_key = b.asdb_plan_key
LEFT JOIN
    maternity AS c
        ON st.asdb_member_key = c.asdb_member_key
        AND st.asdb_plan_key = c.asdb_plan_key
'

#-- Summarize OP Visits
bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_op_yr2`'

bq query \
--use_legacy_sql=false \
'
CREATE TABLE  `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_op_yr2`
PARTITION BY RANGE_BUCKET(asdb_plan_key, GENERATE_ARRAY(0,100,1))
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
WITH clm AS(
    SELECT 
        *
        -- flag one claim for each visit to avoid double counting
        -- at member, plan, date, and facility level
        , CASE WHEN ROW_NUMBER() OVER(PARTITION BY asdb_member_key, asdb_plan_key, asdb_svc_prov_key, asdb_incurred_dt) = 1 THEN 1 
            ELSE 0 
            END AS op_ct
    FROM 
        `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_med_claims_yr2`
    WHERE 
        TRIM(asdb_coe_general_type) != "Inpatient"     --- remove IP, ED to get op claims
        AND TRIM(emis_cat) != "Institutional Services"
        AND TRIM(emis_cat) != "Emergency"
)
SELECT 
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
    , COALESCE(SUM(clm.paid_amt), 0) AS sum_op_cost
    , COALESCE(SUM(clm.op_ct), 0) AS sum_op_visits
    , MAX(CASE WHEN clm.op_ct=1 THEN 1 ELSE 0 END) AS op_flag
FROM 
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `'$ST'`) AS st
LEFT JOIN 
    clm
        ON st.asdb_member_key = clm.asdb_member_key
        AND st.asdb_plan_key = clm.asdb_plan_key
GROUP BY 
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
'

#-------- Other Cost & Utilization by Claims --------
bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_med_claims_flag_yr2`'

bq query \
--use_legacy_sql=false \
'
CREATE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_med_claims_flag_yr2`
--PARTITION BY RANGE_BUCKET(asdb_plan_key, GENERATE_ARRAY(0,100,1))
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
WITH clm AS (
    SELECT 
        *
        , CASE WHEN TRIM(asdb_coe_general_type)="Inpatient" OR TRIM(emis_cat)="Institutional Services" THEN "Inpatient"
            WHEN TRIM(emis_cat)="Emergency" THEN "Emergency"
            ELSE "Outpatient" 
            END AS plc_svc_ctg
    FROM 
        `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_med_claims_yr2`

-- no available view for this table

)
SELECT 
    clm.*
    , fac.prov_specialty
    ---- Cost Metrics
    , CASE WHEN plc_svc_ctg="Inpatient" THEN paid_amt 
        ELSE 0 
        END AS inpatient_cost
    , CASE WHEN plc_svc_ctg="Emergency" THEN paid_amt 
        ELSE 0 
        END AS emergency_cost
    , CASE WHEN plc_svc_ctg="Outpatient" THEN paid_amt 
        ELSE 0 
        END AS outpatient_cost
    , CASE WHEN TRIM(emis_cat)="Community-Based Services" THEN paid_amt 
        ELSE 0 
        END AS emis_community_cost
    , CASE WHEN TRIM(emis_cat)="Emergency" THEN paid_amt 
        ELSE 0 
        END AS emis_ed_cost
    , CASE WHEN TRIM(emis_cat)="Home Health" THEN paid_amt 
        ELSE 0 
        END AS emis_hh_cost
    , CASE WHEN TRIM(emis_cat)="Home-Based Services" THEN paid_amt  
        ELSE 0 
        END AS emis_home_cost
    , CASE WHEN TRIM(emis_cat)="Inpatient Facility" THEN paid_amt 
        ELSE 0 
        END AS emis_ip_cost
    , CASE WHEN TRIM(emis_cat)="Institutional Services" THEN paid_amt 
        ELSE 0 
        END AS emis_ins_cost
    , CASE WHEN TRIM(emis_cat)="Laboratory" THEN paid_amt   
        ELSE 0 
        END AS emis_lab_cost
    , CASE WHEN TRIM(emis_cat)="Medical Pharmacy" THEN paid_amt 
        ELSE 0 
        END AS emis_mrx_cost
    , CASE WHEN TRIM(emis_cat)="Mental Health" THEN paid_amt 
        ELSE 0 
        END AS emis_mh_cost
    , CASE WHEN TRIM(emis_cat)="Misc. Medical" THEN paid_amt 
        ELSE 0 
        END AS emis_misc_cost
    , CASE WHEN TRIM(emis_cat)="Primary Physician" THEN paid_amt 
        ELSE 0 
        END AS emis_pcp_cost
    , CASE WHEN TRIM(emis_cat)="Radiology" THEN paid_amt 
        ELSE 0 
        END AS emis_radio_cost
    , CASE WHEN TRIM(emis_cat)="Selected Ambulatory Facility" THEN paid_amt 
        ELSE 0 
        END AS emis_ambul_cost
    , CASE WHEN TRIM(emis_cat)="Specialist Physician" THEN paid_amt 
        ELSE 0 
        END AS emis_spec_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" THEN paid_amt 
        ELSE 0 
        END AS coe_ltc_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Inpatient" AND TRIM(asdb_coe_sub_cat)="Hospital" THEN paid_amt 
        ELSE 0 
        END AS coe_ip_hos_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Inpatient" AND TRIM(asdb_coe_sub_cat)="Non Hospital" THEN paid_amt 
        ELSE 0 
        END AS coe_ip_non_hos_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Laboratory" AND TRIM(asdb_coe_sub_cat)="Professional" THEN paid_amt 
        ELSE 0 
        END AS coe_lab_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" AND TRIM(asdb_coe_sub_cat)="Community Based Services" THEN paid_amt 
        ELSE 0 
        END AS coe_ltc_community_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" AND TRIM(asdb_coe_sub_cat)="Home Based Services" THEN paid_amt 
        ELSE 0 
        END AS coe_ltc_home_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" AND TRIM(asdb_coe_sub_cat)="Institution" THEN paid_amt 
        ELSE 0 
        END AS coe_ltc_ins_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Other" AND TRIM(asdb_coe_sub_cat)="Professional" THEN paid_amt 
        ELSE 0 
        END AS coe_other_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Outpatient" AND TRIM(asdb_coe_sub_cat)="Hospital" THEN paid_amt 
        ELSE 0 
        END AS coe_op_hos_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Outpatient" AND TRIM(asdb_coe_sub_cat)="Non Hospital" THEN paid_amt 
        ELSE 0 
        END AS coe_op_non_hos_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Anesthesia" THEN paid_amt 
        ELSE 0 
        END AS coe_anesth_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Evaluation & Management" THEN paid_amt 
        ELSE 0 
        END AS coe_eval_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Maternity" THEN paid_amt 
        ELSE 0 
        END AS coe_maternity_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Medicine" THEN paid_amt 
        ELSE 0 
        END AS coe_mrx_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Mental Health" THEN paid_amt 
        ELSE 0 
        END AS coe_mh_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Physician" THEN paid_amt 
        ELSE 0 
        END AS coe_phy_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Surgery" THEN paid_amt 
        ELSE 0 
        END AS coe_surg_cost
    , CASE WHEN TRIM(asdb_coe_general_type)="Radiology" AND TRIM(asdb_coe_sub_cat)="Professional" THEN paid_amt 
        ELSE 0 
        END AS coe_radio_cost
    , CASE WHEN TRIM(fac.prov_specialty)="Urgent Care" OR TRIM(location)="20" OR TRIM(servcode)="S9083" THEN paid_amt 
        ELSE 0 
        END AS uc_cost
    ---- Utilization Metrics
    , CASE WHEN TRIM(emis_cat)="Community-Based Services" THEN 1 
        ELSE 0 
        END AS emis_community_clm
    , CASE WHEN TRIM(emis_cat)="Emergency" THEN 1 
        ELSE 0 
        END AS emis_ed_clm
    , CASE WHEN TRIM(emis_cat)="Home Health" THEN 1 
        ELSE 0 
        END AS emis_hh_clm
    , CASE WHEN TRIM(emis_cat)="Home-Based Services" THEN 1 
        ELSE 0 
        END AS emis_home_clm
    , CASE WHEN TRIM(emis_cat)="Inpatient Facility" THEN 1 
        ELSE 0 
        END AS emis_ip_clm
    , CASE WHEN TRIM(emis_cat)="Institutional Services" THEN 1 
        ELSE 0 
        END AS emis_ins_clm
    , CASE WHEN TRIM(emis_cat)="Laboratory" THEN 1 
        ELSE 0 
        END AS emis_lab_clm
    , CASE WHEN TRIM(emis_cat)="Medical Pharmacy" THEN 1 
        ELSE 0 
        END AS emis_mrx_clm
    , CASE WHEN TRIM(emis_cat)="Mental Health" THEN 1 
        ELSE 0 
        END AS emis_mh_clm
    , CASE WHEN TRIM(emis_cat)="Misc. Medical" THEN 1 
        ELSE 0 
        END AS emis_misc_clm
    , CASE WHEN TRIM(emis_cat)="Primary Physician" THEN 1 
        ELSE 0 
        END AS emis_pcp_clm
    , CASE WHEN TRIM(emis_cat)="Radiology" THEN 1 
        ELSE 0 
        END AS emis_radio_clm
    , CASE WHEN TRIM(emis_cat)="Selected Ambulatory Facility" THEN 1 
        ELSE 0 
        END AS emis_ambul_clm
    , CASE WHEN TRIM(emis_cat)="Specialist Physician" THEN 1 
        ELSE 0 
        END AS emis_spec_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" THEN 1 
        ELSE 0 
        END as ltc_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Inpatient" AND TRIM(asdb_coe_sub_cat)="Hospital" THEN 1 
        ELSE 0 
        END AS coe_ip_hos_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Inpatient" AND TRIM(asdb_coe_sub_cat)="Non Hospital" THEN 1 
        ELSE 0 
        END AS coe_ip_non_hos_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Laboratory" AND TRIM(asdb_coe_sub_cat)="Professional" THEN 1 
        ELSE 0 
        END AS coe_lab_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" AND TRIM(asdb_coe_sub_cat)="Community Based Services" THEN 1 
        ELSE 0 
        END AS coe_ltc_community_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" AND TRIM(asdb_coe_sub_cat)="Home Based Services" THEN 1 
        ELSE 0 
        END AS coe_ltc_home_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Long Term Care" AND TRIM(asdb_coe_sub_cat)="Institution" THEN 1 
        ELSE 0 
        END AS coe_ltc_ins_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Other" AND TRIM(asdb_coe_sub_cat)="Professional" THEN 1 
        ELSE 0 
        END AS coe_other_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Outpatient" AND TRIM(asdb_coe_sub_cat)="Hospital" THEN 1 
        ELSE 0 
        END AS coe_op_hos_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Outpatient" AND TRIM(asdb_coe_sub_cat)="Non Hospital" THEN 1 
        ELSE 0 
        END AS coe_op_non_hos_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Anesthesia" THEN 1 
        ELSE 0 
        END AS coe_anesth_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Evaluation & Management" THEN 1 
        ELSE 0 
        END AS coe_eval_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Maternity" THEN 1 
        ELSE 0 
        END AS coe_maternity_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Medicine" THEN 1 
        ELSE 0 
        END AS coe_mrx_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Mental Health" THEN 1 
        ELSE 0 
        END AS coe_mh_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Physician" THEN 1 
        ELSE 0 
        END AS coe_phy_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Physician" AND TRIM(asdb_coe_sub_cat)="Surgery" THEN 1 
        ELSE 0 
        END AS coe_surg_clm
    , CASE WHEN TRIM(asdb_coe_general_type)="Radiology" AND TRIM(asdb_coe_sub_cat)="Professional" THEN 1 
        ELSE 0 
        END AS coe_radio_clm
    , CASE WHEN TRIM(fac.prov_specialty)="Urgent Care" OR TRIM(location)="20" OR TRIM(servcode)="S9083" THEN 1 
        ELSE 0 
        END AS uc_clm
    , CASE WHEN (TRIM(clm.revcode) in ("0760","0761","0762","0769") AND (TRIM(clm.billtype) like "13%" or TRIM(clm.billtype) like "85%") 
            AND (TRIM(clm.servcode)) in ("99217","99218","99219","99202","G0378","G0379","")) THEN 1 
        ELSE 0 
        END AS obs_clm
FROM 
    clm
LEFT JOIN 
    `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_SVC_PROV` AS fac
        ON clm.asdb_svc_prov_key = fac.asdb_svc_prov_key 
        AND clm.asdb_plan_key = fac.asdb_plan_key
'

#-- Summarize
bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_other_cost_utilization_yr2`'

bq query \
--use_legacy_sql=false \
'
CREATE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_other_cost_utilization_yr2`
--PARTITION BY RANGE_BUCKET(asdb_plan_key, GENERATE_ARRAY(0,100,1))
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
SELECT
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
    , COUNT(DISTINCT clm.claimid) AS claim_cnt
     ,COUNT(*) AS claim_line_cnt
    -- COALESCE(SUM(clm.allowed_amt), 0) AS sum_allowed_amt,
    , COALESCE(SUM(clm.paid_amt), 0) AS sum_paid_amt  --use paid amt for cost in Medicaid Analyses
    -- COALESCE(SUM(clm.copay_amt), 0) AS sum_copay_amt,
    -- Cost
    , COALESCE(SUM(clm.inpatient_cost), 0) AS inpatient_cost
    , COALESCE(SUM(clm.emergency_cost), 0) AS emergency_cost
    , COALESCE(SUM(clm.outpatient_cost), 0) AS outpatient_cost
    , COALESCE(SUM(clm.emis_community_cost), 0) AS emis_community_cost
    , COALESCE(SUM(clm.emis_ed_cost), 0) AS emis_ed_cost
    , COALESCE(SUM(clm.emis_hh_cost), 0) AS emis_hh_cost
    , COALESCE(SUM(clm.emis_home_cost), 0) AS emis_home_cost
    , COALESCE(SUM(clm.emis_ip_cost), 0) AS emis_ip_cost
    , COALESCE(SUM(clm.emis_ins_cost), 0) AS emis_ins_cost
    , COALESCE(SUM(clm.emis_lab_cost), 0) AS emis_lab_cost
    , COALESCE(SUM(clm.emis_mrx_cost), 0) AS emis_mrx_cost
    , COALESCE(SUM(clm.emis_mh_cost), 0) AS emis_mh_cost
    , COALESCE(SUM(clm.emis_misc_cost), 0) AS emis_misc_cost
    , COALESCE(SUM(clm.emis_pcp_cost), 0) AS emis_pcp_cost
    , COALESCE(SUM(clm.emis_radio_cost), 0) AS emis_radio_cost
    , COALESCE(SUM(clm.emis_ambul_cost), 0) AS emis_ambul_cost
    , COALESCE(SUM(clm.emis_spec_cost), 0) AS emis_spec_cost
    , COALESCE(SUM(clm.coe_ltc_cost), 0) AS coe_ltc_cost
    , COALESCE(SUM(clm.coe_ip_hos_cost), 0) AS coe_ip_hos_cost
    , COALESCE(SUM(clm.coe_ip_non_hos_cost), 0) AS coe_ip_non_hos_cost
    , COALESCE(SUM(clm.coe_lab_cost), 0) AS coe_lab_cost
    , COALESCE(SUM(clm.coe_ltc_community_cost), 0) AS coe_ltc_community_cost
    , COALESCE(SUM(clm.coe_ltc_home_cost), 0) AS coe_ltc_home_cost
    , COALESCE(SUM(clm.coe_ltc_ins_cost), 0) AS coe_ltc_ins_cost
    , COALESCE(SUM(clm.coe_other_cost), 0) AS coe_other_cost
    , COALESCE(SUM(clm.coe_op_hos_cost), 0) AS coe_op_hos_cost
    , COALESCE(SUM(clm.coe_op_non_hos_cost), 0) AS coe_op_non_hos_cost
    , COALESCE(SUM(clm.coe_anesth_cost), 0) AS coe_anesth_cost
    , COALESCE(SUM(clm.coe_eval_cost), 0) AS coe_eval_cost
    , COALESCE(SUM(clm.coe_maternity_cost), 0) AS coe_maternity_cost
    , COALESCE(SUM(clm.coe_mrx_cost), 0) AS coe_mrx_cost
    , COALESCE(SUM(clm.coe_mh_cost), 0) AS coe_mh_cost
    , COALESCE(SUM(clm.coe_phy_cost), 0) AS coe_phy_cost
    , COALESCE(SUM(clm.coe_surg_cost), 0) AS coe_surg_cost
    , COALESCE(SUM(clm.coe_radio_cost), 0) AS coe_radio_cost
    , COALESCE(SUM(clm.uc_cost), 0) AS uc_cost
    -- Utilization
    , COALESCE(SUM(clm.emis_community_clm), 0) AS emis_community_clm
    , COALESCE(SUM(clm.emis_ed_clm), 0) AS emis_ed_clm
    , COALESCE(SUM(clm.emis_hh_clm), 0) AS emis_hh_clm
    , COALESCE(SUM(clm.emis_home_clm), 0) AS emis_home_clm
    , COALESCE(SUM(clm.emis_ip_clm), 0) AS emis_ip_clm
    , COALESCE(SUM(clm.emis_ins_clm), 0) AS emis_ins_clm
    , COALESCE(SUM(clm.emis_lab_clm), 0) AS emis_lab_clm
    , COALESCE(SUM(clm.emis_mrx_clm), 0) AS emis_mrx_clm
    , COALESCE(SUM(clm.emis_mh_clm), 0) AS emis_mh_clm
    , COALESCE(SUM(clm.emis_misc_clm), 0) AS emis_misc_clm
    , COALESCE(SUM(clm.emis_pcp_clm), 0) AS emis_pcp_clm
    , COALESCE(SUM(clm.emis_radio_clm), 0) AS emis_radio_clm
    , COALESCE(SUM(clm.emis_ambul_clm), 0) AS emis_ambul_clm
    , COALESCE(SUM(clm.emis_spec_clm), 0) AS emis_spec_clm
    , COALESCE(SUM(clm.ltc_clm), 0) AS ltc_clm
    , COALESCE(SUM(clm.coe_ip_hos_clm), 0) AS coe_ip_hos_clm
    , COALESCE(SUM(clm.coe_ip_non_hos_clm), 0) AS coe_ip_non_hos_clm
    , COALESCE(SUM(clm.coe_lab_clm), 0) AS coe_lab_clm
    , COALESCE(SUM(clm.coe_ltc_community_clm), 0) AS coe_ltc_community_clm
    , COALESCE(SUM(clm.coe_ltc_home_clm), 0) AS coe_ltc_home_clm
    , COALESCE(SUM(clm.coe_ltc_ins_clm), 0) AS coe_ltc_ins_clm
    , COALESCE(SUM(clm.coe_other_clm), 0) AS coe_other_clm
    , COALESCE(SUM(clm.coe_op_hos_clm), 0) AS coe_op_hos_clm
    , COALESCE(SUM(clm.coe_op_non_hos_clm), 0) AS coe_op_non_hos_clm
    , COALESCE(SUM(clm.coe_anesth_clm), 0) AS coe_anesth_clm
    , COALESCE(SUM(clm.coe_eval_clm), 0) AS coe_eval_clm
    , COALESCE(SUM(clm.coe_maternity_clm), 0) AS coe_maternity_clm
    , COALESCE(SUM(clm.coe_mrx_clm), 0) AS coe_mrx_clm
    , COALESCE(SUM(clm.coe_mh_clm), 0) AS coe_mh_clm
    , COALESCE(SUM(clm.coe_phy_clm), 0) AS coe_phy_clm
    , COALESCE(SUM(clm.coe_surg_clm), 0) AS coe_surg_clm
    , COALESCE(SUM(clm.coe_radio_clm), 0) AS coe_radio_clm
    , COALESCE(SUM(clm.uc_clm), 0) AS uc_clm
    , COALESCE(SUM(clm.obs_clm), 0) AS obs_clm
FROM 
    (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `'$ST'`) AS st
LEFT JOIN 
    `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_med_claims_flag_yr2` AS clm
        ON st.asdb_member_key=clm.asdb_member_key
        AND st.asdb_plan_key=clm.asdb_plan_key
GROUP BY 
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
'

# -- QA 

bq query \
--use_legacy_sql=false \
'
SELECT 
    AVG(ed_flag) AS avg_ed_flag
    , AVG(sum_ed_visits) AS avg_sum_ed_visits
    , AVG(sum_ed_cost) AS avg_sum_ed_cost
FROM `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_ed_yr2`
'

bq query \
--use_legacy_sql=false \
'
SELECT
    AVG(acute_ip_flag) AS avg_acute_ip_flag
    , AVG(sum_acute_ip_admits) AS avg_sum_acute_ip_admits
    , AVG(sum_acute_ip_cost) AS avg_sum_acute_ip_cost
    , AVG(non_acute_ip_flag) AS avg_non_acute_ip_flag
    , AVG(sum_non_acute_ip_admits) AS avg_sum_non_acute_ip_admits
    , AVG(sum_non_acute_ip_cost) AS avg_sum_non_acute_ip_cost
FROM `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_ip_yr2`
'

bq query \
--use_legacy_sql=false \
'
SELECT
    AVG(op_flag) AS avg_op_flag
    , AVG(sum_op_visits) AS avg_sum_op_visits
    , AVG(sum_op_cost) AS avg_sum_op_cost
FROM `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_op_yr2`
'
