CREATE OR REPLACE TABLE `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_outcome_ip_cases_w_drg`
OPTIONS (labels = [("owner", "yourname_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
SELECT 
    mc.asdb_member_key
    , mc.asdb_event_start_dt
    , mc.asdb_event_end_dt
    , mc.final_discharge_dt
    , mc.prindiag
    , mc.admit_drg
    , drg.drg_type
    , CASE WHEN mc.asdb_coe_id IN (10200,10700,10800) THEN "Acute"
        ELSE "Non-Acute"
        END AS ip_type
    , DATE_DIFF(mc.final_discharge_dt, mc.asdb_event_start_dt, DAY) AS calc_los
    , mc.event_ct
    , mc.admit_los
    , mc.paid_los
    , mc.cost AS ip_paid_amt
FROM
    (SELECT DISTINCT asdb_member_key, run_dt AS index_dt FROM `anbc-hcb-dev.cm_medicaid_hcb_dev.a091749_all_cohort`) AS st
INNER JOIN 
    `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ICE_IP`  AS mc
        ON st.asdb_member_key=mc.asdb_member_key
LEFT JOIN
    `edp-prod-hcbstorage.edp_hcb_core_src.T_PS_IMPACTABLE_ADMIT_DRG` AS drg
        ON mc.admit_drg = drg.drg
WHERE 
    CAST(mc.asdb_event_start_dt AS DATE) BETWEEN DATE_ADD(index_dt, INTERVAL 1 DAY) AND DATE_ADD(index_dt, INTERVAL 181 DAY)
    AND mc.event_ct=1
;
