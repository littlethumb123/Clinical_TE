--modify to use DRG impactability for final metrics we care about
--PQI checks...

CREATE OR REPLACE TABLE `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_outcome_ip_cases`
OPTIONS (labels = [("owner", "yourname_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
SELECT 
    mc.asdb_member_key
    , mc.asdb_event_start_dt
    , mc.asdb_event_end_dt
    , mc.final_discharge_dt
    , mc.prindiag
    , CASE WHEN mc.asdb_coe_id IN (10200,10700,10800) THEN "Acute"
        ELSE "Non-Acute"
        END AS ip_type
    , DATE_DIFF(mc.final_discharge_dt, mc.asdb_event_start_dt, DAY) AS calc_los
    , mc.event_ct
    , mc.admit_los
    , mc.paid_los
    , mc.cost AS ip_paid_amt
FROM
    (SELECT DISTINCT asdb_member_key, index_dt FROM `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_member_index`) AS st
INNER JOIN 
    `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ICE_IP`  AS mc

-- view already references table (edp_hcb_mdcd_core_src.T_ASDB_ASDB_ICE_IP), but it is not a SNAP table

        ON st.asdb_member_key=mc.asdb_member_key
WHERE 
    CAST(mc.asdb_event_start_dt AS DATE) BETWEEN DATE_ADD(index_dt, INTERVAL 1 DAY) AND DATE_ADD(index_dt, INTERVAL 181 DAY)
    AND mc.event_ct=1
;
    
CREATE OR REPLACE TABLE `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_outcome_ip`
OPTIONS (labels = [("owner", "yourname_aetna_com"),("cost_center", "13070")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY))
AS
WITH acute AS (
    SELECT
        asdb_member_key
        , CASE WHEN SUM(event_ct) > 0 THEN 1 
            ELSE 0 
            END AS acute_ip_flag
        , SUM(event_ct) AS sum_acute_ip_admits
        , SUM(calc_los) AS sum_acute_calc_los
        , SUM(admit_los) AS sum_acute_admit_los
        , SUM(paid_los) AS sum_acute_paid_los
        , SUM(ip_paid_amt) AS sum_acute_ip_cost
    FROM  
        `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_outcome_ip_cases`
    WHERE 
        ip_type = "Acute"
    GROUP BY 
        asdb_member_key
)
SELECT 
    st.asdb_member_key
    , st.index_dt
    , COALESCE(a.acute_ip_flag, 0) AS acute_ip_flag
    , COALESCE(a.sum_acute_ip_admits, 0) AS sum_acute_ip_admits
    , COALESCE(a.sum_acute_calc_los, 0) AS sum_acute_calc_los
    , COALESCE(a.sum_acute_admit_los, 0) AS sum_acute_admit_los
    , COALESCE(a.sum_acute_paid_los, 0) AS sum_acute_paid_los
    , COALESCE(a.sum_acute_ip_cost, 0) AS sum_acute_ip_cost
FROM 
    (SELECT DISTINCT asdb_member_key, index_dt FROM `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_member_index`) AS st
LEFT JOIN 
    acute AS a
        ON st.asdb_member_key = a.asdb_member_key
;