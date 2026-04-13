bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_conditions`'

bq query \
--use_legacy_sql=false \
'
CREATE OR REPLACE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_conditions`
--PARTITION BY RANGE_BUCKET(asdb_plan_key, GENERATE_ARRAY(0,100,1))
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
SELECT
    *
    , (abdominal_pain+AID+ANX+OST+AST+AUT+CHO+burns+cad+Cancer+narc+CBD+CHF+CRF+VNA+CHD+
        COP+CYS+DEP+DIA+EDO+esrd+EPL+CRO+MOH+HEM+HepC+immune+intel_dsblty+meta_cancer+
        liver_dis+MSS+OBE+oud+liver_other+paralysis+PAR+hmd+PVD+autoimmune+DEM+SCA+
        sleep_apnea+spinal_inj+back+substance+ALC+bipolar+psychoses) AS major_chronic_cnt
FROM 
    (SELECT
        st.asdb_member_key
        , st.asdb_plan_key
        , st.index_dt
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

        -- bh
        , MAX(CASE WHEN b.cond_rank=22 THEN 1 ELSE 0 END) AS substance
        , MAX(CASE WHEN b.cond_rank=14 THEN 1 ELSE 0 END) AS ALC
        , MAX(CASE WHEN b.cond_rank=36 THEN 1 ELSE 0 END) AS bipolar 
        , MAX(CASE WHEN b.cond_rank=25 THEN 1 ELSE 0 END) AS psychoses
    FROM 
        (SELECT DISTINCT asdb_member_key, asdb_plan_key, index_dt FROM `'$ST'`) AS st
    LEFT JOIN 
        `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_PPM_MEMBER_CONDITION_HISTORY` AS b 

-- view already references SNAP table (edp_hcb_mdcd_core_src.SNAP_T_ASDB_PPM_MEMBER_CONDITION_HISTORY_20230626T183935)

            ON st.asdb_member_key = b.ppm_member_key
            AND st.asdb_plan_key = b.ppm_plan_key
            AND DATE_TRUNC(st.index_dt, MONTH) BETWEEN DATE_ADD(LAST_DAY(CAST(b.rpt_end_dt AS DATE), MONTH), INTERVAL 1 DAY) 
                AND DATE_ADD(LAST_DAY(CAST(b.rpt_end_dt AS DATE), MONTH), INTERVAL 12 MONTH)
    GROUP BY 
        st.asdb_member_key
        , st.asdb_plan_key
        , st.index_dt
) tb
'


#--QA check
bq query \
--use_legacy_sql=false \
'
SELECT
    SUM(chf)
    , SUM(dia)
    , SUM(hyp)
FROM `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_conditions`
'
