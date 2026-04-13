#!/bin/bash
#---- BRFSS Date
bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_preventative`'

bq query \
--use_legacy_sql=false \
'
CREATE OR REPLACE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_preventative`
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
SELECT
    asdb_member_key
    , asdb_plan_key
    , claimid
    , index_dt
    , asdb_incurred_dt
    , CASE WHEN plc_svc_ctg = "Outpatient"
            AND (asdb_coe_id IN (63000, 63100, 63200, 63300, 63400, 63500, 63600, 63999)
                OR TRIM(prindiag) in ("V20.2","V20.31","V20.32","V70.0","V70.3","V70.5","V70.6","V70.8","V70.9","V72.31","V72.3",
                    "Z00.110","Z00.111","Z00.129", "Z00.8","Z01.411","Z01.419","Z01.42"))
            AND TRIM(emis_cat) = "Primary Physician"
        THEN 1 ELSE 0 END AS pcp_op_visit
    , CASE WHEN plc_svc_ctg = "Outpatient"
            AND (asdb_coe_id IN (63000, 63100, 63200, 63300, 63400, 63500, 63600, 63999)
                OR TRIM(prindiag) in ("V20.2","V20.31","V20.32","V70.0","V70.3","V70.5","V70.6","V70.8","V70.9","V72.31","V72.3",
                    "Z00.110","Z00.111","Z00.129", "Z00.8","Z01.411","Z01.419","Z01.42"))
            AND NOT (TRIM(emis_cat) = "Primary Physician" OR TRIM(lower(prov_specialty)) LIKE "%gynecol%")
        THEN 1 ELSE 0 END AS spec_op_visit
    , CASE WHEN plc_svc_ctg = "Outpatient"
            AND (asdb_coe_id IN (63000, 63100, 63200, 63300, 63400, 63500, 63600, 63999)
                OR TRIM(prindiag) in ("V20.2","V20.31","V20.32","V70.0","V70.3","V70.5","V70.6","V70.8","V70.9","V72.31","V72.3",
                    "Z00.110","Z00.111","Z00.129", "Z00.8","Z01.411","Z01.419","Z01.42"))
            AND (TRIM(lower(prov_specialty)) LIKE "%midwif%" OR TRIM(lower(prov_specialty)) LIKE "%gynecol%")
        THEN 1 ELSE 0 END AS obgyn_mw_op_visit
    , CASE WHEN TRIM(servcode) in ("99377","99378","G0182","G0337","Q5003","Q5004","Q5005","Q5006",
            "Q5007","Q5008","Q5010","S0255","S0271","S9126","T2042","T2043","T2044","T2045","T2046","G9054")
        THEN 1 ELSE 0 END AS hospice_claim_flag
    , CASE WHEN TRIM(servcode) in
                      ("E0185","E0188","E0189","E0194",
                       "E0197","E0198","E0199","E0250",
                       "E0251","E0255","E0256","E0260",
                       "E0261","E0265","E0266","E0290",
                       "E0291","E0292","E0293","E0294",
                       "E0295","E0296","E0297","E0300",
                       "E0301","E0302","E0303","E0304",
                       "E0424","E0431","E0433","E0434",
                       "E0439","E0441","E0442","E0443",
                       "E0444","E0450","E0460","E0461",
                       "E0462","E0463","E0464","E0470",
                       "E0471","E0472","E0480","E0482",
                       "E0483","E0484","E0570","E0575",
                       "E0580","E0585","E0601","E0607",
                       "E0627","E0628","E0629","E0636",
                       "E0650","E0651","E0652","E0655",
                       "E0656","E0657","E0660","E0665",
                       "E0666","E0667","E0668","E0669",
                       "E0671","E0672","E0673","E0675",
                       "E0692","E0693","E0694","E0720",
                       "E0730","E0731","E0740","E0744",
                       "E0745","E0747","E0748","E0749",
                       "E0760","E0762","E0764","E0765",
                       "E0782","E0783","E0784","E0786",
                       "E0840","E0849","E0850","E0855",
                       "E0856","E0958","E0959","E0960",
                       "E0961","E0966","E0967","E0968",
                       "E0969","E0971","E0973","E0974",
                       "E0978","E0980","E0981","E0982",
                       "E0983","E0984","E0985","E0986",
                       "E0990","E0992","E0994","E1014",
                       "E1015","E1020","E1028","E1029",
                       "E1030","E1031","E1035","E1036",
                       "E1037","E1038","E1039","E1161",
                       "E1227","E1228","E1232","E1233",
                       "E1234","E1235","E1236","E1237",
                       "E1238","E1296","E1297","E1298",
                       "E1310","E2502","E2506","E2508",
                       "E2510","E2227","K0001","K0002",
                       "K0003","K0004","K0005","K0006",
                       "K0007","K0009","K0606","K0730")
        THEN 1 ELSE 0 END AS dme_significant
    , CASE WHEN TRIM(servcode) in ("80061","83715","83716","83721","83718","83700","83701","83704","3048F","3049F","3050F")
        THEN 1 ELSE 0 END AS cholest_screen_claim
    , CASE WHEN TRIM(servcode) in ("83036","83037","3044F","3045F","3046F")
        THEN 1 ELSE 0 END AS hba1c_test_claim
    , CASE WHEN TRIM(servcode) in ("C9287","J0178","Q2048","Q2049","Q2043","C9296","Q2050","J8510","J8520","J8521","J8530","J8560","J8600","J8610","J8700","J8705","J8999")
      OR TRIM(servcode) LIKE "J9%" AND TRIM(servcode) not in ("J9202","J9395","J9217")
        THEN 1 ELSE 0 END AS chemo_clm_flg
    , CASE WHEN TRIM(servcode) in ("90653", "90654", "90656", "90660", "90661", "90662", "90672", "90673", "90686", "90688","Q2033", "Q2034", "Q2035", "Q2036", "Q2037", "Q2038", "Q2039", "90655", "90657", "90685", "90687", "G0008","90630", "90656", "90664", "90666", "90667", "90668", "90674", "90658", "90682", "90756")
        THEN 1 ELSE 0 END AS flu_shot
    , CASE WHEN TRIM(servcode) in ("G0442","G0443")
        THEN 1 ELSE 0 END AS CMS_Alcohol_Misuse_Screening_Counseling
    , CASE WHEN TRIM(servcode) in ("G0438","G0439","G0468","99497","99498")
        THEN 1 ELSE 0 END AS CMS_Annual_Wellness_Visit
    , CASE WHEN TRIM(servcode) in ("0554T","0555T","0556T","0557T","0558T","76977","77078","77080","77081","77085","G0130")
            AND TRIM(prindiag) in ("E21.0","E21.3","E23.0","E34.2","E89.40","E89.41","M80.08xA","M80.88xA","M84.58xA",
                "M84.68xA","N95.8","N95.9","Q78.0","S34.3xxA","Z78.0","Z79.3","Z79.51","Z79.52","Z79.811","Z79.818","Z79.83",
                "Z87.310","E24","E28.3","M48","M81","M85.8","Q96","S12","S14","S22","S24","S32.0","S32.1","S32.2","S34.1")
        THEN 1 ELSE 0 END AS CMS_Bone_Mass_Measurements
     ,CASE WHEN TRIM(servcode) in ("82465","83718","84478")
            AND TRIM(prindiag) = "Z13.6"
        THEN 1 ELSE 0 END AS CMS_Cardiovascular_Disease_Screening
    , CASE WHEN TRIM(servcode) in ("81528","82270","G0104","G0105","G0106","G0120","G0121","G0328")
            AND TRIM(prindiag) IN ("Z86.004","Z12.11","Z12.12")
        THEN 1 ELSE 0 END AS CMS_Colorectal_Cancer_Screening
    , CASE WHEN TRIM(servcode) in ("99406","99407")
            AND TRIM(prindiag) in ("F17.210","F17.211","F17.213","F17.218","F17.219","F17.220","F17.221",
                "F17.223","F17.228","F17.229","F17.290","F17.291","F17.293","F17.298","F17.299","T65.211A",
                "T65.212A","T65.213A","T65.214A","T65.221A","T65.222A","T65.223A","T65.224A","T65.291A",
                "T65.292A","T65.293A","T65.294A","Z87.891")
        THEN 1 ELSE 0 END AS CMS_tobacco_use_counseling
    , CASE WHEN TRIM(servcode) in ("G0444")
        THEN 1 ELSE 0 END AS CMS_depression_screening
    , CASE WHEN TRIM(servcode) in ("82947","82950","82951")
            AND TRIM(prindiag) ="Z13.1"
        THEN 1 ELSE 0 END AS CMS_Diabetes_Screening
    , CASE WHEN TRIM(servcode) in ("G0499")
        THEN 1 ELSE 0 END AS CMS_hep_b_virus_Screening
    , CASE WHEN TRIM(servcode) in ("90739","90740","90743","90744","90746","90747","G0010")
            AND TRIM(prindiag) ="Z23"
        THEN 1 ELSE 0 END AS CMS_Hep_B_Virus_Vax
    , CASE WHEN TRIM(servcode) in ("G0472")
            AND TRIM(prindiag) IN ("Z72.89","F19.20")
        THEN 1 ELSE 0 END AS CMS_hep_c_Screening
    , CASE WHEN TRIM(servcode) in ("80081","G0432","G0433","G0435")
            AND TRIM(prindiag) in ("Z11.4","Z11.4","Z72.51","Z72.52","Z72.53","Z72.89")
        THEN 1 ELSE 0 END AS CMS_hiv_Screening
    , CASE WHEN TRIM(servcode) in ("G0446")
        THEN 1 ELSE 0 END AS CMS_IBT_for_CVD
    , CASE WHEN TRIM(servcode) in ("G0447","G0473")
            AND TRIM(prindiag) in ("Z68.30","Z68.31","Z68.32","Z68.33","Z68.34","Z68.35","Z68.36","Z68.37","Z68.38","Z68.39","Z68.41","Z68.42","Z68.43","Z68.44","Z68.45")
        THEN 1 ELSE 0 END AS CMS_IBT_for_obesity
    , CASE WHEN TRIM(servcode) in ("90630","90653","90654","90655","90656","90657","90658","90660","90662","90672","90673","90674","90682","90685","90686","90687","90688","90689","90694","90756","Q2034","Q2035","Q2036","Q2037","Q2038","Q2039","G0008")
            AND TRIM(prindiag) = "Z23"
        THEN 1 ELSE 0 END AS CMS_Influenza_Virus_Vaccine
    , CASE WHEN TRIM(servcode) in ("G0402","G0468")
        THEN 1 ELSE 0 END AS CMS_welcome_to_medicare_Examination
    , CASE WHEN TRIM(servcode) in ("G0296","G0297")
            AND TRIM(prindiag) in ("F17.210","F17.211","F17.213","F17.218","F17.219","Z87.891")
        THEN 1 ELSE 0 END AS CMS_lung_Cancer_Screening
    , CASE WHEN TRIM(servcode) in ("97802","97803","97804","G0270","G0271")
        THEN 1 ELSE 0 END AS CMS_nutrition_therapy
    , CASE WHEN TRIM(servcode) IN ("G9873","G9874","G9875","G9876","G9877","G9878","G9879","G9880","G9881","G9882","G9883","G9884","G9885","G9890","G9891")
        THEN 1 ELSE 0 END AS CMS_Medicare_Diabetes_Prevention_Program
    , CASE WHEN TRIM(servcode) in ("90670","90732","G0009")
            AND TRIM(prindiag) = "Z23"
        THEN 1 ELSE 0 END AS CMS_Pneumococcal_Vaccine
    , CASE WHEN TRIM(servcode) in ("G0513","G0514")
        THEN 1 ELSE 0 END AS CMS_Prolonged_Preventive_Services
    , CASE WHEN TRIM(servcode) in ("G0476")
            AND TRIM(prindiag) in ("Z11.51","Z01.411","Z01.419")
        THEN 1 ELSE 0 END AS CMS_Cervical_cancer_hpv
    , CASE WHEN TRIM(servcode) in ("86631","86632","87110","87270","87320","87490","87491","87810","87800","87590","87591","87850","87800","86592","86593","86780","87340","87341","G0445")
            AND TRIM(prindiag) in ("Z11.3","Z11.59","Z34.00","Z34.01","Z34.02","Z34.03","Z34.80","Z34.81","Z34.82","Z34.83","Z34.90","Z34.91","Z34.92","Z34.93","Z72.51","Z72.52","Z72.53","Z72.89","O09.90","O09.91","O09.92","O09.93")
        THEN 1 ELSE 0 END AS CMS_sti_screening
    , CASE WHEN TRIM(servcode) in ("77063","77067")
            AND TRIM(prindiag) in ("N63.15","N63.25","Z12.31")
        THEN 1 ELSE 0 END AS CMS_screening_Mammography
    , CASE WHEN TRIM(servcode) in ("G0123","G0124","G0141","G0143","G0144","G0145","G0147","G0148","P3000","P3001")
            AND TRIM(prindiag) in ("Z72.51","Z72.52","Z72.53","Z77.29","Z77.9","Z91.89","Z92.89","Z01.411","Z01.419",
                "Z12.4","Z12.72","Z12.79","Z12.89")
        THEN 1 ELSE 0 END AS CMS_screening_pap
    , CASE WHEN TRIM(servcode) = "G0101"
            AND TRIM(prindiag) in ("Z72.51","Z72.52","Z72.53","Z77.29","Z77.9","Z91.89","Z92.89","Z77.22","Z77.29",
                "Z72.89","Z92.89","Z01.411","Z01.419","Z12.4","Z12.72","Z12.79","Z12.89")
        THEN 1 ELSE 0 END AS CMS_Screening_Pelvic_Exams
    , CASE WHEN TRIM(servcode) in ("76706","G0389")
        THEN 1 ELSE 0 END AS CMS_Abdominal_Aortic_Aneurysm_screening
    , CASE WHEN TRIM(servcode) in ("G0108","G0109")
        THEN 1 ELSE 0 END AS CMS_Diabetes_Self_Management_Training
    , CASE WHEN TRIM(servcode) in ("G0117","G0118")
            AND TRIM(prindiag) = "Z13.5"
        THEN 1 ELSE 0 END AS CMS_Glaucoma_Screening
    , CASE WHEN TRIM(servcode) in ("G0403","G0404","G0405")
        THEN 1 ELSE 0 END AS CMS_welcome_to_medicare_eeg
    , CASE WHEN TRIM(servcode) = "G0102"
            AND TRIM(prindiag) = "Z12.5"
        THEN 1 ELSE 0 END AS CMS_prostate_cancer_rectal_examination
    , CASE WHEN TRIM(prindiag) in ("585.6","N18.6","I12.0","I13.11","I13.2","N18.5","Z91.15","Z99.2","N19","N18.4","N18.5")
        THEN 1 ELSE 0 END AS ESRD
    , CASE WHEN TRIM(prindiag) in ("792.5","996.56","996.68","V56.2","V56.32","V56.8")
            OR asdb_coe_id IN (25200, 67300)
        THEN 1 ELSE 0 END AS dialysis_flag
    , CASE WHEN TRIM(substr(prindiag,0,1)) = "J"
        THEN 1 ELSE 0 END AS J_code_injectables
    , CASE WHEN TRIM(servcode) in ("80061","83715","83716","83721","83718","83700","83701","83704","3048F","3049F","3050F")
        THEN paid_amt ELSE null END AS cholest_lab_clm
    , CASE WHEN TRIM(servcode) in ("83036","83037","3044F","3045F","3046F")
        THEN paid_amt ELSE null END AS hba1c_lab_clm
FROM `anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_med_claims_flag_yr1`
'



bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_preventative_summary`'

bq query \
--use_legacy_sql=false \
'
CREATE OR REPLACE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_preventative_summary`
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
SELECT
    asdb_member_key
    , asdb_plan_key
    , index_dt
    , MIN(asdb_incurred_dt) AS first_prv_dt
    , MAX(asdb_incurred_dt) AS last_prv_dt
    , SUM(pcp_op_visit) AS sum_pcp
    , SUM(spec_op_visit) AS sum_spec
    , SUM(obgyn_mw_op_visit) AS sum_ob
    , SUM(dme_significant) AS sum_dme
    , SUM(cholest_screen_claim) AS sum_chol_lab
    , SUM(hba1c_test_claim) AS sum_a1c_lab
    , SUM(chemo_clm_flg) AS sum_chemo
    , SUM(cms_alcohol_misuse_screening_counseling) AS cms_alc_scrn
    , SUM(cms_bone_mass_measurements) AS cms_bone_scrn
    , SUM(cms_cardiovascular_disease_screening) AS cms_cvd_scrn
    , SUM(cms_colorectal_cancer_screening) AS cms_col_scrn
    , SUM(cms_tobacco_use_counseling) AS cms_tobacco
    , SUM(cms_depression_screening) AS cms_dep_scrn
    , SUM(cms_diabetes_screening) AS cms_t2d_scrn
    , SUM(cms_hep_b_virus_screening) AS cms_hepb_scrn
    , SUM(cms_hep_b_virus_vax) AS cms_hepb_vax
    , SUM(cms_ibt_for_cvd) AS cms_ibt_cvd
    , SUM(cms_ibt_for_obesity) AS cms_ibt_obese
    , SUM(cms_influenza_virus_vaccine) AS cms_flu_vax
    , SUM(cms_lung_cancer_screening) AS cms_lung_cancer_scrn
    , SUM(cms_nutrition_therapy) AS cms_nutrition
    , SUM(cms_pneumococcal_vaccine) AS cms_pneum_vax
    , SUM(cms_cervical_cancer_hpv) AS cms_hpv_scrn
    , SUM(cms_sti_screening) AS cms_sti_scrn
    , SUM(cms_screening_mammography) AS cms_mam_scrn
    , SUM(cms_screening_pap) AS cms_pap
    , SUM(cms_screening_pelvic_exams) AS cms_pelvic
    , SUM(cms_diabetes_self_management_training) AS cms_t2d_train
    , SUM(cms_prostate_cancer_rectal_examination) AS cms_prost_cancer_scrn
FROM
    `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_preventative`
GROUP BY
    asdb_member_key
    , asdb_plan_key
    , index_dt
'


bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_preventative`'
