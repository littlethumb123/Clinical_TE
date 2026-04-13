/*==============================================================================
  COMMERCIAL CLINICAL TRANSFORMER - IP OUTCOME CREATION
  
  Purpose: Create acute inpatient admission outcome labels for downstream
           classification evaluation of transformer embeddings (linear probe)
  
    Timeline Alignment with Formal Evaluation Features:
    - Uses SAME membership base: a964286_commercial_embedding_raw_features_20241120_20250930
  - Uses SAME index_dt per member (ONE random eligibility month)
  - Features lookback: 36 months before index_dt (ends 90 days before index_dt)
  - Outcome windows: 90, 180, or 365 days AFTER index_dt
  
  Methodology from Commercial IP Model (cp_ip_yc_finetune/bq/070_ip_post.bq):
  - Acute IP: med_cs_ps_ctg_cd = 'I'
  - Event window: index_dt + 1 day to index_dt + X days (90/180/365)
  - Binary flag: 1 if ANY acute IP admission in window, 0 otherwise
  - Exclusions: Maternity, trauma, transplant, non-impactible conditions
  
  Table Retention Policy:
  - Intermediate tables (51): Expire after 1 DAY (auto-cleanup)
  - Final dataset (1): Expires after 180 DAYS (for analysis)
  - Final table: a834793_Commercial_final_dataset_4_te_formal_evaluation_20241120_20250930
  
  Team: Clinical & Social Determinants Intelligence (CSDI)
  Owner: Pritha Ghosh (pritha.ghosh@cvshealth.com)
  Cost Center: 13070
  
  ⚠️ REGENERATION: When rebuilding this formal evaluation dataset, regenerate it AFTER
      regenerating a964286_commercial_embedding_raw_features_20241120_20250930
      to ensure index_dt alignment.
  
==============================================================================*/

/*==============================================================================
  CONFIGURATION: PREDICTION HORIZONS
  
  This script focuses exclusively on 6-month prediction (180 days):
  - ip6: 180 days (6 months) - ONLY OUTCOME
  
  All feature tables expire after 1 day.
  Only the final merged dataset (with ip6 outcome) is retained for 180 days.
  
==============================================================================*/

-- Declare prediction horizon (days from index_dt)
DECLARE predict_6mo_days INT64 DEFAULT 180;


/*==============================================================================
  STEP 0: CREATE BASE MEMBERSHIP COHORT WITH DEMOGRAPHICS
  
    Purpose: Extract the membership base from the formal evaluation raw feature table and
           enrich with demographics and membership attributes
  
    Source: a964286_commercial_embedding_raw_features_20241120_20250930 (formal evaluation raw feature cohort)
  
  Additional Data Sources:
  - INDVDL_CUST_DIST: Get member_id for joins
  - MEMBER: Get birth_dt, gender_cd, zip, county
  - EMIS_MEMBERSHIP: Get drug_ind, fund_ctg_cd, product_ln_cd, group_nbr
  - GROUP_CONTROL: Get customer segment info
  
  Output: One row per member with:
  - individual_id, member_id, index_dt (identifiers)
  - gender_cd, birth_dt, age, age_in_months (demographics)
  - drug_ind, fund_ctg_cd, product_ln_cd, group_nbr (membership)
  - cust_subseg_cd (customer info)
  - feature_end_dt (for consistency with original IP model)
  
  Note: This ensures we use the EXACT SAME cohort and index_dt values that
      feed the formal evaluation raw feature table, guaranteeing perfect alignment.
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH base AS (
    -- Get formal evaluation cohort (individual_id, index_dt)
    SELECT DISTINCT 
        individual_id
        , index_dt 
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_commercial_embedding_raw_features_20241120_20250930`
),
base_with_member AS (
    -- Join to get member_id (needed for membership/demographic joins)
    SELECT 
        base.individual_id
        , base.index_dt
        , icd.member_id
        -- Use most recent member_id if individual has multiple
        , ROW_NUMBER() OVER (PARTITION BY base.individual_id, base.index_dt 
                             ORDER BY icd.member_id DESC) AS rn
    FROM base
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` AS icd
        ON base.individual_id = icd.individual_id
),
base_dedup AS (
    -- Keep only one member_id per individual_id (most recent)
    SELECT 
        individual_id
        , index_dt
        , member_id
    FROM base_with_member
    WHERE rn = 1
)
SELECT 
    -- Identifiers
    base.individual_id
    , base.member_id
    , base.index_dt
    , base.index_dt AS feature_end_dt  -- For consistency with original IP model
    , MOD(base.individual_id, 10) AS ind_id_last_digit  -- For train/test splits
    
    -- Demographics
    , mem.gender_cd
    , mem.birth_dt
    , CASE 
        WHEN em.age_nbr IS NULL THEN DATE_DIFF(base.index_dt, CAST(mem.birth_dt AS DATE), YEAR)
        ELSE em.age_nbr 
      END AS age
    , CASE 
        WHEN em.age_nbr IS NULL THEN DATE_DIFF(base.index_dt, CAST(mem.birth_dt AS DATE), YEAR) * 12
        ELSE em.age_nbr * 12 
      END AS age_in_months
    
    -- Membership attributes
    , em.drug_ind
    , em.fund_ctg_cd
    , em.product_ln_cd
    , pl.prod_ctg_cd
    , em.group_nbr
    
    -- Customer segment
    , gc.cust_subseg_cd
    
FROM base_dedup AS base

-- Get demographics
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEMBER` AS mem
    ON base.member_id = mem.member_id

-- Get membership details (at index_dt)
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_MEMBERSHIP` AS em
    ON base.member_id = em.member_id
    AND base.index_dt = em.eff_dt
    AND em.business_ln_cd = 'CP'
    AND em.file_id <> 'C2'

-- Get product line details for category filter
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PRODUCT_LINE` AS pl
    ON em.product_ln_cd = pl.product_ln_cd

-- Get customer segment
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GROUP_CONTROL` AS gc
    ON em.group_nbr = gc.group_nbr

WHERE mem.birth_dt IS NOT NULL
  AND mem.gender_cd IS NOT NULL
  AND CASE 
        WHEN em.age_nbr IS NULL THEN DATE_DIFF(base.index_dt, CAST(mem.birth_dt AS DATE), YEAR)
        ELSE em.age_nbr 
      END < 150  -- Exclude unrealistic ages
  AND (pl.prod_ctg_cd IS NULL OR TRIM(pl.prod_ctg_cd) <> '02')  -- Exclude product category 02
;


/*==============================================================================
  STEP 0b: EXTRACT HPD (HEALTH PREDICTIVE DISEASE) CONDITIONS
  
  Purpose: Extract chronic condition flags for full model comparison
  
  Source: INDVDL_MSTR_INTGTN (HPD system)
  
  Why We Need This:
  - Original IP model uses 90+ HPD condition flags as features
  - To compare transformer embeddings vs original model fairly, we need
    to replicate the original model's feature set exactly
  
  Timeline: 
  - Only include conditions identified BEFORE or AT feature_end_dt
  - Ensures no data leakage (conditions must exist before prediction point)
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_hpd_extract_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_hpd_extract_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT DISTINCT 
    st.individual_id
    , st.member_id
    , st.index_dt
    , hpd.disease_cd
    , hpd.dm_eligibility_cd
    , hpd.first_indctn_dt
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_MSTR_INTGTN` AS hpd 
    ON st.member_id = hpd.individual_id
WHERE hpd.disease_cd <> 'SUM'  -- Exclude summary records
  AND hpd.dm_eligibility_cd IN ('Y', 'H')  -- Active or historical eligibility
  AND hpd.first_indctn_dt <= st.feature_end_dt  -- Condition before index date
;


/*==============================================================================
  STEP 0c: CREATE HPD CONDITION FLAGS (90+ BINARY FEATURES)
  
  Purpose: Transform disease codes into binary flags for model features
  
  Output: One row per member with 94 columns:
  - 90+ individual disease flags (e.g., Diabetes_Mellitus, Heart_Failure)
  - 4 aggregated flags (cancer, bh, cerebrovascular_condition, hpd_major_flag)
  
  Usage: Join to embeddings for full model comparison
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_hpd_features_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_hpd_features_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    st.individual_id
    , st.index_dt
    
    -- Individual condition flags (90+ conditions)
    , MAX(CASE WHEN he.disease_cd='ADD' THEN 1 ELSE 0 END) AS Attention_Deficit_Disorders
    , MAX(CASE WHEN he.disease_cd='AFF' THEN 1 ELSE 0 END) AS Atrial_Fibrillation
    , MAX(CASE WHEN he.disease_cd='AID' THEN 1 ELSE 0 END) AS HIV_AIDS
    , MAX(CASE WHEN he.disease_cd='ALC' THEN 1 ELSE 0 END) AS Alcoholism
    , MAX(CASE WHEN he.disease_cd='ALG' THEN 1 ELSE 0 END) AS Allergy
    , MAX(CASE WHEN he.disease_cd='ANX' THEN 1 ELSE 0 END) AS Anxiety
    , MAX(CASE WHEN he.disease_cd='AST' THEN 1 ELSE 0 END) AS Asthma
    , MAX(CASE WHEN he.disease_cd='AUT' THEN 1 ELSE 0 END) AS Autism
    , MAX(CASE WHEN he.disease_cd='BIP' THEN 1 ELSE 0 END) AS Bipolar
    , MAX(CASE WHEN he.disease_cd='BLC' THEN 1 ELSE 0 END) AS Bladder_Cancer
    , MAX(CASE WHEN he.disease_cd='BNC' THEN 1 ELSE 0 END) AS Brain_Cancer
    , MAX(CASE WHEN he.disease_cd='BPH' THEN 1 ELSE 0 END) AS Benign_Prostatic_Hypertrophy
    , MAX(CASE WHEN he.disease_cd='BRC' THEN 1 ELSE 0 END) AS Breast_Cancer
    , MAX(CASE WHEN he.disease_cd='CAN' THEN 1 ELSE 0 END) AS Other_Cancer
    , MAX(CASE WHEN he.disease_cd='CAT' THEN 1 ELSE 0 END) AS Cataract
    , MAX(CASE WHEN he.disease_cd='CBD' THEN 1 ELSE 0 END) AS Cerebrovascular_Disease
    , MAX(CASE WHEN he.disease_cd='CDO' THEN 1 ELSE 0 END) AS Disruptive_Childhood_Disorders
    , MAX(CASE WHEN he.disease_cd='CFS' THEN 1 ELSE 0 END) AS Chronic_Fatigue_Syndrome
    , MAX(CASE WHEN he.disease_cd='CHD' THEN 1 ELSE 0 END) AS Congential_Heart_Disease
    , MAX(CASE WHEN he.disease_cd='CHF' THEN 1 ELSE 0 END) AS Heart_Failure
    , MAX(CASE WHEN he.disease_cd='CHO' THEN 1 ELSE 0 END) AS Cholelithiasis_Cholecystitis
    , MAX(CASE WHEN he.disease_cd='COC' THEN 1 ELSE 0 END) AS Colorectal_Cancer
    , MAX(CASE WHEN he.disease_cd='COP' THEN 1 ELSE 0 END) AS Chronic_Obstructive_Pulmonary_Disease
    , MAX(CASE WHEN he.disease_cd='COV' THEN 1 ELSE 0 END) AS COVID_19
    , MAX(CASE WHEN he.disease_cd='CRF' THEN 1 ELSE 0 END) AS Chronic_Renal_Failure
    , MAX(CASE WHEN he.disease_cd='CRO' THEN 1 ELSE 0 END) AS Inflammatory_Bowel_Disease
    , MAX(CASE WHEN he.disease_cd='CTD' THEN 1 ELSE 0 END) AS Chronic_Thyroid_Disorders
    , MAX(CASE WHEN he.disease_cd='CVC' THEN 1 ELSE 0 END) AS Cervical_Cancer
    , MAX(CASE WHEN he.disease_cd='CYS' THEN 1 ELSE 0 END) AS Cystic_Fibrosis
    , MAX(CASE WHEN he.disease_cd='DEM' THEN 1 ELSE 0 END) AS Dementia
    , MAX(CASE WHEN he.disease_cd='DEP' THEN 1 ELSE 0 END) AS Depression
    , MAX(CASE WHEN he.disease_cd='DIA' THEN 1 ELSE 0 END) AS Diabetes_Mellitus
    , MAX(CASE WHEN he.disease_cd='DNS' THEN 1 ELSE 0 END) AS Down_s_Syndrome
    , MAX(CASE WHEN he.disease_cd='DTD' THEN 1 ELSE 0 END) AS Diverticular_Disease
    , MAX(CASE WHEN he.disease_cd='EDO' THEN 1 ELSE 0 END) AS Eating_Disorders
    , MAX(CASE WHEN he.disease_cd='EDT' THEN 1 ELSE 0 END) AS Endometriosis
    , MAX(CASE WHEN he.disease_cd='ENC' THEN 1 ELSE 0 END) AS Endometrial_Cancer
    , MAX(CASE WHEN he.disease_cd='EPL' THEN 1 ELSE 0 END) AS Epilepsy
    , MAX(CASE WHEN he.disease_cd='ESC' THEN 1 ELSE 0 END) AS Esophageal_Cancer
    , MAX(CASE WHEN he.disease_cd='FIB' THEN 1 ELSE 0 END) AS Fibromyalgia
    , MAX(CASE WHEN he.disease_cd='FIF' THEN 1 ELSE 0 END) AS Female_Infertility
    , MAX(CASE WHEN he.disease_cd='GLC' THEN 1 ELSE 0 END) AS Glaucoma
    , MAX(CASE WHEN he.disease_cd='HAE' THEN 1 ELSE 0 END) AS Hereditary_Angioedema
    , MAX(CASE WHEN he.disease_cd='HCG' THEN 1 ELSE 0 END) AS Hypercoaguable_Syndrome
    , MAX(CASE WHEN he.disease_cd='HDL' THEN 1 ELSE 0 END) AS Hodgkin_s_Disease_Lymphoma
    , MAX(CASE WHEN he.disease_cd='HEM' THEN 1 ELSE 0 END) AS Hemophilia_Congenital_Coagulopathies
    , MAX(CASE WHEN he.disease_cd='HEP' THEN 1 ELSE 0 END) AS Hepatitis
    , MAX(CASE WHEN he.disease_cd='HNC' THEN 1 ELSE 0 END) AS Head_Neck_Cancer
    , MAX(CASE WHEN he.disease_cd='HYC' THEN 1 ELSE 0 END) AS Hyperlipidemia
    , MAX(CASE WHEN he.disease_cd='HYP' THEN 1 ELSE 0 END) AS Hypertension
    , MAX(CASE WHEN he.disease_cd='IDA' THEN 1 ELSE 0 END) AS Iron_Deficiency_Anemia
    , MAX(CASE WHEN he.disease_cd='IHD' THEN 1 ELSE 0 END) AS Ischemic_Heart_Disease
    , MAX(CASE WHEN he.disease_cd='KST' THEN 1 ELSE 0 END) AS Kidney_Stones
    , MAX(CASE WHEN he.disease_cd='LBP' THEN 1 ELSE 0 END) AS Low_Back_Pain
    , MAX(CASE WHEN he.disease_cd='LBW' THEN 1 ELSE 0 END) AS Maternal_Hist_of_LowBirth_Weight_or_Preterm_Birth
    , MAX(CASE WHEN he.disease_cd='LEU' THEN 1 ELSE 0 END) AS Leukemia_Myeloma
    , MAX(CASE WHEN he.disease_cd='LUC' THEN 1 ELSE 0 END) AS Lung_Cancer
    , MAX(CASE WHEN he.disease_cd='LVB' THEN 1 ELSE 0 END) AS Low_Vision_and_Blindness
    , MAX(CASE WHEN he.disease_cd='LYM' THEN 1 ELSE 0 END) AS Lyme_Disease
    , MAX(CASE WHEN he.disease_cd='MLM' THEN 1 ELSE 0 END) AS Malignant_Melanoma
    , MAX(CASE WHEN he.disease_cd='MNP' THEN 1 ELSE 0 END) AS Menopause
    , MAX(CASE WHEN he.disease_cd='MOH' THEN 1 ELSE 0 END) AS Migraine_and_Other_Headaches
    , MAX(CASE WHEN he.disease_cd='MSS' THEN 1 ELSE 0 END) AS Multiple_Sclerosis
    , MAX(CASE WHEN he.disease_cd='MSX' THEN 1 ELSE 0 END) AS Metabolic_Syndrome
    , MAX(CASE WHEN he.disease_cd='NEU' THEN 1 ELSE 0 END) AS Neurosis
    , MAX(CASE WHEN he.disease_cd='NGD' THEN 1 ELSE 0 END) AS Nonspecific_Gastritis_Dyspepsia
    , MAX(CASE WHEN he.disease_cd='OBE' THEN 1 ELSE 0 END) AS Obesity
    , MAX(CASE WHEN he.disease_cd='OMD' THEN 1 ELSE 0 END) AS Otitis_Media
    , MAX(CASE WHEN he.disease_cd='ORC' THEN 1 ELSE 0 END) AS Oral_Cancer
    , MAX(CASE WHEN he.disease_cd='OSP' THEN 1 ELSE 0 END) AS Osteoporosis
    , MAX(CASE WHEN he.disease_cd='OST' THEN 1 ELSE 0 END) AS Osteoarthritis
    , MAX(CASE WHEN he.disease_cd='OVC' THEN 1 ELSE 0 END) AS Ovarian_Cancer
    , MAX(CASE WHEN he.disease_cd='PAN' THEN 1 ELSE 0 END) AS Pancreatitis
    , MAX(CASE WHEN he.disease_cd='PAR' THEN 1 ELSE 0 END) AS Parkinson_s_Disease
    , MAX(CASE WHEN he.disease_cd='PER' THEN 1 ELSE 0 END) AS Periodontal_Disease
    , MAX(CASE WHEN he.disease_cd='PMC' THEN 1 ELSE 0 END) AS Psychiatric_Disorders_related_to_Med_Conditions
    , MAX(CASE WHEN he.disease_cd='PNC' THEN 1 ELSE 0 END) AS Pancreatic_Cancer
    , MAX(CASE WHEN he.disease_cd='PPD' THEN 1 ELSE 0 END) AS Post_Partum_BH_Disorder
    , MAX(CASE WHEN he.disease_cd='PRC' THEN 1 ELSE 0 END) AS Prostate_Cancer
    , MAX(CASE WHEN he.disease_cd='PSY' THEN 1 ELSE 0 END) AS Psychoses
    , MAX(CASE WHEN he.disease_cd='PUD' THEN 1 ELSE 0 END) AS Peptic_Ulcer_Disease
    , MAX(CASE WHEN he.disease_cd='PVD' THEN 1 ELSE 0 END) AS Peripheral_Artery_Disease
    , MAX(CASE WHEN he.disease_cd='RHA' THEN 1 ELSE 0 END) AS Rheumatoid_Arthritis
    , MAX(CASE WHEN he.disease_cd='SCA' THEN 1 ELSE 0 END) AS Sickle_Cell_Anemia
    , MAX(CASE WHEN he.disease_cd='SDO' THEN 1 ELSE 0 END) AS Substances_Related_Disorders
    , MAX(CASE WHEN he.disease_cd='SKC' THEN 1 ELSE 0 END) AS Skin_Cancer
    , MAX(CASE WHEN he.disease_cd='SLE' THEN 1 ELSE 0 END) AS Systemic_Lupus_Erythematosus
    , MAX(CASE WHEN he.disease_cd='STC' THEN 1 ELSE 0 END) AS Stomach_Cancer
    , MAX(CASE WHEN he.disease_cd='VNA' THEN 1 ELSE 0 END) AS Ventricular_Arrhythmia
    
    -- Aggregated condition flags
    , MAX(CASE WHEN he.disease_cd IN ('BLC','BNC','BRC','CAN','COC','CVC','ENC','ESC','HDL','HNC',
                                      'LEU','LUC','MLM','ORC','OVC','PNC','PRC','STC') 
               THEN 1 ELSE 0 END) AS cancer
    , MAX(CASE WHEN LOWER(he.disease_cd) IN ('alc','anx','bip','dep','sdo','psy','neu') 
               THEN 1 ELSE 0 END) AS bh
    , MAX(CASE WHEN LOWER(he.disease_cd) IN ('chf', 'ihd', 'hyp', 'aff', 'pvd', 'vna') 
               THEN 1 ELSE 0 END) AS cerebrovascular_condition
    , MAX(CASE WHEN LOWER(he.disease_cd) IN ('aff','aid','alc','bip','cbd','chf','cop','cro','cys','dem',
                                              'dia','edo','fib','hem','hep','ihd','pan','par','psy','pvd',
                                              'sca','sdo','sle','rha','mss') 
               THEN 1 ELSE 0 END) AS hpd_major_flag
    
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_hpd_extract_4_te_formal_evaluation_20241120_20250930` AS he 
    ON st.member_id = he.member_id 
    AND st.index_dt = he.index_dt
GROUP BY st.individual_id, st.index_dt
;


/*==============================================================================
  STEP 0d: EXTRACT MEDICAL CLAIMS DATA (24-MONTH LOOKBACK)
  
  Purpose: Extract claim-level data for feature engineering
  
  Data Sources: 
  - EMIS_CLAIM_LINE (current)
  - T_EDW_EMIS_CLAIM_LINE_Y2020 (archived 2020 data)
  - INDVDL_CUST_DIST (to link individual_id across member_ids)
  
  Extracts:
  - Diagnosis codes (ICD9) and groups
  - Procedure codes and groups
  - Costs (allowed_amt, paid_amt)
  - Service specialty, place of service
  - Revenue codes
  
  Timeline: 24 months before feature_end_dt
  
  Note: Uses UNION DISTINCT to combine current and archived claims data
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mxclm_extract_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mxclm_extract_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
-- Current claims data
SELECT 
    st.member_id
    , st.individual_id
    , st.index_dt
    , st.feature_end_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 24 MONTH) AS feature_end_dt_2yr_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 12 MONTH) AS feature_end_dt_1yr_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 6 MONTH) AS feature_end_dt_6mo_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 3 MONTH) AS feature_end_dt_3mo_dt
    , eecl.claim_line_id
    , eecl.srv_start_dt
    , eecl.adjn_dt
    , eecl.allowed_amt
    , eecl.paid_amt
    , eecl.pri_icd9_dx_cd
    , eid.icd9_dx_group_nbr
    , eidg.icd9_dx_ctg_cd
    , eecl.prcdr_cd
    , ep.prcdr_group_nbr
    , epg.prcdr_ctg_cd
    , eecl.plc_srv_ctg_cd
    , eecl.med_cost_subctg_cd
    , emcc.med_cost_ctg_cd
    , eecl.revenue_cd
    , eecl.srv_spclty_ctg_cd
    , eecl.hcfa_plc_srv_cd
    , eecl.srv_prvdr_id
    , eecl.clm_ln_status_cd
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` AS eicd 
    ON st.member_id = eicd.member_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` AS eicd2 
    ON eicd.individual_id = eicd2.individual_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_CLAIM_LINE` AS eecl 
    ON eicd2.member_id = eecl.member_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` AS eid 
    ON eecl.pri_icd9_dx_cd = eid.icd9_dx_cd
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DX_GROUP` AS eidg 
    ON eid.icd9_dx_group_nbr = eidg.icd9_dx_group_nbr
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PROCEDURE` AS ep 
    ON eecl.prcdr_cd = ep.prcdr_cd
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PROCEDURE_GROUP` AS epg 
    ON ep.prcdr_group_nbr = epg.prcdr_group_nbr
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MED_COST_CATEGORY` AS emcc 
    ON eecl.med_cost_subctg_cd = emcc.med_cost_subctg_cd
WHERE eecl.summarized_srv_ind = 'Y' 
  AND eecl.duplicate_ind = 'N' 
  AND eecl.srv_start_dt <= st.feature_end_dt 
  AND DATE_SUB(st.feature_end_dt, INTERVAL 24 MONTH) <= eecl.srv_start_dt

UNION DISTINCT

-- Archived 2020 claims data
SELECT 
    st.member_id
    , st.individual_id
    , st.index_dt
    , st.feature_end_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 24 MONTH) AS feature_end_dt_2yr_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 12 MONTH) AS feature_end_dt_1yr_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 6 MONTH) AS feature_end_dt_6mo_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 3 MONTH) AS feature_end_dt_3mo_dt
    , eecl.claim_line_id
    , eecl.srv_start_dt
    , eecl.adjn_dt
    , eecl.allowed_amt
    , eecl.paid_amt
    , eecl.pri_icd9_dx_cd
    , eid.icd9_dx_group_nbr
    , eidg.icd9_dx_ctg_cd
    , eecl.prcdr_cd
    , ep.prcdr_group_nbr
    , epg.prcdr_ctg_cd
    , eecl.plc_srv_ctg_cd
    , eecl.med_cost_subctg_cd
    , emcc.med_cost_ctg_cd
    , eecl.revenue_cd
    , eecl.srv_spclty_ctg_cd
    , eecl.hcfa_plc_srv_cd
    , eecl.srv_prvdr_id
    , eecl.clm_ln_status_cd
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST_Y2020` AS eicd 
    ON st.member_id = eicd.member_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST_Y2020` AS eicd2 
    ON eicd.individual_id = eicd2.individual_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cns.T_EDW_EMIS_CLAIM_LINE_Y2020` AS eecl 
    ON eicd2.member_id = eecl.member_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` AS eid 
    ON eecl.pri_icd9_dx_cd = eid.icd9_dx_cd
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DX_GROUP` AS eidg 
    ON eid.icd9_dx_group_nbr = eidg.icd9_dx_group_nbr
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PROCEDURE` AS ep 
    ON eecl.prcdr_cd = ep.prcdr_cd
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PROCEDURE_GROUP` AS epg 
    ON ep.prcdr_group_nbr = epg.prcdr_group_nbr
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MED_COST_CATEGORY` AS emcc 
    ON eecl.med_cost_subctg_cd = emcc.med_cost_subctg_cd
WHERE eecl.summarized_srv_ind = 'Y' 
  AND eecl.duplicate_ind = 'N' 
  AND eecl.srv_start_dt <= st.feature_end_dt 
  AND DATE_SUB(st.feature_end_dt, INTERVAL 24 MONTH) <= eecl.srv_start_dt
;


/*==============================================================================
  STEP 0e: EXTRACT MEDICAL CASES DATA (24-MONTH LOOKBACK)
  
  Purpose: Extract medical case-level data for IP, Newborn IP, and ER cases
  
  Data Sources:
  - MEDICAL_CASE (current)
  - MEDICAL_CASE_Y2020 (archived 2020 data)
  
  Case Types:
  - 'I' = Inpatient
  - 'N' = Newborn Inpatient
  - 'E' = Emergency Room
  
  Extracts:
  - Case dates (start, stop)
  - Diagnosis, procedure, DRG, MDC
  - Length of stay (los_day_cnt)
  - Admission type, discharge status
  - Total allowed amount
  
  Timeline: Cases overlapping with 24 months before feature_end_dt
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_extract_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_extract_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
-- Current medical cases
SELECT 
    st.member_id
    , st.index_dt
    , st.feature_end_dt
    , st.individual_id
    , DATE_SUB(st.feature_end_dt, INTERVAL 24 MONTH) AS feature_end_dt_2yr_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 12 MONTH) AS feature_end_dt_1yr_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 6 MONTH) AS feature_end_dt_6mo_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 3 MONTH) AS feature_end_dt_3mo_dt
    , emc.medical_case_id
    , emc.med_case_start_dt
    , emc.med_case_stop_dt
    , emc.med_cs_ps_ctg_cd
    , emc.icd9_dx_cd
    , emc.prcdr_cd
    , emc.drg_cd
    , emc.mdc_cd
    , emc.los_day_cnt
    , emc.med_cs_admit_ty_cd
    , emc.dschrg_status_cd
    , emc.mng_spclst_cls_cd
    , emc.mng_spclty_ctg_cd
    , emc.total_allowed_amt
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEDICAL_CASE` AS emc 
    ON st.member_id = emc.member_id
WHERE emc.med_cs_ps_ctg_cd IN ('I', 'N', 'E') 
  AND emc.med_case_start_dt <= st.feature_end_dt 
  AND DATE_SUB(st.feature_end_dt, INTERVAL 24 MONTH) <= emc.med_case_stop_dt

UNION DISTINCT

-- Archived 2020 medical cases
SELECT 
    st.member_id
    , st.index_dt
    , st.feature_end_dt
    , st.individual_id
    , DATE_SUB(st.feature_end_dt, INTERVAL 24 MONTH) AS feature_end_dt_2yr_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 12 MONTH) AS feature_end_dt_1yr_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 6 MONTH) AS feature_end_dt_6mo_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 3 MONTH) AS feature_end_dt_3mo_dt
    , emc.medical_case_id
    , emc.med_case_start_dt
    , emc.med_case_stop_dt
    , emc.med_cs_ps_ctg_cd
    , emc.icd9_dx_cd
    , emc.prcdr_cd
    , emc.drg_cd
    , emc.mdc_cd
    , emc.los_day_cnt
    , emc.med_cs_admit_ty_cd
    , emc.dschrg_status_cd
    , emc.mng_spclst_cls_cd
    , emc.mng_spclty_ctg_cd
    , emc.total_allowed_amt
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEDICAL_CASE_Y2020` AS emc 
    ON st.member_id = emc.member_id
WHERE emc.med_cs_ps_ctg_cd IN ('I', 'N', 'E') 
  AND emc.med_case_start_dt <= st.feature_end_dt 
  AND DATE_SUB(st.feature_end_dt, INTERVAL 24 MONTH) <= emc.med_case_stop_dt
;


/*==============================================================================
  STEP 0f: EXTRACT MEDICAL CASE FACILITY INFORMATION
  
  Purpose: Link medical cases to facility provider specialty
  
  Data Source: MED_CASE_X_PRVDR (medical case provider crosswalk)
  
  Filters:
  - Only Inpatient ('I') and Newborn Inpatient ('N') cases (not ER)
  - med_cs_prvdr_rl_cd = '2' (Facility provider role)
  
  Output: medical_case_id -> facility specialty
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcsfac_extract_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcsfac_extract_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
-- Current facility data
SELECT 
    st.member_id
    , st.index_dt
    , st.feature_end_dt
    , st.individual_id
    , DATE_SUB(st.feature_end_dt, INTERVAL 24 MONTH) AS feature_end_dt_2yr_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 12 MONTH) AS feature_end_dt_1yr_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 6 MONTH) AS feature_end_dt_6mo_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 3 MONTH) AS feature_end_dt_3mo_dt
    , emc.medical_case_id
    , emc.med_case_start_dt
    , emc.med_case_stop_dt
    , emc.med_cs_ps_ctg_cd
    , emcp.provider_id AS fac_provider_id
    , emcp.specialty_ctg_cd AS fac_spec_ctg_cd
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` AS eicd 
    ON st.member_id = eicd.member_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` AS eicd2 
    ON eicd.individual_id = eicd2.individual_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEDICAL_CASE` AS emc 
    ON eicd2.member_id = emc.member_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MED_CASE_X_PRVDR` AS emcp 
    ON emc.member_id = emcp.member_id 
    AND emc.medical_case_id = emcp.medical_case_id
WHERE emc.med_cs_ps_ctg_cd IN ('I', 'N') 
  AND emc.med_case_start_dt <= st.feature_end_dt 
  AND DATE_SUB(st.feature_end_dt, INTERVAL 24 MONTH) <= emc.med_case_stop_dt 
  AND emcp.med_cs_prvdr_rl_cd = '2'  -- Facility provider

UNION DISTINCT

-- Archived 2020 facility data
SELECT 
    st.member_id
    , st.index_dt
    , st.feature_end_dt
    , st.individual_id
    , DATE_SUB(st.feature_end_dt, INTERVAL 24 MONTH) AS feature_end_dt_2yr_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 12 MONTH) AS feature_end_dt_1yr_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 6 MONTH) AS feature_end_dt_6mo_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 3 MONTH) AS feature_end_dt_3mo_dt
    , emc.medical_case_id
    , emc.med_case_start_dt
    , emc.med_case_stop_dt
    , emc.med_cs_ps_ctg_cd
    , emcp.provider_id AS fac_provider_id
    , emcp.specialty_ctg_cd AS fac_spec_ctg_cd
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST_Y2020` AS eicd 
    ON st.member_id = eicd.member_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST_Y2020` AS eicd2 
    ON eicd.individual_id = eicd2.individual_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEDICAL_CASE_Y2020` AS emc 
    ON eicd2.member_id = emc.member_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MED_CASE_X_PRVDR` AS emcp 
    ON emc.member_id = emcp.member_id 
    AND emc.medical_case_id = emcp.medical_case_id
WHERE emc.med_cs_ps_ctg_cd IN ('I', 'N') 
  AND emc.med_case_start_dt <= st.feature_end_dt 
  AND DATE_SUB(st.feature_end_dt, INTERVAL 24 MONTH) <= emc.med_case_stop_dt 
  AND emcp.med_cs_prvdr_rl_cd = '2'  -- Facility provider
;


/*==============================================================================
  STEP 0g: LINK MEDICAL CASES TO CLAIM LINES
  
  Purpose: Create medical_case_id -> claim_line_id mapping for adjudication dates
  
  Data Source: MED_CASE_X_CLM_LN (medical case to claim line crosswalk)
  
  Note: 030_medical_case.bq references this table but doesn't create it.
        In 070_ip_post.bq, they create it using MED_CASE_X_CLM_LN as source.
        Using the same table name to match original implementation.
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcsclm_extract_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcsclm_extract_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT DISTINCT
    emc.member_id
    , emc.medical_case_id
    , emccl.claim_line_id
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_extract_4_te_formal_evaluation_20241120_20250930` AS emc
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MED_CASE_X_CLM_LN` AS emccl
    ON emc.member_id = emccl.member_id
    AND emc.medical_case_id = emccl.medical_case_id
;


/*==============================================================================
  STEP 0h: JOIN MEDICAL CASES TO CLAIMS (GET BASIC CASE INFO)
  
  Purpose: Link medical cases to claim lines for subsequent processing
  
  Output: medical_case_id, claim_line_id, DRG, MDC, length of stay
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs1_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs1_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    me1.member_id
    , me1.index_dt
    , me1.individual_id
    , me1.feature_end_dt
    , me1.feature_end_dt_2yr_dt
    , me1.feature_end_dt_1yr_dt
    , me1.feature_end_dt_6mo_dt
    , me1.feature_end_dt_3mo_dt
    , me1.medical_case_id
    , me1.med_cs_ps_ctg_cd
    , me1.med_case_start_dt
    , me1.med_case_stop_dt
    , me2.claim_line_id
    , me1.drg_cd
    , me1.mdc_cd
    , me1.los_day_cnt
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_extract_4_te_formal_evaluation_20241120_20250930` AS me1
INNER JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcsclm_extract_4_te_formal_evaluation_20241120_20250930` AS me2 
    ON me1.member_id = me2.member_id 
    AND me1.medical_case_id = me2.medical_case_id
;


/*==============================================================================
  STEP 0i: GET MAX ADJUDICATION DATE PER MEDICAL CASE
  
  Purpose: Determine the latest claim adjudication date for each medical case
  
  Logic: Group by medical_case_id and take MAX(adjn_dt) from all claim lines
  
  Note: Used for temporal filtering to ensure claims are fully adjudicated
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs2_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs2_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    m.member_id
    , m.index_dt
    , m.individual_id
    , m.feature_end_dt
    , m.feature_end_dt_2yr_dt
    , m.feature_end_dt_1yr_dt
    , m.feature_end_dt_6mo_dt
    , m.feature_end_dt_3mo_dt
    , m.medical_case_id
    , m.med_cs_ps_ctg_cd
    , m.med_case_start_dt
    , m.med_case_stop_dt
    , MAX(me.adjn_dt) AS max_adjn
    , m.drg_cd
    , m.mdc_cd
    , m.los_day_cnt
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs1_4_te_formal_evaluation_20241120_20250930` AS m
INNER JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mxclm_extract_4_te_formal_evaluation_20241120_20250930` AS me 
    ON m.member_id = me.member_id 
    AND m.claim_line_id = me.claim_line_id
GROUP BY 
    m.member_id, m.index_dt, m.individual_id, m.feature_end_dt
    , m.feature_end_dt_2yr_dt, m.feature_end_dt_1yr_dt, m.feature_end_dt_6mo_dt, m.feature_end_dt_3mo_dt
    , m.medical_case_id, m.med_cs_ps_ctg_cd, m.med_case_start_dt, m.med_case_stop_dt
    , m.drg_cd, m.mdc_cd, m.los_day_cnt
;


/*==============================================================================
  STEP 0j: ADD FACILITY SPECIALTY TO MEDICAL CASES
  
  Purpose: Enrich medical cases with facility specialty information
  
  Logic: LEFT JOIN to facility extract to preserve all medical cases
  
  Output: Complete medical case record with facility specialty
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs3_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs3_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    m.member_id
    , m.index_dt
    , m.individual_id
    , m.feature_end_dt
    , m.feature_end_dt_2yr_dt
    , m.feature_end_dt_1yr_dt
    , m.feature_end_dt_6mo_dt
    , m.feature_end_dt_3mo_dt
    , m.med_cs_ps_ctg_cd
    , m.med_case_start_dt
    , m.med_case_stop_dt
    , m.max_adjn
    , m.drg_cd
    , m.mdc_cd
    , m.los_day_cnt
    , me.fac_spec_ctg_cd
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs2_4_te_formal_evaluation_20241120_20250930` AS m
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcsfac_extract_4_te_formal_evaluation_20241120_20250930` AS me 
    ON m.member_id = me.member_id 
    AND m.medical_case_id = me.medical_case_id
;


/*==============================================================================
  STEP 0k: AGGREGATE MEDICAL CASE COUNTS - 2 YEAR LOOKBACK
  
  Purpose: Create utilization features for 2-year historical window
  
  Features Created (per member):
  - ercs_2yr_cnt: Count of ER cases in last 2 years
  - ercs_2yr: Binary flag for any ER case
  - ip_2yr_cnt: Count of IP admissions
  - ip_2yr: Binary flag for any IP admission
  - ip_2yr_days: Total IP days
  - nip_2yr_cnt: Count of newborn IP cases
  - nip_2yr: Binary flag for any newborn IP
  - nip_2yr_days: Total newborn IP days
  
  Temporal Logic:
  - Case start date: within [feature_end_dt - 2yr, feature_end_dt]
  - Adjudication date: within [feature_end_dt - 2yr, feature_end_dt]
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_2yr_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_2yr_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    st.individual_id
    , st.index_dt
    , CAST(SUM(CASE WHEN m.med_cs_ps_ctg_cd='E' THEN 1 ELSE 0 END) AS INT64) AS ercs_2yr_cnt
    , CAST(MAX(CASE WHEN m.med_cs_ps_ctg_cd='E' THEN 1 ELSE 0 END) AS INT64) AS ercs_2yr
    , CAST(SUM(CASE WHEN m.med_cs_ps_ctg_cd='I' THEN 1 ELSE 0 END) AS INT64) AS ip_2yr_cnt
    , CAST(MAX(CASE WHEN m.med_cs_ps_ctg_cd='I' THEN 1 ELSE 0 END) AS INT64) AS ip_2yr
    , CAST(SUM(CASE WHEN m.med_cs_ps_ctg_cd='I' THEN m.los_day_cnt ELSE 0 END) AS INT64) AS ip_2yr_days
    , CAST(SUM(CASE WHEN m.med_cs_ps_ctg_cd='N' THEN 1 ELSE 0 END) AS INT64) AS nip_2yr_cnt
    , CAST(MAX(CASE WHEN m.med_cs_ps_ctg_cd='N' THEN 1 ELSE 0 END) AS INT64) AS nip_2yr
    , CAST(SUM(CASE WHEN m.med_cs_ps_ctg_cd='N' THEN m.los_day_cnt ELSE 0 END) AS INT64) AS nip_2yr_days
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs3_4_te_formal_evaluation_20241120_20250930` AS m 
    ON st.member_id = m.member_id 
    AND st.index_dt = m.index_dt 
    AND m.feature_end_dt_2yr_dt <= m.med_case_start_dt 
    AND m.med_case_start_dt <= m.feature_end_dt 
    AND m.feature_end_dt_2yr_dt <= m.max_adjn 
    AND m.max_adjn <= m.feature_end_dt
GROUP BY st.individual_id, st.index_dt
;


/*==============================================================================
  STEP 0l: AGGREGATE MEDICAL CASE COUNTS - 1 YEAR LOOKBACK
  
  Purpose: Create utilization features for 1-year historical window
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_1yr_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_1yr_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    st.individual_id
    , st.index_dt
    , CAST(SUM(CASE WHEN m.med_cs_ps_ctg_cd='E' THEN 1 ELSE 0 END) AS INT64) AS ercs_1yr_cnt
    , CAST(MAX(CASE WHEN m.med_cs_ps_ctg_cd='E' THEN 1 ELSE 0 END) AS INT64) AS ercs_1yr
    , CAST(SUM(CASE WHEN m.med_cs_ps_ctg_cd='I' THEN 1 ELSE 0 END) AS INT64) AS ip_1yr_cnt
    , CAST(MAX(CASE WHEN m.med_cs_ps_ctg_cd='I' THEN 1 ELSE 0 END) AS INT64) AS ip_1yr
    , CAST(SUM(CASE WHEN m.med_cs_ps_ctg_cd='I' THEN m.los_day_cnt ELSE 0 END) AS INT64) AS ip_1yr_days
    , CAST(SUM(CASE WHEN m.med_cs_ps_ctg_cd='N' THEN 1 ELSE 0 END) AS INT64) AS nip_1yr_cnt
    , CAST(MAX(CASE WHEN m.med_cs_ps_ctg_cd='N' THEN 1 ELSE 0 END) AS INT64) AS nip_1yr
    , CAST(SUM(CASE WHEN m.med_cs_ps_ctg_cd='N' THEN m.los_day_cnt ELSE 0 END) AS INT64) AS nip_1yr_days
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs3_4_te_formal_evaluation_20241120_20250930` AS m 
    ON st.member_id = m.member_id 
    AND st.index_dt = m.index_dt 
    AND m.feature_end_dt_1yr_dt <= m.med_case_start_dt 
    AND m.med_case_start_dt <= m.feature_end_dt 
    AND m.feature_end_dt_1yr_dt <= m.max_adjn 
    AND m.max_adjn <= m.feature_end_dt
GROUP BY st.individual_id, st.index_dt
;


/*==============================================================================
  STEP 0m: AGGREGATE MEDICAL CASE COUNTS - 6 MONTH LOOKBACK
  
  Purpose: Create utilization features for 6-month historical window
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_6mo_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_6mo_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    st.individual_id
    , st.index_dt
    , CAST(SUM(CASE WHEN m.med_cs_ps_ctg_cd='E' THEN 1 ELSE 0 END) AS INT64) AS ercs_6mo_cnt
    , CAST(MAX(CASE WHEN m.med_cs_ps_ctg_cd='E' THEN 1 ELSE 0 END) AS INT64) AS ercs_6mo
    , CAST(SUM(CASE WHEN m.med_cs_ps_ctg_cd='I' THEN 1 ELSE 0 END) AS INT64) AS ip_6mo_cnt
    , CAST(MAX(CASE WHEN m.med_cs_ps_ctg_cd='I' THEN 1 ELSE 0 END) AS INT64) AS ip_6mo
    , CAST(SUM(CASE WHEN m.med_cs_ps_ctg_cd='I' THEN m.los_day_cnt ELSE 0 END) AS INT64) AS ip_6mo_days
    , CAST(SUM(CASE WHEN m.med_cs_ps_ctg_cd='N' THEN 1 ELSE 0 END) AS INT64) AS nip_6mo_cnt
    , CAST(MAX(CASE WHEN m.med_cs_ps_ctg_cd='N' THEN 1 ELSE 0 END) AS INT64) AS nip_6mo
    , CAST(SUM(CASE WHEN m.med_cs_ps_ctg_cd='N' THEN m.los_day_cnt ELSE 0 END) AS INT64) AS nip_6mo_days
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs3_4_te_formal_evaluation_20241120_20250930` AS m 
    ON st.member_id = m.member_id 
    AND st.index_dt = m.index_dt 
    AND m.feature_end_dt_6mo_dt <= m.med_case_start_dt 
    AND m.med_case_start_dt <= m.feature_end_dt 
    AND m.feature_end_dt_6mo_dt <= m.max_adjn 
    AND m.max_adjn <= m.feature_end_dt
GROUP BY st.individual_id, st.index_dt
;


/*==============================================================================
  STEP 0n: AGGREGATE MEDICAL CASE COUNTS - 3 MONTH LOOKBACK
  
  Purpose: Create utilization features for 3-month historical window
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_3mo_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_3mo_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    st.individual_id
    , st.index_dt
    , CAST(SUM(CASE WHEN m.med_cs_ps_ctg_cd='E' THEN 1 ELSE 0 END) AS INT64) AS ercs_3mo_cnt
    , CAST(MAX(CASE WHEN m.med_cs_ps_ctg_cd='E' THEN 1 ELSE 0 END) AS INT64) AS ercs_3mo
    , CAST(SUM(CASE WHEN m.med_cs_ps_ctg_cd='I' THEN 1 ELSE 0 END) AS INT64) AS ip_3mo_cnt
    , CAST(MAX(CASE WHEN m.med_cs_ps_ctg_cd='I' THEN 1 ELSE 0 END) AS INT64) AS ip_3mo
    , CAST(SUM(CASE WHEN m.med_cs_ps_ctg_cd='I' THEN m.los_day_cnt ELSE 0 END) AS INT64) AS ip_3mo_days
    , CAST(SUM(CASE WHEN m.med_cs_ps_ctg_cd='N' THEN 1 ELSE 0 END) AS INT64) AS nip_3mo_cnt
    , CAST(MAX(CASE WHEN m.med_cs_ps_ctg_cd='N' THEN 1 ELSE 0 END) AS INT64) AS nip_3mo
    , CAST(SUM(CASE WHEN m.med_cs_ps_ctg_cd='N' THEN m.los_day_cnt ELSE 0 END) AS INT64) AS nip_3mo_days
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs3_4_te_formal_evaluation_20241120_20250930` AS m 
    ON st.member_id = m.member_id 
    AND st.index_dt = m.index_dt 
    AND m.feature_end_dt_3mo_dt <= m.med_case_start_dt 
    AND m.med_case_start_dt <= m.feature_end_dt 
    AND m.feature_end_dt_3mo_dt <= m.max_adjn 
    AND m.max_adjn <= m.feature_end_dt
GROUP BY st.individual_id, st.index_dt
;


/*==============================================================================
  STEP 0o: COMBINE ALL TIME-PERIOD MEDICAL CASE FEATURES
  
  Purpose: Join all temporal aggregations into single feature table
  
  Output: 32 medical case utilization features (8 per time period × 4 periods)
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_features_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_features_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    m1.*
    , m2.ercs_1yr_cnt, m2.ercs_1yr, m2.ip_1yr_cnt, m2.ip_1yr, m2.ip_1yr_days
    , m2.nip_1yr_cnt, m2.nip_1yr, m2.nip_1yr_days
    , m3.ercs_6mo_cnt, m3.ercs_6mo, m3.ip_6mo_cnt, m3.ip_6mo, m3.ip_6mo_days
    , m3.nip_6mo_cnt, m3.nip_6mo, m3.nip_6mo_days
    , m4.ercs_3mo_cnt, m4.ercs_3mo, m4.ip_3mo_cnt, m4.ip_3mo, m4.ip_3mo_days
    , m4.nip_3mo_cnt, m4.nip_3mo, m4.nip_3mo_days
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_2yr_4_te_formal_evaluation_20241120_20250930` AS m1
INNER JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_1yr_4_te_formal_evaluation_20241120_20250930` AS m2 
    ON m1.individual_id = m2.individual_id 
    AND m1.index_dt = m2.index_dt
INNER JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_6mo_4_te_formal_evaluation_20241120_20250930` AS m3 
    ON m1.individual_id = m3.individual_id 
    AND m1.index_dt = m3.index_dt
INNER JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_3mo_4_te_formal_evaluation_20241120_20250930` AS m4 
    ON m1.individual_id = m4.individual_id 
    AND m1.index_dt = m4.index_dt
;


/*==============================================================================
  STEP 0p: MDC-SPECIFIC COUNTS BY TIME PERIOD (GROUPED BY MDC_CD)
  
  Purpose: Aggregate IP cases by Major Diagnostic Category (MDC) for each time period
  
  MDC Categories: 25 categories (01-25) representing major body systems:
  - 01: Nervous System
  - 04: Respiratory System
  - 05: Circulatory System
  - 08: Musculoskeletal System
  - etc.
  
  Creates 4 intermediate tables (one per time period) with columns:
  - individual_id, index_dt, mdc_cd, cnt, days
  
  Note: Only includes Inpatient ('I') cases, not ER or Newborn
  
==============================================================================*/

-- 2-year MDC counts
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc2yr_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc2yr_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    m.individual_id
    , m.index_dt
    , m.mdc_cd
    , COUNT(*) AS cnt
    , SUM(m.los_day_cnt) AS days
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs3_4_te_formal_evaluation_20241120_20250930` AS m
WHERE m.med_cs_ps_ctg_cd = 'I' 
  AND m.feature_end_dt_2yr_dt <= m.med_case_start_dt 
  AND m.med_case_start_dt <= m.feature_end_dt 
  AND m.feature_end_dt_2yr_dt <= m.max_adjn 
  AND m.max_adjn <= m.feature_end_dt
GROUP BY m.individual_id, m.index_dt, m.mdc_cd
;

-- 1-year MDC counts
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc1yr_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc1yr_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    m.individual_id
    , m.index_dt
    , m.mdc_cd
    , COUNT(*) AS cnt
    , SUM(m.los_day_cnt) AS days
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs3_4_te_formal_evaluation_20241120_20250930` AS m
WHERE m.med_cs_ps_ctg_cd = 'I' 
  AND m.feature_end_dt_1yr_dt <= m.med_case_start_dt 
  AND m.med_case_start_dt <= m.feature_end_dt 
  AND m.feature_end_dt_1yr_dt <= m.max_adjn 
  AND m.max_adjn <= m.feature_end_dt
GROUP BY m.individual_id, m.index_dt, m.mdc_cd
;

-- 6-month MDC counts
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc6mo_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc6mo_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    m.individual_id
    , m.index_dt
    , m.mdc_cd
    , COUNT(*) AS cnt
    , SUM(m.los_day_cnt) AS days
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs3_4_te_formal_evaluation_20241120_20250930` AS m
WHERE m.med_cs_ps_ctg_cd = 'I' 
  AND m.feature_end_dt_6mo_dt <= m.med_case_start_dt 
  AND m.med_case_start_dt <= m.feature_end_dt 
  AND m.feature_end_dt_6mo_dt <= m.max_adjn 
  AND m.max_adjn <= m.feature_end_dt
GROUP BY m.individual_id, m.index_dt, m.mdc_cd
;

-- 3-month MDC counts
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc3mo_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc3mo_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    m.individual_id
    , m.index_dt
    , m.mdc_cd
    , COUNT(*) AS cnt
    , SUM(m.los_day_cnt) AS days
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs3_4_te_formal_evaluation_20241120_20250930` AS m
WHERE m.med_cs_ps_ctg_cd = 'I' 
  AND m.feature_end_dt_3mo_dt <= m.med_case_start_dt 
  AND m.med_case_start_dt <= m.feature_end_dt 
  AND m.feature_end_dt_3mo_dt <= m.max_adjn 
  AND m.max_adjn <= m.feature_end_dt
GROUP BY m.individual_id, m.index_dt, m.mdc_cd
;


/*==============================================================================
  STEP 0q: MDC PIVOT TABLES - CREATE INDIVIDUAL COLUMNS FOR EACH MDC
  
  Purpose: Transform grouped MDC data into individual columns for modeling
  
  Creates 4 tables (one per time period), each with 77 columns:
  - individual_id, index_dt (2 columns)
  - 25 × count columns (ipmdc01_Xyr_cnt ... ipmdc25_Xyr_cnt)
  - 25 × binary flag columns (ipmdc01_Xyr ... ipmdc25_Xyr)
  - 25 × days columns (ipmdc01_Xyr_days ... ipmdc25_Xyr_days)
  
  MDC Categories (01-25):
  - 01: Nervous System
  - 02: Eye
  - 03: Ear, Nose, Mouth, Throat
  - 04: Respiratory System
  - 05: Circulatory System
  - 06: Digestive System
  - 07: Hepatobiliary System & Pancreas
  - 08: Musculoskeletal System & Connective Tissue
  - 09: Skin, Subcutaneous Tissue & Breast
  - 10: Endocrine, Nutritional & Metabolic
  - 11: Kidney & Urinary Tract
  - 12: Male Reproductive System
  - 13: Female Reproductive System
  - 14: Pregnancy, Childbirth & Puerperium
  - 15: Newborns & Other Neonates
  - 16: Blood, Blood Forming Organs, Immunological Disorders
  - 17: Myeloproliferative & Poorly Differentiated Neoplasm
  - 18: Infectious & Parasitic Diseases
  - 19: Mental Diseases & Disorders
  - 20: Alcohol/Drug Use & Induced Organic Mental Disorders
  - 21: Injuries, Poisonings & Toxic Effects of Drugs
  - 22: Burns
  - 23: Factors Influencing Health Status
  - 24: Multiple Significant Trauma
  - 25: Human Immunodeficiency Virus Infections
  
  Note: This is a LOT of columns but necessary for feature parity with original model
  
==============================================================================*/

-- 2-year MDC pivot
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc2yr2_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc2yr2_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH st AS (
    SELECT DISTINCT individual_id, index_dt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930`
)
SELECT 
        st.individual_id
        , st.index_dt
        , CAST(SUM(CASE WHEN md.mdc_cd='01' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc01_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='02' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc02_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='03' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc03_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='04' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc04_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='05' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc05_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='06' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc06_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='07' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc07_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='08' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc08_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='09' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc09_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='10' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc10_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='11' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc11_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='12' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc12_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='13' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc13_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='14' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc14_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='15' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc15_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='16' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc16_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='17' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc17_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='18' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc18_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='19' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc19_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='20' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc20_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='21' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc21_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='22' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc22_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='23' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc23_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='24' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc24_2yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='25' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc25_2yr_cnt
        , CAST(MAX(CASE WHEN md.mdc_cd='01' THEN 1 ELSE 0 END) AS INT64) AS ipmdc01_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='02' THEN 1 ELSE 0 END) AS INT64) AS ipmdc02_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='03' THEN 1 ELSE 0 END) AS INT64) AS ipmdc03_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='04' THEN 1 ELSE 0 END) AS INT64) AS ipmdc04_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='05' THEN 1 ELSE 0 END) AS INT64) AS ipmdc05_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='06' THEN 1 ELSE 0 END) AS INT64) AS ipmdc06_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='07' THEN 1 ELSE 0 END) AS INT64) AS ipmdc07_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='08' THEN 1 ELSE 0 END) AS INT64) AS ipmdc08_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='09' THEN 1 ELSE 0 END) AS INT64) AS ipmdc09_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='10' THEN 1 ELSE 0 END) AS INT64) AS ipmdc10_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='11' THEN 1 ELSE 0 END) AS INT64) AS ipmdc11_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='12' THEN 1 ELSE 0 END) AS INT64) AS ipmdc12_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='13' THEN 1 ELSE 0 END) AS INT64) AS ipmdc13_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='14' THEN 1 ELSE 0 END) AS INT64) AS ipmdc14_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='15' THEN 1 ELSE 0 END) AS INT64) AS ipmdc15_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='16' THEN 1 ELSE 0 END) AS INT64) AS ipmdc16_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='17' THEN 1 ELSE 0 END) AS INT64) AS ipmdc17_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='18' THEN 1 ELSE 0 END) AS INT64) AS ipmdc18_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='19' THEN 1 ELSE 0 END) AS INT64) AS ipmdc19_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='20' THEN 1 ELSE 0 END) AS INT64) AS ipmdc20_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='21' THEN 1 ELSE 0 END) AS INT64) AS ipmdc21_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='22' THEN 1 ELSE 0 END) AS INT64) AS ipmdc22_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='23' THEN 1 ELSE 0 END) AS INT64) AS ipmdc23_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='24' THEN 1 ELSE 0 END) AS INT64) AS ipmdc24_2yr
        , CAST(MAX(CASE WHEN md.mdc_cd='25' THEN 1 ELSE 0 END) AS INT64) AS ipmdc25_2yr
        , CAST(SUM(CASE WHEN md.mdc_cd='01' THEN md.days ELSE 0 END) AS INT64) AS ipmdc01_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='02' THEN md.days ELSE 0 END) AS INT64) AS ipmdc02_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='03' THEN md.days ELSE 0 END) AS INT64) AS ipmdc03_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='04' THEN md.days ELSE 0 END) AS INT64) AS ipmdc04_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='05' THEN md.days ELSE 0 END) AS INT64) AS ipmdc05_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='06' THEN md.days ELSE 0 END) AS INT64) AS ipmdc06_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='07' THEN md.days ELSE 0 END) AS INT64) AS ipmdc07_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='08' THEN md.days ELSE 0 END) AS INT64) AS ipmdc08_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='09' THEN md.days ELSE 0 END) AS INT64) AS ipmdc09_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='10' THEN md.days ELSE 0 END) AS INT64) AS ipmdc10_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='11' THEN md.days ELSE 0 END) AS INT64) AS ipmdc11_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='12' THEN md.days ELSE 0 END) AS INT64) AS ipmdc12_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='13' THEN md.days ELSE 0 END) AS INT64) AS ipmdc13_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='14' THEN md.days ELSE 0 END) AS INT64) AS ipmdc14_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='15' THEN md.days ELSE 0 END) AS INT64) AS ipmdc15_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='16' THEN md.days ELSE 0 END) AS INT64) AS ipmdc16_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='17' THEN md.days ELSE 0 END) AS INT64) AS ipmdc17_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='18' THEN md.days ELSE 0 END) AS INT64) AS ipmdc18_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='19' THEN md.days ELSE 0 END) AS INT64) AS ipmdc19_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='20' THEN md.days ELSE 0 END) AS INT64) AS ipmdc20_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='21' THEN md.days ELSE 0 END) AS INT64) AS ipmdc21_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='22' THEN md.days ELSE 0 END) AS INT64) AS ipmdc22_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='23' THEN md.days ELSE 0 END) AS INT64) AS ipmdc23_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='24' THEN md.days ELSE 0 END) AS INT64) AS ipmdc24_2yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='25' THEN md.days ELSE 0 END) AS INT64) AS ipmdc25_2yr_days
FROM st
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc2yr_4_te_formal_evaluation_20241120_20250930` AS md 
    ON st.individual_id = md.individual_id 
    AND st.index_dt = md.index_dt
GROUP BY st.individual_id, st.index_dt
;

-- 1-year MDC pivot
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc1yr2_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc1yr2_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH st AS (
    SELECT DISTINCT individual_id, index_dt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930`
)
SELECT 
        st.individual_id
        , st.index_dt
        , CAST(SUM(CASE WHEN md.mdc_cd='01' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc01_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='02' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc02_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='03' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc03_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='04' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc04_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='05' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc05_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='06' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc06_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='07' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc07_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='08' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc08_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='09' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc09_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='10' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc10_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='11' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc11_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='12' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc12_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='13' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc13_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='14' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc14_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='15' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc15_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='16' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc16_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='17' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc17_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='18' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc18_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='19' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc19_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='20' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc20_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='21' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc21_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='22' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc22_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='23' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc23_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='24' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc24_1yr_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='25' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc25_1yr_cnt
        , CAST(MAX(CASE WHEN md.mdc_cd='01' THEN 1 ELSE 0 END) AS INT64) AS ipmdc01_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='02' THEN 1 ELSE 0 END) AS INT64) AS ipmdc02_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='03' THEN 1 ELSE 0 END) AS INT64) AS ipmdc03_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='04' THEN 1 ELSE 0 END) AS INT64) AS ipmdc04_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='05' THEN 1 ELSE 0 END) AS INT64) AS ipmdc05_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='06' THEN 1 ELSE 0 END) AS INT64) AS ipmdc06_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='07' THEN 1 ELSE 0 END) AS INT64) AS ipmdc07_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='08' THEN 1 ELSE 0 END) AS INT64) AS ipmdc08_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='09' THEN 1 ELSE 0 END) AS INT64) AS ipmdc09_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='10' THEN 1 ELSE 0 END) AS INT64) AS ipmdc10_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='11' THEN 1 ELSE 0 END) AS INT64) AS ipmdc11_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='12' THEN 1 ELSE 0 END) AS INT64) AS ipmdc12_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='13' THEN 1 ELSE 0 END) AS INT64) AS ipmdc13_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='14' THEN 1 ELSE 0 END) AS INT64) AS ipmdc14_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='15' THEN 1 ELSE 0 END) AS INT64) AS ipmdc15_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='16' THEN 1 ELSE 0 END) AS INT64) AS ipmdc16_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='17' THEN 1 ELSE 0 END) AS INT64) AS ipmdc17_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='18' THEN 1 ELSE 0 END) AS INT64) AS ipmdc18_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='19' THEN 1 ELSE 0 END) AS INT64) AS ipmdc19_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='20' THEN 1 ELSE 0 END) AS INT64) AS ipmdc20_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='21' THEN 1 ELSE 0 END) AS INT64) AS ipmdc21_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='22' THEN 1 ELSE 0 END) AS INT64) AS ipmdc22_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='23' THEN 1 ELSE 0 END) AS INT64) AS ipmdc23_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='24' THEN 1 ELSE 0 END) AS INT64) AS ipmdc24_1yr
        , CAST(MAX(CASE WHEN md.mdc_cd='25' THEN 1 ELSE 0 END) AS INT64) AS ipmdc25_1yr
        , CAST(SUM(CASE WHEN md.mdc_cd='01' THEN md.days ELSE 0 END) AS INT64) AS ipmdc01_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='02' THEN md.days ELSE 0 END) AS INT64) AS ipmdc02_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='03' THEN md.days ELSE 0 END) AS INT64) AS ipmdc03_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='04' THEN md.days ELSE 0 END) AS INT64) AS ipmdc04_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='05' THEN md.days ELSE 0 END) AS INT64) AS ipmdc05_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='06' THEN md.days ELSE 0 END) AS INT64) AS ipmdc06_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='07' THEN md.days ELSE 0 END) AS INT64) AS ipmdc07_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='08' THEN md.days ELSE 0 END) AS INT64) AS ipmdc08_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='09' THEN md.days ELSE 0 END) AS INT64) AS ipmdc09_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='10' THEN md.days ELSE 0 END) AS INT64) AS ipmdc10_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='11' THEN md.days ELSE 0 END) AS INT64) AS ipmdc11_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='12' THEN md.days ELSE 0 END) AS INT64) AS ipmdc12_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='13' THEN md.days ELSE 0 END) AS INT64) AS ipmdc13_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='14' THEN md.days ELSE 0 END) AS INT64) AS ipmdc14_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='15' THEN md.days ELSE 0 END) AS INT64) AS ipmdc15_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='16' THEN md.days ELSE 0 END) AS INT64) AS ipmdc16_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='17' THEN md.days ELSE 0 END) AS INT64) AS ipmdc17_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='18' THEN md.days ELSE 0 END) AS INT64) AS ipmdc18_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='19' THEN md.days ELSE 0 END) AS INT64) AS ipmdc19_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='20' THEN md.days ELSE 0 END) AS INT64) AS ipmdc20_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='21' THEN md.days ELSE 0 END) AS INT64) AS ipmdc21_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='22' THEN md.days ELSE 0 END) AS INT64) AS ipmdc22_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='23' THEN md.days ELSE 0 END) AS INT64) AS ipmdc23_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='24' THEN md.days ELSE 0 END) AS INT64) AS ipmdc24_1yr_days
        , CAST(SUM(CASE WHEN md.mdc_cd='25' THEN md.days ELSE 0 END) AS INT64) AS ipmdc25_1yr_days
FROM st
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc1yr_4_te_formal_evaluation_20241120_20250930` AS md 
    ON st.individual_id = md.individual_id 
    AND st.index_dt = md.index_dt
GROUP BY st.individual_id, st.index_dt
;

-- 6-month MDC pivot
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc6mo2_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc6mo2_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH st AS (
    SELECT DISTINCT individual_id, index_dt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930`
)
SELECT 
        st.individual_id
        , st.index_dt
        , CAST(SUM(CASE WHEN md.mdc_cd='01' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc01_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='02' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc02_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='03' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc03_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='04' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc04_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='05' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc05_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='06' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc06_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='07' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc07_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='08' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc08_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='09' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc09_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='10' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc10_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='11' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc11_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='12' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc12_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='13' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc13_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='14' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc14_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='15' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc15_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='16' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc16_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='17' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc17_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='18' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc18_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='19' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc19_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='20' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc20_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='21' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc21_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='22' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc22_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='23' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc23_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='24' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc24_6mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='25' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc25_6mo_cnt
        , CAST(MAX(CASE WHEN md.mdc_cd='01' THEN 1 ELSE 0 END) AS INT64) AS ipmdc01_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='02' THEN 1 ELSE 0 END) AS INT64) AS ipmdc02_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='03' THEN 1 ELSE 0 END) AS INT64) AS ipmdc03_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='04' THEN 1 ELSE 0 END) AS INT64) AS ipmdc04_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='05' THEN 1 ELSE 0 END) AS INT64) AS ipmdc05_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='06' THEN 1 ELSE 0 END) AS INT64) AS ipmdc06_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='07' THEN 1 ELSE 0 END) AS INT64) AS ipmdc07_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='08' THEN 1 ELSE 0 END) AS INT64) AS ipmdc08_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='09' THEN 1 ELSE 0 END) AS INT64) AS ipmdc09_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='10' THEN 1 ELSE 0 END) AS INT64) AS ipmdc10_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='11' THEN 1 ELSE 0 END) AS INT64) AS ipmdc11_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='12' THEN 1 ELSE 0 END) AS INT64) AS ipmdc12_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='13' THEN 1 ELSE 0 END) AS INT64) AS ipmdc13_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='14' THEN 1 ELSE 0 END) AS INT64) AS ipmdc14_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='15' THEN 1 ELSE 0 END) AS INT64) AS ipmdc15_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='16' THEN 1 ELSE 0 END) AS INT64) AS ipmdc16_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='17' THEN 1 ELSE 0 END) AS INT64) AS ipmdc17_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='18' THEN 1 ELSE 0 END) AS INT64) AS ipmdc18_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='19' THEN 1 ELSE 0 END) AS INT64) AS ipmdc19_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='20' THEN 1 ELSE 0 END) AS INT64) AS ipmdc20_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='21' THEN 1 ELSE 0 END) AS INT64) AS ipmdc21_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='22' THEN 1 ELSE 0 END) AS INT64) AS ipmdc22_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='23' THEN 1 ELSE 0 END) AS INT64) AS ipmdc23_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='24' THEN 1 ELSE 0 END) AS INT64) AS ipmdc24_6mo
        , CAST(MAX(CASE WHEN md.mdc_cd='25' THEN 1 ELSE 0 END) AS INT64) AS ipmdc25_6mo
        , CAST(SUM(CASE WHEN md.mdc_cd='01' THEN md.days ELSE 0 END) AS INT64) AS ipmdc01_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='02' THEN md.days ELSE 0 END) AS INT64) AS ipmdc02_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='03' THEN md.days ELSE 0 END) AS INT64) AS ipmdc03_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='04' THEN md.days ELSE 0 END) AS INT64) AS ipmdc04_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='05' THEN md.days ELSE 0 END) AS INT64) AS ipmdc05_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='06' THEN md.days ELSE 0 END) AS INT64) AS ipmdc06_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='07' THEN md.days ELSE 0 END) AS INT64) AS ipmdc07_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='08' THEN md.days ELSE 0 END) AS INT64) AS ipmdc08_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='09' THEN md.days ELSE 0 END) AS INT64) AS ipmdc09_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='10' THEN md.days ELSE 0 END) AS INT64) AS ipmdc10_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='11' THEN md.days ELSE 0 END) AS INT64) AS ipmdc11_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='12' THEN md.days ELSE 0 END) AS INT64) AS ipmdc12_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='13' THEN md.days ELSE 0 END) AS INT64) AS ipmdc13_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='14' THEN md.days ELSE 0 END) AS INT64) AS ipmdc14_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='15' THEN md.days ELSE 0 END) AS INT64) AS ipmdc15_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='16' THEN md.days ELSE 0 END) AS INT64) AS ipmdc16_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='17' THEN md.days ELSE 0 END) AS INT64) AS ipmdc17_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='18' THEN md.days ELSE 0 END) AS INT64) AS ipmdc18_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='19' THEN md.days ELSE 0 END) AS INT64) AS ipmdc19_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='20' THEN md.days ELSE 0 END) AS INT64) AS ipmdc20_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='21' THEN md.days ELSE 0 END) AS INT64) AS ipmdc21_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='22' THEN md.days ELSE 0 END) AS INT64) AS ipmdc22_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='23' THEN md.days ELSE 0 END) AS INT64) AS ipmdc23_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='24' THEN md.days ELSE 0 END) AS INT64) AS ipmdc24_6mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='25' THEN md.days ELSE 0 END) AS INT64) AS ipmdc25_6mo_days
FROM st
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc6mo_4_te_formal_evaluation_20241120_20250930` AS md 
    ON st.individual_id = md.individual_id 
    AND st.index_dt = md.index_dt
GROUP BY st.individual_id, st.index_dt
;

-- 3-month MDC pivot
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc3mo2_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc3mo2_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH st AS (
    SELECT DISTINCT individual_id, index_dt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930`
)
SELECT 
        st.individual_id
        , st.index_dt
        , CAST(SUM(CASE WHEN md.mdc_cd='01' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc01_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='02' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc02_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='03' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc03_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='04' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc04_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='05' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc05_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='06' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc06_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='07' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc07_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='08' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc08_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='09' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc09_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='10' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc10_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='11' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc11_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='12' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc12_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='13' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc13_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='14' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc14_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='15' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc15_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='16' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc16_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='17' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc17_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='18' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc18_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='19' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc19_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='20' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc20_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='21' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc21_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='22' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc22_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='23' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc23_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='24' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc24_3mo_cnt
        , CAST(SUM(CASE WHEN md.mdc_cd='25' THEN md.cnt ELSE 0 END) AS INT64) AS ipmdc25_3mo_cnt
        , CAST(MAX(CASE WHEN md.mdc_cd='01' THEN 1 ELSE 0 END) AS INT64) AS ipmdc01_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='02' THEN 1 ELSE 0 END) AS INT64) AS ipmdc02_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='03' THEN 1 ELSE 0 END) AS INT64) AS ipmdc03_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='04' THEN 1 ELSE 0 END) AS INT64) AS ipmdc04_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='05' THEN 1 ELSE 0 END) AS INT64) AS ipmdc05_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='06' THEN 1 ELSE 0 END) AS INT64) AS ipmdc06_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='07' THEN 1 ELSE 0 END) AS INT64) AS ipmdc07_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='08' THEN 1 ELSE 0 END) AS INT64) AS ipmdc08_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='09' THEN 1 ELSE 0 END) AS INT64) AS ipmdc09_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='10' THEN 1 ELSE 0 END) AS INT64) AS ipmdc10_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='11' THEN 1 ELSE 0 END) AS INT64) AS ipmdc11_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='12' THEN 1 ELSE 0 END) AS INT64) AS ipmdc12_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='13' THEN 1 ELSE 0 END) AS INT64) AS ipmdc13_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='14' THEN 1 ELSE 0 END) AS INT64) AS ipmdc14_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='15' THEN 1 ELSE 0 END) AS INT64) AS ipmdc15_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='16' THEN 1 ELSE 0 END) AS INT64) AS ipmdc16_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='17' THEN 1 ELSE 0 END) AS INT64) AS ipmdc17_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='18' THEN 1 ELSE 0 END) AS INT64) AS ipmdc18_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='19' THEN 1 ELSE 0 END) AS INT64) AS ipmdc19_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='20' THEN 1 ELSE 0 END) AS INT64) AS ipmdc20_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='21' THEN 1 ELSE 0 END) AS INT64) AS ipmdc21_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='22' THEN 1 ELSE 0 END) AS INT64) AS ipmdc22_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='23' THEN 1 ELSE 0 END) AS INT64) AS ipmdc23_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='24' THEN 1 ELSE 0 END) AS INT64) AS ipmdc24_3mo
        , CAST(MAX(CASE WHEN md.mdc_cd='25' THEN 1 ELSE 0 END) AS INT64) AS ipmdc25_3mo
        , CAST(SUM(CASE WHEN md.mdc_cd='01' THEN md.days ELSE 0 END) AS INT64) AS ipmdc01_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='02' THEN md.days ELSE 0 END) AS INT64) AS ipmdc02_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='03' THEN md.days ELSE 0 END) AS INT64) AS ipmdc03_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='04' THEN md.days ELSE 0 END) AS INT64) AS ipmdc04_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='05' THEN md.days ELSE 0 END) AS INT64) AS ipmdc05_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='06' THEN md.days ELSE 0 END) AS INT64) AS ipmdc06_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='07' THEN md.days ELSE 0 END) AS INT64) AS ipmdc07_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='08' THEN md.days ELSE 0 END) AS INT64) AS ipmdc08_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='09' THEN md.days ELSE 0 END) AS INT64) AS ipmdc09_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='10' THEN md.days ELSE 0 END) AS INT64) AS ipmdc10_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='11' THEN md.days ELSE 0 END) AS INT64) AS ipmdc11_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='12' THEN md.days ELSE 0 END) AS INT64) AS ipmdc12_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='13' THEN md.days ELSE 0 END) AS INT64) AS ipmdc13_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='14' THEN md.days ELSE 0 END) AS INT64) AS ipmdc14_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='15' THEN md.days ELSE 0 END) AS INT64) AS ipmdc15_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='16' THEN md.days ELSE 0 END) AS INT64) AS ipmdc16_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='17' THEN md.days ELSE 0 END) AS INT64) AS ipmdc17_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='18' THEN md.days ELSE 0 END) AS INT64) AS ipmdc18_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='19' THEN md.days ELSE 0 END) AS INT64) AS ipmdc19_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='20' THEN md.days ELSE 0 END) AS INT64) AS ipmdc20_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='21' THEN md.days ELSE 0 END) AS INT64) AS ipmdc21_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='22' THEN md.days ELSE 0 END) AS INT64) AS ipmdc22_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='23' THEN md.days ELSE 0 END) AS INT64) AS ipmdc23_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='24' THEN md.days ELSE 0 END) AS INT64) AS ipmdc24_3mo_days
        , CAST(SUM(CASE WHEN md.mdc_cd='25' THEN md.days ELSE 0 END) AS INT64) AS ipmdc25_3mo_days
FROM st
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc3mo_4_te_formal_evaluation_20241120_20250930` AS md 
    ON st.individual_id = md.individual_id 
    AND st.index_dt = md.index_dt
GROUP BY st.individual_id, st.index_dt
;


/*==============================================================================
  STEP 0r: COMBINE ALL MDC FEATURES INTO SINGLE TABLE
  
  Purpose: Join all time-period MDC pivot tables into comprehensive feature set
  
  Output: ~300 MDC feature columns:
  - 25 MDCs × 3 metrics (count, flag, days) × 4 time periods = 300 columns
  - Plus individual_id, index_dt = 302 total columns
  
  This provides detailed medical utilization history by diagnostic category
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc_features_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc_features_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    mm1.*
    -- 1-year MDC features (75 columns)
    , mm2.ipmdc01_1yr_cnt, mm2.ipmdc02_1yr_cnt, mm2.ipmdc03_1yr_cnt, mm2.ipmdc04_1yr_cnt, mm2.ipmdc05_1yr_cnt
    , mm2.ipmdc06_1yr_cnt, mm2.ipmdc07_1yr_cnt, mm2.ipmdc08_1yr_cnt, mm2.ipmdc09_1yr_cnt, mm2.ipmdc10_1yr_cnt
    , mm2.ipmdc11_1yr_cnt, mm2.ipmdc12_1yr_cnt, mm2.ipmdc13_1yr_cnt, mm2.ipmdc14_1yr_cnt, mm2.ipmdc15_1yr_cnt
    , mm2.ipmdc16_1yr_cnt, mm2.ipmdc17_1yr_cnt, mm2.ipmdc18_1yr_cnt, mm2.ipmdc19_1yr_cnt, mm2.ipmdc20_1yr_cnt
    , mm2.ipmdc21_1yr_cnt, mm2.ipmdc22_1yr_cnt, mm2.ipmdc23_1yr_cnt, mm2.ipmdc24_1yr_cnt, mm2.ipmdc25_1yr_cnt
    , mm2.ipmdc01_1yr, mm2.ipmdc02_1yr, mm2.ipmdc03_1yr, mm2.ipmdc04_1yr, mm2.ipmdc05_1yr
    , mm2.ipmdc06_1yr, mm2.ipmdc07_1yr, mm2.ipmdc08_1yr, mm2.ipmdc09_1yr, mm2.ipmdc10_1yr
    , mm2.ipmdc11_1yr, mm2.ipmdc12_1yr, mm2.ipmdc13_1yr, mm2.ipmdc14_1yr, mm2.ipmdc15_1yr
    , mm2.ipmdc16_1yr, mm2.ipmdc17_1yr, mm2.ipmdc18_1yr, mm2.ipmdc19_1yr, mm2.ipmdc20_1yr
    , mm2.ipmdc21_1yr, mm2.ipmdc22_1yr, mm2.ipmdc23_1yr, mm2.ipmdc24_1yr, mm2.ipmdc25_1yr
    , mm2.ipmdc01_1yr_days, mm2.ipmdc02_1yr_days, mm2.ipmdc03_1yr_days, mm2.ipmdc04_1yr_days, mm2.ipmdc05_1yr_days
    , mm2.ipmdc06_1yr_days, mm2.ipmdc07_1yr_days, mm2.ipmdc08_1yr_days, mm2.ipmdc09_1yr_days, mm2.ipmdc10_1yr_days
    , mm2.ipmdc11_1yr_days, mm2.ipmdc12_1yr_days, mm2.ipmdc13_1yr_days, mm2.ipmdc14_1yr_days, mm2.ipmdc15_1yr_days
    , mm2.ipmdc16_1yr_days, mm2.ipmdc17_1yr_days, mm2.ipmdc18_1yr_days, mm2.ipmdc19_1yr_days, mm2.ipmdc20_1yr_days
    , mm2.ipmdc21_1yr_days, mm2.ipmdc22_1yr_days, mm2.ipmdc23_1yr_days, mm2.ipmdc24_1yr_days, mm2.ipmdc25_1yr_days
    -- 6-month MDC features (75 columns)
    , mm3.ipmdc01_6mo_cnt, mm3.ipmdc02_6mo_cnt, mm3.ipmdc03_6mo_cnt, mm3.ipmdc04_6mo_cnt, mm3.ipmdc05_6mo_cnt
    , mm3.ipmdc06_6mo_cnt, mm3.ipmdc07_6mo_cnt, mm3.ipmdc08_6mo_cnt, mm3.ipmdc09_6mo_cnt, mm3.ipmdc10_6mo_cnt
    , mm3.ipmdc11_6mo_cnt, mm3.ipmdc12_6mo_cnt, mm3.ipmdc13_6mo_cnt, mm3.ipmdc14_6mo_cnt, mm3.ipmdc15_6mo_cnt
    , mm3.ipmdc16_6mo_cnt, mm3.ipmdc17_6mo_cnt, mm3.ipmdc18_6mo_cnt, mm3.ipmdc19_6mo_cnt, mm3.ipmdc20_6mo_cnt
    , mm3.ipmdc21_6mo_cnt, mm3.ipmdc22_6mo_cnt, mm3.ipmdc23_6mo_cnt, mm3.ipmdc24_6mo_cnt, mm3.ipmdc25_6mo_cnt
    , mm3.ipmdc01_6mo, mm3.ipmdc02_6mo, mm3.ipmdc03_6mo, mm3.ipmdc04_6mo, mm3.ipmdc05_6mo
    , mm3.ipmdc06_6mo, mm3.ipmdc07_6mo, mm3.ipmdc08_6mo, mm3.ipmdc09_6mo, mm3.ipmdc10_6mo
    , mm3.ipmdc11_6mo, mm3.ipmdc12_6mo, mm3.ipmdc13_6mo, mm3.ipmdc14_6mo, mm3.ipmdc15_6mo
    , mm3.ipmdc16_6mo, mm3.ipmdc17_6mo, mm3.ipmdc18_6mo, mm3.ipmdc19_6mo, mm3.ipmdc20_6mo
    , mm3.ipmdc21_6mo, mm3.ipmdc22_6mo, mm3.ipmdc23_6mo, mm3.ipmdc24_6mo, mm3.ipmdc25_6mo
    , mm3.ipmdc01_6mo_days, mm3.ipmdc02_6mo_days, mm3.ipmdc03_6mo_days, mm3.ipmdc04_6mo_days, mm3.ipmdc05_6mo_days
    , mm3.ipmdc06_6mo_days, mm3.ipmdc07_6mo_days, mm3.ipmdc08_6mo_days, mm3.ipmdc09_6mo_days, mm3.ipmdc10_6mo_days
    , mm3.ipmdc11_6mo_days, mm3.ipmdc12_6mo_days, mm3.ipmdc13_6mo_days, mm3.ipmdc14_6mo_days, mm3.ipmdc15_6mo_days
    , mm3.ipmdc16_6mo_days, mm3.ipmdc17_6mo_days, mm3.ipmdc18_6mo_days, mm3.ipmdc19_6mo_days, mm3.ipmdc20_6mo_days
    , mm3.ipmdc21_6mo_days, mm3.ipmdc22_6mo_days, mm3.ipmdc23_6mo_days, mm3.ipmdc24_6mo_days, mm3.ipmdc25_6mo_days
    -- 3-month MDC features (75 columns)
    , mm4.ipmdc01_3mo_cnt, mm4.ipmdc02_3mo_cnt, mm4.ipmdc03_3mo_cnt, mm4.ipmdc04_3mo_cnt, mm4.ipmdc05_3mo_cnt
    , mm4.ipmdc06_3mo_cnt, mm4.ipmdc07_3mo_cnt, mm4.ipmdc08_3mo_cnt, mm4.ipmdc09_3mo_cnt, mm4.ipmdc10_3mo_cnt
    , mm4.ipmdc11_3mo_cnt, mm4.ipmdc12_3mo_cnt, mm4.ipmdc13_3mo_cnt, mm4.ipmdc14_3mo_cnt, mm4.ipmdc15_3mo_cnt
    , mm4.ipmdc16_3mo_cnt, mm4.ipmdc17_3mo_cnt, mm4.ipmdc18_3mo_cnt, mm4.ipmdc19_3mo_cnt, mm4.ipmdc20_3mo_cnt
    , mm4.ipmdc21_3mo_cnt, mm4.ipmdc22_3mo_cnt, mm4.ipmdc23_3mo_cnt, mm4.ipmdc24_3mo_cnt, mm4.ipmdc25_3mo_cnt
    , mm4.ipmdc01_3mo, mm4.ipmdc02_3mo, mm4.ipmdc03_3mo, mm4.ipmdc04_3mo, mm4.ipmdc05_3mo
    , mm4.ipmdc06_3mo, mm4.ipmdc07_3mo, mm4.ipmdc08_3mo, mm4.ipmdc09_3mo, mm4.ipmdc10_3mo
    , mm4.ipmdc11_3mo, mm4.ipmdc12_3mo, mm4.ipmdc13_3mo, mm4.ipmdc14_3mo, mm4.ipmdc15_3mo
    , mm4.ipmdc16_3mo, mm4.ipmdc17_3mo, mm4.ipmdc18_3mo, mm4.ipmdc19_3mo, mm4.ipmdc20_3mo
    , mm4.ipmdc21_3mo, mm4.ipmdc22_3mo, mm4.ipmdc23_3mo, mm4.ipmdc24_3mo, mm4.ipmdc25_3mo
    , mm4.ipmdc01_3mo_days, mm4.ipmdc02_3mo_days, mm4.ipmdc03_3mo_days, mm4.ipmdc04_3mo_days, mm4.ipmdc05_3mo_days
    , mm4.ipmdc06_3mo_days, mm4.ipmdc07_3mo_days, mm4.ipmdc08_3mo_days, mm4.ipmdc09_3mo_days, mm4.ipmdc10_3mo_days
    , mm4.ipmdc11_3mo_days, mm4.ipmdc12_3mo_days, mm4.ipmdc13_3mo_days, mm4.ipmdc14_3mo_days, mm4.ipmdc15_3mo_days
    , mm4.ipmdc16_3mo_days, mm4.ipmdc17_3mo_days, mm4.ipmdc18_3mo_days, mm4.ipmdc19_3mo_days, mm4.ipmdc20_3mo_days
    , mm4.ipmdc21_3mo_days, mm4.ipmdc22_3mo_days, mm4.ipmdc23_3mo_days, mm4.ipmdc24_3mo_days, mm4.ipmdc25_3mo_days
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc2yr2_4_te_formal_evaluation_20241120_20250930` AS mm1
INNER JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc1yr2_4_te_formal_evaluation_20241120_20250930` AS mm2 
    ON mm1.individual_id = mm2.individual_id 
    AND mm1.index_dt = mm2.index_dt
INNER JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc6mo2_4_te_formal_evaluation_20241120_20250930` AS mm3 
    ON mm1.individual_id = mm3.individual_id 
    AND mm1.index_dt = mm3.index_dt
INNER JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc3mo2_4_te_formal_evaluation_20241120_20250930` AS mm4 
    ON mm1.individual_id = mm4.individual_id 
    AND mm1.index_dt = mm4.index_dt
;


/*==============================================================================
  STEP 0s: LAB RESULTS EXTRACT (24-MONTH LOOKBACK)
  
  Purpose: Extract lab results from LAB_RESULTS table
  
  Data Source: LAB_RESULTS and LAB_RESULTS_REF (current + archived 2020)
  
  Temporal Window: 24 months before feature_end_dt
  
  Lab Tests: ALT/SGPT, BILIRUB, CEA, CHOL/HDL, CHOLEST, CREAT, CRP, GGT, 
             GLUCOSE, HBA1C, HDL, LDL, MAGNESIU, PSA, SED RATE, SODIUM, TRIGLYC
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_labrslt_extract_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_labrslt_extract_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
-- Current lab data
SELECT 
    st.member_id
    , st.index_dt
    , st.feature_end_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 24 MONTH) AS feature_end_dt_2yr_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 12 MONTH) AS feature_end_dt_1yr_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 6 MONTH) AS feature_end_dt_6mo_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 3 MONTH) AS feature_end_dt_3mo_dt
    , elr.lab_create_dt
    , elr.srv_start_dt
    , elr.lab_low_range
    , elr.lab_high_range
    , CAST(elr.lab_result_nbr AS FLOAT64) AS lab_result_nbr
    , elrr.loinc_class_cd
    , elrr.prcdr_cd
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` AS eicd 
    ON st.member_id = eicd.member_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` AS eicd2 
    ON eicd.individual_id = eicd2.individual_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.LAB_RESULTS` AS elr 
    ON eicd2.member_id = elr.member_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.LAB_RESULTS_REF` AS elrr 
    ON elr.lab_loinc_cd = elrr.lab_loinc_cd
WHERE elr.srv_start_dt <= st.feature_end_dt 
  AND DATE_SUB(st.feature_end_dt, INTERVAL 24 MONTH) <= elr.srv_start_dt

UNION DISTINCT

-- Archived 2020 lab data
SELECT 
    st.member_id
    , st.index_dt
    , st.feature_end_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 24 MONTH) AS feature_end_dt_2yr_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 12 MONTH) AS feature_end_dt_1yr_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 6 MONTH) AS feature_end_dt_6mo_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 3 MONTH) AS feature_end_dt_3mo_dt
    , elr.lab_create_dt
    , elr.srv_start_dt
    , elr.lab_low_range
    , elr.lab_high_range
    , CAST(elr.lab_result_nbr AS FLOAT64) AS lab_result_nbr
    , elrr.loinc_class_cd
    , elrr.prcdr_cd
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST_Y2020` AS eicd 
    ON st.member_id = eicd.member_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST_Y2020` AS eicd2 
    ON eicd.individual_id = eicd2.individual_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.LAB_RESULTS_Y2020` AS elr 
    ON eicd2.member_id = elr.member_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.LAB_RESULTS_REF_Y2020` AS elrr 
    ON elr.lab_loinc_cd = elrr.lab_loinc_cd
WHERE elr.srv_start_dt <= st.feature_end_dt 
  AND DATE_SUB(st.feature_end_dt, INTERVAL 24 MONTH) <= elr.srv_start_dt
;


/*==============================================================================
  STEP 0t: LAB RESULTS 2-YEAR AGGREGATION
  
  Purpose: Aggregate lab results over 2-year lookback window
  
  Features per lab test:
  - Maximum value (minimum for HDL, sodium, magnesium)
  - Elevated/abnormal flag based on clinical thresholds
  
  Lab-specific thresholds:
  - ALT/SGPT > 72, BILIRUB > 1.2, CEA > 10, CHOL/HDL >= 5, CHOLEST >= 200
  - CREAT > 1.5, CRP > 6, GGT > 130, GLUCOSE > 100, HBA1C > 7
  - HDL < 40, LDL >= 130, PSA > 4, SED RATE > 20, TRIGLYC > 300
  - MAGNESIU abnormal if < 1.2 or > 2.7, SODIUM abnormal if < 130 or > 148
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_lab2yr_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_lab2yr_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    st.individual_id
    , st.index_dt
    -- Maximum values
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='ALT/SGPT' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_altsgpt_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='BILIRUB' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_bilirub_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='CEA' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_cea_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='CHOL/HDL' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_cholhdl_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='CHOLEST' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_cholest_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='CREAT' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_creat_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='CRP' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_crp_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='GGT' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_ggt_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='GLUCOSE' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_glucose_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='HBA1C' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_hba1c_2yr
    , MIN(CASE WHEN TRIM(le.loinc_class_cd)='HDL' THEN le.lab_result_nbr ELSE 0 END) AS lab_min_hdl_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='LDL' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_ldl_2yr
    , MIN(CASE WHEN TRIM(le.loinc_class_cd)='MAGNESIU' THEN le.lab_result_nbr ELSE 0 END) AS lab_min_magnesiu_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='MAGNESIU' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_magnesiu_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='PSA' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_psa_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='SED RATE' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_sedrate_2yr
    , MIN(CASE WHEN TRIM(le.loinc_class_cd)='SODIUM' THEN le.lab_result_nbr ELSE 0 END) AS lab_min_sodium_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='SODIUM' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_sodium_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='TRIGLYC' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_triglyc_2yr
    -- Elevated/abnormal flags
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='ALT/SGPT' AND le.lab_result_nbr>72 THEN 1 ELSE 0 END) AS lab_elev_altsgpt_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='BILIRUB' AND le.lab_result_nbr>1.2 THEN 1 ELSE 0 END) AS lab_elev_bilirub_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='CEA' AND le.lab_result_nbr>10 THEN 1 ELSE 0 END) AS lab_elev_cea_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='CHOL/HDL' AND le.lab_result_nbr>=5 THEN 1 ELSE 0 END) AS lab_elev_cholhdl_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='CHOLEST' AND le.lab_result_nbr>=200 THEN 1 ELSE 0 END) AS lab_elev_cholest_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='CREAT' AND le.lab_result_nbr>1.5 THEN 1 ELSE 0 END) AS lab_elev_creat_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='CRP' AND le.lab_result_nbr>6 THEN 1 ELSE 0 END) AS lab_elev_crp_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='GGT' AND le.lab_result_nbr>130 THEN 1 ELSE 0 END) AS lab_elev_ggt_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='GLUCOSE' AND le.lab_result_nbr>100 THEN 1 ELSE 0 END) AS lab_elev_glucose_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='HBA1C' AND le.lab_result_nbr>7 THEN 1 ELSE 0 END) AS lab_elev_hba1c_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='HDL' AND le.lab_result_nbr<40 THEN 1 ELSE 0 END) AS lab_low_hdl_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='LDL' AND le.lab_result_nbr>=130 THEN 1 ELSE 0 END) AS lab_elev_ldl_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='MAGNESIU' AND (le.lab_result_nbr<1.2 OR le.lab_result_nbr>2.7) THEN 1 ELSE 0 END) AS lab_nnorm_magnesiu_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='PSA' AND le.lab_result_nbr>4 THEN 1 ELSE 0 END) AS lab_elev_psa_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='SED RATE' AND le.lab_result_nbr>20 THEN 1 ELSE 0 END) AS lab_elev_sedrate_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='SODIUM' AND (le.lab_result_nbr<130 OR le.lab_result_nbr>148) THEN 1 ELSE 0 END) AS lab_nnorm_sodium_2yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='TRIGLYC' AND le.lab_result_nbr>300 THEN 1 ELSE 0 END) AS lab_elev_triglyc_2yr
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_labrslt_extract_4_te_formal_evaluation_20241120_20250930` AS le 
    ON st.member_id = le.member_id 
    AND st.index_dt = le.index_dt 
    AND le.feature_end_dt_2yr_dt <= le.lab_create_dt 
    AND le.lab_create_dt <= le.feature_end_dt
GROUP BY st.individual_id, st.index_dt
;


/*==============================================================================
  STEP 0u: LAB RESULTS 1-YEAR AGGREGATION
  
  Purpose: Aggregate lab results over 1-year lookback window
  
  Features per lab test:
  - Maximum value (minimum for HDL, sodium, magnesium)
  - Elevated/abnormal flag based on clinical thresholds
  
  Same clinical thresholds as 2-year aggregation
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_lab1yr_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_lab1yr_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    st.individual_id
    , st.index_dt
    -- Maximum values
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='ALT/SGPT' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_altsgpt_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='BILIRUB' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_bilirub_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='CEA' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_cea_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='CHOL/HDL' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_cholhdl_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='CHOLEST' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_cholest_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='CREAT' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_creat_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='CRP' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_crp_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='GGT' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_ggt_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='GLUCOSE' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_glucose_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='HBA1C' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_hba1c_1yr
    , MIN(CASE WHEN TRIM(le.loinc_class_cd)='HDL' THEN le.lab_result_nbr ELSE 0 END) AS lab_min_hdl_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='LDL' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_ldl_1yr
    , MIN(CASE WHEN TRIM(le.loinc_class_cd)='MAGNESIU' THEN le.lab_result_nbr ELSE 0 END) AS lab_min_magnesiu_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='MAGNESIU' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_magnesiu_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='PSA' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_psa_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='SED RATE' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_sedrate_1yr
    , MIN(CASE WHEN TRIM(le.loinc_class_cd)='SODIUM' THEN le.lab_result_nbr ELSE 0 END) AS lab_min_sodium_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='SODIUM' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_sodium_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='TRIGLYC' THEN le.lab_result_nbr ELSE 0 END) AS lab_max_triglyc_1yr
    -- Elevated/abnormal flags
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='ALT/SGPT' AND le.lab_result_nbr>72 THEN 1 ELSE 0 END) AS lab_altsgpt_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='BILIRUB' AND le.lab_result_nbr>1.2 THEN 1 ELSE 0 END) AS lab_bilirub_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='CEA' AND le.lab_result_nbr>10 THEN 1 ELSE 0 END) AS lab_cea_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='CHOL/HDL' AND le.lab_result_nbr>=5 THEN 1 ELSE 0 END) AS lab_cholhdl_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='CHOLEST' AND le.lab_result_nbr>=200 THEN 1 ELSE 0 END) AS lab_cholest_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='CREAT' AND le.lab_result_nbr>1.5 THEN 1 ELSE 0 END) AS lab_creat_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='CRP' AND le.lab_result_nbr>6 THEN 1 ELSE 0 END) AS lab_crp_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='GGT' AND le.lab_result_nbr>130 THEN 1 ELSE 0 END) AS lab_ggt_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='GLUCOSE' AND le.lab_result_nbr>100 THEN 1 ELSE 0 END) AS lab_glucose_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='HBA1C' AND le.lab_result_nbr>7 THEN 1 ELSE 0 END) AS lab_hba1c_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='HDL' AND le.lab_result_nbr<40 THEN 1 ELSE 0 END) AS lab_hdl_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='LDL' AND le.lab_result_nbr>=130 THEN 1 ELSE 0 END) AS lab_ldl_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='MAGNESIU' AND (le.lab_result_nbr<1.2 OR le.lab_result_nbr>2.7) THEN 1 ELSE 0 END) AS lab_magnesiu_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='PSA' AND le.lab_result_nbr>4 THEN 1 ELSE 0 END) AS lab_psa_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='SED RATE' AND le.lab_result_nbr>20 THEN 1 ELSE 0 END) AS lab_sedrate_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='SODIUM' AND (le.lab_result_nbr<130 OR le.lab_result_nbr>148) THEN 1 ELSE 0 END) AS lab_sodium_1yr
    , MAX(CASE WHEN TRIM(le.loinc_class_cd)='TRIGLYC' AND le.lab_result_nbr>300 THEN 1 ELSE 0 END) AS lab_triglyc_1yr
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_labrslt_extract_4_te_formal_evaluation_20241120_20250930` AS le 
    ON st.member_id = le.member_id 
    AND st.index_dt = le.index_dt 
    AND le.feature_end_dt_1yr_dt <= le.lab_create_dt 
    AND le.lab_create_dt <= le.feature_end_dt
    AND le.feature_end_dt_1yr_dt <= le.srv_start_dt
GROUP BY st.individual_id, st.index_dt
;


/*==============================================================================
  STEP 0v: COMBINE LAB RESULT FEATURES
  
  Purpose: Merge 2-year and 1-year lab aggregations
  
  Output: ~72 lab-based features
  - 38 features from 2-year window (19 max/min values + 19 elevated flags)
  - 34 features from 1-year window (17 max/min values + 17 elevated flags)
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_labrslt_features_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_labrslt_features_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    l1.*
    , l2.lab_max_altsgpt_1yr
    , l2.lab_max_bilirub_1yr
    , l2.lab_max_cea_1yr
    , l2.lab_max_cholhdl_1yr
    , l2.lab_max_cholest_1yr
    , l2.lab_max_creat_1yr
    , l2.lab_max_crp_1yr
    , l2.lab_max_ggt_1yr
    , l2.lab_max_glucose_1yr
    , l2.lab_max_hba1c_1yr
    , l2.lab_min_hdl_1yr
    , l2.lab_max_ldl_1yr
    , l2.lab_min_magnesiu_1yr
    , l2.lab_max_magnesiu_1yr
    , l2.lab_max_psa_1yr
    , l2.lab_max_sedrate_1yr
    , l2.lab_min_sodium_1yr
    , l2.lab_max_sodium_1yr
    , l2.lab_max_triglyc_1yr
    , l2.lab_altsgpt_1yr
    , l2.lab_bilirub_1yr
    , l2.lab_cea_1yr
    , l2.lab_cholhdl_1yr
    , l2.lab_cholest_1yr
    , l2.lab_creat_1yr
    , l2.lab_crp_1yr
    , l2.lab_ggt_1yr
    , l2.lab_glucose_1yr
    , l2.lab_hba1c_1yr
    , l2.lab_hdl_1yr
    , l2.lab_ldl_1yr
    , l2.lab_magnesiu_1yr
    , l2.lab_psa_1yr
    , l2.lab_sedrate_1yr
    , l2.lab_sodium_1yr
    , l2.lab_triglyc_1yr
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_lab2yr_4_te_formal_evaluation_20241120_20250930` AS l1
INNER JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_lab1yr_4_te_formal_evaluation_20241120_20250930` AS l2 
    ON l1.individual_id = l2.individual_id 
    AND l1.index_dt = l2.index_dt
;


/*==============================================================================
  STEP 0w: MEMBERSHIP HISTORY EXTRACT (24-MONTH LOOKBACK)
  
  Purpose: Extract active membership months for member month calculations
  
  Data Source: EMIS_MEMBERSHIP (current + archived 2020)
  
  Filters: product_type_cd = 'M' (Medical), within 24-month lookback
  
  Fields: member_id, index_dt, individual_id, feature_end_dt, eff_dt,
          cust_segment_cd, cust_subseg_cd, vision_ind, mental_health_ind
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
-- Current membership data
SELECT 
    st.member_id
    , st.index_dt
    , st.individual_id
    , st.feature_end_dt
    , eem.eff_dt
    , eem.cust_segment_cd
    , eem.cust_subseg_cd
    , eem.vision_ind
    , eem.mental_health_ind
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` AS eicd 
    ON st.member_id = eicd.member_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` AS eicd2 
    ON eicd.individual_id = eicd2.individual_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_MEMBERSHIP` AS eem 
    ON eicd2.member_id = eem.member_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PRODUCT_LINE` AS epl 
    ON eem.product_ln_cd = epl.product_ln_cd
WHERE TRIM(epl.product_type_cd) = 'M' 
  AND DATE_SUB(st.feature_end_dt, INTERVAL 24 MONTH) < eem.eff_dt 
  AND eem.eff_dt <= st.feature_end_dt

UNION DISTINCT

-- Archived 2020 membership data
SELECT 
    st.member_id
    , st.index_dt
    , st.individual_id
    , st.feature_end_dt
    , eem.eff_dt
    , eem.cust_segment_cd
    , eem.cust_subseg_cd
    , eem.vision_ind
    , eem.mental_health_ind
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` AS eicd 
    ON st.member_id = eicd.member_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` AS eicd2 
    ON eicd.individual_id = eicd2.individual_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_MEMBERSHIP_Y2020` AS eem 
    ON eicd2.member_id = eem.member_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PRODUCT_LINE_Y2020` AS epl 
    ON eem.product_ln_cd = epl.product_ln_cd
WHERE TRIM(epl.product_type_cd) = 'M' 
  AND DATE_SUB(st.feature_end_dt, INTERVAL 24 MONTH) < eem.eff_dt 
  AND eem.eff_dt <= st.feature_end_dt
;


/*==============================================================================
  STEP 0x: MEMBER MONTH AGGREGATIONS
  
  Purpose: Calculate member month counts for 2yr, 1yr, 6mo, 3mo windows
  
  Note: Member months are used to normalize cost/utilization features
  
==============================================================================*/

-- 2-year member months with latest segment/subsegment codes
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm2yr_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm2yr_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH other AS (
    SELECT *
        , ROW_NUMBER() OVER (PARTITION BY individual_id, index_dt ORDER BY eff_dt DESC) AS mem_id
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm_4_te_formal_evaluation_20241120_20250930`
)
SELECT 
    m.individual_id
    , m.index_dt
    , CAST(COUNT(*) AS INT64) AS mm_2yr_cnt
    , MAX(o.cust_segment_cd) AS cust_segment_cd
    , MAX(o.cust_subseg_cd) AS cust_subseg_cd
    , MAX(o.mental_health_ind) AS mental_health_ind
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm_4_te_formal_evaluation_20241120_20250930` AS m
JOIN other o
    ON m.individual_id = o.individual_id 
    AND m.index_dt = o.index_dt
GROUP BY m.individual_id, m.index_dt
;

-- 1-year member months
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm1yr_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm1yr_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    m.individual_id
    , m.index_dt
    , CAST(COUNT(*) AS INT64) AS mm_1yr_cnt
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm_4_te_formal_evaluation_20241120_20250930` AS m
WHERE DATE_SUB(m.feature_end_dt, INTERVAL 12 MONTH) < m.eff_dt
GROUP BY m.individual_id, m.index_dt
;

-- 6-month member months
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm6mo_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm6mo_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    m.individual_id
    , m.index_dt
    , CAST(COUNT(*) AS INT64) AS mm_6mo_cnt
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm_4_te_formal_evaluation_20241120_20250930` AS m
WHERE DATE_SUB(m.feature_end_dt, INTERVAL 6 MONTH) < m.eff_dt
GROUP BY m.individual_id, m.index_dt
;

-- 3-month member months
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm3mo_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm3mo_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    m.individual_id
    , m.index_dt
    , CAST(COUNT(*) AS INT64) AS mm_3mo_cnt
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm_4_te_formal_evaluation_20241120_20250930` AS m
WHERE DATE_SUB(m.feature_end_dt, INTERVAL 3 MONTH) < m.eff_dt
GROUP BY m.individual_id, m.index_dt
;

-- Combined member month features
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm_features_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm_features_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH st AS (
    SELECT DISTINCT individual_id, index_dt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930`
)
SELECT 
    st.individual_id
    , st.index_dt
    , CASE WHEN m1.mm_2yr_cnt > 0 THEN m1.mm_2yr_cnt ELSE 0 END AS mm_2yr_cnt
    , CASE WHEN m2.mm_1yr_cnt > 0 THEN m2.mm_1yr_cnt ELSE 0 END AS mm_1yr_cnt
    , CASE WHEN m3.mm_6mo_cnt > 0 THEN m3.mm_6mo_cnt ELSE 0 END AS mm_6mo_cnt
    , CASE WHEN m4.mm_3mo_cnt > 0 THEN m4.mm_3mo_cnt ELSE 0 END AS mm_3mo_cnt
FROM st
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm2yr_4_te_formal_evaluation_20241120_20250930` AS m1 
    ON st.individual_id = m1.individual_id AND st.index_dt = m1.index_dt
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm1yr_4_te_formal_evaluation_20241120_20250930` AS m2 
    ON st.individual_id = m2.individual_id AND st.index_dt = m2.index_dt
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm6mo_4_te_formal_evaluation_20241120_20250930` AS m3 
    ON st.individual_id = m3.individual_id AND st.index_dt = m3.index_dt
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm3mo_4_te_formal_evaluation_20241120_20250930` AS m4 
    ON st.individual_id = m4.individual_id AND st.index_dt = m4.index_dt
;


/*==============================================================================
  STEP 0y: MEMBERSHIP GEOGRAPHIC/DEMOGRAPHIC FEATURES
  
  Purpose: Extract geographic (division, region, state, county) and 
           membership attributes (vision, mental health,segment, etc.)
  
  Data Source: EMIS_MEMBERSHIP, MEMBER, PRODUCT_LINE, GROUP_CONTROL, ZIP_X_ST_X_COUNTY
  
  Geographic Divisions:
  - New England (CT, ME, MA, NH, RI, VT)
  - Mid Atlantic (NJ, NY, PA)
  - East North Central (IN, IL, MI, OH, WI)
  - And 6 more divisions covering all US states
  
==============================================================================*/

-- Extract membership details at index date
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mbrshp_extract_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mbrshp_extract_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT DISTINCT 
    st.member_id
    , st.individual_id
    , st.index_dt
    , eem.drug_ind
    , eem.business_ln_cd
    , eem.fund_ctg_cd
    , eem.product_ln_cd
    , eem.cust_segment_cd
    , eem.cust_subseg_cd
    , eem.vision_ind
    , eem.mental_health_ind
    , epl.prod_ctg_cd
    , eem.group_nbr
    , em.birth_dt
    , CASE 
        WHEN eem.age_nbr IS NULL THEN DATE_DIFF(eem.eff_dt, CAST(em.birth_dt AS DATE), YEAR)
        ELSE eem.age_nbr
      END AS age
    , em.gender_cd
    , em.zip_cd
    , em.county_cd
    , ezsc.state_postal_cd
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_MEMBERSHIP` AS eem 
    ON st.member_id = eem.member_id 
    AND EXTRACT(MONTH FROM st.index_dt) = EXTRACT(MONTH FROM eem.eff_dt) 
    AND EXTRACT(YEAR FROM st.index_dt) = EXTRACT(YEAR FROM eem.eff_dt)
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PRODUCT_LINE` AS epl 
    ON eem.product_ln_cd = epl.product_ln_cd
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GROUP_CONTROL` AS egc 
    ON eem.group_nbr = egc.group_nbr
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEMBER` AS em 
    ON eem.member_id = em.member_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ZIP_X_ST_X_COUNTY` AS ezsc 
    ON em.zip_cd = ezsc.zip_cd
;

-- Create geographic and membership features
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mbrshp_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mbrshp_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH ind AS (
    SELECT 
        st.individual_id
        , st.index_dt
        , CASE 
            WHEN me.state_postal_cd IN ('CT','ME','MA','NH','RI','VT') THEN 'divNewEngland'
            WHEN me.state_postal_cd IN ('NJ','NY','PA') THEN 'divMidAtlantic'
            WHEN me.state_postal_cd IN ('IN','IL','MI','OH','WI') THEN 'divENCentral'
            WHEN me.state_postal_cd IN ('IA','KS','MN','MO','NE','ND','SD') THEN 'divWNCentral'
            WHEN me.state_postal_cd IN ('DE','DC','FL','GA','MD','NC','SC','VA','WV') THEN 'divSouthAtlantic'
            WHEN me.state_postal_cd IN ('AL','KY','MS','TN') THEN 'divESCentral'
            WHEN me.state_postal_cd IN ('AR','LA','OK','TX') THEN 'divWSCentral'
            WHEN me.state_postal_cd IN ('AZ','CO','ID','NM','MT','UT','NV','WY') THEN 'divMountain'
            WHEN me.state_postal_cd IN ('AK','CA','HI','OR','WA') THEN 'divPacific'
            ELSE 'missing' 
          END AS division
        , CASE 
            WHEN me.state_postal_cd IN ('CT','ME','MA','NH','RI','VT','NJ','NY','PA') THEN 'regNE'
            WHEN me.state_postal_cd IN ('IN','IL','MI','OH','WI','IA','KS','MN','MO','NE','ND','SD') THEN 'regMW'
            WHEN me.state_postal_cd IN ('DE','DC','FL','GA','MD','NC','SC','VA','WV','AL','KY','MS','TN','AR','LA','OK','TX') THEN 'regS'
            WHEN me.state_postal_cd IN ('AZ','CO','ID','NM','MT','UT','NV','WY','AK','CA','HI','OR','WA') THEN 'regW'
            ELSE 'missing' 
          END AS region
        , COALESCE(me.cust_segment_cd, 'missing') AS cust_segment_cd
        , COALESCE(me.cust_subseg_cd, 'missing') AS cust_subseg_cd
        , COALESCE(me.vision_ind, "0") AS vision_ind
        , COALESCE(me.mental_health_ind, "0") AS mental_health_ind
        , COALESCE(st.fund_ctg_cd, 'missing') AS fund_ctg_cd
        , COALESCE(county_cd, 'missing') AS county_cd
        , COALESCE(state_postal_cd, 'missing') AS state_postal_cd
        , ROW_NUMBER() OVER (PARTITION BY st.individual_id ORDER BY st.index_dt DESC) AS ord
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` AS eicd 
        ON st.member_id = eicd.member_id
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` AS eicd2 
        ON eicd.individual_id = eicd2.individual_id
    LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mbrshp_extract_4_te_formal_evaluation_20241120_20250930` AS me 
        ON eicd2.individual_id = me.individual_id 
        AND st.index_dt = me.index_dt
)
SELECT 
    individual_id
    , index_dt
    , division
    , region
    , cust_segment_cd
    , cust_subseg_cd
    , vision_ind
    , mental_health_ind
    , fund_ctg_cd
    , county_cd
    , state_postal_cd
FROM ind
WHERE ord = 1
;

-- Combined membership features
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mbrshp_features_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mbrshp_features_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH st AS (
    SELECT DISTINCT individual_id, index_dt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930`
)
SELECT 
    st.*
    , mf.* EXCEPT(individual_id, index_dt)
    , mm.* EXCEPT(individual_id, index_dt)
FROM st
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mbrshp_4_te_formal_evaluation_20241120_20250930` AS mf 
    ON st.individual_id = mf.individual_id AND st.index_dt = mf.index_dt
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm_features_4_te_formal_evaluation_20241120_20250930` AS mm 
    ON st.individual_id = mm.individual_id AND st.index_dt = mm.index_dt
;


/*==============================================================================
  STEP 0z: MEMBER RELATIONSHIP FEATURES
  
  Purpose: Determine member's role in family (employee, partner, dependent)
  
  Data Source: T_EDW_BASE_UNMASK_MEMBER (links members via subscriber_id)
  
  Member Types:
  - Employee: mbr_rtp_type_cd = 'E'
  - Partner: mbr_rtp_type_cd in ('P','S','X')
  - Dependent: mbr_rtp_type_cd in ('C','F','G','L')
  
==============================================================================*/

-- Extract family relationships
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mbrrtp_extract_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mbrrtp_extract_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT DISTINCT 
    st.member_id
    , st.individual_id
    , st.index_dt
    , st.feature_end_dt
    , em2.member_id AS fam_member_id
    , em2.birth_dt
    , em2.covg_cncl_dt
    , em2.mbr_rtp_type_cd
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` AS eicd 
    ON st.member_id = eicd.member_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` AS eicd2 
    ON eicd.individual_id = eicd2.individual_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cns.T_EDW_BASE_UNMASK_MEMBER` AS em 
    ON eicd2.member_id = em.member_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cns.T_EDW_BASE_UNMASK_MEMBER` AS em2 
    ON em.subscriber_id = em2.subscriber_id
;

-- Filter to active family members at index date
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mbrrtp1_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mbrrtp1_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT DISTINCT 
    me.member_id
    , me.individual_id
    , me.index_dt
    , me.fam_member_id
    , me.mbr_rtp_type_cd
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mbrrtp_extract_4_te_formal_evaluation_20241120_20250930` AS me
WHERE CAST(me.birth_dt AS DATE) <= CAST(me.feature_end_dt AS DATE) 
  AND CAST(me.feature_end_dt AS DATE) <= me.covg_cncl_dt
;

-- Create member relationship features
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mbrrtp_features_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mbrrtp_features_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    st.individual_id
    , st.index_dt
    , MAX(CASE 
        WHEN m.mbr_rtp_type_cd = 'E' THEN 'employee'
        WHEN m.mbr_rtp_type_cd IN ('P','S','X') THEN 'partner'
        WHEN m.mbr_rtp_type_cd IN ('C','F','G','L') THEN 'dependent'
        ELSE 'missing'
      END) AS member_type
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mbrrtp1_4_te_formal_evaluation_20241120_20250930` AS m 
    ON st.member_id = m.member_id 
    AND m.member_id = m.fam_member_id
GROUP BY st.individual_id, st.index_dt
;


/*==============================================================================
  STEP 0aa: CLAIM EXTRACT FOR COST/UTILIZATION (24-MONTH LOOKBACK)
  
  Purpose: Extract claim-level data for cost and utilization analysis
  
  Data Source: CLAIM_LINE (current + archived 2020) + related reference tables
  
  Fields: claim_line_id, procedure codes, diagnosis codes, revenue codes,
          service dates, specialty, place of service, provider info, costs
  
  Filters:
  - summarized_srv_ind = 'Y' or NULL
  - duplicate_ind = 'N' or NULL
  - clm_ln_status_cd <> 'D' (exclude denied claims)
  - adjn_dt <= feature_end_dt
  - srv_start_dt within 24-month lookback
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_claim_extract_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_claim_extract_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
-- Current claim data
SELECT
    st.member_id
    , st.individual_id
    , st.index_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 24 MONTH) AS feature_end_dt_2yr_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 12 MONTH) AS feature_end_dt_1yr_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 6 MONTH) AS feature_end_dt_6mo_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 3 MONTH) AS feature_end_dt_3mo_dt
    , st.feature_end_dt
    , CLM.claim_line_id
    , CLM.prcdr_cd
    , CLM.pri_icd9_dx_cd
    , CLM.revenue_cd
    , CLM.srv_start_dt
    , CLM.srv_spclty_ctg_cd
    , CLM.plc_srv_ctg_cd
    , CLM.srv_prvdr_nsa_id
    , CLM.hcfa_plc_srv_cd
    , CLM.allowed_amt
    , CLM.paid_prvdr_par_cd
    , CLM.paid_amt
    , CLM.srv_copay_amt
    , PROC.prcdr_group_nbr
    , SC.spclty_ctg_cls_cd
    , PROV.provider_type_cd
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` st
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` AS I
    ON st.member_id = I.member_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` AS II
    ON I.individual_id = II.individual_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLAIM_LINE` AS CLM
    ON II.member_id = CLM.member_id
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PROCEDURE` AS PROC
    ON CLM.prcdr_cd = PROC.prcdr_cd
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.SPECIALTY_CATEGORY` AS SC 
    ON CLM.srv_spclty_ctg_cd = SC.specialty_ctg_cd
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PROVIDER_DM` AS PROV
    ON CLM.srv_prvdr_id = PROV.provider_id
WHERE CLM.adjn_dt <= st.feature_end_dt
    AND CLM.srv_start_dt BETWEEN DATE_SUB(st.feature_end_dt, INTERVAL 24 MONTH) AND st.feature_end_dt
    AND (TRIM(CLM.summarized_srv_ind) = 'Y' 
        OR TRIM(CLM.summarized_srv_ind) = '' 
        OR CLM.summarized_srv_ind IS NULL)
    AND (TRIM(CLM.duplicate_ind) = 'N' 
        OR TRIM(CLM.duplicate_ind) = '' 
        OR CLM.duplicate_ind IS NULL)
    AND TRIM(CLM.clm_ln_status_cd) <> 'D'

UNION DISTINCT 

-- Archived 2020 claim data
SELECT
    st.member_id
    , st.individual_id
    , st.index_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 24 MONTH) AS feature_end_dt_2yr_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 12 MONTH) AS feature_end_dt_1yr_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 6 MONTH) AS feature_end_dt_6mo_dt
    , DATE_SUB(st.feature_end_dt, INTERVAL 3 MONTH) AS feature_end_dt_3mo_dt
    , st.feature_end_dt
    , CLM.claim_line_id
    , CLM.prcdr_cd
    , CLM.pri_icd9_dx_cd
    , CLM.revenue_cd
    , CLM.srv_start_dt
    , CLM.srv_spclty_ctg_cd
    , CLM.plc_srv_ctg_cd
    , CLM.srv_prvdr_nsa_id
    , CLM.hcfa_plc_srv_cd
    , CLM.allowed_amt
    , CLM.paid_prvdr_par_cd
    , CLM.paid_amt
    , CLM.srv_copay_amt
    , PROC.prcdr_group_nbr
    , SC.spclty_ctg_cls_cd
    , PROV.provider_type_cd
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` st
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` AS I
    ON st.member_id = I.member_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` AS II
    ON I.individual_id = II.individual_id
INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLAIM_LINE_Y2020` AS CLM
    ON II.member_id = CLM.member_id
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PROCEDURE_Y2020` AS PROC
    ON CLM.prcdr_cd = PROC.prcdr_cd
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.SPECIALTY_CATEGORY_Y2020` AS SC 
    ON CLM.srv_spclty_ctg_cd = SC.specialty_ctg_cd
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PROVIDER_DM_Y2020` AS PROV
    ON CLM.srv_prvdr_id = PROV.provider_id
WHERE CLM.adjn_dt <= st.feature_end_dt
    AND CLM.srv_start_dt BETWEEN DATE_SUB(st.feature_end_dt, INTERVAL 24 MONTH) AND st.feature_end_dt
    AND (TRIM(CLM.summarized_srv_ind) = 'Y' 
        OR TRIM(CLM.summarized_srv_ind) = '' 
        OR CLM.summarized_srv_ind IS NULL)
    AND (TRIM(CLM.duplicate_ind) = 'N' 
        OR TRIM(CLM.duplicate_ind) = '' 
        OR CLM.duplicate_ind IS NULL)
    AND TRIM(CLM.clm_ln_status_cd) <> 'D'
;


/*==============================================================================
  STEP 0ab: COST/UTILIZATION AGGREGATIONS BY TIME PERIOD
  
  Purpose: Aggregate claims to create utilization and cost features
  
  Features per time period (2yr, 1yr, 6mo, 3mo):
  - Claim line counts
  - Procedure code counts (total and unique)
  - Diagnosis code counts (total and unique)
  - Revenue code counts (total and unique)
  - Visit type counts: lab, PCP, specialist, urgent care, ER
  - Cost amounts: allowed, paid, copay (total and in-network)
  
  Visit Type Definitions:
  - Lab: prcdr_group_nbr in (145, 218, 219, 220)
  - PCP: prcdr_group_nbr = 161 AND spclty in ('FP','I','P') AND place = 'O'
  - Specialist: prcdr_group_nbr = 161 AND spclty NOT in ('FP','I','P') AND place = 'O'
  - Urgent Care: place <> 'E' AND (hcfa_plc_srv = '20' OR provider_type in ('UC','UM') OR prcdr = 'S9083')
  - ER: place = 'E'
  
==============================================================================*/

-- 2-year aggregation
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_cost_2y_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_cost_2y_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT
    st.individual_id
    , st.index_dt
    , IFNULL(COUNT(claim_line_id), 0) AS clm_ln_cnt_2yr
    , IFNULL(COUNT(prcdr_cd), 0) AS proc_cd_cnt_2yr
    , IFNULL(COUNT(DISTINCT prcdr_cd), 0) AS uniq_proc_cd_cnt_2yr
    , IFNULL(COUNT(pri_icd9_dx_cd), 0) AS dx_cd_cnt_2yr
    , IFNULL(COUNT(DISTINCT pri_icd9_dx_cd), 0) AS uniq_dx_cd_cnt_2yr
    , IFNULL(COUNT(revenue_cd), 0) AS rev_cd_cnt_2yr
    , IFNULL(COUNT(DISTINCT revenue_cd), 0) AS uniq_rev_cd_cnt_2yr
    , IFNULL(COUNT(IF(prcdr_group_nbr IN (145, 218, 219, 220), srv_start_dt, NULL)), 0) AS lab_visit_cnt_2yr
    , IFNULL(COUNT(IF(prcdr_group_nbr = 161
        AND TRIM(srv_spclty_ctg_cd) IN ('FP', 'I', 'P')
        AND TRIM(plc_srv_ctg_cd) = 'O', srv_start_dt, NULL)), 0) AS pcp_visit_cnt_2yr
    , IFNULL(COUNT(IF(prcdr_group_nbr = 161
        AND TRIM(spclty_ctg_cls_cd) IN ('M', 'S', 'O')
        AND TRIM(srv_spclty_ctg_cd) NOT IN ('FP', 'I', 'P')
        AND TRIM(plc_srv_ctg_cd) = 'O', srv_start_dt, NULL)), 0) AS spec_visit_cnt_2yr
    , IFNULL(COUNT(IF(TRIM(plc_srv_ctg_cd) <> 'E'
        AND TRIM(srv_prvdr_nsa_id) NOT BETWEEN '03651' AND '03656'
        AND TRIM(srv_prvdr_nsa_id) NOT BETWEEN '03661' AND '03666'
        AND (TRIM(hcfa_plc_srv_cd) = '20'
            OR TRIM(provider_type_cd) IN ('UC', 'UM')
            OR TRIM(prcdr_cd) = 'S9083'), srv_start_dt, NULL)), 0) AS urg_visit_cnt_2yr
    , IFNULL(CAST(SUM(CASE WHEN TRIM(plc_srv_ctg_cd) = 'E' THEN 1 ELSE 0 END) AS INT64), 0) AS er_clm_cnt_2yr
    , CAST(IFNULL(SUM(allowed_amt), 0) AS FLOAT64) AS clm_allowed_amt_2yr
    , CAST(IFNULL(SUM(IF(TRIM(paid_prvdr_par_cd) = 'Y', allowed_amt, NULL)), 0) AS FLOAT64) AS clm_par_allowed_amt_2yr
    , CAST(IFNULL(SUM(paid_amt), 0) AS FLOAT64) AS clm_paid_amt_2yr
    , CAST(IFNULL(SUM(IF(TRIM(paid_prvdr_par_cd) = 'Y', paid_amt, NULL)), 0) AS FLOAT64) AS clm_par_paid_amt_2yr
    , CAST(IFNULL(SUM(srv_copay_amt), 0) AS FLOAT64) AS clm_srv_copay_amt_2yr
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_claim_extract_4_te_formal_evaluation_20241120_20250930` st
GROUP BY st.individual_id, st.index_dt
;

-- 1-year aggregation
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_cost_1y_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_cost_1y_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT
    st.individual_id
    , st.index_dt
    , IFNULL(COUNT(claim_line_id), 0) AS clm_ln_cnt_1yr
    , IFNULL(COUNT(prcdr_cd), 0) AS proc_cd_cnt_1yr
    , IFNULL(COUNT(DISTINCT prcdr_cd), 0) AS uniq_proc_cd_cnt_1yr
    , IFNULL(COUNT(pri_icd9_dx_cd), 0) AS dx_cd_cnt_1yr
    , IFNULL(COUNT(DISTINCT pri_icd9_dx_cd), 0) AS uniq_dx_cd_cnt_1yr
    , IFNULL(COUNT(revenue_cd), 0) AS rev_cd_cnt_1yr
    , IFNULL(COUNT(DISTINCT revenue_cd), 0) AS uniq_rev_cd_cnt_1yr
    , IFNULL(COUNT(IF(prcdr_group_nbr IN (145, 218, 219, 220), srv_start_dt, NULL)), 0) AS lab_visit_cnt_1yr
    , IFNULL(COUNT(IF(prcdr_group_nbr = 161
        AND TRIM(srv_spclty_ctg_cd) IN ('FP', 'I', 'P')
        AND TRIM(plc_srv_ctg_cd) = 'O', srv_start_dt, NULL)), 0) AS pcp_visit_cnt_1yr
    , IFNULL(COUNT(IF(prcdr_group_nbr = 161
        AND TRIM(spclty_ctg_cls_cd) IN ('M', 'S', 'O')
        AND TRIM(srv_spclty_ctg_cd) NOT IN ('FP', 'I', 'P')
        AND TRIM(plc_srv_ctg_cd) = 'O', srv_start_dt, NULL)), 0) AS spec_visit_cnt_1yr
    , IFNULL(COUNT(IF(TRIM(plc_srv_ctg_cd) <> 'E'
        AND TRIM(srv_prvdr_nsa_id) NOT BETWEEN '03651' AND '03656'
        AND TRIM(srv_prvdr_nsa_id) NOT BETWEEN '03661' AND '03666'
        AND (TRIM(hcfa_plc_srv_cd) = '20'
            OR TRIM(provider_type_cd) IN ('UC', 'UM')
            OR TRIM(prcdr_cd) = 'S9083'), srv_start_dt, NULL)), 0) AS urg_visit_cnt_1yr
    , IFNULL(CAST(SUM(CASE WHEN TRIM(plc_srv_ctg_cd) = 'E' THEN 1 ELSE 0 END) AS INT64), 0) AS er_clm_cnt_1yr
    , CAST(IFNULL(SUM(allowed_amt), 0) AS FLOAT64) AS clm_allowed_amt_1yr
    , CAST(IFNULL(SUM(IF(TRIM(paid_prvdr_par_cd) = 'Y', allowed_amt, NULL)), 0) AS FLOAT64) AS clm_par_allowed_amt_1yr
    , CAST(IFNULL(SUM(paid_amt), 0) AS FLOAT64) AS clm_paid_amt_1yr
    , CAST(IFNULL(SUM(IF(TRIM(paid_prvdr_par_cd) = 'Y', paid_amt, NULL)), 0) AS FLOAT64) AS clm_par_paid_amt_1yr
    , CAST(IFNULL(SUM(srv_copay_amt), 0) AS FLOAT64) AS clm_srv_copay_amt_1yr
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_claim_extract_4_te_formal_evaluation_20241120_20250930` st
WHERE st.srv_start_dt BETWEEN st.feature_end_dt_1yr_dt AND st.feature_end_dt
GROUP BY st.individual_id, st.index_dt
;

-- 6-month aggregation
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_cost_6mo_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_cost_6mo_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT
    st.individual_id
    , st.index_dt
    , IFNULL(COUNT(claim_line_id), 0) AS clm_ln_cnt_6mo
    , IFNULL(COUNT(prcdr_cd), 0) AS proc_cd_cnt_6mo
    , IFNULL(COUNT(DISTINCT prcdr_cd), 0) AS uniq_proc_cd_cnt_6mo
    , IFNULL(COUNT(pri_icd9_dx_cd), 0) AS dx_cd_cnt_6mo
    , IFNULL(COUNT(DISTINCT pri_icd9_dx_cd), 0) AS uniq_dx_cd_cnt_6mo
    , IFNULL(COUNT(revenue_cd), 0) AS rev_cd_cnt_6mo
    , IFNULL(COUNT(DISTINCT revenue_cd), 0) AS uniq_rev_cd_cnt_6mo
    , IFNULL(COUNT(IF(prcdr_group_nbr IN (145, 218, 219, 220), srv_start_dt, NULL)), 0) AS lab_visit_cnt_6mo
    , IFNULL(COUNT(IF(prcdr_group_nbr = 161
        AND TRIM(srv_spclty_ctg_cd) IN ('FP', 'I', 'P')
        AND TRIM(plc_srv_ctg_cd) = 'O', srv_start_dt, NULL)), 0) AS pcp_visit_cnt_6mo
    , IFNULL(COUNT(IF(prcdr_group_nbr = 161
        AND TRIM(spclty_ctg_cls_cd) IN ('M', 'S', 'O')
        AND TRIM(srv_spclty_ctg_cd) NOT IN ('FP', 'I', 'P')
        AND TRIM(plc_srv_ctg_cd) = 'O', srv_start_dt, NULL)), 0) AS spec_visit_cnt_6mo
    , IFNULL(COUNT(IF(TRIM(plc_srv_ctg_cd) <> 'E'
        AND TRIM(srv_prvdr_nsa_id) NOT BETWEEN '03651' AND '03656'
        AND TRIM(srv_prvdr_nsa_id) NOT BETWEEN '03661' AND '03666'
        AND (TRIM(hcfa_plc_srv_cd) = '20'
            OR TRIM(provider_type_cd) IN ('UC', 'UM')
            OR TRIM(prcdr_cd) = 'S9083'), srv_start_dt, NULL)), 0) AS urg_visit_cnt_6mo
    , IFNULL(CAST(SUM(CASE WHEN TRIM(plc_srv_ctg_cd) = 'E' THEN 1 ELSE 0 END) AS INT64), 0) AS er_clm_cnt_6mo
    , CAST(IFNULL(SUM(allowed_amt), 0) AS FLOAT64) AS clm_allowed_amt_6mo
    , CAST(IFNULL(SUM(IF(TRIM(paid_prvdr_par_cd) = 'Y', allowed_amt, NULL)), 0) AS FLOAT64) AS clm_par_allowed_amt_6mo
    , CAST(IFNULL(SUM(paid_amt), 0) AS FLOAT64) AS clm_paid_amt_6mo
    , CAST(IFNULL(SUM(IF(TRIM(paid_prvdr_par_cd) = 'Y', paid_amt, NULL)), 0) AS FLOAT64) AS clm_par_paid_amt_6mo
    , CAST(IFNULL(SUM(srv_copay_amt), 0) AS FLOAT64) AS clm_srv_copay_amt_6mo
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_claim_extract_4_te_formal_evaluation_20241120_20250930` st
WHERE st.srv_start_dt BETWEEN st.feature_end_dt_6mo_dt AND st.feature_end_dt
GROUP BY st.individual_id, st.index_dt
;

-- 3-month aggregation
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_cost_3mo_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_cost_3mo_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT
    st.individual_id
    , st.index_dt
    , IFNULL(COUNT(claim_line_id), 0) AS clm_ln_cnt_3mo
    , IFNULL(COUNT(prcdr_cd), 0) AS proc_cd_cnt_3mo
    , IFNULL(COUNT(DISTINCT prcdr_cd), 0) AS uniq_proc_cd_cnt_3mo
    , IFNULL(COUNT(pri_icd9_dx_cd), 0) AS dx_cd_cnt_3mo
    , IFNULL(COUNT(DISTINCT pri_icd9_dx_cd), 0) AS uniq_dx_cd_cnt_3mo
    , IFNULL(COUNT(revenue_cd), 0) AS rev_cd_cnt_3mo
    , IFNULL(COUNT(DISTINCT revenue_cd), 0) AS uniq_rev_cd_cnt_3mo
    , IFNULL(COUNT(IF(prcdr_group_nbr IN (145, 218, 219, 220), srv_start_dt, NULL)), 0) AS lab_visit_cnt_3mo
    , IFNULL(COUNT(IF(prcdr_group_nbr = 161
        AND TRIM(srv_spclty_ctg_cd) IN ('FP', 'I', 'P')
        AND TRIM(plc_srv_ctg_cd) = 'O', srv_start_dt, NULL)), 0) AS pcp_visit_cnt_3mo
    , IFNULL(COUNT(IF(prcdr_group_nbr = 161
        AND TRIM(spclty_ctg_cls_cd) IN ('M', 'S', 'O')
        AND TRIM(srv_spclty_ctg_cd) NOT IN ('FP', 'I', 'P')
        AND TRIM(plc_srv_ctg_cd) = 'O', srv_start_dt, NULL)), 0) AS spec_visit_cnt_3mo
    , IFNULL(COUNT(IF(TRIM(plc_srv_ctg_cd) <> 'E'
        AND TRIM(srv_prvdr_nsa_id) NOT BETWEEN '03651' AND '03656'
        AND TRIM(srv_prvdr_nsa_id) NOT BETWEEN '03661' AND '03666'
        AND (TRIM(hcfa_plc_srv_cd) = '20'
            OR TRIM(provider_type_cd) IN ('UC', 'UM')
            OR TRIM(prcdr_cd) = 'S9083'), srv_start_dt, NULL)), 0) AS urg_visit_cnt_3mo
    , IFNULL(CAST(SUM(CASE WHEN TRIM(plc_srv_ctg_cd) = 'E' THEN 1 ELSE 0 END) AS INT64), 0) AS er_clm_cnt_3mo
    , CAST(IFNULL(SUM(allowed_amt), 0) AS FLOAT64) AS clm_allowed_amt_3mo
    , CAST(IFNULL(SUM(IF(TRIM(paid_prvdr_par_cd) = 'Y', allowed_amt, NULL)), 0) AS FLOAT64) AS clm_par_allowed_amt_3mo
    , CAST(IFNULL(SUM(paid_amt), 0) AS FLOAT64) AS clm_paid_amt_3mo
    , CAST(IFNULL(SUM(IF(TRIM(paid_prvdr_par_cd) = 'Y', paid_amt, NULL)), 0) AS FLOAT64) AS clm_par_paid_amt_3mo
    , CAST(IFNULL(SUM(srv_copay_amt), 0) AS FLOAT64) AS clm_srv_copay_amt_3mo
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_claim_extract_4_te_formal_evaluation_20241120_20250930` st
WHERE st.srv_start_dt BETWEEN st.feature_end_dt_3mo_dt AND st.feature_end_dt
GROUP BY st.individual_id, st.index_dt
;


/*==============================================================================
  STEP 0ac: COST/UTILIZATION PER MEMBER MONTH FEATURES
  
  Purpose: Normalize cost/utilization by dividing by member months
  
  This creates per-member-month rates which are more comparable across members
  with different enrollment durations
  
  Final Output: ~68 features (17 per time period x 4 time periods)
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_cost_visit_features_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_cost_visit_features_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH yr2 AS (
    SELECT cs.individual_id, cs.index_dt,
        clm_ln_cnt_2yr/mm_2yr_cnt AS clm_ln_cnt_2yr,
        proc_cd_cnt_2yr/mm_2yr_cnt AS proc_cd_cnt_2yr,
        uniq_proc_cd_cnt_2yr/mm_2yr_cnt AS uniq_proc_cd_cnt_2yr,
        dx_cd_cnt_2yr/mm_2yr_cnt AS dx_cd_cnt_2yr,
        uniq_dx_cd_cnt_2yr/mm_2yr_cnt AS uniq_dx_cd_cnt_2yr,
        rev_cd_cnt_2yr/mm_2yr_cnt AS rev_cd_cnt_2yr,
        uniq_rev_cd_cnt_2yr/mm_2yr_cnt AS uniq_rev_cd_cnt_2yr,
        lab_visit_cnt_2yr/mm_2yr_cnt AS lab_visit_cnt_2yr,
        pcp_visit_cnt_2yr/mm_2yr_cnt AS pcp_visit_cnt_2yr,
        spec_visit_cnt_2yr/mm_2yr_cnt AS spec_visit_cnt_2yr,
        urg_visit_cnt_2yr/mm_2yr_cnt AS urg_visit_cnt_2yr,
        er_clm_cnt_2yr/mm_2yr_cnt AS er_clm_cnt_2yr,
        clm_allowed_amt_2yr/mm_2yr_cnt AS clm_allowed_amt_2yr,
        clm_par_allowed_amt_2yr/mm_2yr_cnt AS clm_par_allowed_amt_2yr,
        clm_paid_amt_2yr/mm_2yr_cnt AS clm_paid_amt_2yr,
        clm_par_paid_amt_2yr/mm_2yr_cnt AS clm_par_paid_amt_2yr,
        clm_srv_copay_amt_2yr/mm_2yr_cnt AS clm_srv_copay_amt_2yr
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_cost_2y_4_te_formal_evaluation_20241120_20250930` cs
    JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm2yr_4_te_formal_evaluation_20241120_20250930` mm
        ON cs.individual_id = mm.individual_id AND cs.index_dt = mm.index_dt
)
, yr1 AS (
    SELECT cs.individual_id, cs.index_dt,
        clm_ln_cnt_1yr/mm_1yr_cnt AS clm_ln_cnt_1yr,
        proc_cd_cnt_1yr/mm_1yr_cnt AS proc_cd_cnt_1yr,
        uniq_proc_cd_cnt_1yr/mm_1yr_cnt AS uniq_proc_cd_cnt_1yr,
        dx_cd_cnt_1yr/mm_1yr_cnt AS dx_cd_cnt_1yr,
        uniq_dx_cd_cnt_1yr/mm_1yr_cnt AS uniq_dx_cd_cnt_1yr,
        rev_cd_cnt_1yr/mm_1yr_cnt AS rev_cd_cnt_1yr,
        uniq_rev_cd_cnt_1yr/mm_1yr_cnt AS uniq_rev_cd_cnt_1yr,
        lab_visit_cnt_1yr/mm_1yr_cnt AS lab_visit_cnt_1yr,
        pcp_visit_cnt_1yr/mm_1yr_cnt AS pcp_visit_cnt_1yr,
        spec_visit_cnt_1yr/mm_1yr_cnt AS spec_visit_cnt_1yr,
        urg_visit_cnt_1yr/mm_1yr_cnt AS urg_visit_cnt_1yr,
        er_clm_cnt_1yr/mm_1yr_cnt AS er_clm_cnt_1yr,
        clm_allowed_amt_1yr/mm_1yr_cnt AS clm_allowed_amt_1yr,
        clm_par_allowed_amt_1yr/mm_1yr_cnt AS clm_par_allowed_amt_1yr,
        clm_paid_amt_1yr/mm_1yr_cnt AS clm_paid_amt_1yr,
        clm_par_paid_amt_1yr/mm_1yr_cnt AS clm_par_paid_amt_1yr,
        clm_srv_copay_amt_1yr/mm_1yr_cnt AS clm_srv_copay_amt_1yr
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_cost_1y_4_te_formal_evaluation_20241120_20250930` cs
    JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm1yr_4_te_formal_evaluation_20241120_20250930` mm
        ON cs.individual_id = mm.individual_id AND cs.index_dt = mm.index_dt
)
, mon6 AS (
    SELECT cs.individual_id, cs.index_dt,
        clm_ln_cnt_6mo/mm_6mo_cnt AS clm_ln_cnt_6mo,
        proc_cd_cnt_6mo/mm_6mo_cnt AS proc_cd_cnt_6mo,
        uniq_proc_cd_cnt_6mo/mm_6mo_cnt AS uniq_proc_cd_cnt_6mo,
        dx_cd_cnt_6mo/mm_6mo_cnt AS dx_cd_cnt_6mo,
        uniq_dx_cd_cnt_6mo/mm_6mo_cnt AS uniq_dx_cd_cnt_6mo,
        rev_cd_cnt_6mo/mm_6mo_cnt AS rev_cd_cnt_6mo,
        uniq_rev_cd_cnt_6mo/mm_6mo_cnt AS uniq_rev_cd_cnt_6mo,
        lab_visit_cnt_6mo/mm_6mo_cnt AS lab_visit_cnt_6mo,
        pcp_visit_cnt_6mo/mm_6mo_cnt AS pcp_visit_cnt_6mo,
        spec_visit_cnt_6mo/mm_6mo_cnt AS spec_visit_cnt_6mo,
        urg_visit_cnt_6mo/mm_6mo_cnt AS urg_visit_cnt_6mo,
        er_clm_cnt_6mo/mm_6mo_cnt AS er_clm_cnt_6mo,
        clm_allowed_amt_6mo/mm_6mo_cnt AS clm_allowed_amt_6mo,
        clm_par_allowed_amt_6mo/mm_6mo_cnt AS clm_par_allowed_amt_6mo,
        clm_paid_amt_6mo/mm_6mo_cnt AS clm_paid_amt_6mo,
        clm_par_paid_amt_6mo/mm_6mo_cnt AS clm_par_paid_amt_6mo,
        clm_srv_copay_amt_6mo/mm_6mo_cnt AS clm_srv_copay_amt_6mo
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_cost_6mo_4_te_formal_evaluation_20241120_20250930` cs
    JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm6mo_4_te_formal_evaluation_20241120_20250930` mm
        ON cs.individual_id = mm.individual_id AND cs.index_dt = mm.index_dt
)
, mon3 AS (
    SELECT cs.individual_id, cs.index_dt,
       clm_ln_cnt_3mo/mm_3mo_cnt AS clm_ln_cnt_3mo,
        proc_cd_cnt_3mo/mm_3mo_cnt AS proc_cd_cnt_3mo,
        uniq_proc_cd_cnt_3mo/mm_3mo_cnt AS uniq_proc_cd_cnt_3mo,
        dx_cd_cnt_3mo/mm_3mo_cnt AS dx_cd_cnt_3mo,
        uniq_dx_cd_cnt_3mo/mm_3mo_cnt AS uniq_dx_cd_cnt_3mo,
        rev_cd_cnt_3mo/mm_3mo_cnt AS rev_cd_cnt_3mo,
        uniq_rev_cd_cnt_3mo/mm_3mo_cnt AS uniq_rev_cd_cnt_3mo,
        lab_visit_cnt_3mo/mm_3mo_cnt AS lab_visit_cnt_3mo,
        pcp_visit_cnt_3mo/mm_3mo_cnt AS pcp_visit_cnt_3mo,
        spec_visit_cnt_3mo/mm_3mo_cnt AS spec_visit_cnt_3mo,
        urg_visit_cnt_3mo/mm_3mo_cnt AS urg_visit_cnt_3mo,
        er_clm_cnt_3mo/mm_3mo_cnt AS er_clm_cnt_3mo,
        clm_allowed_amt_3mo/mm_3mo_cnt AS clm_allowed_amt_3mo,
        clm_par_allowed_amt_3mo/mm_3mo_cnt AS clm_par_allowed_amt_3mo,
        clm_paid_amt_3mo/mm_3mo_cnt AS clm_paid_amt_3mo,
        clm_par_paid_amt_3mo/mm_3mo_cnt AS clm_par_paid_amt_3mo,
        clm_srv_copay_amt_3mo/mm_3mo_cnt AS clm_srv_copay_amt_3mo
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_cost_3mo_4_te_formal_evaluation_20241120_20250930` cs
    JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mm3mo_4_te_formal_evaluation_20241120_20250930` mm
        ON cs.individual_id = mm.individual_id AND cs.index_dt = mm.index_dt
)
, ind AS (
    SELECT DISTINCT individual_id, index_dt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930`
)
SELECT ind.*,
    -- 3-month features
    ROUND(IFNULL(clm_ln_cnt_3mo, 0), 2) AS clm_ln_cnt_3mo,
    ROUND(IFNULL(proc_cd_cnt_3mo, 0), 2) AS proc_cd_cnt_3mo,
    ROUND(IFNULL(uniq_proc_cd_cnt_3mo, 0), 2) AS uniq_proc_cd_cnt_3mo,
    ROUND(IFNULL(dx_cd_cnt_3mo, 0), 2) AS dx_cd_cnt_3mo,
    ROUND(IFNULL(uniq_dx_cd_cnt_3mo, 0), 2) AS uniq_dx_cd_cnt_3mo,
    ROUND(IFNULL(rev_cd_cnt_3mo, 0), 2) AS rev_cd_cnt_3mo,
    ROUND(IFNULL(uniq_rev_cd_cnt_3mo, 0), 2) AS uniq_rev_cd_cnt_3mo,
    ROUND(IFNULL(lab_visit_cnt_3mo, 0), 2) AS lab_visit_cnt_3mo,
    ROUND(IFNULL(pcp_visit_cnt_3mo, 0), 2) AS pcp_visit_cnt_3mo,
    ROUND(IFNULL(spec_visit_cnt_3mo, 0), 2) AS spec_visit_cnt_3mo,
    ROUND(IFNULL(urg_visit_cnt_3mo, 0), 2) AS urg_visit_cnt_3mo,
    ROUND(IFNULL(er_clm_cnt_3mo, 0), 2) AS er_clm_cnt_3mo,
    ROUND(IFNULL(clm_allowed_amt_3mo, 0), 2) AS clm_allowed_amt_3mo,
    ROUND(IFNULL(clm_par_allowed_amt_3mo, 0), 2) AS clm_par_allowed_amt_3mo,
    ROUND(IFNULL(clm_paid_amt_3mo, 0), 2) AS clm_paid_amt_3mo,
    ROUND(IFNULL(clm_par_paid_amt_3mo, 0), 2) AS clm_par_paid_amt_3mo,
    ROUND(IFNULL(clm_srv_copay_amt_3mo,  0), 2) AS clm_srv_copay_amt_3mo,
    -- 6-month features
    ROUND(IFNULL(clm_ln_cnt_6mo, 0), 2) AS clm_ln_cnt_6mo,
    ROUND(IFNULL(proc_cd_cnt_6mo, 0), 2) AS proc_cd_cnt_6mo,
    ROUND(IFNULL(uniq_proc_cd_cnt_6mo, 0), 2) AS uniq_proc_cd_cnt_6mo,
    ROUND(IFNULL(dx_cd_cnt_6mo, 0), 2) AS dx_cd_cnt_6mo,
    ROUND(IFNULL(uniq_dx_cd_cnt_6mo, 0), 2) AS uniq_dx_cd_cnt_6mo,
    ROUND(IFNULL(rev_cd_cnt_6mo, 0), 2) AS rev_cd_cnt_6mo,
    ROUND(IFNULL(uniq_rev_cd_cnt_6mo, 0), 2) AS uniq_rev_cd_cnt_6mo,
    ROUND(IFNULL(lab_visit_cnt_6mo, 0), 2) AS lab_visit_cnt_6mo,
    ROUND(IFNULL(pcp_visit_cnt_6mo, 0), 2) AS pcp_visit_cnt_6mo,
    ROUND(IFNULL(spec_visit_cnt_6mo, 0), 2) AS spec_visit_cnt_6mo,
    ROUND(IFNULL(urg_visit_cnt_6mo, 0), 2) AS urg_visit_cnt_6mo,
    ROUND(IFNULL(er_clm_cnt_6mo, 0), 2) AS er_clm_cnt_6mo,
    ROUND(IFNULL(clm_allowed_amt_6mo, 0), 2) AS clm_allowed_amt_6mo,
    ROUND(IFNULL(clm_par_allowed_amt_6mo, 0), 2) AS clm_par_allowed_amt_6mo,
    ROUND(IFNULL(clm_paid_amt_6mo, 0), 2) AS clm_paid_amt_6mo,
    ROUND(IFNULL(clm_par_paid_amt_6mo, 0), 2) AS clm_par_paid_amt_6mo,
    ROUND(IFNULL(clm_srv_copay_amt_6mo,  0), 2) AS clm_srv_copay_amt_6mo,
    -- 1-year features
    ROUND(IFNULL(clm_ln_cnt_1yr, 0), 2) AS clm_ln_cnt_1yr,
    ROUND(IFNULL(proc_cd_cnt_1yr, 0), 2) AS proc_cd_cnt_1yr,
    ROUND(IFNULL(uniq_proc_cd_cnt_1yr, 0), 2) AS uniq_proc_cd_cnt_1yr,
    ROUND(IFNULL(dx_cd_cnt_1yr, 0), 2) AS dx_cd_cnt_1yr,
    ROUND(IFNULL(uniq_dx_cd_cnt_1yr, 0), 2) AS uniq_dx_cd_cnt_1yr,
    ROUND(IFNULL(rev_cd_cnt_1yr, 0), 2) AS rev_cd_cnt_1yr,
    ROUND(IFNULL(uniq_rev_cd_cnt_1yr, 0), 2) AS uniq_rev_cd_cnt_1yr,
    ROUND(IFNULL(lab_visit_cnt_1yr, 0), 2) AS lab_visit_cnt_1yr,
    ROUND(IFNULL(pcp_visit_cnt_1yr, 0), 2) AS pcp_visit_cnt_1yr,
    ROUND(IFNULL(spec_visit_cnt_1yr, 0), 2) AS spec_visit_cnt_1yr,
    ROUND(IFNULL(urg_visit_cnt_1yr, 0), 2) AS urg_visit_cnt_1yr,
    ROUND(IFNULL(er_clm_cnt_1yr, 0), 2) AS er_clm_cnt_1yr,
    ROUND(IFNULL(clm_allowed_amt_1yr, 0), 2) AS clm_allowed_amt_1yr,
    ROUND(IFNULL(clm_par_allowed_amt_1yr, 0), 2) AS clm_par_allowed_amt_1yr,
    ROUND(IFNULL(clm_paid_amt_1yr, 0), 2) AS clm_paid_amt_1yr,
    ROUND(IFNULL(clm_par_paid_amt_1yr, 0), 2) AS clm_par_paid_amt_1yr,
    ROUND(IFNULL(clm_srv_copay_amt_1yr,  0), 2) AS clm_srv_copay_amt_1yr,
    -- 2-year features
    ROUND(IFNULL(clm_ln_cnt_2yr, 0), 2) AS clm_ln_cnt_2yr,
    ROUND(IFNULL(proc_cd_cnt_2yr, 0), 2) AS proc_cd_cnt_2yr,
    ROUND(IFNULL(uniq_proc_cd_cnt_2yr, 0), 2) AS uniq_proc_cd_cnt_2yr,
    ROUND(IFNULL(dx_cd_cnt_2yr, 0), 2) AS dx_cd_cnt_2yr,
    ROUND(IFNULL(uniq_dx_cd_cnt_2yr, 0), 2) AS uniq_dx_cd_cnt_2yr,
    ROUND(IFNULL(rev_cd_cnt_2yr, 0), 2) AS rev_cd_cnt_2yr,
    ROUND(IFNULL(uniq_rev_cd_cnt_2yr, 0), 2) AS uniq_rev_cd_cnt_2yr,
    ROUND(IFNULL(lab_visit_cnt_2yr, 0), 2) AS lab_visit_cnt_2yr,
    ROUND(IFNULL(pcp_visit_cnt_2yr, 0), 2) AS pcp_visit_cnt_2yr,
    ROUND(IFNULL(spec_visit_cnt_2yr, 0), 2) AS spec_visit_cnt_2yr,
    ROUND(IFNULL(urg_visit_cnt_2yr, 0), 2) AS urg_visit_cnt_2yr,
    ROUND(IFNULL(er_clm_cnt_2yr, 0), 2) AS er_clm_cnt_2yr,
    ROUND(IFNULL(clm_allowed_amt_2yr, 0), 2) AS clm_allowed_amt_2yr,
    ROUND(IFNULL(clm_par_allowed_amt_2yr, 0), 2) AS clm_par_allowed_amt_2yr,
    ROUND(IFNULL(clm_paid_amt_2yr, 0), 2) AS clm_paid_amt_2yr,
    ROUND(IFNULL(clm_par_paid_amt_2yr, 0), 2) AS clm_par_paid_amt_2yr,
    ROUND(IFNULL(clm_srv_copay_amt_2yr,  0), 2) AS clm_srv_copay_amt_2yr
FROM ind
LEFT JOIN yr2 ON ind.individual_id = yr2.individual_id AND ind.index_dt = yr2.index_dt
LEFT JOIN yr1 ON ind.individual_id = yr1.individual_id AND ind.index_dt = yr1.index_dt
LEFT JOIN mon6 ON ind.individual_id = mon6.individual_id AND ind.index_dt = mon6.index_dt
LEFT JOIN mon3 ON ind.individual_id = mon3.individual_id AND ind.index_dt = mon3.index_dt
;


/*==============================================================================
  STEP 0ad: POST-PERIOD MEMBERSHIP CONTINUITY FLAGS
  
  Purpose: Check if members have continuous enrollment in outcome periods
  
  This creates flags indicating whether members maintained continuous Commercial
  enrollment for 3, 6, or 12 months after the index date, which is needed to
  ensure valid outcome observation windows.
  
  Criteria:
  - Business line = 'CP' (Commercial)
  - Product type = 'M' (Medical)
  - Product category != '02' (exclude indemnity)
  - Continuous monthly enrollment records for full outcome period
  
  Output: mon_3_include, mon_6_include, mon_12_include (binary flags)
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_post_status_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_post_status_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH 
-- Check 3-month post-period continuity
post_3mo AS (
    SELECT mm.individual_id, mm.index_dt
        , COUNT(DISTINCT mb.eff_dt) AS cnt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` mm
    JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_MEMBERSHIP` mb
        ON mm.member_id = mb.member_id
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PRODUCT_LINE` c  
        ON mb.product_ln_cd = c.product_ln_cd 
    WHERE TRIM(mb.BUSINESS_LN_CD) = 'CP' 
      AND c.prod_ctg_cd != '02' 
      AND TRIM(c.product_type_cd) = 'M'
      AND mb.eff_dt BETWEEN DATE_ADD(mm.feature_end_dt, INTERVAL 1 DAY) 
                        AND DATE_ADD(mm.feature_end_dt, INTERVAL 3 MONTH)
    GROUP BY mm.individual_id, mm.index_dt
)
-- Check 6-month post-period continuity
, post_6mo AS (
    SELECT mm.individual_id, mm.index_dt
        , COUNT(DISTINCT mb.eff_dt) AS cnt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` mm
    JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_MEMBERSHIP` mb
        ON mm.member_id = mb.member_id
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PRODUCT_LINE` c  
        ON mb.product_ln_cd = c.product_ln_cd 
    WHERE TRIM(mb.BUSINESS_LN_CD) = 'CP' 
      AND c.prod_ctg_cd != '02' 
      AND TRIM(c.product_type_cd) = 'M'
      AND mb.eff_dt BETWEEN DATE_ADD(mm.feature_end_dt, INTERVAL 1 DAY) 
                        AND DATE_ADD(mm.feature_end_dt, INTERVAL 6 MONTH)
    GROUP BY mm.individual_id, mm.index_dt
)
-- Check 12-month post-period continuity
, post_12mo AS (
    SELECT mm.individual_id, mm.index_dt
        , COUNT(DISTINCT mb.eff_dt) AS cnt
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` mm
    JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_MEMBERSHIP` mb
        ON mm.member_id = mb.member_id
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PRODUCT_LINE` c  
        ON mb.product_ln_cd = c.product_ln_cd 
    WHERE TRIM(mb.BUSINESS_LN_CD) = 'CP' 
      AND c.prod_ctg_cd != '02' 
      AND TRIM(c.product_type_cd) = 'M'
      AND mb.eff_dt BETWEEN DATE_ADD(mm.feature_end_dt, INTERVAL 1 DAY) 
                        AND DATE_ADD(mm.feature_end_dt, INTERVAL 12 MONTH)
    GROUP BY mm.individual_id, mm.index_dt
)
SELECT 
    st.individual_id
    , st.index_dt
    , CASE WHEN p3.cnt = 3 THEN 1 ELSE 0 END AS mon_3_include
    , CASE WHEN p6.cnt = 6 THEN 1 ELSE 0 END AS mon_6_include
    , CASE WHEN p12.cnt = 12 THEN 1 ELSE 0 END AS mon_12_include
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
LEFT JOIN post_3mo p3 
    ON st.individual_id = p3.individual_id AND st.index_dt = p3.index_dt
LEFT JOIN post_6mo p6 
    ON st.individual_id = p6.individual_id AND st.index_dt = p6.index_dt
LEFT JOIN post_12mo p12 
    ON st.individual_id = p12.individual_id AND st.index_dt = p12.index_dt
;


/*==============================================================================
  STEP 1: EXTRACT ACUTE IP ADMISSION CASES (ALL HORIZONS)
  
  Purpose: Identify all acute inpatient admissions within outcome windows
  
  Data Source: MEDICAL_CASE (Commercial)
  
  Acute IP Definition (from Commercial IP model):
  - med_cs_ps_ctg_cd = 'I' (Inpatient)
  - dummy_mbr_id_ind = 'N' (exclude dummy records)
  
  Timeline:
  - Start: index_dt + 1 day (no overlap with features)
  - End: index_dt + 180 days (6-month prediction horizon)
  
  Note: Only 6-month outcomes, so we extract cases within 180-day window
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_outcome_ip_cases_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_outcome_ip_cases_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    -- Member identifier (matches formal evaluation cohort)
    st.individual_id
    , st.index_dt
    
    -- IP admission details
    , mc.med_case_start_dt
    , mc.med_case_stop_dt
    , mc.medical_case_id
    , mc.med_cs_ps_ctg_cd
    
    -- Length of stay
    , mc.los_day_cnt
    , mc.acu_pd_day_cnt
    
    -- Diagnosis and procedure
    , mc.icd9_dx_cd AS prindiag
    , mc.icd9_dx_group_nbr
    , mc.prcdr_cd
    , mc.prcdr_group_nbr
    , mc.drg_cd
    , mc.mdc_cd
    
    -- Admission type
    , mc.med_cs_admit_ty_cd
    , mc.dschrg_status_cd

FROM
    -- Transformer membership base (created in Step 0)
    `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st

INNER JOIN 
    -- Get member_id from individual_id for joining to claims
    `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` AS icd
        ON st.individual_id = icd.individual_id

INNER JOIN 
    -- Medical Case table (Commercial)
    `edp-prod-hcbstorage.edp_hcb_core_cnsv.MEDICAL_CASE` AS mc
        ON icd.member_id = mc.member_id

WHERE 
    -- OUTCOME WINDOW: Up to 180 days AFTER index_dt (6-month horizon)
    -- Start: 1 day after index_dt (ensures no overlap with features)
    -- End: 180 days after index_dt (6-month prediction horizon)
    CAST(mc.med_case_start_dt AS DATE) 
        BETWEEN DATE_ADD(st.index_dt, INTERVAL 1 DAY) 
            AND DATE_ADD(st.index_dt, INTERVAL 180 DAY)
    
    -- Only inpatient admissions (acute)
    AND mc.med_cs_ps_ctg_cd = 'I'
    
    -- Exclude dummy member records
    AND mc.dummy_mbr_id_ind = 'N'
;


/*==============================================================================
  STEP 2: IDENTIFY EXCLUSION FLAGS (MATERNITY, TRAUMA, TRANSPLANT, ETC.)
  
  Purpose: Flag admissions that should be EXCLUDED from outcomes
  
  Exclusion Criteria (from Commercial IP model):
  - Maternity: pregnancy/delivery-related admissions
  - Trauma: acute injury/accident admissions  
  - Transplant: organ transplant admissions
  - Not impactible: terminal conditions, palliative care
  
  Logic Source: cp_ip_yc_finetune/bq/070_ip_post.bq (lines 110-158)
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_outcome_ip_cases_exclusions`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_outcome_ip_cases_exclusions`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH 
-- Step 2a: Get all claim lines associated with medical cases
case_claims AS (
    SELECT 
        cases.individual_id
        , cases.index_dt
        , cases.medical_case_id
        , cases.med_case_start_dt
        , mccl.claim_line_id
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_outcome_ip_cases_4_te_formal_evaluation_20241120_20250930` cases
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.INDVDL_CUST_DIST` icd
        ON cases.individual_id = icd.individual_id
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.MED_CASE_X_CLM_LN` mccl
        ON icd.member_id = mccl.member_id
        AND cases.medical_case_id = mccl.medical_case_id
)
,
-- Step 2b: Get claim details and DRG codes for exclusion logic
claim_details AS (
    SELECT 
        cc.individual_id
        , cc.index_dt
        , cc.medical_case_id
        , ecl.claim_line_id
        , ecl.pri_icd9_dx_cd
        , eid.icd9_dx_group_nbr
        , ep.prcdr_group_nbr
        , drg.drg_cd
    FROM case_claims cc
    INNER JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.EMIS_CLAIM_LINE` ecl
        ON cc.claim_line_id = ecl.claim_line_id
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` eid
        ON ecl.pri_icd9_dx_cd = eid.icd9_dx_cd
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PROCEDURE` ep
        ON ecl.prcdr_cd = ep.prcdr_cd
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.CLM_LN_X_DRG_TYPE` drg
        ON ecl.claim_line_id = drg.claim_line_id
    WHERE ecl.summarized_srv_ind = 'Y'
      AND ecl.duplicate_ind = 'N'
)
-- Step 2c: Apply exclusion logic
SELECT
    individual_id
    , index_dt
    , medical_case_id
    
    -- MATERNITY FLAG
    , MAX(CASE
        WHEN prcdr_group_nbr IN (59, 61, 63, 64, 67, 69)
         OR icd9_dx_group_nbr IN (80, 82, 83, 84, 86, 87, 90, 94, 96, 97, 102, 104, 107, 108, 110)
         OR TRIM(drg_cd) IN ('765', '766', '767', '768', '770', '774', '775', '777', '778', '779', '780', '795')
        THEN 1
        ELSE 0
      END) AS maternity_flag
    
    -- TRAUMA FLAG
    , MAX(CASE
        WHEN icd9_dx_group_nbr IN (2, 9, 15, 63, 64, 70, 72, 144, 185, 195, 196, 212, 261, 284, 303, 304)
         OR drg_cd IN ('001', '002', '005', '006', '007', '008', '010', '183', '184', '185', 
                       '461', '462', '483', '484', '604', '605', '652', '904', '905', '907', 
                       '908', '909', '913', '914', '927', '928', '929', '933', '934', '935', 
                       '955', '956', '957', '958', '959', '963', '964', '965')
        THEN 1
        ELSE 0
      END) AS trauma_flag
    
    -- TRANSPLANT FLAG
    , MAX(CASE
        WHEN TRIM(pri_icd9_dx_cd) IN ('V42.0', 'V49.83')
         OR prcdr_group_nbr IN (165, 184, 185, 186, 194)
        THEN 1
        ELSE 0
      END) AS transplant_flag
    
    -- NOT IMPACTIBLE FLAG (terminal/palliative conditions)
    , MAX(CASE
        WHEN icd9_dx_group_nbr IN (32, 36, 41, 204, 235, 242, 290)
        THEN 1
        ELSE 0
      END) AS not_impactible_flag

FROM claim_details
GROUP BY individual_id, index_dt, medical_case_id
;


/*==============================================================================
  STEP 3: CREATE MEMBER-LEVEL OUTCOME FLAGS (6-MONTH HORIZON)
  
  Purpose: Aggregate IP cases into member-level binary flags for 6-month window
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_outcome_ip6_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_outcome_ip6_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
WITH acute_ip6 AS (
    SELECT
        cases.individual_id
        , cases.index_dt
        , COUNT(DISTINCT cases.medical_case_id) AS sum_ip6_admits
        , SUM(cases.los_day_cnt) AS sum_ip6_los
        , SUM(cases.acu_pd_day_cnt) AS sum_ip6_acu_days
    FROM  
        `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_outcome_ip_cases_4_te_formal_evaluation_20241120_20250930` cases
    LEFT JOIN 
        `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_outcome_ip_cases_exclusions` excl
        ON cases.individual_id = excl.individual_id
        AND cases.medical_case_id = excl.medical_case_id
    WHERE 
        -- 6-month window filter
        cases.med_case_start_dt <= DATE_ADD(cases.index_dt, INTERVAL 180 DAY)
        
        -- Exclude maternity, trauma, transplant, non-impactible
        AND COALESCE(excl.maternity_flag, 0) = 0
        AND COALESCE(excl.trauma_flag, 0) = 0
        AND COALESCE(excl.transplant_flag, 0) = 0
        AND COALESCE(excl.not_impactible_flag, 0) = 0
    GROUP BY 
        cases.individual_id, cases.index_dt
)
SELECT 
    st.individual_id
    , st.index_dt
    , CASE WHEN a.sum_ip6_admits > 0 THEN 1 ELSE 0 END AS ip6
    , COALESCE(a.sum_ip6_admits, 0) AS sum_ip6_admits
    , COALESCE(a.sum_ip6_los, 0) AS sum_ip6_los
    , COALESCE(a.sum_ip6_acu_days, 0) AS sum_ip6_acu_days
FROM 
    -- Full formal evaluation cohort (created in Step 0)
    `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st
LEFT JOIN 
    acute_ip6 AS a
        ON st.individual_id = a.individual_id
        AND st.index_dt = a.index_dt
;


/*==============================================================================
  STEP 4: FINAL 6-MONTH OUTCOMES TABLE
  
  Purpose: Create outcome table with 6-month prediction horizon
  
  Output: One row per member with ip6, sum_ip6_admits, sum_ip6_los
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_outcome_6mo_final_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_outcome_6mo_final_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
AS
SELECT 
    st.individual_id
    , st.index_dt
    
    -- 6-month outcomes (180-day prediction window)
    , COALESCE(ip6.ip6, 0) AS ip6
    , COALESCE(ip6.sum_ip6_admits, 0) AS sum_ip6_admits
    , COALESCE(ip6.sum_ip6_los, 0) AS sum_ip6_los

FROM 
    -- Full formal evaluation cohort (created in Step 0)
    `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930` AS st

LEFT JOIN 
    `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_outcome_ip6_4_te_formal_evaluation_20241120_20250930` ip6
        ON st.individual_id = ip6.individual_id
        AND st.index_dt = ip6.index_dt
;


/*==============================================================================
  STEP 5: VALIDATION QUERIES (Run after table creation)
  
  Purpose: Verify 6-month outcome table quality and distribution
  
==============================================================================*/

-- Query 1: Check 6-month outcome distribution
SELECT 
    COUNT(*) AS total_members,
    SUM(ip6) AS members_with_ip6,
    ROUND(AVG(ip6) * 100, 2) AS ip6_rate_pct,
    ROUND(AVG(sum_ip6_admits), 2) AS avg_ip6_admits,
    ROUND(AVG(sum_ip6_los), 2) AS avg_ip6_los_days,
    MIN(sum_ip6_admits) AS min_admits,
    MAX(sum_ip6_admits) AS max_admits,
    MIN(sum_ip6_los) AS min_los,
    MAX(sum_ip6_los) AS max_los
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_outcome_6mo_final_4_te_formal_evaluation_20241120_20250930`;

-- Query 2: Verify member count matches base cohort
SELECT 
    (SELECT COUNT(DISTINCT individual_id) FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930`) AS base_cohort_count,
    (SELECT COUNT(DISTINCT individual_id) FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_outcome_6mo_final_4_te_formal_evaluation_20241120_20250930`) AS outcome_count,
    CASE WHEN 
        (SELECT COUNT(DISTINCT individual_id) FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930`) =
        (SELECT COUNT(DISTINCT individual_id) FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_outcome_6mo_final_4_te_formal_evaluation_20241120_20250930`)
    THEN 'MATCH ✓' ELSE 'MISMATCH ✗' END AS alignment_check;

-- Query 3: Distribution by index_dt month
SELECT
    FORMAT_DATE('%Y-%m', index_dt) AS index_month,
    COUNT(*) AS members,
    ROUND(AVG(ip6) * 100, 1) AS ip6_rate_pct,
    SUM(ip6) AS ip6_positives
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_outcome_6mo_final_4_te_formal_evaluation_20241120_20250930`
GROUP BY index_month
ORDER BY index_month;

-- Query 4: Check for members with high utilization
SELECT
    'High Utilizers' AS category,
    COUNT(*) AS members_with_multiple_admits,
    ROUND(AVG(sum_ip6_admits), 2) AS avg_admits,
    ROUND(AVG(sum_ip6_los), 2) AS avg_los
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_outcome_6mo_final_4_te_formal_evaluation_20241120_20250930`
WHERE sum_ip6_admits >= 2;


/*==============================================================================
  STEP 6: FINAL MERGE - COMBINE ALL FEATURES AND 6-MONTH OUTCOMES
  
  Purpose: Create the final evaluation table with all features and 6-month outcomes
  
  ⭐ THIS IS THE PRIMARY OUTPUT TABLE ⭐
  - Retained for 180 DAYS (all other tables expire after 1 day)
  - Contains all baseline features and 6-month outcomes
  - Ready for transformer embedding evaluation
  
  This replicates the logic from 260_merge.bq, joining:
  - Base demographics (14 columns)
  - HPD chronic conditions (~94 features)
  - Medical case utilization + MDC features (~332 features)
  - Lab result features (~72 features)
  - Membership/geographic features (~19 features)
  - Member relationship features (1 feature)
  - 6-month outcome labels (3 variables: ip6, sum_ip6_admits, sum_ip6_los)
  - Post-period continuity flags (3 features)
  - Cost/utilization features (~68 features)
  
  Total: ~599 features + 3 outcomes + 3 flags = ~605 columns
  
==============================================================================*/

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_final_dataset_4_te_formal_evaluation_20241120_20250930`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_final_dataset_4_te_formal_evaluation_20241120_20250930`
OPTIONS (
    labels = [("owner", "pritha_ghosh_cvshealth_com"),("cost_center", "13070")]
    , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
)
AS
WITH st AS (
  SELECT  
      individual_id
      , member_id
      , index_dt
      , gender_cd
      , drug_ind 
      , age
      , age_in_months
      , ind_id_last_digit
      , birth_dt
      , feature_end_dt
      , fund_ctg_cd
      , product_ln_cd
      , group_nbr
      , cust_subseg_cd
  FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930`
)
SELECT 
    -- Base identifiers and demographics (14 columns)
    st.*
    
    -- *** OUTCOMES: 6-Month IP Prediction (3 columns) ***
    , lbl.* EXCEPT(individual_id, index_dt) 
    
    -- *** CONTINUITY FLAGS: Post-period enrollment (3 columns) ***
    , inc.* EXCEPT(individual_id, index_dt) 
    
    -- Feature sets (599 columns)
    , hpd.* EXCEPT(individual_id, index_dt) 
    , mdc.* EXCEPT(individual_id, index_dt)
    , labrslt.* EXCEPT(individual_id, index_dt) 
    , mbrshp.* EXCEPT(individual_id, index_dt, cust_subseg_cd, fund_ctg_cd) 
    , mbrrpt.* EXCEPT(individual_id, index_dt) 
    , cv.* EXCEPT(individual_id, index_dt) 
FROM st
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_outcome_6mo_final_4_te_formal_evaluation_20241120_20250930` lbl
  ON st.individual_id = lbl.individual_id AND st.index_dt = lbl.index_dt
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_post_status_4_te_formal_evaluation_20241120_20250930` inc
  ON st.individual_id = inc.individual_id AND st.index_dt = inc.index_dt
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_hpd_features_4_te_formal_evaluation_20241120_20250930` hpd
  ON st.individual_id = hpd.individual_id AND st.index_dt = hpd.index_dt
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc_features_4_te_formal_evaluation_20241120_20250930` mdc
  ON st.individual_id = mdc.individual_id AND st.index_dt = mdc.index_dt
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_labrslt_features_4_te_formal_evaluation_20241120_20250930` labrslt
  ON st.individual_id = labrslt.individual_id AND st.index_dt = labrslt.index_dt
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mbrshp_features_4_te_formal_evaluation_20241120_20250930` mbrshp
  ON st.individual_id = mbrshp.individual_id AND st.index_dt = mbrshp.index_dt 
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mbrrtp_features_4_te_formal_evaluation_20241120_20250930` mbrrpt
  ON st.individual_id = mbrrpt.individual_id AND st.index_dt = mbrrpt.index_dt
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_cost_visit_features_4_te_formal_evaluation_20241120_20250930` cv
  ON st.individual_id = cv.individual_id AND st.index_dt = cv.index_dt
;


/*==============================================================================
  STEP 7: FINAL TABLE VALIDATION
  
  Purpose: Verify the final merged table has correct structure and completeness
  
==============================================================================*/

-- DEBUG: Check row counts of each feature table
SELECT 
    'Base Table' AS table_name,
    COUNT(*) AS row_count
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930`
UNION ALL
SELECT 
    'HPD Features' AS table_name,
    COUNT(*) AS row_count
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_hpd_features_4_te_formal_evaluation_20241120_20250930`
UNION ALL
SELECT 
    'MDC Features' AS table_name,
    COUNT(*) AS row_count
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_medcs_mdc_features_4_te_formal_evaluation_20241120_20250930`
UNION ALL
SELECT 
    'Lab Features' AS table_name,
    COUNT(*) AS row_count
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_labrslt_features_4_te_formal_evaluation_20241120_20250930`
UNION ALL
SELECT 
    'Membership Features' AS table_name,
    COUNT(*) AS row_count
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mbrshp_features_4_te_formal_evaluation_20241120_20250930`
UNION ALL
SELECT 
    'Member Relationship' AS table_name,
    COUNT(*) AS row_count
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_mbrrtp_features_4_te_formal_evaluation_20241120_20250930`
UNION ALL
SELECT 
    'Outcome 6mo' AS table_name,
    COUNT(*) AS row_count
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_outcome_6mo_final_4_te_formal_evaluation_20241120_20250930`
UNION ALL
SELECT 
    'Post Status' AS table_name,
    COUNT(*) AS row_count
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_post_status_4_te_formal_evaluation_20241120_20250930`
UNION ALL
SELECT 
    'Cost/Visit Features' AS table_name,
    COUNT(*) AS row_count
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_cost_visit_features_4_te_formal_evaluation_20241120_20250930`
UNION ALL
SELECT 
    'Final Dataset' AS table_name,
    COUNT(*) AS row_count
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_final_dataset_4_te_formal_evaluation_20241120_20250930`
ORDER BY row_count;

-- Check row count matches base
SELECT 
    'Row Count Check' AS check_name,
    (SELECT COUNT(*) FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930`) AS base_count,
    (SELECT COUNT(*) FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_final_dataset_4_te_formal_evaluation_20241120_20250930`) AS final_count,
    CASE WHEN 
        (SELECT COUNT(*) FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_base_4_te_formal_evaluation_20241120_20250930`) =
        (SELECT COUNT(*) FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_final_dataset_4_te_formal_evaluation_20241120_20250930`)
    THEN 'PASS ✓' ELSE 'FAIL ✗' END AS status;

-- Check 6-month outcome distribution in final table
SELECT 
    'Outcome Distribution' AS check_name,
    COUNT(*) AS total_rows,
    SUM(ip6) AS ip6_positives,
    ROUND(AVG(ip6) * 100, 2) AS ip6_rate_pct,
    ROUND(AVG(sum_ip6_admits), 2) AS avg_admits_per_member,
    ROUND(AVG(sum_ip6_los), 2) AS avg_los_per_member,
    SUM(mon_6_include) AS members_with_6mo_continuity,
    ROUND(AVG(mon_6_include) * 100, 2) AS pct_with_6mo_continuity
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_final_dataset_4_te_formal_evaluation_20241120_20250930`;

-- Sample first 10 rows to verify column structure (key columns)
SELECT 
    individual_id,
    member_id,
    index_dt,
    age,
    gender_cd,
    ip6,
    sum_ip6_admits,
    sum_ip6_los,
    mon_6_include,
    diabetes,
    ip_2yr_cnt,
    alt_max_2yr
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_final_dataset_4_te_formal_evaluation_20241120_20250930`
LIMIT 10;


/*==============================================================================
  USAGE IN TRANSFORMER LINEAR PROBE EVALUATION
  
  After running this SQL, join the outcome to transformer embeddings:
  
  Python pseudocode:
  
  # Load the final dataset with ALL features and outcomes
  final_df = pd.read_gbq(
      """
      SELECT * 
      FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_final_dataset_4_te_formal_evaluation_20241120_20250930`
      """
  )
  
  # This table contains:
  # - individual_id, member_id, index_dt (keys)
  # - ~599 baseline features (demographics, HPD, medical cases, labs, costs, etc.)
  # - 3 outcome variables for 6-month prediction (ip6, sum_ip6_admits, sum_ip6_los)
  # - 3 continuity flags (mon_3_include, mon_6_include, mon_12_include)
  
  # Load transformer embeddings (from o3_train_ending)
  # Extract embeddings using your trained model
  embeddings_df = ...  # Your transformer embedding extraction code
  
  # Join on individual_id + index_dt
  merged_df = embeddings_df.merge(
      final_df, 
      on=['individual_id', 'index_dt'], 
      how='inner'
  )
==============================================================================*/
