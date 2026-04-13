
#!/bin/bash
bq query \
--use_legacy_sql=false \
'DROP TABLE IF EXISTS `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_non_embedding_features_v2`'

bq query \
--use_legacy_sql=false \
'
CREATE OR REPLACE TABLE `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_non_embedding_features_v2`
OPTIONS (labels = [("owner", "'$OWNER'"),("cost_center", "'$COST_CENTER'")]
         , expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), '$DEFAULT_EXP'))
AS
SELECT
    st.asdb_member_key
    , st.asdb_plan_key
    , st.index_dt
    , st.coa_population_category
    , st.coa_population_group
    , ed1.sum_ed_visits AS sum_ed_visits_yr1
    , ed1.ed_flag AS ed_flag_yr1
    , ed1.sum_avoidable AS sum_avoidable_yr1
    , ed1.sum_unnecessary AS sum_unnecessary_yr1
    , ed1.sum_preventable AS sum_preventable_yr1
    , ed1.low_sev_ed_visits AS low_sev_ed_visits_yr1
    , ed1.low_med_sev_ed_visits AS low_med_sev_ed_visits_yr1
    , ed1.med_sev_ed_visits AS med_sev_ed_visits_yr1
    , ed1.med_high_sev_ed_visits AS med_high_sev_ed_visits_yr1
    , ed1.high_sev_ed_visits AS high_sev_ed_visits_yr1
    , ed1.low_sev_ed_flag AS low_sev_ed_flag_yr1
    , ed1.low_med_sev_ed_flag AS low_med_sev_ed_flag_yr1
    , ed1.med_sev_ed_flag AS med_sev_ed_flag_yr1
    , ed1.med_high_sev_ed_flag AS med_high_sev_ed_flag_yr1
    , ed1.high_sev_ed_flag AS high_sev_ed_flag_yr1
    , ed2.ed_flag AS ed_flag_yr2   
    , ed2.sum_ed_visits AS sum_ed_visits_yr2
    , ed2.sum_avoidable AS sum_avoidable_yr2
    , ed2.sum_unnecessary AS sum_unnecessary_yr2
    , ed2.sum_preventable AS sum_preventable_yr2
    , ed2.low_sev_ed_visits AS low_sev_ed_visits_yr2
    , ed2.low_med_sev_ed_visits AS low_med_sev_ed_visits_yr2
    , ed2.med_sev_ed_visits AS med_sev_ed_visits_yr2
    , ed2.med_high_sev_ed_visits AS med_high_sev_ed_visits_yr2
    , ed2.high_sev_ed_visits AS high_sev_ed_visits_yr2
    , ed2.low_sev_ed_flag AS low_sev_ed_flag_yr2
    , ed2.low_med_sev_ed_flag AS low_med_sev_ed_flag_yr2
    , ed2.med_sev_ed_flag AS med_sev_ed_flag_yr2
    , ed2.med_high_sev_ed_flag AS med_high_sev_ed_flag_yr2
    , ed2.high_sev_ed_flag AS high_sev_ed_flag_yr2
    , ip1.acute_ip_flag AS acute_ip_flag_yr1
    , ip1.sum_acute_ip_admits AS sum_acute_ip_admits_yr1
    , ip1.sum_acute_calc_los  AS sum_acute_calc_los_yr1
    , ip2.acute_ip_flag AS acute_ip_flag_yr2
    , ip2.sum_acute_ip_admits AS sum_acute_ip_admits_yr2
    , ip2.sum_acute_calc_los AS sum_acute_calc_los_yr2
    , op1.sum_op_visits AS sum_op_visits_yr1
    , op2.sum_op_visits AS sum_op_visits_yr2
    , ut1.emis_community_clm AS emis_community_clm_yr1
    , ut1.emis_ed_clm AS emis_ed_clm_yr1
    , ut1.emis_hh_clm AS emis_hh_clm_yr1
    , ut1.emis_home_clm AS emis_home_clm_yr1
    , ut1.emis_ip_clm AS emis_ip_clm_yr1
    , ut1.emis_ins_clm AS emis_ins_clm_yr1
    , ut1.emis_lab_clm AS emis_lab_clm_yr1
    , ut1.emis_mrx_clm AS emis_mrx_clm_yr1
    , ut1.emis_mh_clm AS emis_mh_clm_yr1
    , ut1.emis_misc_clm AS emis_misc_clm_yr1
    , ut1.emis_pcp_clm AS emis_pcp_clm_yr1
    , ut1.emis_radio_clm AS emis_radio_clm_yr1
    , ut1.emis_ambul_clm AS emis_ambul_clm_yr1
    , ut1.emis_spec_clm AS emis_spec_clm_yr1
    , ut1.ltc_clm AS ltc_clm_yr1
    , ut1.coe_ip_hos_clm AS coe_ip_hos_clm_yr1
    , ut1.coe_ip_non_hos_clm AS coe_ip_non_hos_clm_yr1
    , ut1.coe_lab_clm AS coe_lab_clm_yr1
    , ut1.coe_ltc_community_clm AS coe_ltc_community_clm_yr1
    , ut1.coe_ltc_home_clm AS coe_ltc_home_clm_yr1
    , ut1.coe_ltc_ins_clm AS coe_ltc_ins_clm_yr1
    , ut1.coe_other_clm AS coe_other_clm_yr1
    , ut1.coe_op_hos_clm AS coe_op_hos_clm_yr1
    , ut1.coe_op_non_hos_clm AS coe_op_non_hos_clm_yr1
    , ut1.coe_anesth_clm AS coe_anesth_clm_yr1
    , ut1.coe_eval_clm AS coe_eval_clm_yr1
    , ut1.coe_maternity_clm AS coe_maternity_clm_yr1
    , ut1.coe_mrx_clm AS coe_mrx_clm_yr1
    , ut1.coe_mh_clm AS coe_mh_clm_yr1
    , ut1.coe_phy_clm AS coe_phy_clm_yr1
    , ut1.coe_surg_clm AS coe_surg_clm_yr1
    , ut1.coe_radio_clm AS coe_radio_clm_yr1
    , ut1.uc_clm AS uc_clm_yr1
    , ut1.obs_clm AS obs_clm_yr1   
    , ut2.emis_community_clm AS emis_community_clm_yr2
    , ut2.emis_ed_clm AS emis_ed_clm_yr2
    , ut2.emis_hh_clm AS emis_hh_clm_yr2
    , ut2.emis_home_clm AS emis_home_clm_yr2
    , ut2.emis_ip_clm AS emis_ip_clm_yr2
    , ut2.emis_ins_clm AS emis_ins_clm_yr2
    , ut2.emis_lab_clm AS emis_lab_clm_yr2
    , ut2.emis_mrx_clm AS emis_mrx_clm_yr2
    , ut2.emis_mh_clm AS emis_mh_clm_yr2
    , ut2.emis_misc_clm AS emis_misc_clm_yr2
    , ut2.emis_pcp_clm AS emis_pcp_clm_yr2
    , ut2.emis_radio_clm AS emis_radio_clm_yr2
    , ut2.emis_ambul_clm AS emis_ambul_clm_yr2
    , ut2.emis_spec_clm AS emis_spec_clm_yr2
    , ut2.ltc_clm AS ltc_clm_yr2
    , ut2.coe_ip_hos_clm AS coe_ip_hos_clm_yr2
    , ut2.coe_ip_non_hos_clm AS coe_ip_non_hos_clm_yr2
    , ut2.coe_lab_clm AS coe_lab_clm_yr2
    , ut2.coe_ltc_community_clm AS coe_ltc_community_clm_yr2
    , ut2.coe_ltc_home_clm AS coe_ltc_home_clm_yr2
    , ut2.coe_ltc_ins_clm AS coe_ltc_ins_clm_yr2
    , ut2.coe_other_clm AS coe_other_clm_yr2
    , ut2.coe_op_hos_clm AS coe_op_hos_clm_yr2
    , ut2.coe_op_non_hos_clm AS coe_op_non_hos_clm_yr2
    , ut2.coe_anesth_clm AS coe_anesth_clm_yr2
    , ut2.coe_eval_clm AS coe_eval_clm_yr2
    , ut2.coe_maternity_clm AS coe_maternity_clm_yr2
    , ut2.coe_mrx_clm AS coe_mrx_clm_yr2
    , ut2.coe_mh_clm AS coe_mh_clm_yr2
    , ut2.coe_phy_clm AS coe_phy_clm_yr2
    , ut2.coe_surg_clm AS coe_surg_clm_yr2
    , ut2.coe_radio_clm AS coe_radio_clm_yr2
    , ut2.uc_clm AS uc_clm_yr2
    , ut2.obs_clm AS obs_clm_yr2
    , cond.abdominal_pain
    , cond.AID
    , cond.IDA
    , cond.ANX
    , cond.OST
    , cond.AST
    , cond.AUT
    , cond.CHO
    , cond.burns
    , cond.cad
    , cond.Cancer
    , cond.narc
    , cond.CBD
    , cond.CHF
    , cond.CRF
    , cond.VNA
    , cond.CHD
    , cond.COP
    , cond.CYS
    , cond.DEP
    , cond.DIA
    , cond.EDO
    , cond.esrd
    , cond.EPL
    , cond.CRO
    , cond.MOH
    , cond.HEM
    , cond.HepC
    , cond.HYP
    , cond.HYC
    , cond.immune
    , cond.intel_dsblty
    , cond.meta_cancer
    , cond.liver_dis
    , cond.MSS
    , cond.OBE
    , cond.oud
    , liver_other
    , cond.paralysis
    , cond.PAR
    , cond.PUD
    , cond.hmd
    , cond.PVD
    , cond.autoimmune
    , cond.DEM
    , cond.SCA
    , cond.sleep_apnea
    , cond.spinal_inj
    , cond.back
    , cond.substance
    , cond.ALC
    , cond.bipolar 
    , cond.psychoses
    , cond.major_chronic_cnt 
    , rx1.rx_claim_cnt AS rx_claim_cnt_yr1
    , rx1.days_supply_sum AS days_supply_sum_yr1
    , rx1.ndc_cnt AS ndc_cnt_yr1
    , rx1.gpi_cnt AS gpi_cnt_yr1
    , rx1.gpi4_cnt AS gpi4_cnt_yr1
    , rx1.gpi2_cnt AS gpi2_cnt_yr1
    , rx1.retail_fills AS retail_fills_yr1
    , rx1.mail_order_fills AS mail_order_fills_yr1
    , rx1.generic_fills AS generic_fills_yr1
    , rx1.branded_generic_fills AS branded_generic_fills_yr1
    , rx1.otc_fills AS otc_fills_yr1
    , rx1.ss_brand_fills AS ss_brand_fills_yr1
    , rx1.ms_brand_fills AS ms_brand_fills_yr1
    , rx1.formulary_fills AS formulary_fills_yr1
    , rx1.maint_drug_fills AS maint_drug_fills_yr1
    , rx1.antidiabetic_scripts AS antidiabetic_scripts_yr1
    , rx1.antidiabetic_days_supply AS antidiabetic_days_supply_yr1
    , rx1.beta_blocker_scripts AS beta_blocker_scripts_yr1
    , rx1.beta_blocker_days_supply AS beta_blocker_days_supply_yr1
    , rx1.antihypertensive_scripts AS antihypertensive_scripts_yr1
    , rx1.antihypertensive_days_supply AS antihypertensive_days_supply_yr1
    , rx1.lipid_lowering_scripts AS lipid_lowering_scripts_yr1
    , rx1.lipid_lowering_days_supply AS lipid_lowering_days_supply_yr1
    , rx1.calcium_channel_blk_scripts AS calcium_channel_blk_scripts_yr1
    , rx1.calcium_channel_blk_days_supply AS calcium_channel_blk_days_supply_yr1
    , rx1.diuretic_scripts AS diuretic_scripts_yr1
    , rx1.diuretic_days_supply AS diuretic_days_supply_yr1
    , rx1.antianginal_agent_scripts AS antianginal_agent_scripts_yr1
    , rx1.antianginal_agent_days_supply AS antianginal_agent_days_supply_yr1
    , rx1.antidepressant_scripts AS antidepressant_scripts_yr1
    , rx1.antidepressant_days_supply AS antidepressant_days_supply_yr1
    , rx1.antipsychotic_scripts AS antipsychotic_scripts_yr1
    , rx1.antipsychotic_days_supply AS antipsychotic_days_supply_yr1
    , rx1.antianxiety_scripts AS antianxiety_scripts_yr1
    , rx1.antianxiety_days_supply AS antianxiety_days_supply_yr1
    , rx1.anticonvulsant_scripts AS anticonvulsant_scripts_yr1
    , rx1.anticonvulsant_days_supply AS anticonvulsant_days_supply_yr1
    , rx1.inhaled_steroid_scripts AS inhaled_steroid_scripts_yr1
    , rx1.inhaled_steroid_days_supply AS inhaled_steroid_days_supply_yr1
    , rx2.rx_claim_cnt AS rx_claim_cnt_yr2
    , rx2.days_supply_sum AS days_supply_sum_yr2
    , rx2.ndc_cnt AS ndc_cnt_yr2
    , rx2.gpi_cnt AS gpi_cnt_yr2
    , rx2.gpi4_cnt AS gpi4_cnt_yr2
    , rx2.gpi2_cnt AS gpi2_cnt_yr2
    , rx2.retail_fills AS retail_fills_yr2
    , rx2.mail_order_fills AS mail_order_fills_yr2
    , rx2.generic_fills AS generic_fills_yr2
    , rx2.branded_generic_fills AS branded_generic_fills_yr2
    , rx2.otc_fills AS otc_fills_yr2
    , rx2.ss_brand_fills AS ss_brand_fills_yr2
    , rx2.ms_brand_fills AS ms_brand_fills_yr2
    , rx2.formulary_fills AS formulary_fills_yr2
    , rx2.maint_drug_fills AS maint_drug_fills_yr2
    , rx2.antidiabetic_scripts AS antidiabetic_scripts_yr2
    , rx2.antidiabetic_days_supply AS antidiabetic_days_supply_yr2
    , rx2.beta_blocker_scripts AS beta_blocker_scripts_yr2
    , rx2.beta_blocker_days_supply AS beta_blocker_days_supply_yr2
    , rx2.antihypertensive_scripts AS antihypertensive_scripts_yr2
    , rx2.antihypertensive_days_supply AS antihypertensive_days_supply_yr2
    , rx2.lipid_lowering_scripts AS lipid_lowering_scripts_yr2
    , rx2.lipid_lowering_days_supply AS lipid_lowering_days_supply_yr2
    , rx2.calcium_channel_blk_scripts AS calcium_channel_blk_scripts_yr2
    , rx2.calcium_channel_blk_days_supply AS calcium_channel_blk_days_supply_yr2
    , rx2.diuretic_scripts AS diuretic_scripts_yr2
    , rx2.diuretic_days_supply AS diuretic_days_supply_yr2
    , rx2.antianginal_agent_scripts AS antianginal_agent_scripts_yr2
    , rx2.antianginal_agent_days_supply AS antianginal_agent_days_supply_yr2
    , rx2.antidepressant_scripts AS antidepressant_scripts_yr2
    , rx2.antidepressant_days_supply AS antidepressant_days_supply_yr2
    , rx2.antipsychotic_scripts AS antipsychotic_scripts_yr2
    , rx2.antipsychotic_days_supply AS antipsychotic_days_supply_yr2
    , rx2.antianxiety_scripts AS antianxiety_scripts_yr2
    , rx2.antianxiety_days_supply AS antianxiety_days_supply_yr2
    , rx2.anticonvulsant_scripts AS anticonvulsant_scripts_yr2
    , rx2.anticonvulsant_days_supply AS anticonvulsant_days_supply_yr2
    , rx2.inhaled_steroid_scripts AS inhaled_steroid_scripts_yr2
    , rx2.inhaled_steroid_days_supply AS inhaled_steroid_days_supply_yr2
    , demo.agenbr
    , demo.gender
    , demo.ethnicity_code
    , demo.primarylanguage_desc
    , demo.tenure_yr1
    , demo.tenure_yr2
    , demo.post_mnths
    , demo.urbsubr
    , demo.zip_weight_avg_medinc
    , acs.social_risk_score AS acs_social_risk_score
    , acs.sdi_score
    , acs.svi_score
    , acs.adi_score
    , csdi.citizenship_index
    , csdi.education_index
    , csdi.food_access
    , csdi.health_access
    , csdi.health_habits
    , csdi.housing_desert
    , csdi.housing_ownership     
    , csdi.housing_quality     
    , csdi.income_index    
    , csdi.income_inequality    
    , csdi.language_score    
    , csdi.natural_disaster 
    , csdi.poverty_score 
    , csdi.proactive_health
    , csdi.racial_diversity
    , csdi.social_isolation    
    , csdi.technology_access    
    , csdi.transport_access     
    , csdi.unemployment_index    
    , csdi.water_quality     
    , csdi.disability_score    
    , csdi.health_infra    
    , csdi.social_risk_score AS csdi_social_risk_score   
    , prev.first_prv_dt
    , prev.last_prv_dt
    , prev.sum_pcp
    , prev.sum_spec
    , prev.sum_ob
    , prev.sum_dme
    , prev.sum_chol_lab
    , prev.sum_a1c_lab
    , prev.sum_chemo
    , prev.cms_alc_scrn
    , prev.cms_bone_scrn
    , prev.cms_cvd_scrn
    , prev.cms_col_scrn
    , prev.cms_tobacco
    , prev.cms_dep_scrn
    , prev.cms_t2d_scrn
    , prev.cms_hepb_scrn
    , prev.cms_hepb_vax
    , prev.cms_ibt_cvd
    , prev.cms_ibt_obese
    , prev.cms_flu_vax
    , prev.cms_lung_cancer_scrn
    , prev.cms_nutrition
    , prev.cms_pneum_vax
    , prev.cms_hpv_scrn
    , prev.cms_sti_scrn
    , prev.cms_mam_scrn
    , prev.cms_pap
    , prev.cms_pelvic
    , prev.cms_t2d_train
    , prev.cms_prost_cancer_scrn
FROM `'$ST'` AS st
LEFT JOIN `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_ed_yr1` AS ed1
    ON st.asdb_member_key = ed1.asdb_member_key
LEFT JOIN `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_ed_yr2` AS ed2
    ON st.asdb_member_key = ed2.asdb_member_key
LEFT JOIN `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_ip_yr1` AS ip1
    ON st.asdb_member_key = ip1.asdb_member_key
LEFT JOIN `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_ip_yr1` AS ip2
    ON st.asdb_member_key = ip2.asdb_member_key
LEFT JOIN `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_op_yr1` AS op1
    ON st.asdb_member_key = op1.asdb_member_key
LEFT JOIN `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_op_yr1` AS op2
    ON st.asdb_member_key = op2.asdb_member_key
LEFT JOIN `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_other_cost_utilization_yr1` AS ut1
    ON st.asdb_member_key = ut1.asdb_member_key
LEFT JOIN `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_other_cost_utilization_yr2` AS ut2
    ON st.asdb_member_key = ut2.asdb_member_key
LEFT JOIN `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_conditions` AS cond
    ON st.asdb_member_key = cond.asdb_member_key
LEFT JOIN `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_rx_yr1` AS rx1
    ON st.asdb_member_key = rx1.asdb_member_key
LEFT JOIN `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_rx_yr2` AS rx2
    ON st.asdb_member_key = rx2.asdb_member_key
LEFT JOIN `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_demographics` AS demo
    ON st.asdb_member_key = demo.asdb_member_key
LEFT JOIN `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_acs` AS acs
    ON st.asdb_member_key = acs.asdb_member_key
LEFT JOIN `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_csdi` AS csdi
    ON st.asdb_member_key = csdi.asdb_member_key
LEFT JOIN `'$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_preventative_summary` AS prev
    ON st.asdb_member_key = prev.asdb_member_key
'
