#!/usr/bin/env python
# coding: utf-8

"""
Medicaid IP Downstream Evaluation Pipeline

This module implements the evaluation pipeline for testing transformer embeddings
on the Medicaid Inpatient (IP) hospitalization prediction task.

The pipeline replicates the exact methodology from the original Medicaid IP model
developed during Eric Ma's internship, including:
- Same data sources and filtering criteria
- Same selected features (from RFECV CatBoost feature selection)
- Same tuned hyperparameters (from Optuna optimization)
- Same evaluation metrics (ROC-AUC, Lift@1%, Lift@10%, PPV, Sensitivity)
- Same class imbalance handling (undersampling with 0.2 ratio)

Supports three evaluation modes:
1. embedding_only: Uses only transformer embeddings (256 dimensions)
2. tabular_only: Uses only hand-crafted tabular features (~243 features)
3. hybrid: Uses both embeddings and tabular features (~499 features)

Original Model Reference:
- CatBoost AUC: ~0.8737
- CatBoost 1% Lift: ~19-20x
- Optimal undersampling ratio: 0.2 (CatBoost)

Data Sources (BigQuery - HELDOUT tables for downstream eval):
- Features: edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_heldout_non_embedding_features
- Outcomes: edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_heldout_outcome_ip
- TE Input: edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_heldout_te_inference_input

Note: Heldout members are those NOT in the 10% TE pretraining sample, ensuring
no data leakage between pretraining and downstream evaluation.

Author: Adapted from Eric Ma's Medicaid IP Model Refresh
"""

"""
-- ============================================================================
-- CREATE TE INFERENCE INPUT TABLE FOR HELDOUT MEDICAID IP MEMBERS
-- ============================================================================
-- Purpose: Extract raw TE input sequences (cd, gender_cd, age_in_months) for 
--          heldout members to generate new embeddings via transformer inference.
--
-- ============================================================================
-- CRITICAL: ID MATCHING CAVEAT FOR MEDICAID
-- ============================================================================
-- 
-- PROBLEM: Medicaid uses TWO different ID formats that must NOT be confused:
--
--   1. asdb_member_key (8-9 digits): The primary key in downstream feature tables
--      (a964286_medicaid_ip_*). This is the ID we use to join features + outcomes.
--
--   2. individual_id in TE sequence tables (10-16 digits): A TRANSFORMED identifier
--      used internally by the TE pipeline (a834793_Medicaid_o3_train_ending).
--      This ID does NOT directly match asdb_member_key.
--
-- The 10% pretraining sample (a834793_Combined_All_LOB_o3_train_10pct_sample) 
-- uses individual_id that EQUALS asdb_member_key (8-9 digit format), which is
-- DIFFERENT from the transformed individual_id in member_train_ending.
--
-- SOLUTION: Use member_train_ending as a crosswalk table that maps between:
--   - member_id (= asdb_member_key, 8-9 digit format)
--   - individual_id (transformed 10-16 digit format used in TE sequences)
--
-- Without this crosswalk, direct matching yields <1% match rate. With the 
-- crosswalk, we achieve proper matching for downstream evaluation.
--
-- ID Mapping Chain (how the join works):
--   heldout.asdb_member_key 
--   → member_train_ending.member_id (= asdb_member_key, 8-9 digit)
--   → member_train_ending.individual_id (transformed, 10-16 digit)
--   → o3_train_ending.individual_id (TE sequences with same transformed ID)
--
-- NOTE: Commercial IP does NOT have this complexity because its individual_id
-- format is consistent across all tables. Medicaid requires extra care.
-- ============================================================================

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_heldout_te_inference_input`;

CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_heldout_te_inference_input`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_aetna_com"), ("cost_center", "13070")],
    description = "TE inference input for Medicaid IP heldout members (3.35M) - use to generate embeddings",
    expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
) AS
SELECT 
    -- Join key back to Medicaid IP features
    CAST(m.member_id AS INT64) AS asdb_member_key,
    
    -- TE identifiers
    o.individual_id,
    o.index_dt,
    
    -- TE input sequences (required for transformer inference)
    o.gender_cd,       -- "*"-separated gender sequence per day
    o.age_in_months,   -- "*"-separated age sequence per day  
    o.cd,              -- "*"-separated medical code sequences (input to transformer)
    o.dt_cnt           -- Number of days in sequence
    
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_o3_train_ending` o
INNER JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_member_train_ending` m
    ON CAST(o.individual_id AS STRING) = CAST(m.individual_id AS STRING)
-- Only include heldout members (not in 10% sample)
WHERE CAST(m.member_id AS STRING) IN (
    SELECT CAST(asdb_member_key AS STRING)
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_heldout_non_embedding_features`
);

-- ============================================================================
-- VERIFY: Check counts and data quality
-- ============================================================================

SELECT 
    'te_inference_input' AS table_name,
    COUNT(*) AS row_count,
    COUNT(DISTINCT asdb_member_key) AS unique_members,
    AVG(dt_cnt) AS avg_days_per_member,
    MIN(dt_cnt) AS min_days,
    MAX(dt_cnt) AS max_days
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_heldout_te_inference_input`;

-- Check for any members in heldout that DON'T have TE sequences
SELECT 
    'heldout_without_te_data' AS check_type,
    COUNT(*) AS members_missing_te_data
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_heldout_non_embedding_features` h
LEFT JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_heldout_te_inference_input` t
    ON h.asdb_member_key = t.asdb_member_key
WHERE t.asdb_member_key IS NULL;

"""




"""
-- ============================================================================
-- CREATE HELDOUT TABLES (EXCLUDING MEMBERS IN 10% PRETRAINING SAMPLE)
-- ============================================================================
-- Purpose: Create feature and outcome tables for members NOT used in TE 
--          pretraining, ensuring no data leakage in downstream evaluation.
--
-- ============================================================================
-- KEY INSIGHT: 10% Sample Uses asdb_member_key Format
-- ============================================================================
-- 
-- Unlike the o3_train_ending tables which use a transformed 10-16 digit 
-- individual_id, the 10% pretraining sample (a834793_Combined_All_LOB_o3_train_10pct_sample)
-- stores individual_id in the SAME format as asdb_member_key (8-9 digits).
--
-- This means for HELDOUT EXCLUSION, we can directly match:
--   feature_table.asdb_member_key = pretrain_sample.individual_id
--
-- This is different from the TE INFERENCE INPUT step above, which requires
-- the member_train_ending crosswalk because o3_train_ending uses transformed IDs.
--
-- Summary of when to use which matching:
--   - Excluding pretrain members: Direct match (asdb_member_key = individual_id)
--   - Linking to TE sequences: Use member_train_ending crosswalk
-- ============================================================================

-- Step 1: Verify the direct match works (should show ~10% of members)
SELECT 
    'Direct_Match_Test' AS test,
    COUNT(DISTINCT f.asdb_member_key) AS ip_members,
    COUNT(DISTINCT s.individual_id) AS matched_in_10pct
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_non_embedding_features` f
INNER JOIN `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_10pct_sample` s
    ON CAST(f.asdb_member_key AS STRING) = CAST(s.individual_id AS STRING)
WHERE s.lob = 'Medicaid';

-- ============================================================================
-- Step 2: Create heldout non-embedding features (excluding pretrain members)
-- ============================================================================

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_heldout_non_embedding_features`;

CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_heldout_non_embedding_features`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_aetna_com"), ("cost_center", "13070")],
    expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
) AS
SELECT f.*
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_non_embedding_features` f
WHERE CAST(f.asdb_member_key AS STRING) NOT IN (
    -- Match directly: 10% sample individual_id = asdb_member_key
    SELECT CAST(individual_id AS STRING)
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_10pct_sample`
    WHERE lob = 'Medicaid'
);

-- ============================================================================
-- Step 3: Create CORRECTED heldout outcome table
-- ============================================================================

DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_heldout_outcome_ip`;

CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_heldout_outcome_ip`
OPTIONS (
    labels = [("owner", "zhaopeng_xing_aetna_com"), ("cost_center", "13070")],
    expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
) AS
SELECT o.*
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_outcome_ip` o
WHERE CAST(o.asdb_member_key AS STRING) NOT IN (
    SELECT CAST(individual_id AS STRING)
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_10pct_sample`
    WHERE lob = 'Medicaid'
);

-- ============================================================================
-- Step 4: Verify counts
-- ============================================================================

SELECT 'heldout_features' AS table_name, COUNT(*) AS row_count, COUNT(DISTINCT asdb_member_key) AS unique_members
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_heldout_non_embedding_features`
UNION ALL
SELECT 'heldout_outcome', COUNT(*), COUNT(DISTINCT asdb_member_key)
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_medicaid_ip_heldout_outcome_ip`;

"""








# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import glob
import time
import copy
import warnings
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.base import clone
from catboost import CatBoostClassifier, Pool

# BigQuery
import google.auth
from google.cloud import bigquery

# PyTorch imports (for embedding generation)
import torch
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# IMPORT EMBEDDING GENERATION UTILITIES FROM COMMERCIAL MODULE
# =============================================================================
# These functions are shared between Commercial and Medicaid downstream pipelines.
# We import from the commercial module to avoid code duplication.
# See moe_flashattn_3_commercial_downstream.py for full implementation details.

try:
    from moe_flashattn_3_core import (
        # Configurations
        BaseConfig,
        FlashAttentionConfig,
        MoEConfig,
        
        # Models
        BaselineTransformer,
        FlashAttentionTransformer,
        FlashMoETransformer,
        
        # Data utilities
        create_collate_fn,
        conv_cd,
        conv_age_gender,
        conv_lob,
        conv_target,
        
        # Embedding extraction
        EmbeddingExtractor,
        
        # GPU utilities
        cleanup_gpu_memory,
    )
    CORE_MODULE_AVAILABLE = True
except ImportError:
    CORE_MODULE_AVAILABLE = False
    print("⚠️  Warning: moe_flashattn_3_core not found. Embedding generation will not be available.")
    print("   To use embedding generation, ensure moe_flashattn_3_core.py is in the same directory.")

# Import embedding generation functions from commercial module
# These are generic and can be reused for Medicaid with minor adaptations
try:
    from moe_flashattn_3_commercial_downstream import (
        load_model_from_checkpoint,
        LazyClinicalDataset,
        save_embeddings,
        save_embeddings_to_bigquery,
    )
    COMMERCIAL_MODULE_AVAILABLE = True
except ImportError:
    COMMERCIAL_MODULE_AVAILABLE = False
    print("⚠️  Warning: moe_flashattn_3_commercial_downstream not found.")
    print("   Embedding generation functions will be defined locally.")


# =============================================================================
# CONFIGURATION - MEDICAID IP MODEL PARAMETERS
# =============================================================================
# These parameters replicate the original Medicaid IP model exactly

# BigQuery Tables - CORRECTED
PROJECT_ID = "edp-prod-storage"
DATASET_ID = "edp_ent_sdoheir_cns"

# =============================================================================
# HELDOUT TABLES (members NOT in TE pretraining 10% sample)
# These are used for downstream evaluation to avoid data leakage
# =============================================================================
HELDOUT_FEATURES_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a964286_medicaid_ip_heldout_non_embedding_features"
HELDOUT_OUTCOME_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a964286_medicaid_ip_heldout_outcome_ip"
HELDOUT_TE_INPUT_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a964286_medicaid_ip_heldout_te_inference_input"

# =============================================================================
# FULL DATASET TABLES (all members, including those in pretrain)
# Only use these for reference/comparison, NOT for downstream evaluation
# =============================================================================
FULL_FEATURES_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_non_embedding_features"
FULL_OUTCOME_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a964286_medicaid_ip_final_dataset_4_te_experiment_2023_outcome_ip"

# =============================================================================
# TE CROSSWALK TABLES (for ID mapping between formats)
# =============================================================================
MEMBER_CROSSWALK_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a834793_Medicaid_member_train_ending"
TE_SEQUENCE_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a834793_Medicaid_o3_train_ending"
PRETRAIN_10PCT_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a834793_Combined_All_LOB_o3_train_10pct_sample"

# Legacy table references (from Eric Ma's original pipeline - for reference only)
# These are from the anbc-hcb-dev project, not used in current downstream eval
# LEGACY_FEATURES_TABLE = "anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_non_embedding_features"
# LEGACY_EMBEDDINGS_TABLE = "anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_embeddings"
# LEGACY_OUTCOME_TABLE = "anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_outcome_ip"

# Default to HELDOUT tables for downstream evaluation
FEATURES_TABLE = HELDOUT_FEATURES_TABLE
OUTCOME_TABLE = HELDOUT_OUTCOME_TABLE

# Note: Embeddings will come from transformer inference on HELDOUT_TE_INPUT_TABLE
# No pre-computed embeddings table for heldout - we generate them fresh

# Target variable
TARGET_COLUMN = "acute_ip_flag"

# Member ID column (primary key for joining)
MEMBER_KEY = "asdb_member_key"

# Random seeds for reproducibility (matches original Eric Ma pipeline)
RANDOM_STATE = 35  # For train/test split (matches Eric's train_test_split random_state)
UNDERSAMPLE_RANDOM_STATE = 53  # For undersampling (Eric uses 53 for RandomUnderSampler)
CATBOOST_RANDOM_SEED = 53  # For CatBoost model (Eric uses 53 for random_seed)

# Class imbalance handling
# Original finding: CatBoost works better with 0.2 undersampling ratio
CATBOOST_UNDERSAMPLE_RATIO = 0.2  # 20% minority-to-majority (5:1)
XGBOOST_UNDERSAMPLE_RATIO = 0.03  # 3% minority-to-majority (~33:1)

# Train/Val/Test split ratios (original: 80/10/10)
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1

# =============================================================================
# OUT-OF-TIME (OOT) VALIDATION CONFIGURATION
# =============================================================================
# For time-based train/test split similar to commercial IP:
# - Data before cutoff: used for train/val/test (stratified random split)
# - Data after cutoff: used for OOT (out-of-time) validation
# This tests temporal generalization of the model.
#
# Note: Commercial IP uses ind_id_last_digit for deterministic splitting.
# Medicaid doesn't have this column, so we use stratified random split
# for train/val/test, matching Eric's original Medicaid IP pipeline.
OOT_CUTOFF_DATE = "2023-10-16"  # Same as commercial IP for consistency

# Sampling fraction for efficient embedding generation (optional)
# Set to None to use full data, or 0.3 for 30% sample like commercial
EMBEDDING_SAMPLE_FRAC = None  # Use full data for Medicaid heldout (already ~90% of total)

# =============================================================================
# CATBOOST TUNED HYPERPARAMETERS
# =============================================================================
# From Optuna optimization in original pipeline (optuna_results_catboost.csv)
# Best trial 42: AUC = 0.8737 (from optuna_catboost.log)
# Note: Eric's final model uses params from lines 513-521 of catboost.py
CATBOOST_TUNED_PARAMS = {
    'learning_rate': 0.015742881221129403,
    'iterations': 2665,
    'l2_leaf_reg': 0.222046549398224,
    'depth': 7,
    'random_seed': CATBOOST_RANDOM_SEED,  # Eric uses 53 for CatBoost
    'verbose': 0,
    'thread_count': -1,  # Use all available threads (-1), Eric used 15
    'use_best_model': True,
    # Note: Original didn't use auto_class_weights, relied on undersampling instead
}

# Alternative: Balanced class weights model (without undersampling)
CATBOOST_BALANCED_PARAMS = {
    'iterations': 2500,
    'depth': 7,
    'learning_rate': 0.025,
    'grow_policy': 'SymmetricTree',
    'auto_class_weights': 'Balanced',
    'od_wait': 80,
    'use_best_model': True,
    'random_seed': CATBOOST_RANDOM_SEED,  # Use same seed as tuned params
    'verbose': 0,
    'thread_count': -1,
}

# =============================================================================
# SELECTED FEATURES FROM RFECV
# =============================================================================
# These 243 non-embedding features were selected by RFECV in the original pipeline
# The full list (499) includes these + 256 embedding features (emb0-emb255)

SELECTED_TABULAR_FEATURES = [
    # COA Population Group (categorical)
    'coa_population_group',
    
    # ED Visits - Year 1
    'sum_ed_visits_yr1', 'ed_flag_yr1', 'sum_avoidable_yr1', 'sum_unnecessary_yr1',
    'sum_preventable_yr1', 'low_sev_ed_visits_yr1', 'low_med_sev_ed_visits_yr1',
    'med_sev_ed_visits_yr1', 'med_high_sev_ed_visits_yr1', 'high_sev_ed_visits_yr1',
    'high_sev_ed_flag_yr1',
    
    # ED Visits - Year 2
    'sum_ed_visits_yr2', 'sum_avoidable_yr2', 'sum_preventable_yr2',
    'med_sev_ed_visits_yr2', 'med_high_sev_ed_visits_yr2', 'high_sev_ed_visits_yr2',
    
    # IP Admits
    'sum_acute_ip_admits_yr1', 'sum_acute_calc_los_yr1',
    'sum_acute_ip_admits_yr2', 'sum_acute_calc_los_yr2',
    
    # OP Visits
    'sum_op_visits_yr1', 'sum_op_visits_yr2',
    
    # EMIS Claims - Year 1
    'emis_community_clm_yr1', 'emis_ed_clm_yr1', 'emis_hh_clm_yr1', 'emis_home_clm_yr1',
    'emis_ip_clm_yr1', 'emis_ins_clm_yr1', 'emis_lab_clm_yr1', 'emis_mrx_clm_yr1',
    'emis_mh_clm_yr1', 'emis_misc_clm_yr1', 'emis_pcp_clm_yr1', 'emis_radio_clm_yr1',
    'emis_ambul_clm_yr1', 'emis_spec_clm_yr1',
    
    # LTC and COE Claims - Year 1
    'ltc_clm_yr1', 'coe_ip_hos_clm_yr1', 'coe_ip_non_hos_clm_yr1', 'coe_lab_clm_yr1',
    'coe_ltc_community_clm_yr1', 'coe_ltc_home_clm_yr1', 'coe_ltc_ins_clm_yr1',
    'coe_other_clm_yr1', 'coe_op_hos_clm_yr1', 'coe_op_non_hos_clm_yr1',
    'coe_anesth_clm_yr1', 'coe_eval_clm_yr1', 'coe_maternity_clm_yr1',
    'coe_mrx_clm_yr1', 'coe_mh_clm_yr1', 'coe_phy_clm_yr1', 'coe_surg_clm_yr1',
    'coe_radio_clm_yr1', 'uc_clm_yr1', 'obs_clm_yr1',
    
    # EMIS Claims - Year 2
    'emis_community_clm_yr2', 'emis_ed_clm_yr2', 'emis_hh_clm_yr2', 'emis_home_clm_yr2',
    'emis_ip_clm_yr2', 'emis_ins_clm_yr2', 'emis_lab_clm_yr2', 'emis_mrx_clm_yr2',
    'emis_mh_clm_yr2', 'emis_misc_clm_yr2', 'emis_pcp_clm_yr2', 'emis_radio_clm_yr2',
    'emis_ambul_clm_yr2', 'emis_spec_clm_yr2',
    
    # LTC and COE Claims - Year 2
    'ltc_clm_yr2', 'coe_ip_hos_clm_yr2', 'coe_ip_non_hos_clm_yr2', 'coe_lab_clm_yr2',
    'coe_other_clm_yr2', 'coe_op_hos_clm_yr2', 'coe_op_non_hos_clm_yr2',
    'coe_anesth_clm_yr2', 'coe_eval_clm_yr2', 'coe_maternity_clm_yr2',
    'coe_mrx_clm_yr2', 'coe_mh_clm_yr2', 'coe_phy_clm_yr2', 'coe_surg_clm_yr2',
    'coe_radio_clm_yr2', 'uc_clm_yr2', 'obs_clm_yr2',
    
    # Chronic Conditions (binary flags)
    'IDA', 'ANX', 'OST', 'AST', 'CHO', 'burns', 'CBD', 'CHF', 'CRF', 'CHD',
    'COP', 'DIA', 'esrd', 'EPL', 'CRO', 'MOH', 'HepC', 'HYP', 'HYC',
    'meta_cancer', 'liver_dis', 'MSS', 'OBE', 'oud', 'paralysis', 'hmd',
    'PVD', 'autoimmune', 'SCA', 'spinal_inj', 'back', 'substance', 'ALC', 'psychoses',
    'major_chronic_cnt',
    
    # Pharmacy Features - Year 1
    'rx_claim_cnt_yr1', 'days_supply_sum_yr1', 'ndc_cnt_yr1', 'gpi_cnt_yr1',
    'gpi4_cnt_yr1', 'gpi2_cnt_yr1', 'retail_fills_yr1', 'mail_order_fills_yr1',
    'generic_fills_yr1', 'branded_generic_fills_yr1', 'ss_brand_fills_yr1',
    'ms_brand_fills_yr1', 'formulary_fills_yr1', 'maint_drug_fills_yr1',
    'antidiabetic_scripts_yr1', 'antidiabetic_days_supply_yr1',
    'beta_blocker_scripts_yr1', 'beta_blocker_days_supply_yr1',
    'antihypertensive_scripts_yr1', 'antihypertensive_days_supply_yr1',
    'lipid_lowering_scripts_yr1', 'lipid_lowering_days_supply_yr1',
    'calcium_channel_blk_scripts_yr1', 'calcium_channel_blk_days_supply_yr1',
    'diuretic_scripts_yr1', 'diuretic_days_supply_yr1',
    'antianginal_agent_scripts_yr1', 'antianginal_agent_days_supply_yr1',
    'antidepressant_scripts_yr1', 'antidepressant_days_supply_yr1',
    'antipsychotic_scripts_yr1', 'antipsychotic_days_supply_yr1',
    'antianxiety_days_supply_yr1', 'anticonvulsant_scripts_yr1',
    'anticonvulsant_days_supply_yr1', 'inhaled_steroid_scripts_yr1',
    'inhaled_steroid_days_supply_yr1',
    
    # Pharmacy Features - Year 2
    'rx_claim_cnt_yr2', 'days_supply_sum_yr2', 'ndc_cnt_yr2', 'gpi_cnt_yr2',
    'gpi4_cnt_yr2', 'gpi2_cnt_yr2', 'retail_fills_yr2', 'generic_fills_yr2',
    'branded_generic_fills_yr2', 'ss_brand_fills_yr2', 'ms_brand_fills_yr2',
    'formulary_fills_yr2', 'maint_drug_fills_yr2',
    'antidiabetic_scripts_yr2', 'antidiabetic_days_supply_yr2',
    'beta_blocker_scripts_yr2', 'beta_blocker_days_supply_yr2',
    'antihypertensive_scripts_yr2', 'antihypertensive_days_supply_yr2',
    'lipid_lowering_scripts_yr2', 'lipid_lowering_days_supply_yr2',
    'calcium_channel_blk_scripts_yr2', 'calcium_channel_blk_days_supply_yr2',
    'diuretic_days_supply_yr2', 'antianginal_agent_scripts_yr2',
    'antidepressant_scripts_yr2', 'antidepressant_days_supply_yr2',
    'antipsychotic_scripts_yr2', 'antipsychotic_days_supply_yr2',
    'antianxiety_scripts_yr2', 'antianxiety_days_supply_yr2',
    'anticonvulsant_scripts_yr2', 'anticonvulsant_days_supply_yr2',
    'inhaled_steroid_days_supply_yr2',
    
    # Demographics
    'agenbr', 'gender', 'ethnicity_code', 'primarylanguage_desc',
    'tenure_yr1', 'tenure_yr2', 'urbsubr',
    
    # SDOH Scores
    'zip_weight_avg_medinc', 'acs_social_risk_score', 'sdi_score', 'svi_score',
    'adi_score', 'citizenship_index', 'education_index', 'food_access',
    'health_access', 'health_habits', 'housing_desert', 'housing_ownership',
    'housing_quality', 'income_index', 'income_inequality', 'language_score',
    'natural_disaster', 'poverty_score', 'proactive_health', 'racial_diversity',
    'social_isolation', 'technology_access', 'transport_access',
    'unemployment_index', 'water_quality', 'disability_score', 'health_infra',
    'csdi_social_risk_score',
    
    # Healthcare Utilization
    'sum_pcp', 'sum_spec', 'sum_ob', 'sum_dme', 'sum_chol_lab', 'sum_a1c_lab',
    'sum_chemo',
    
    # CMS Screening Flags
    'cms_alc_scrn', 'cms_col_scrn', 'cms_hepb_scrn', 'cms_nutrition',
    'cms_sti_scrn', 'cms_mam_scrn',
]

# Embedding features (256 dimensions)
EMBEDDING_FEATURES = [f'emb{i}' for i in range(256)]

# Categorical features requiring special handling
CATEGORICAL_FEATURES = [
    'coa_population_group', 'gender', 'ethnicity_code',
    'primarylanguage_desc', 'urbsubr',
    'cms_alc_scrn', 'cms_col_scrn', 'cms_hepb_scrn', 'cms_nutrition',
    'cms_sti_scrn', 'cms_mam_scrn',
]


# =============================================================================
# EVALUATION METRICS FUNCTIONS
# =============================================================================
# These match the original Medicaid IP model evaluation exactly

def lift_at_percentage(y_true: np.ndarray, y_prob: np.ndarray, pct: float) -> float:
    """
    Calculate lift at top percentile.
    
    Lift = precision@k / baseline_prevalence
    
    This is the primary metric for the Medicaid IP model, measuring how
    many times better the model is at identifying positives in the top k%.
    
    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities
        pct: Percentile (0.01 for 1%, 0.10 for 10%)
        
    Returns:
        Lift value (e.g., 20 means 20x better than random)
    """
    n = len(y_true)
    k = max(1, int(n * pct))
    top_k_indices = np.argsort(y_prob)[::-1][:k]
    precision_at_k = y_true[top_k_indices].mean()
    baseline = y_true.mean()
    return precision_at_k / baseline if baseline > 0 else 0.0


def true_positives_at_percentage(y_true: np.ndarray, y_prob: np.ndarray, pct: float) -> int:
    """Count true positives in top percentile."""
    n = len(y_true)
    k = max(1, int(n * pct))
    top_k_indices = np.argsort(y_prob)[::-1][:k]
    return int(y_true[top_k_indices].sum())


def precision_at_percentage(y_true: np.ndarray, y_prob: np.ndarray, pct: float) -> float:
    """Calculate precision at top percentile (PPV@k)."""
    n = len(y_true)
    k = max(1, int(n * pct))
    top_k_indices = np.argsort(y_prob)[::-1][:k]
    return float(y_true[top_k_indices].mean())


def sensitivity_at_percentage(y_true: np.ndarray, y_prob: np.ndarray, pct: float) -> float:
    """
    Calculate sensitivity at top percentile.
    
    Sensitivity = TP / (TP + FN) for members in top k%
    """
    n = len(y_true)
    k = max(1, int(n * pct))
    top_k_indices = np.argsort(y_prob)[::-1][:k]
    
    # Binary prediction: 1 for top k%, 0 for rest
    y_pred_binary = np.zeros(n, dtype=int)
    y_pred_binary[top_k_indices] = 1
    
    tp = (y_true[top_k_indices] == 1).sum()
    fn = ((y_true == 1) & (y_pred_binary == 0)).sum()
    
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def compute_split_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a single split.
    
    This function replicates the exact metrics from the original Medicaid IP model:
    - ROC-AUC: Overall discrimination ability
    - AUC-PR: Precision-Recall AUC (important for imbalanced data)
    - Brier Score: Calibration metric
    - Lift@1%, Lift@5%, Lift@10%: Key business metrics
    - PPV@1%, PPV@10%: Precision at top percentiles
    - Sensitivity@1%, Sensitivity@10%: Recall at top percentiles
    - TP@1%: True positives captured in top 1%
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        
    Returns:
        Dict with metric names as keys
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    return {
        # Discrimination metrics
        'auc_roc': roc_auc_score(y_true, y_prob),
        'auc_pr': average_precision_score(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob),
        
        # Lift metrics (primary business metrics)
        'lift_1pct': lift_at_percentage(y_true, y_prob, 0.01),
        'lift_5pct': lift_at_percentage(y_true, y_prob, 0.05),
        'lift_10pct': lift_at_percentage(y_true, y_prob, 0.10),
        
        # PPV (Precision) at percentiles
        'ppv_1pct': precision_at_percentage(y_true, y_prob, 0.01) * 100,
        'ppv_10pct': precision_at_percentage(y_true, y_prob, 0.10) * 100,
        
        # Sensitivity (Recall) at percentiles
        'sensitivity_1pct': sensitivity_at_percentage(y_true, y_prob, 0.01) * 100,
        'sensitivity_10pct': sensitivity_at_percentage(y_true, y_prob, 0.10) * 100,
        
        # True positives captured
        'tp_1pct': true_positives_at_percentage(y_true, y_prob, 0.01),
        
        # Sample info
        'n_samples': len(y_true),
        'n_positives': int(y_true.sum()),
        'prevalence': float(y_true.mean()),
    }


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_medicaid_heldout_data(
    sample_frac: Optional[float] = None,
    random_state: int = RANDOM_STATE,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load HELDOUT Medicaid IP data from BigQuery for downstream evaluation.
    
    This function loads from the HELDOUT tables which contain members
    NOT in the TE pretraining 10% sample, ensuring no data leakage.
    
    Tables loaded:
    - HELDOUT_FEATURES_TABLE: Non-embedding features (already filtered)
    - HELDOUT_OUTCOME_TABLE: Target variable (acute_ip_flag)
    
    Note: Embeddings are NOT loaded here - they must be generated via
    transformer inference on HELDOUT_TE_INPUT_TABLE and merged separately.
    
    Args:
        sample_frac: Optional sampling fraction for testing (e.g., 0.1 for 10%)
        random_state: Random seed for sampling
        verbose: Print progress information
        
    Returns:
        DataFrame with features and outcome (no embeddings)
    """
    client = bigquery.Client()
    
    if verbose:
        print(f"\n{'='*70}")
        print("LOADING MEDICAID IP HELDOUT DATA FROM BIGQUERY")
        print(f"{'='*70}")
        print(f"Features table: {HELDOUT_FEATURES_TABLE}")
        print(f"Outcome table: {HELDOUT_OUTCOME_TABLE}")
    
    # Step 1: Load heldout non-embedding features (already filtered)
    if verbose:
        print("\n[Step 1/3] Loading heldout non-embedding features...")
    
    features_sql = f"""
    SELECT *
    FROM `{HELDOUT_FEATURES_TABLE}`
    """
    
    df_features = client.query(features_sql).to_dataframe()
    if verbose:
        print(f"  Features loaded: {len(df_features):,} rows, {len(df_features.columns)} columns")
    
    # Step 2: Load heldout outcomes
    if verbose:
        print("\n[Step 2/3] Loading heldout outcomes...")
    
    outcomes_sql = f"""
    SELECT
        asdb_member_key,
        acute_ip_flag
    FROM `{HELDOUT_OUTCOME_TABLE}`
    """
    
    df_outcomes = client.query(outcomes_sql).to_dataframe()
    if verbose:
        print(f"  Outcomes loaded: {len(df_outcomes):,} rows")
        print(f"  Positive rate: {df_outcomes[TARGET_COLUMN].mean()*100:.2f}%")
    
    # Step 3: Merge features with outcomes
    if verbose:
        print("\n[Step 3/3] Merging features and outcomes...")
    
    df_features = df_features.set_index(MEMBER_KEY)
    df_outcomes = df_outcomes.set_index(MEMBER_KEY)
    
    df_merged = df_features.merge(df_outcomes, left_index=True, right_index=True, how='inner')
    df_merged = df_merged.reset_index()
    
    if verbose:
        print(f"  Merged dataset: {len(df_merged):,} rows, {len(df_merged.columns)} columns")
    
    # Optional sampling for testing
    if sample_frac is not None:
        if verbose:
            print(f"\n  Sampling {sample_frac*100:.0f}% of data...")
        df_merged = df_merged.sample(frac=sample_frac, random_state=random_state)
        if verbose:
            print(f"  Sampled dataset: {len(df_merged):,} rows")
    
    if verbose:
        print(f"\n✅ Data loading complete!")
        print(f"   Final shape: {df_merged.shape}")
        print(f"   Positive rate: {df_merged[TARGET_COLUMN].mean()*100:.2f}%")
        print(f"   ⚠️  Note: Embeddings NOT included - merge from NPZ after inference")
        print(f"{'='*70}\n")
    
    return df_merged


def load_te_inference_input(
    sample_frac: Optional[float] = None,
    random_state: int = RANDOM_STATE,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load TE inference input data for generating embeddings for heldout members.
    
    This table contains the raw TE sequences (cd, gender_cd, age_in_months)
    needed to run transformer inference and generate embeddings.
    
    Returns:
        DataFrame with columns: asdb_member_key, individual_id, index_dt,
        gender_cd, age_in_months, cd, dt_cnt
    """
    client = bigquery.Client()
    
    if verbose:
        print(f"\n{'='*70}")
        print("LOADING TE INFERENCE INPUT FOR HELDOUT MEMBERS")
        print(f"{'='*70}")
        print(f"Table: {HELDOUT_TE_INPUT_TABLE}")
    
    sql = f"""
    SELECT *
    FROM `{HELDOUT_TE_INPUT_TABLE}`
    """
    
    df = client.query(sql).to_dataframe()
    
    if verbose:
        print(f"  Loaded: {len(df):,} rows")
        print(f"  Unique members: {df[MEMBER_KEY].nunique():,}")
        print(f"  Avg sequence days: {df['dt_cnt'].mean():.1f}")
    
    # Optional sampling
    if sample_frac is not None:
        if verbose:
            print(f"\n  Sampling {sample_frac*100:.0f}% of data...")
        df = df.sample(frac=sample_frac, random_state=random_state)
        if verbose:
            print(f"  Sampled: {len(df):,} rows")
    
    if verbose:
        print(f"{'='*70}\n")
    
    return df


# Legacy function for backward compatibility with original Eric Ma pipeline
def load_medicaid_data_from_bigquery(
    sample_frac: Optional[float] = None,
    random_state: int = RANDOM_STATE,
    verbose: bool = True
) -> pd.DataFrame:
    """
    DEPRECATED: Use load_medicaid_heldout_data() for downstream evaluation.
    
    This function is kept for backward compatibility but now loads from
    heldout tables. For the full pipeline, use load_medicaid_heldout_data()
    and merge embeddings separately.
    """
    if verbose:
        print("⚠️  Note: load_medicaid_data_from_bigquery() now loads from HELDOUT tables")
        print("   For new embeddings, use load_medicaid_heldout_data() + merge_new_embeddings_with_features()")
    
    return load_medicaid_heldout_data(
        sample_frac=sample_frac,
        random_state=random_state,
        verbose=verbose
    )


def preprocess_features(
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Preprocess features replicating the original Medicaid IP pipeline.
    
    Preprocessing steps:
    1. Fill embedding columns (emb0-emb255) with 0
    2. Fill numeric columns with 0
    3. Fill string/categorical columns with empty string
    4. Encode gender: M -> 1, F -> 0, other -> -1
    
    Note: CatBoost handles categorical features natively, so we don't
    need to one-hot encode them. We just ensure proper types.
    
    Args:
        df: Raw DataFrame
        verbose: Print progress
        
    Returns:
        Preprocessed DataFrame
    """
    import re
    from pandas.api.types import is_integer_dtype, is_float_dtype
    
    df = df.copy()
    
    if verbose:
        print("Preprocessing features...")
    
    # Step 1: Fill embedding columns with 0
    emb_pattern = r'^emb\d+$'
    emb_cols = [col for col in df.columns if re.match(emb_pattern, col)]
    if emb_cols:
        df[emb_cols] = df[emb_cols].fillna(0)
        if verbose:
            print(f"  Filled {len(emb_cols)} embedding columns with 0")
    
    # Step 2: Fill numeric columns with 0, string columns with ''
    numeric_filled = 0
    string_filled = 0
    
    for col in df.columns:
        if col in emb_cols or col == TARGET_COLUMN or col == MEMBER_KEY:
            continue
            
        if is_integer_dtype(df[col]) or is_float_dtype(df[col]):
            df[col] = df[col].fillna(0)
            numeric_filled += 1
        else:
            try:
                df[col] = df[col].fillna('')
                string_filled += 1
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not process column {col}: {e}")
    
    if verbose:
        print(f"  Filled {numeric_filled} numeric columns with 0")
        print(f"  Filled {string_filled} string columns with ''")
    
    # Step 3: Encode gender (matches original)
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'M': 1, 'F': 0}).fillna(-1).astype(int)
        if verbose:
            print("  Encoded gender: M->1, F->0, other->-1")
    
    if verbose:
        print("✅ Preprocessing complete!")
    
    return df


def downsample_negatives(
    X: pd.DataFrame,
    y: pd.Series,
    ratio: float = CATBOOST_UNDERSAMPLE_RATIO,
    random_state: int = UNDERSAMPLE_RANDOM_STATE  # Eric uses 53 for undersampling
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Downsample negative class to achieve target ratio.
    
    The original Medicaid IP model used undersampling as the primary
    class imbalance strategy. CatBoost worked best with 0.2 ratio
    (20% minority, meaning 5:1 negative-to-positive ratio).
    
    Args:
        X: Feature DataFrame
        y: Target Series
        ratio: Minority class ratio (e.g., 0.2 for 5:1)
        random_state: Random seed
        
    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    np.random.seed(random_state)
    
    pos_mask = y == 1
    neg_mask = y == 0
    
    pos_indices = X.index[pos_mask].tolist()
    neg_indices = X.index[neg_mask].tolist()
    
    n_positives = len(pos_indices)
    n_negatives = len(neg_indices)
    
    # Calculate target number of negatives based on ratio
    # ratio = n_positives / (n_positives + n_negatives_target)
    # Solving: n_negatives_target = n_positives * (1 - ratio) / ratio
    target_n_negatives = int(n_positives * (1 - ratio) / ratio)
    
    if n_negatives <= target_n_negatives:
        print(f"  Downsampling: No action needed (current ratio: {n_positives/(n_positives+n_negatives):.3f})")
        return X, y
    
    # Randomly sample negatives
    sampled_neg_indices = np.random.choice(neg_indices, size=target_n_negatives, replace=False)
    keep_indices = pos_indices + sampled_neg_indices.tolist()
    
    X_resampled = X.loc[keep_indices].copy()
    y_resampled = y.loc[keep_indices].copy()
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_resampled))
    X_resampled = X_resampled.iloc[shuffle_idx].reset_index(drop=True)
    y_resampled = y_resampled.iloc[shuffle_idx].reset_index(drop=True)
    
    new_ratio = y_resampled.sum() / len(y_resampled)
    print(f"  Downsampling: {n_negatives}:{n_positives} -> "
          f"{target_n_negatives}:{n_positives} (ratio: {new_ratio:.3f})")
    
    return X_resampled, y_resampled


# =============================================================================
# DATA PREPARATION
# =============================================================================

@dataclass
class MedicaidPreparedData:
    """
    Container for prepared Medicaid IP evaluation data.
    
    Prepare data once using prepare_medicaid_evaluation_data(), 
    then evaluate multiple models using evaluate_with_prepared_data().
    """
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    feature_cols: List[str]
    embedding_features: List[str]
    tabular_features: List[str]
    cat_feature_indices: List[int]
    feature_set: str
    downsampled: bool
    original_train_size: int
    

def create_train_val_test_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Create stratified train/validation/test splits.
    
    Replicates the original Medicaid IP model split strategy:
    - 80% train, 10% validation, 10% test
    - Stratified by target variable to preserve class distribution
    
    Args:
        df: Full DataFrame with features and target
        test_size: Fraction for test set (default 0.1)
        val_size: Fraction for validation set (default 0.1)
        random_state: Random seed
        verbose: Print info
        
    Returns:
        Dict with 'train', 'val', 'test' DataFrames
    """
    if verbose:
        print("\nCreating stratified train/val/test splits...")
    
    # First split: train+val vs test
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN]
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=train_val[TARGET_COLUMN]
    )
    
    splits = {
        'train': train.reset_index(drop=True),
        'val': val.reset_index(drop=True),
        'test': test.reset_index(drop=True),
    }
    
    if verbose:
        for name, split_df in splits.items():
            prevalence = split_df[TARGET_COLUMN].mean() * 100
            print(f"  {name}: {len(split_df):,} rows, "
                  f"{int(split_df[TARGET_COLUMN].sum()):,} positives ({prevalence:.2f}%)")
    
    return splits


def prepare_medicaid_evaluation_data(
    df: pd.DataFrame,
    feature_set: str = 'hybrid',
    apply_downsampling: bool = True,
    downsample_ratio: float = CATBOOST_UNDERSAMPLE_RATIO,
    split_random_state: int = RANDOM_STATE,
    undersample_random_state: int = UNDERSAMPLE_RANDOM_STATE,
    verbose: bool = True
) -> MedicaidPreparedData:
    """
    Prepare Medicaid IP data for model evaluation.
    
    This function encapsulates the complete data preparation pipeline
    from the original Medicaid IP model:
    1. Preprocessing (missing values, encoding)
    2. Train/val/test splitting (stratified)
    3. Feature selection based on feature_set
    4. Optional downsampling of training set
    
    Args:
        df: Raw Medicaid IP DataFrame
        feature_set: One of 'embedding_only', 'tabular_only', 'hybrid'
        apply_downsampling: Whether to downsample training set
        downsample_ratio: Undersampling ratio (default 0.2 for CatBoost)
        split_random_state: Random seed for train/test split (default 35, Eric's)
        undersample_random_state: Random seed for undersampling (default 53, Eric's)
        verbose: Print progress
        
    Returns:
        MedicaidPreparedData object ready for model evaluation
    """
    start_time = time.time()
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"PREPARING MEDICAID IP DATA FOR EVALUATION")
        print(f"{'='*70}")
        print(f"Feature set: {feature_set}")
        print(f"Downsampling: {apply_downsampling} (ratio: {downsample_ratio})")
    
    # Validate feature_set
    valid_feature_sets = {'embedding_only', 'tabular_only', 'hybrid'}
    if feature_set not in valid_feature_sets:
        raise ValueError(f"feature_set must be one of {valid_feature_sets}")
    
    # Step 1: Preprocess
    if verbose:
        print("\n[Step 1/4] Preprocessing features...")
    df_processed = preprocess_features(df, verbose=verbose)
    
    # Step 2: Split data (using split_random_state=35 to match Eric's train_test_split)
    if verbose:
        print("\n[Step 2/4] Creating train/val/test splits...")
    splits = create_train_val_test_split(df_processed, random_state=split_random_state, verbose=verbose)
    
    # Step 3: Select features based on feature_set
    if verbose:
        print(f"\n[Step 3/4] Selecting features for '{feature_set}'...")
    
    # Identify available features
    available_tabular = [f for f in SELECTED_TABULAR_FEATURES if f in df_processed.columns]
    available_embedding = [f for f in EMBEDDING_FEATURES if f in df_processed.columns]
    
    if verbose:
        print(f"  Available tabular features: {len(available_tabular)}")
        print(f"  Available embedding features: {len(available_embedding)}")
    
    if feature_set == 'embedding_only':
        feature_cols = available_embedding
    elif feature_set == 'tabular_only':
        feature_cols = available_tabular
    else:  # hybrid
        feature_cols = available_tabular + available_embedding
    
    if verbose:
        print(f"  Selected features: {len(feature_cols)}")
    
    # Prepare X and y for each split
    X_train = splits['train'][feature_cols].copy()
    X_val = splits['val'][feature_cols].copy()
    X_test = splits['test'][feature_cols].copy()
    y_train = splits['train'][TARGET_COLUMN].astype(int)
    y_val = splits['val'][TARGET_COLUMN].astype(int)
    y_test = splits['test'][TARGET_COLUMN].astype(int)
    
    original_train_size = len(X_train)
    
    # Step 4: Apply downsampling to training set (using undersample_random_state=53 to match Eric's)
    downsampled = False
    if apply_downsampling:
        if verbose:
            print(f"\n[Step 4/4] Applying downsampling to training set...")
        X_train, y_train = downsample_negatives(
            X_train, y_train, 
            ratio=downsample_ratio, 
            random_state=undersample_random_state  # Eric uses 53 for undersampling
        )
        downsampled = True
    else:
        if verbose:
            print(f"\n[Step 4/4] Skipping downsampling...")
    
    # Identify categorical columns for CatBoost
    cat_feature_indices = []
    if feature_set != 'embedding_only':
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
        cat_feature_indices = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
        if verbose:
            print(f"  Categorical features for CatBoost: {len(cat_feature_indices)}")
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\n✅ Data preparation complete! ({elapsed:.1f}s)")
        print(f"   Train: {len(X_train):,} samples ({y_train.sum():,} positives)")
        print(f"   Val: {len(X_val):,} samples ({y_val.sum():,} positives)")
        print(f"   Test: {len(X_test):,} samples ({y_test.sum():,} positives)")
        print(f"{'='*70}\n")
    
    return MedicaidPreparedData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_cols=feature_cols,
        embedding_features=available_embedding,
        tabular_features=available_tabular,
        cat_feature_indices=cat_feature_indices,
        feature_set=feature_set,
        downsampled=downsampled,
        original_train_size=original_train_size,
    )


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model_on_splits(
    model: Any,
    prepared_data: MedicaidPreparedData,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Train model on training set and evaluate on all splits.
    
    Args:
        model: CatBoostClassifier or compatible model
        prepared_data: MedicaidPreparedData from prepare_medicaid_evaluation_data()
        verbose: Print progress
        
    Returns:
        Dict with split names as keys, metrics dict as values
    """
    start_time = time.time()
    
    # Clone model to avoid modifying original
    model = clone(model)
    model_type = type(model).__name__
    
    if verbose:
        print(f"\nTraining {model_type}...")
    
    # Train model
    if model_type == 'CatBoostClassifier':
        train_pool = Pool(
            prepared_data.X_train, 
            prepared_data.y_train,
            cat_features=prepared_data.cat_feature_indices if prepared_data.cat_feature_indices else None
        )
        val_pool = Pool(
            prepared_data.X_val,
            prepared_data.y_val,
            cat_features=prepared_data.cat_feature_indices if prepared_data.cat_feature_indices else None
        )
        model.fit(train_pool, eval_set=val_pool, verbose=0)
    else:
        model.fit(prepared_data.X_train, prepared_data.y_train)
    
    train_time = time.time() - start_time
    if verbose:
        print(f"  Training completed in {train_time:.1f}s")
    
    # Evaluate on all splits
    results = {}
    splits_data = {
        'train': (prepared_data.X_train, prepared_data.y_train),
        'val': (prepared_data.X_val, prepared_data.y_val),
        'test': (prepared_data.X_test, prepared_data.y_test),
    }
    
    for split_name, (X_split, y_split) in splits_data.items():
        if verbose:
            print(f"  Evaluating on {split_name}...")
        
        # Predict probabilities
        if model_type == 'CatBoostClassifier' and prepared_data.cat_feature_indices:
            pool = Pool(X_split, cat_features=prepared_data.cat_feature_indices)
            y_prob = model.predict_proba(pool)[:, 1]
        else:
            y_prob = model.predict_proba(X_split)[:, 1]
        
        # Compute metrics
        results[split_name] = compute_split_metrics(np.array(y_split), y_prob)
    
    # Add training time
    results['_training_time_sec'] = train_time
    
    return results


def evaluate_with_prepared_data(
    prepared_data: MedicaidPreparedData,
    model: Any,
    exp_name: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a model using pre-prepared data.
    
    Args:
        prepared_data: MedicaidPreparedData from prepare_medicaid_evaluation_data()
        model: Pre-configured model (e.g., CatBoostClassifier)
        exp_name: Experiment name for result identification
        verbose: Print progress
        
    Returns:
        Dict with exp_name, model_type, feature_set, and all metrics
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"EVALUATING: {exp_name}")
        print(f"{'='*70}")
    
    # Evaluate model
    split_results = evaluate_model_on_splits(model, prepared_data, verbose=verbose)
    
    # Build output dictionary
    output = {
        'exp_name': exp_name,
        'model_type': type(model).__name__,
        'feature_set': prepared_data.feature_set,
        'n_features': len(prepared_data.feature_cols),
        'n_embedding_features': len(prepared_data.embedding_features),
        'n_tabular_features': len(prepared_data.tabular_features),
        'downsampled': prepared_data.downsampled,
        'original_train_size': prepared_data.original_train_size,
        'actual_train_size': len(prepared_data.X_train),
        'training_time_sec': split_results.pop('_training_time_sec', 0),
    }
    
    # Flatten split results with prefixes
    for split_name, metrics in split_results.items():
        for metric_name, value in metrics.items():
            output[f'{split_name}_{metric_name}'] = value
    
    if verbose:
        print(f"\n📊 Key Results for {exp_name}:")
        print(f"   Test AUC-ROC: {output.get('test_auc_roc', 0):.4f}")
        print(f"   Test Lift@1%: {output.get('test_lift_1pct', 0):.2f}x")
        print(f"   Test Lift@10%: {output.get('test_lift_10pct', 0):.2f}x")
        print(f"   Test PPV@1%: {output.get('test_ppv_1pct', 0):.2f}%")
        print(f"{'='*70}\n")
    
    return output


# =============================================================================
# EMBEDDING INTEGRATION
# =============================================================================

def load_embeddings_from_npz(
    embedding_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load embeddings from NPZ file (generated by embedding generation pipeline).
    
    Args:
        embedding_path: Path to NPZ file or directory containing NPZ files
        
    Returns:
        Tuple of (embeddings, individual_ids, index_dts)
    """
    if os.path.isdir(embedding_path):
        npz_files = glob.glob(os.path.join(embedding_path, "embeddings_*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No NPZ files found in {embedding_path}")
        npz_path = sorted(npz_files)[-1]  # Use most recent
    else:
        npz_path = embedding_path
    
    print(f"Loading embeddings from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    return (
        data['embeddings'],
        data['individual_ids'],
        data['index_dts']
    )


def merge_new_embeddings_with_features(
    df_features: pd.DataFrame,
    embeddings: np.ndarray,
    individual_ids: np.ndarray,
    merge_key: str = MEMBER_KEY
) -> pd.DataFrame:
    """
    Replace existing embeddings with new transformer embeddings.
    
    This function is used when you want to evaluate new embeddings
    (e.g., from a new transformer model) against the same Medicaid IP task.
    
    Args:
        df_features: DataFrame with existing features (may include old embeddings)
        embeddings: New embeddings array [num_members, embedding_dim]
        individual_ids: Member IDs corresponding to embeddings
        merge_key: Column name to join on
        
    Returns:
        DataFrame with old embeddings replaced by new embeddings
    """
    # Create embedding DataFrame
    embedding_dim = embeddings.shape[1]
    embedding_cols = [f'emb{i}' for i in range(embedding_dim)]
    
    df_emb = pd.DataFrame(embeddings, columns=embedding_cols)
    df_emb[merge_key] = individual_ids
    
    # Remove old embeddings from features
    old_emb_cols = [c for c in df_features.columns if c.startswith('emb')]
    df_features_no_emb = df_features.drop(columns=old_emb_cols, errors='ignore')
    
    # Merge new embeddings
    df_merged = df_features_no_emb.merge(df_emb, on=merge_key, how='inner')
    
    print(f"Merged new embeddings: {len(df_merged):,} rows (from {len(df_features):,})")
    print(f"  Embedding dim: {embedding_dim}")
    
    return df_merged


# =============================================================================
# EMBEDDING GENERATION FOR MEDICAID
# =============================================================================
# These functions generate embeddings from trained transformer models.
# They reuse code from moe_flashattn_3_commercial_downstream.py with
# Medicaid-specific adaptations for ID columns and LOB handling.

# If commercial module is not available, define LazyClinicalDataset locally
if not COMMERCIAL_MODULE_AVAILABLE:
    class LazyClinicalDataset(Dataset):
        """
        Memory-efficient dataset that parses data on-the-fly.
        
        Medicaid Adaptation:
        - Adds 'lob'='Medicaid' if not present in the DataFrame
        - Uses asdb_member_key as the primary ID for Medicaid data
        
        Note: This is a fallback if commercial module is not available.
        Prefer importing from moe_flashattn_3_commercial_downstream.
        """
        
        def __init__(self, df: pd.DataFrame, config: 'BaseConfig'):
            if not CORE_MODULE_AVAILABLE:
                raise ImportError("Core module required for LazyClinicalDataset. "
                                "Ensure moe_flashattn_3_core.py is in the same directory.")
            
            self.config = config
            self.df = df.reset_index(drop=True)
            
            # Pre-extract columns as lists for faster access
            self.age_strs = self.df['age_in_months'].tolist()
            self.gender_strs = self.df['gender_cd'].tolist()
            self.cd_strs = self.df['cd'].tolist()
            
            # Medicaid-specific: Add 'lob' column if not present
            if 'lob' in self.df.columns:
                self.lob_strs = self.df['lob'].tolist()
            else:
                self.lob_strs = ['Medicaid'] * len(self.df)
                
            self.dt_cnt = self.df['dt_cnt'].tolist()
            self.target_strs = self.df['target'].tolist() if 'target' in self.df.columns else None
            
            print(f"LazyClinicalDataset initialized with {len(self.df):,} samples (lazy loading)")
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            age_list = conv_age_gender(self.age_strs[idx], self.config.len_dy)
            gender_list = conv_age_gender(self.gender_strs[idx], self.config.len_dy, max_val=3)
            cd_list = conv_cd(self.cd_strs[idx], self.config.len_dy, self.config.len_cd)
            lob_list = conv_lob(self.lob_strs[idx], self.config.len_dy)
            
            age = torch.tensor(age_list, dtype=torch.long)
            gender = torch.tensor(gender_list, dtype=torch.long)
            codes = torch.tensor(cd_list, dtype=torch.long)
            lob = torch.tensor(lob_list, dtype=torch.long)
            
            if self.target_strs is not None:
                target_list = conv_target(self.target_strs[idx], self.config.len_dy, self.config.target_cd_cnt)
            else:
                target_list = [[0] for _ in range(self.config.len_dy)]
            
            return {
                'age': age,
                'gender': gender,
                'lob': lob,
                'codes': codes,
                'dt_cnt': self.dt_cnt[idx],
                'target': target_list
            }
    
    def load_model_from_checkpoint(
        model_path: str,
        device: torch.device,
        verbose: bool = True
    ) -> Tuple[torch.nn.Module, 'BaseConfig', Optional['MoEConfig'], bool, str]:
        """
        Load a pretrained model from a checkpoint file.
        
        Note: This is a fallback if commercial module is not available.
        """
        if not CORE_MODULE_AVAILABLE:
            raise ImportError("Core module required for model loading.")
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Loading model from: {model_path}")
        
        checkpoint_data = torch.load(model_path, map_location=device, weights_only=False)
        
        model_type = checkpoint_data.get('model_type', 'Unknown')
        config_dict = checkpoint_data.get('config', {})
        moe_config_dict = checkpoint_data.get('moe_config', None)
        state_dict = checkpoint_data['model_state_dict']
        
        if verbose:
            print(f"  Model type: {model_type}")
            print(f"  Embedding size: {config_dict.get('embedding_size', 256)}")
        
        use_learnt_att_pool_inferred = 'daily_pooling.query' in state_dict
        
        inferred_d_ff = None
        if 'FlashMoE' in model_type:
            for key in state_dict.keys():
                if 'experts.0.ffn.w_gate.weight' in key:
                    weight_shape = state_dict[key].shape
                    d_ff_adjusted = weight_shape[0]
                    inferred_d_ff = (d_ff_adjusted * 3 + 1) // 2
                    break
            if inferred_d_ff is None:
                inferred_d_ff = config_dict.get('nhid', 512)
        
        moe_config = None
        
        if 'FlashMoE' in model_type:
            config = FlashAttentionConfig(
                embedding_size=config_dict.get('embedding_size', 256),
                nhid=config_dict.get('nhid', 512),
                nhead=config_dict.get('nhead', 8),
                nlayers=config_dict.get('nlayers', 6),
                dropout=config_dict.get('dropout', 0.1),
                use_learnt_att_pool=use_learnt_att_pool_inferred,
                use_swiglu=config_dict.get('use_swiglu', True),
                use_rope=config_dict.get('use_rope', True),
                use_flash=config_dict.get('use_flash', True),
            )
            
            if moe_config_dict:
                d_ff_to_use = inferred_d_ff or config_dict.get('nhid', 512)
                moe_config = MoEConfig(
                    d_model=moe_config_dict.get('d_model', config.embedding_size),
                    d_ff=d_ff_to_use,
                    num_experts=moe_config_dict.get('num_experts', 8),
                    num_shared_experts=moe_config_dict.get('num_shared_experts', 1),
                    top_k=moe_config_dict.get('top_k', 2),
                    expert_dropout=moe_config_dict.get('expert_dropout', 0.1),
                    load_balance_strategy=moe_config_dict.get('load_balance_strategy', 'deepseek'),
                    aux_loss_weight=moe_config_dict.get('aux_loss_weight', 0.001),
                    use_moe_from_layer=moe_config_dict.get('use_moe_from_layer', 2),
                    use_swiglu_experts=moe_config_dict.get('use_swiglu_experts', True),
                    router_warmup_steps=moe_config_dict.get('router_warmup_steps', 0),
                    z_loss_weight=moe_config_dict.get('z_loss_weight', 0.005),
                    bias_lr=moe_config_dict.get('bias_lr', 1e-3),
                    bias_momentum=moe_config_dict.get('bias_momentum', 0.6),
                )
            else:
                moe_config = MoEConfig(d_model=config.embedding_size, d_ff=config.nhid)
            
            model = FlashMoETransformer(config, moe_config)
            use_mixed_precision = True
            
        elif 'FlashAttention' in model_type:
            config = FlashAttentionConfig(
                embedding_size=config_dict.get('embedding_size', 256),
                nhid=config_dict.get('nhid', 512),
                nhead=config_dict.get('nhead', 8),
                nlayers=config_dict.get('nlayers', 6),
                dropout=config_dict.get('dropout', 0.1),
                use_learnt_att_pool=config_dict.get('use_learnt_att_pool', True),
                use_swiglu=config_dict.get('use_swiglu', True),
                use_rope=config_dict.get('use_rope', True),
                use_flash=config_dict.get('use_flash', True),
            )
            model = FlashAttentionTransformer(config)
            use_mixed_precision = True
            
        else:
            config = BaseConfig(
                embedding_size=config_dict.get('embedding_size', 256),
                nhid=config_dict.get('nhid', 512),
                nlayers=config_dict.get('nlayers', 6),
                dropout=config_dict.get('dropout', 0.1),
            )
            model = BaselineTransformer(config)
            use_mixed_precision = False
        
        model.load_state_dict(checkpoint_data['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        if verbose:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✅ Model loaded: {total_params:,} parameters")
            print(f"{'='*70}\n")
        
        return model, config, moe_config, use_mixed_precision, model_type


def generate_medicaid_embeddings(
    model: torch.nn.Module,
    config: 'BaseConfig',
    data: pd.DataFrame,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    use_mixed_precision: bool = True,
    verbose: bool = True,
    multi_gpu: bool = False,
    moe_config: Optional['MoEConfig'] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Generate embeddings for Medicaid heldout members.
    
    This function adapts the commercial embedding generation for Medicaid:
    1. Uses asdb_member_key as the primary ID (not individual_id)
    2. Handles the Medicaid lob column (adds 'Medicaid' if missing)
    3. Returns asdb_member_key for joining with downstream features
    
    Args:
        model: Loaded model in eval mode
        config: Model configuration
        data: DataFrame from load_te_inference_input() with columns:
              asdb_member_key, individual_id, index_dt, gender_cd,
              age_in_months, cd, dt_cnt
        device: Primary device
        batch_size: Batch size per GPU
        num_workers: DataLoader workers
        use_mixed_precision: Use FP16 for Flash models
        verbose: Print progress
        multi_gpu: Enable multi-GPU processing
        moe_config: MoE config (required for multi-GPU with MoE models)
        
    Returns:
        embeddings: np.ndarray [num_members, embedding_size]
        member_keys: List of asdb_member_key values (for joining with features)
        index_dts: List of index dates
    """
    if not CORE_MODULE_AVAILABLE:
        raise ImportError("Core module required for embedding generation")
    
    start_time = time.time()
    n_samples = len(data)
    embedding_dim = config.embedding_size
    
    # Detect model type
    has_moe = (hasattr(model, 'forward') and 
               'return_moe_losses' in model.forward.__code__.co_varnames)
    
    n_gpus = torch.cuda.device_count() if multi_gpu else 1
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"MEDICAID EMBEDDING GENERATION")
        print(f"{'='*70}")
        print(f"Samples: {n_samples:,} | Batch: {batch_size} | GPUs: {n_gpus}")
        print(f"Workers: {num_workers} | Mixed precision: {use_mixed_precision}")
    
    # Ensure 'lob' column exists (required by LazyClinicalDataset)
    if 'lob' not in data.columns:
        data = data.copy()
        data['lob'] = 'Medicaid'
        if verbose:
            print("  Added 'lob'='Medicaid' column")
    
    # Pre-allocate pinned memory output
    embeddings_output = torch.empty(
        (n_samples, embedding_dim),
        dtype=torch.float32,
        pin_memory=True
    )
    
    # Extract IDs - use asdb_member_key for Medicaid (primary key for downstream)
    if MEMBER_KEY in data.columns:
        member_keys = data[MEMBER_KEY].astype(str).tolist()
    else:
        # Fallback to individual_id if asdb_member_key not present
        member_keys = data['individual_id'].astype(str).tolist()
    
    index_dts = data['index_dt'].astype(str).tolist()
    
    if n_gpus > 1 and multi_gpu:
        # Multi-GPU path
        return _generate_embeddings_multi_gpu_medicaid(
            model=model,
            config=config,
            data=data,
            embeddings_output=embeddings_output,
            member_keys=member_keys,
            index_dts=index_dts,
            n_gpus=n_gpus,
            batch_size=batch_size,
            num_workers=num_workers,
            use_mixed_precision=use_mixed_precision,
            has_moe=has_moe,
            moe_config=moe_config,
            verbose=verbose,
            start_time=start_time,
        )
    else:
        # Single GPU path
        return _generate_embeddings_single_gpu_medicaid(
            model=model,
            config=config,
            data=data,
            device=device,
            embeddings_output=embeddings_output,
            member_keys=member_keys,
            index_dts=index_dts,
            batch_size=batch_size,
            num_workers=num_workers,
            use_mixed_precision=use_mixed_precision,
            has_moe=has_moe,
            verbose=verbose,
            start_time=start_time,
        )


def _generate_embeddings_single_gpu_medicaid(
    model, config, data, device, embeddings_output,
    member_keys, index_dts, batch_size, num_workers,
    use_mixed_precision, has_moe, verbose, start_time
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Single GPU optimized path for Medicaid embeddings."""
    
    n_samples = len(data)
    model.eval()
    
    dataset = LazyClinicalDataset(data, config)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=create_collate_fn(config),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    
    current_idx = 0
    pbar = tqdm(dataloader, desc="Generating Medicaid embeddings", disable=not verbose)
    
    with torch.inference_mode():
        with EmbeddingExtractor(model) as extractor:
            for batch in pbar:
                batch_size_actual = batch['age'].shape[0]
                batch_start = current_idx
                batch_end = batch_start + batch_size_actual
                
                x = torch.cat([
                    batch['age'].unsqueeze(-1),
                    batch['gender'].unsqueeze(-1),
                    batch['lob'].unsqueeze(-1),
                    batch['codes']
                ], dim=-1).to(device, non_blocking=True)
                
                dt_cnt = batch['dt_cnt']
                
                if use_mixed_precision:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        if has_moe:
                            _ = model(x, return_moe_losses=False)
                        else:
                            _ = model(x)
                else:
                    if has_moe:
                        _ = model(x, return_moe_losses=False)
                    else:
                        _ = model(x)
                
                dt_cnt_list = dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
                patient_embs = extractor.get_patient_embedding(dt_cnt_list)
                
                embeddings_output[batch_start:batch_end].copy_(
                    patient_embs.float(),
                    non_blocking=True
                )
                
                current_idx = batch_end
                
                elapsed = time.time() - start_time
                speed = batch_end / elapsed
                eta = (n_samples - batch_end) / speed if speed > 0 else 0
                pbar.set_postfix({
                    'speed': f'{speed:.0f}/s',
                    'ETA': f'{eta:.0f}s'
                })
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    embeddings = embeddings_output.numpy()
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\n✅ Complete! Time: {elapsed:.1f}s | Speed: {n_samples/elapsed:,.0f} samples/s")
        print(f"   Output: {embeddings.shape}")
    
    return embeddings, member_keys, index_dts


def _generate_embeddings_multi_gpu_medicaid(
    model, config, data, embeddings_output, member_keys, index_dts,
    n_gpus, batch_size, num_workers, use_mixed_precision, has_moe,
    moe_config, verbose, start_time
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Multi-GPU path for Medicaid embeddings with true parallelism.
    
    Strategy:
    - Split data into N chunks (one per GPU)
    - Each GPU processes its chunk independently
    - Write to non-overlapping regions of shared output tensor
    """
    n_samples = len(data)
    
    if verbose:
        print(f"Multi-GPU mode: {n_gpus} GPUs")
    
    # Clone model to each GPU
    models = []
    for gpu_id in range(n_gpus):
        if verbose:
            print(f"  Cloning model to GPU {gpu_id}...")
        
        with torch.cuda.device(gpu_id):
            model_copy = copy.deepcopy(model)
            model_copy = model_copy.to(f'cuda:{gpu_id}')
            model_copy.eval()
            models.append(model_copy)
    
    # Split data into chunks
    chunk_size = (n_samples + n_gpus - 1) // n_gpus
    data_chunks = []
    start_indices = []
    
    for i in range(n_gpus):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_samples)
        data_chunks.append(data.iloc[start_idx:end_idx].reset_index(drop=True))
        start_indices.append(start_idx)
        
        if verbose:
            print(f"  GPU {i}: samples {start_idx:,} to {end_idx:,} ({end_idx - start_idx:,} samples)")
    
    progress_lock = threading.Lock()
    total_processed = [0]
    errors = []
    
    def process_chunk(gpu_id: int, data_chunk: pd.DataFrame, start_idx: int):
        """Process a data chunk on a specific GPU."""
        if len(data_chunk) == 0:
            return
        
        try:
            gpu_device = torch.device(f'cuda:{gpu_id}')
            gpu_model = models[gpu_id]
            
            dataset = LazyClinicalDataset(data_chunk, config)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=create_collate_fn(config),
                num_workers=max(1, num_workers // n_gpus),
                pin_memory=True,
            )
            
            local_idx = start_idx
            
            with torch.inference_mode():
                with EmbeddingExtractor(gpu_model) as extractor:
                    for batch in dataloader:
                        batch_size_actual = batch['age'].shape[0]
                        
                        x = torch.cat([
                            batch['age'].unsqueeze(-1),
                            batch['gender'].unsqueeze(-1),
                            batch['lob'].unsqueeze(-1),
                            batch['codes']
                        ], dim=-1).to(gpu_device, non_blocking=True)
                        
                        dt_cnt = batch['dt_cnt']
                        
                        if use_mixed_precision:
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                if has_moe:
                                    _ = gpu_model(x, return_moe_losses=False)
                                else:
                                    _ = gpu_model(x)
                        else:
                            if has_moe:
                                _ = gpu_model(x, return_moe_losses=False)
                            else:
                                _ = gpu_model(x)
                        
                        dt_cnt_list = dt_cnt.tolist() if isinstance(dt_cnt, torch.Tensor) else dt_cnt
                        patient_embs = extractor.get_patient_embedding(dt_cnt_list)
                        
                        embeddings_output[local_idx:local_idx + batch_size_actual].copy_(
                            patient_embs.float(),
                            non_blocking=True
                        )
                        
                        local_idx += batch_size_actual
                        
                        with progress_lock:
                            total_processed[0] += batch_size_actual
            
            torch.cuda.synchronize(gpu_device)
            
        except Exception as e:
            errors.append((gpu_id, str(e)))
    
    # Launch parallel processing
    if verbose:
        pbar = tqdm(total=n_samples, desc=f"Multi-GPU ({n_gpus} GPUs)")
    
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = [
            executor.submit(process_chunk, gpu_id, data_chunks[gpu_id], start_indices[gpu_id])
            for gpu_id in range(n_gpus)
        ]
        
        last_count = 0
        while not all(f.done() for f in futures):
            with progress_lock:
                current = total_processed[0]
            if verbose:
                pbar.update(current - last_count)
            last_count = current
            time.sleep(0.1)
        
        if verbose:
            pbar.update(n_samples - last_count)
            pbar.close()
        
        for f in futures:
            f.result()
    
    if errors:
        raise RuntimeError(f"GPU errors: {errors}")
    
    # Cleanup
    for m in models:
        del m
    torch.cuda.empty_cache()
    
    embeddings = embeddings_output.numpy()
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\n✅ Complete! Time: {elapsed:.1f}s | Speed: {n_samples/elapsed:,.0f} samples/s")
        print(f"   Effective: {n_samples/elapsed * n_gpus:,.0f} samples/s (across {n_gpus} GPUs)")
        print(f"   Output: {embeddings.shape}")
    
    return embeddings, member_keys, index_dts


def save_medicaid_embeddings(
    embeddings: np.ndarray,
    member_keys: List[str],
    index_dts: List[str],
    output_path: str,
    model_name: str = "",
    additional_metadata: Dict = None
) -> str:
    """
    Save Medicaid embeddings to disk in NPZ format.
    
    Args:
        embeddings: [num_members, embedding_dim] array
        member_keys: List of asdb_member_key values
        index_dts: List of index dates
        output_path: Directory to save files
        model_name: Name of the model for file naming
        additional_metadata: Optional dict of additional info to save
        
    Returns:
        Path to saved NPZ file
    """
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if model_name:
        filename_base = f"embeddings_medicaid_{model_name}_{timestamp}"
    else:
        filename_base = f"embeddings_medicaid_{timestamp}"
    
    # Save NPZ with embeddings and metadata
    npz_path = os.path.join(output_path, f"{filename_base}.npz")
    np.savez_compressed(
        npz_path,
        embeddings=embeddings,
        # Note: We save as 'individual_ids' for compatibility with load_embeddings_from_npz
        # but these are actually asdb_member_keys for Medicaid
        individual_ids=np.array(member_keys, dtype=object),
        index_dts=np.array(index_dts, dtype=object),
        embedding_dim=embeddings.shape[1],
        num_members=len(member_keys),
        lob='Medicaid',
        member_key_type='asdb_member_key',
        **(additional_metadata or {})
    )
    print(f"Embeddings saved to: {npz_path}")
    
    # Save CSV for easy lookup
    csv_path = os.path.join(output_path, f"{filename_base}_ids.csv")
    pd.DataFrame({
        MEMBER_KEY: member_keys,
        'index_dt': index_dts,
        'embedding_idx': range(len(member_keys))
    }).to_csv(csv_path, index=False)
    print(f"ID mapping saved to: {csv_path}")
    
    return npz_path


def save_medicaid_embeddings_to_bigquery(
    embeddings: np.ndarray,
    member_keys: List[str],
    index_dts: List[str],
    project_id: str = PROJECT_ID,
    dataset_id: str = DATASET_ID,
    table_name: str = "",
    exp_name: str = "",
    model_type: str = "",
    if_exists: str = "replace"
) -> str:
    """
    Save Medicaid embeddings to BigQuery.
    
    Args:
        embeddings: numpy array [num_members, embedding_dim]
        member_keys: list of asdb_member_key values
        index_dts: list of index dates
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_name: Table name to create
        exp_name: Experiment name for metadata
        model_type: Model type for metadata
        if_exists: What to do if table exists ('replace', 'append', 'fail')
        
    Returns:
        Full table path
    """
    # Create DataFrame with ID columns
    df = pd.DataFrame({
        MEMBER_KEY: member_keys,
        'index_dt': index_dts,
    })
    
    # Add embedding columns (emb0, emb1, ..., emb255 to match Eric's naming)
    embedding_dim = embeddings.shape[1]
    for i in range(embedding_dim):
        df[f'emb{i}'] = embeddings[:, i].astype(np.float32)
    
    # Add metadata columns
    df['exp_name'] = exp_name
    df['model_type'] = model_type
    df['lob'] = 'Medicaid'
    
    full_table_id = f"{project_id}.{dataset_id}.{table_name}"
    
    print(f"Writing {len(df):,} rows to BigQuery: {full_table_id}")
    print(f"  Columns: {len(df.columns)} (embedding_dim={embedding_dim})")
    
    client = bigquery.Client()
    
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE if if_exists == "replace" 
                          else bigquery.WriteDisposition.WRITE_APPEND if if_exists == "append"
                          else bigquery.WriteDisposition.WRITE_EMPTY,
    )
    
    job = client.load_table_from_dataframe(df, full_table_id, job_config=job_config)
    job.result()
    
    table = client.get_table(full_table_id)
    print(f"✅ Loaded {table.num_rows:,} rows to {full_table_id}")
    
    return full_table_id


# =============================================================================
# TIME-BASED DATA SPLITTING (FOR OOT VALIDATION)
# =============================================================================

def create_time_based_splits(
    df: pd.DataFrame,
    date_column: str = 'index_dt',
    oot_cutoff_date: str = OOT_CUTOFF_DATE,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Create train/val/test/OOT splits based on time.
    
    This replicates the commercial IP splitting strategy with Medicaid adaptations:
    - Data before OOT cutoff: split into train/val/test (stratified random)
    - Data after OOT cutoff: used for OOT (out-of-time) validation
    
    The OOT split tests temporal generalization - how well the model
    performs on future time periods not seen during training.
    
    Note: Unlike commercial IP which uses ind_id_last_digit for deterministic
    splitting, Medicaid uses stratified random split (matching Eric's original
    pipeline). This is because Medicaid doesn't have the ind_id_last_digit column.
    
    Args:
        df: DataFrame with features and target
        date_column: Column containing dates (typically 'index_dt')
        oot_cutoff_date: Date string for OOT split cutoff (default: 2023-10-16)
        test_size: Fraction for test set (from pre-cutoff data)
        val_size: Fraction for validation set (from pre-cutoff data)
        random_state: Random seed for stratified split (default: 35, Eric's value)
        verbose: Print split information
        
    Returns:
        Dict with keys 'train', 'val', 'test', 'oot'
    """
    df = df.copy()
    
    # Parse dates
    df['_date_parsed'] = pd.to_datetime(df[date_column])
    oot_cutoff = pd.to_datetime(oot_cutoff_date)
    
    # Split by time
    df_pre_cutoff = df[df['_date_parsed'] <= oot_cutoff]
    df_oot = df[df['_date_parsed'] > oot_cutoff]
    
    if verbose:
        print(f"\nTime-based split (cutoff: {oot_cutoff_date}):")
        print(f"  Pre-cutoff: {len(df_pre_cutoff):,} rows")
        print(f"  Post-cutoff (OOT): {len(df_oot):,} rows")
    
    # For pre-cutoff data, do stratified random split
    if len(df_pre_cutoff) > 0:
        # First split: train+val vs test
        train_val, test = train_test_split(
            df_pre_cutoff,
            test_size=test_size,
            random_state=random_state,
            stratify=df_pre_cutoff[TARGET_COLUMN]
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=train_val[TARGET_COLUMN]
        )
    else:
        train, val, test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Remove temp column
    for split_df in [train, val, test, df_oot]:
        if '_date_parsed' in split_df.columns:
            split_df.drop(columns=['_date_parsed'], inplace=True)
    
    splits = {
        'train': train.reset_index(drop=True),
        'val': val.reset_index(drop=True),
        'test': test.reset_index(drop=True),
        'oot': df_oot.reset_index(drop=True),
    }
    
    if verbose:
        print("\nFinal splits:")
        for name, split_df in splits.items():
            if len(split_df) > 0:
                prevalence = split_df[TARGET_COLUMN].mean() * 100
                print(f"  {name}: {len(split_df):,} rows, "
                      f"{int(split_df[TARGET_COLUMN].sum()):,} positives ({prevalence:.2f}%)")
            else:
                print(f"  {name}: EMPTY")
    
    return splits


# =============================================================================
# MEDICAID EMBEDDING GENERATION WORKFLOW
# =============================================================================

def run_medicaid_embedding_generation(
    model_paths: Dict[str, str],
    output_dir: str,
    batch_size: int = 64,
    sample_frac: Optional[float] = EMBEDDING_SAMPLE_FRAC,
    multi_gpu: bool = True,
    save_to_bigquery: bool = False,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Generate embeddings for Medicaid heldout members using multiple models.
    
    This is the main entry point for embedding generation, designed to
    run multiple experiments in sequence and save results.
    
    Args:
        model_paths: Dict mapping experiment names to model checkpoint paths
        output_dir: Base directory for saving embeddings
        batch_size: Batch size for inference
        sample_frac: Optional sampling fraction (None = full data)
        multi_gpu: Use all available GPUs
        save_to_bigquery: Also save to BigQuery table
        verbose: Print progress
        
    Returns:
        Dict mapping experiment names to result metadata
        
    Example:
        model_paths = {
            'exp2b_flash_learned_pool': 'logs/.../model.pt',
            'exp6_auxiliary_free': 'logs/.../model.pt',
        }
        results = run_medicaid_embedding_generation(
            model_paths=model_paths,
            output_dir='embedding_output/medicaid/',
            batch_size=64,
            multi_gpu=True
        )
    """
    if not CORE_MODULE_AVAILABLE:
        raise ImportError("Core module required. Ensure moe_flashattn_3_core.py is available.")
    
    # Load TE inference input for heldout Medicaid members
    print("\n" + "="*70)
    print("MEDICAID EMBEDDING GENERATION WORKFLOW")
    print("="*70)
    
    df_te_input = load_te_inference_input(
        sample_frac=sample_frac,
        verbose=verbose
    )
    
    # Show time distribution for OOT planning
    if 'index_dt' in df_te_input.columns:
        df_te_input['index_dt'] = pd.to_datetime(df_te_input['index_dt'])
        df_pre_oot = df_te_input[df_te_input['index_dt'] <= pd.to_datetime(OOT_CUTOFF_DATE)]
        df_oot = df_te_input[df_te_input['index_dt'] > pd.to_datetime(OOT_CUTOFF_DATE)]
        
        if verbose:
            print(f"\nTime distribution (OOT cutoff: {OOT_CUTOFF_DATE}):")
            print(f"  Pre-OOT (≤{OOT_CUTOFF_DATE}): {len(df_pre_oot):,} members")
            print(f"  OOT (>{OOT_CUTOFF_DATE}): {len(df_oot):,} members")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    
    for exp_name, model_path in tqdm(model_paths.items(), desc="Processing models"):
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {exp_name}")
        print(f"{'='*70}")
        
        # Cleanup GPU memory
        if CORE_MODULE_AVAILABLE and torch.cuda.is_available():
            cleanup_gpu_memory(verbose=False)
        
        # Load model
        model, config, moe_config, use_mixed_precision, model_type = load_model_from_checkpoint(
            model_path=model_path,
            device=device,
            verbose=verbose
        )
        
        # Generate embeddings
        inference_start = time.time()
        embeddings, member_keys, index_dts = generate_medicaid_embeddings(
            model=model,
            config=config,
            data=df_te_input,
            device=device,
            batch_size=batch_size,
            use_mixed_precision=use_mixed_precision,
            verbose=verbose,
            multi_gpu=multi_gpu,
            moe_config=moe_config,
        )
        inference_duration = time.time() - inference_start
        
        # Save embeddings to NPZ
        exp_output_dir = os.path.join(output_dir, exp_name)
        embeddings_path = save_medicaid_embeddings(
            embeddings=embeddings,
            member_keys=member_keys,
            index_dts=index_dts,
            output_path=exp_output_dir,
            model_name=exp_name,
            additional_metadata={
                'model_path': model_path,
                'model_type': model_type,
                'use_mixed_precision': use_mixed_precision,
                'sample_frac': sample_frac,
            }
        )
        
        # Optionally save to BigQuery
        bq_table_path = None
        if save_to_bigquery:
            safe_exp_name = exp_name.replace('-', '_').replace('.', '_')
            table_name = f"a964286_te4exp_{safe_exp_name}_medicaid_heldout_embedding"
            bq_table_path = save_medicaid_embeddings_to_bigquery(
                embeddings=embeddings,
                member_keys=member_keys,
                index_dts=index_dts,
                table_name=table_name,
                exp_name=exp_name,
                model_type=model_type,
            )
        
        results[exp_name] = {
            'embeddings_path': embeddings_path,
            'bq_table_path': bq_table_path,
            'embedding_shape': embeddings.shape,
            'model_type': model_type,
            'model_path': model_path,
            'inference_duration_sec': inference_duration,
            'inference_duration_hr': round(inference_duration / 3600, 2),
            'status': 'success'
        }
        
        # Cleanup
        del model
        del embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "="*70)
    print("EMBEDDING GENERATION COMPLETE")
    print("="*70)
    for exp_name, result in results.items():
        print(f"  {exp_name}: {result['embedding_shape']} ({result['inference_duration_hr']:.2f}hr)")
    
    return results


# =============================================================================
# HIGH-LEVEL EVALUATION FUNCTIONS
# =============================================================================

def run_medicaid_ip_evaluation(
    embedding_path: Optional[str] = None,
    feature_set: str = 'hybrid',
    sample_frac: Optional[float] = None,
    use_tuned_params: bool = True,
    apply_downsampling: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete Medicaid IP evaluation pipeline.
    
    This is the main entry point for evaluating embeddings on the
    Medicaid IP hospitalization prediction task.
    
    Args:
        embedding_path: Path to new embeddings NPZ file (None = use original)
        feature_set: 'embedding_only', 'tabular_only', or 'hybrid'
        sample_frac: Sample fraction for testing (None = full data)
        use_tuned_params: Use Optuna-tuned hyperparameters
        apply_downsampling: Apply class imbalance handling
        verbose: Print progress
        
    Returns:
        Dict with all evaluation results
    """
    start_time = time.time()
    
    if verbose:
        print(f"\n{'#'*70}")
        print("MEDICAID IP DOWNSTREAM EVALUATION")
        print(f"{'#'*70}")
        print(f"Feature set: {feature_set}")
        print(f"New embeddings: {embedding_path or 'Using original'}")
        print(f"Sample fraction: {sample_frac or 'Full data'}")
    
    # Step 1: Load data from BigQuery
    df = load_medicaid_data_from_bigquery(
        sample_frac=sample_frac,
        verbose=verbose
    )
    
    # Step 2: Replace embeddings if new path provided
    if embedding_path is not None:
        if verbose:
            print("\nReplacing embeddings with new transformer embeddings...")
        embeddings, individual_ids, index_dts = load_embeddings_from_npz(embedding_path)
        df = merge_new_embeddings_with_features(df, embeddings, individual_ids)
    
    # Step 3: Prepare data
    prepared_data = prepare_medicaid_evaluation_data(
        df=df,
        feature_set=feature_set,
        apply_downsampling=apply_downsampling,
        verbose=verbose
    )
    
    # Step 4: Configure model
    if use_tuned_params:
        model = CatBoostClassifier(**CATBOOST_TUNED_PARAMS)
    else:
        model = CatBoostClassifier(**CATBOOST_BALANCED_PARAMS)
    
    # Step 5: Evaluate
    exp_name = f"medicaid_ip_{feature_set}"
    if embedding_path:
        exp_name += f"_new_emb"
    
    results = evaluate_with_prepared_data(
        prepared_data=prepared_data,
        model=model,
        exp_name=exp_name,
        verbose=verbose
    )
    
    # Add metadata
    results['embedding_path'] = embedding_path
    results['sample_frac'] = sample_frac
    results['use_tuned_params'] = use_tuned_params
    results['total_time_sec'] = time.time() - start_time
    
    return results


def evaluate_multiple_embeddings(
    embedding_paths: Dict[str, str],
    feature_sets: List[str] = ['embedding_only', 'hybrid'],
    sample_frac: Optional[float] = None,
    apply_downsampling: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluate multiple embedding experiments on Medicaid IP task.
    
    This function enables fair comparison of different transformer
    embeddings by evaluating them all on the same Medicaid IP task
    using identical preprocessing, hyperparameters, and metrics.
    
    Args:
        embedding_paths: Dict mapping experiment names to NPZ paths
        feature_sets: List of feature sets to evaluate per experiment
        sample_frac: Sample fraction for testing
        apply_downsampling: Apply class imbalance handling
        verbose: Print progress
        
    Returns:
        DataFrame with one row per experiment, all metrics as columns
    """
    results = []
    
    # Also evaluate tabular-only baseline (no embeddings)
    if 'tabular_only' not in feature_sets and verbose:
        print("Note: Consider adding 'tabular_only' to feature_sets for baseline comparison")
    
    for exp_name, emb_path in tqdm(embedding_paths.items(), desc="Evaluating embeddings"):
        for feature_set in feature_sets:
            if verbose:
                print(f"\n{'='*70}")
                print(f"Experiment: {exp_name} | Feature set: {feature_set}")
            
            result = run_medicaid_ip_evaluation(
                embedding_path=emb_path,
                feature_set=feature_set,
                sample_frac=sample_frac,
                apply_downsampling=apply_downsampling,
                verbose=verbose
            )
            result['experiment'] = exp_name
            results.append(result)
    
    # Create summary DataFrame
    df_results = pd.DataFrame(results)
    
    # Reorder columns
    priority_cols = ['experiment', 'exp_name', 'feature_set', 'test_auc_roc', 
                     'test_lift_1pct', 'test_lift_10pct', 'test_ppv_1pct']
    other_cols = [c for c in df_results.columns if c not in priority_cols]
    df_results = df_results[priority_cols + other_cols]
    
    return df_results


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def compare_embedding_effects(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comparison table showing embedding contribution.
    
    Compares hybrid vs tabular_only performance to quantify
    the value added by transformer embeddings.
    
    Args:
        results_df: Results from evaluate_multiple_embeddings()
        
    Returns:
        Comparison DataFrame with delta metrics
    """
    # Pivot to compare feature sets
    comparison = []
    
    experiments = results_df['experiment'].unique()
    for exp in experiments:
        exp_data = results_df[results_df['experiment'] == exp]
        
        row = {'experiment': exp}
        
        # Get metrics for each feature set
        for fs in ['embedding_only', 'tabular_only', 'hybrid']:
            fs_data = exp_data[exp_data['feature_set'] == fs]
            if len(fs_data) > 0:
                row[f'{fs}_auc'] = fs_data['test_auc_roc'].values[0]
                row[f'{fs}_lift1'] = fs_data['test_lift_1pct'].values[0]
                row[f'{fs}_lift10'] = fs_data['test_lift_10pct'].values[0]
        
        # Compute deltas (hybrid vs tabular_only)
        if 'hybrid_auc' in row and 'tabular_only_auc' in row:
            row['delta_auc'] = row['hybrid_auc'] - row['tabular_only_auc']
            row['delta_lift1'] = row['hybrid_lift1'] - row['tabular_only_lift1']
            row['delta_lift10'] = row['hybrid_lift10'] - row['tabular_only_lift10']
        
        comparison.append(row)
    
    return pd.DataFrame(comparison)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of the Medicaid IP evaluation pipeline.
    
    This demonstrates how to:
    1. Generate embeddings for heldout Medicaid members
    2. Run evaluation with new transformer embeddings
    3. Use time-based (OOT) validation splits
    4. Compare results across experiments
    """
    
    print("\n" + "="*70)
    print("MEDICAID IP DOWNSTREAM EVALUATION PIPELINE")
    print("="*70)
    print("\nThis pipeline evaluates transformer embeddings on the Medicaid IP")
    print("hospitalization prediction task using the exact same methodology")
    print("as the original model (Eric Ma's internship project).")
    print("\n" + "-"*70)
    print("WORKFLOW OVERVIEW")
    print("-"*70)
    print("\nStep 1: EMBEDDING GENERATION")
    print("  - Load trained transformer model from checkpoint")
    print("  - Load heldout Medicaid member TE sequences from BigQuery")
    print("  - Generate embeddings via forward pass")
    print("  - Save to NPZ (and optionally BigQuery)")
    print("\nStep 2: DOWNSTREAM EVALUATION")
    print("  - Load heldout features + outcomes from BigQuery")
    print("  - Merge with generated embeddings")
    print("  - Split: Train/Val/Test (pre-cutoff) + OOT (post-cutoff)")
    print("  - Train CatBoost with tuned hyperparameters")
    print("  - Evaluate on all splits")
    print("\n" + "-"*70)
    print("FEATURE SETS")
    print("-"*70)
    print("  - embedding_only: 256 transformer embedding features")
    print("  - tabular_only: ~243 hand-crafted features (baseline)")
    print("  - hybrid: Both embeddings + tabular (~499 features)")
    print("\n" + "-"*70)
    print("KEY METRICS")
    print("-"*70)
    print("  - ROC-AUC: Overall discrimination")
    print("  - Lift@1%: Business impact metric (20x = 20 times better than random)")
    print("  - PPV@1%: Precision in top 1% of predictions")
    print("\nExpected performance (original CatBoost model):")
    print("  - AUC: ~0.87")
    print("  - Lift@1%: ~19-20x")
    print("\n" + "-"*70)
    print("TIME-BASED SPLIT (OOT VALIDATION)")
    print("-"*70)
    print(f"  - OOT cutoff date: {OOT_CUTOFF_DATE}")
    print("  - Pre-cutoff: Train (80%), Val (10%), Test (10%) - stratified random")
    print("  - Post-cutoff: OOT (out-of-time validation)")
    print("="*70)
    
    # =========================================================================
    # EXAMPLE 1: EMBEDDING GENERATION
    # =========================================================================
    # Uncomment to generate embeddings for Medicaid heldout members:
    #
    # MODEL_PATHS = {
    #     'exp2b_flash_learned_pool': 
    #         'logs/exp_round6_3lobs_3-4M_pretrain_multi_gpu_test_v2/'
    #         'exp2b_flash_learned_pool/saved_models/'
    #         'exp_round6_3lobs_3-4M_pretrain_multi_gpu_test_v2_exp2b_flash_learned_pool_bs128_ep1_d256_final.pt',
    #     
    #     'exp6_auxiliary_free_v3': 
    #         'logs/exp_round5_3lobs_1-5M_pretrain_multi_gpu_test_v2/'
    #         'exp6_auxiliary_free_v3/saved_models/'
    #         'exp_round5_3lobs_pretrain_multi_gpu_test_v2_exp6_auxiliary_free_bs128_ep1_d256_final.pt',
    # }
    # 
    # results = run_medicaid_embedding_generation(
    #     model_paths=MODEL_PATHS,
    #     output_dir='embedding_output/medicaid_heldout/',
    #     batch_size=64,
    #     sample_frac=None,  # Use full data (or 0.3 for testing)
    #     multi_gpu=True,
    #     save_to_bigquery=False,  # Set True to also save to BigQuery
    #     verbose=True
    # )
    
    # =========================================================================
    # EXAMPLE 2: DOWNSTREAM EVALUATION WITH NEW EMBEDDINGS
    # =========================================================================
    # Uncomment to evaluate embeddings on Medicaid IP task:
    #
    # result = run_medicaid_ip_evaluation(
    #     embedding_path='embedding_output/medicaid_heldout/exp2b_flash_learned_pool/',
    #     feature_set='hybrid',  # or 'embedding_only', 'tabular_only'
    #     sample_frac=None,  # Full data, or 0.01 for testing
    #     use_tuned_params=True,  # Use Optuna-tuned hyperparameters
    #     apply_downsampling=True,  # Use 0.2 undersampling ratio
    #     verbose=True
    # )
    # print(f"\nTest AUC: {result['test_auc_roc']:.4f}")
    # print(f"Test Lift@1%: {result['test_lift_1pct']:.2f}x")
    
    # =========================================================================
    # EXAMPLE 3: EVALUATION WITH OOT (OUT-OF-TIME) VALIDATION
    # =========================================================================
    # For time-based evaluation that includes OOT:
    #
    # # Load data
    # df = load_medicaid_data_from_bigquery(sample_frac=0.1)
    # 
    # # Create time-based splits (includes OOT)
    # splits = create_time_based_splits(
    #     df=df,
    #     oot_cutoff_date=OOT_CUTOFF_DATE,  # 2023-10-16
    #     verbose=True
    # )
    # 
    # # Evaluate on OOT split
    # # Note: Train on pre-cutoff, evaluate on all including OOT
    
    # =========================================================================
    # EXAMPLE 4: COMPARE MULTIPLE EMBEDDINGS
    # =========================================================================
    # Uncomment to compare multiple embedding experiments:
    #
    # embedding_paths = {
    #     'exp2b_flash': 'embedding_output/medicaid_heldout/exp2b_flash_learned_pool/',
    #     'exp6_moe': 'embedding_output/medicaid_heldout/exp6_auxiliary_free_v3/',
    # }
    # 
    # results_df = evaluate_multiple_embeddings(
    #     embedding_paths=embedding_paths,
    #     feature_sets=['embedding_only', 'tabular_only', 'hybrid'],
    #     sample_frac=None,
    #     apply_downsampling=True,
    #     verbose=True
    # )
    # 
    # # Compare embedding effects
    # comparison = compare_embedding_effects(results_df)
    # print("\nEmbedding Contribution Analysis:")
    # print(comparison.to_string())
    
    print("\n⚠️  No code executed. Uncomment examples above to run.")
    print("    Or import this module and call functions directly.")

