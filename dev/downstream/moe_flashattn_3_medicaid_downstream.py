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

Data Sources (BigQuery):
- Features: anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_non_embedding_features
- Embeddings: anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_embeddings
- Outcomes: anbc-hcb-dev.cm_medicaid_hcb_dev.a534354_IP_2024_outcome_ip

Author: Adapted from Eric Ma's Medicaid IP Model Refresh
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import glob
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

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


# =============================================================================
# CONFIGURATION - MEDICAID IP MODEL PARAMETERS
# =============================================================================
# These parameters replicate the original Medicaid IP model exactly

# BigQuery Tables
PROJECT_ID = "anbc-hcb-dev"
DATASET_ID = "cm_medicaid_hcb_dev"

# Original table names from Eric Ma's pipeline
FEATURES_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a534354_IP_2024_non_embedding_features"
EMBEDDINGS_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a534354_IP_2024_embeddings"
OUTCOME_TABLE = f"{PROJECT_ID}.{DATASET_ID}.a534354_IP_2024_outcome_ip"

# Data filtering (from original pipeline)
EXCLUDED_PLAN_KEYS = [33, 54]  # Plans to exclude
MIN_POST_MONTHS = 6  # Minimum post-observation months required

# Target variable
TARGET_COLUMN = "acute_ip_flag"

# Member ID column (primary key for joining)
MEMBER_KEY = "asdb_member_key"

# Random seed for reproducibility (matches original)
RANDOM_STATE = 35

# Class imbalance handling
# Original finding: CatBoost works better with 0.2 undersampling ratio
CATBOOST_UNDERSAMPLE_RATIO = 0.2  # 20% minority-to-majority (5:1)
XGBOOST_UNDERSAMPLE_RATIO = 0.03  # 3% minority-to-majority (~33:1)

# Train/Val/Test split ratios (original: 80/10/10)
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1

# =============================================================================
# CATBOOST TUNED HYPERPARAMETERS
# =============================================================================
# From Optuna optimization in original pipeline (optuna_results_catboost.csv)
# Best trial: AUC = 0.8737
CATBOOST_TUNED_PARAMS = {
    'learning_rate': 0.015742881221129403,
    'iterations': 2665,
    'l2_leaf_reg': 0.222046549398224,
    'depth': 7,
    'random_seed': RANDOM_STATE,
    'verbose': 0,
    'thread_count': -1,
    'use_best_model': True,
    'auto_class_weights': 'Balanced',  # Additional class weighting
}

# Alternative: Balanced class weights model
CATBOOST_BALANCED_PARAMS = {
    'iterations': 2500,
    'depth': 7,
    'learning_rate': 0.025,
    'grow_policy': 'SymmetricTree',
    'auto_class_weights': 'Balanced',
    'od_wait': 80,
    'use_best_model': True,
    'random_seed': RANDOM_STATE,
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

def load_medicaid_data_from_bigquery(
    sample_frac: Optional[float] = None,
    random_state: int = RANDOM_STATE,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load Medicaid IP data from BigQuery, replicating original data pipeline.
    
    This function:
    1. Loads non-embedding features from FEATURES_TABLE
    2. Loads embeddings from EMBEDDINGS_TABLE
    3. Loads outcomes from OUTCOME_TABLE
    4. Applies the same filters as the original pipeline:
       - Excludes plan keys 33, 54
       - Requires post_mnths >= 6
    5. Merges all tables on asdb_member_key
    
    Args:
        sample_frac: Optional sampling fraction for testing (e.g., 0.1 for 10%)
        random_state: Random seed for sampling
        verbose: Print progress information
        
    Returns:
        Merged DataFrame with features, embeddings, and outcome
    """
    client = bigquery.Client()
    
    if verbose:
        print(f"\n{'='*70}")
        print("LOADING MEDICAID IP DATA FROM BIGQUERY")
        print(f"{'='*70}")
    
    # Step 1: Load non-embedding features with filters
    if verbose:
        print("\n[Step 1/4] Loading non-embedding features...")
    
    features_sql = f"""
    SELECT
        f.* EXCEPT (asdb_plan_key, post_mnths, first_prv_dt, last_prv_dt, index_dt)
    FROM 
        `{FEATURES_TABLE}` AS f
    WHERE 1=1
        AND NOT asdb_plan_key IN ({','.join(map(str, EXCLUDED_PLAN_KEYS))})
        AND post_mnths >= {MIN_POST_MONTHS}
    """
    
    df_features = client.query(features_sql).to_dataframe()
    if verbose:
        print(f"  Features loaded: {len(df_features):,} rows, {len(df_features.columns)} columns")
    
    # Step 2: Load embeddings
    if verbose:
        print("\n[Step 2/4] Loading embeddings...")
    
    embeddings_sql = f"""
    SELECT
        e.*
    FROM 
        `{EMBEDDINGS_TABLE}` AS e
    WHERE e.individual_id IN (
        SELECT asdb_member_key 
        FROM `{FEATURES_TABLE}` 
        WHERE NOT asdb_plan_key IN ({','.join(map(str, EXCLUDED_PLAN_KEYS))})
            AND post_mnths >= {MIN_POST_MONTHS}
    )
    """
    
    df_embeddings = client.query(embeddings_sql).to_dataframe()
    # Rename ID column to match features table
    df_embeddings = df_embeddings.rename(columns={'individual_id': MEMBER_KEY})
    if verbose:
        print(f"  Embeddings loaded: {len(df_embeddings):,} rows, {len(df_embeddings.columns)} columns")
    
    # Step 3: Load outcomes
    if verbose:
        print("\n[Step 3/4] Loading outcomes...")
    
    outcomes_sql = f"""
    SELECT
        asdb_member_key,
        acute_ip_flag
    FROM 
        `{OUTCOME_TABLE}` AS o
    WHERE 1=1 
        AND o.asdb_member_key IN (
            SELECT asdb_member_key 
            FROM `{FEATURES_TABLE}` 
            WHERE NOT asdb_plan_key IN ({','.join(map(str, EXCLUDED_PLAN_KEYS))})
                AND post_mnths >= {MIN_POST_MONTHS}
        )
    """
    
    df_outcomes = client.query(outcomes_sql).to_dataframe()
    if verbose:
        print(f"  Outcomes loaded: {len(df_outcomes):,} rows")
        print(f"  Positive rate: {df_outcomes[TARGET_COLUMN].mean()*100:.2f}%")
    
    # Step 4: Merge all tables
    if verbose:
        print("\n[Step 4/4] Merging tables...")
    
    df_features = df_features.set_index(MEMBER_KEY)
    df_embeddings = df_embeddings.set_index(MEMBER_KEY)
    df_outcomes = df_outcomes.set_index(MEMBER_KEY)
    
    # Merge features with embeddings
    df_merged = df_features.merge(df_embeddings, left_index=True, right_index=True, how='inner')
    # Merge with outcomes
    df_merged = df_merged.merge(df_outcomes, left_index=True, right_index=True, how='inner')
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
        print(f"{'='*70}\n")
    
    return df_merged


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
    random_state: int = RANDOM_STATE
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
    random_state: int = RANDOM_STATE,
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
        random_state: Random seed
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
    
    # Step 2: Split data
    if verbose:
        print("\n[Step 2/4] Creating train/val/test splits...")
    splits = create_train_val_test_split(df_processed, random_state=random_state, verbose=verbose)
    
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
    
    # Step 4: Apply downsampling to training set
    downsampled = False
    if apply_downsampling:
        if verbose:
            print(f"\n[Step 4/4] Applying downsampling to training set...")
        X_train, y_train = downsample_negatives(
            X_train, y_train, 
            ratio=downsample_ratio, 
            random_state=random_state
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
    1. Run evaluation with original embeddings
    2. Run evaluation with new transformer embeddings
    3. Compare results across experiments
    """
    
    print("\n" + "="*70)
    print("MEDICAID IP DOWNSTREAM EVALUATION PIPELINE")
    print("="*70)
    print("\nThis pipeline evaluates transformer embeddings on the Medicaid IP")
    print("hospitalization prediction task using the exact same methodology")
    print("as the original model (Eric Ma's internship project).")
    print("\nSupported feature sets:")
    print("  - embedding_only: 256 transformer embedding features")
    print("  - tabular_only: ~243 hand-crafted features (baseline)")
    print("  - hybrid: Both embeddings + tabular (~499 features)")
    print("\nKey metrics:")
    print("  - ROC-AUC: Overall discrimination")
    print("  - Lift@1%: Business impact metric (20x = 20 times better than random)")
    print("  - PPV@1%: Precision in top 1% of predictions")
    print("\nExpected performance (original CatBoost model):")
    print("  - AUC: ~0.87")
    print("  - Lift@1%: ~19-20x")
    print("="*70)
    
    # Example: Run with a sample for testing
    # Uncomment to run:
    #
    # result = run_medicaid_ip_evaluation(
    #     embedding_path=None,  # Use original embeddings
    #     feature_set='hybrid',
    #     sample_frac=0.01,  # 1% sample for quick testing
    #     verbose=True
    # )
    # print(f"\nTest AUC: {result['test_auc_roc']:.4f}")
    # print(f"Test Lift@1%: {result['test_lift_1pct']:.2f}x")

