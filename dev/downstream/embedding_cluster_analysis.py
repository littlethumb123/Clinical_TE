"""
Transformer Embedding Cluster Analysis Module
==============================================

Purpose: Analyze if transformer embeddings cluster meaningfully by clinical outcome.

This module provides reusable functions for:
1. Data ingestion (local NPZ files, BigQuery tables)
2. Point-biserial correlation analysis between embedding dimensions and binary outcome
3. Dataset balancing (downsampling majority class)
4. UMAP dimensionality reduction (256D → 2D)
5. Cluster visualization (4-panel view)
6. Cluster separation quantification (Cohen's D)
7. Summary reporting

Three Usage Patterns:
---------------------

OPTION 1: Full Pipeline with Data Loading
    from embedding_cluster_analysis import load_and_prepare_data, run_cluster_analysis
    
    # Load data (embeddings + outcomes)
    df = load_and_prepare_data(
        embedding_source="embedding_output/exp2b_flash_learned_pool/",
        outcome_source="project.dataset.features_table",
        outcome_column='ip6'
    )
    
    # Run analysis
    result = run_cluster_analysis(df, 'ip6', outcome_label='IP Admission')

OPTION 2: Full Pipeline (Pre-loaded Data)
    from embedding_cluster_analysis import run_cluster_analysis
    result = run_cluster_analysis(df, 'ip6', outcome_label='IP Admission')

OPTION 3: Procedural (Step-by-Step)
    from embedding_cluster_analysis import (
        load_embeddings_from_local,
        create_embedding_dataframe,
        join_embeddings_with_outcomes,
        identify_embedding_columns,
        compute_embedding_correlations,
        balance_dataset_for_clustering,
        run_umap_projection,
        compute_cluster_separation,
        visualize_clusters,
        generate_cluster_summary,
        ClusterConfig, UMAPConfig
    )
    
    # Step 0: Load and join data
    embeddings, ids, dates = load_embeddings_from_local("path/to/embeddings/")
    df_emb = create_embedding_dataframe(embeddings, ids, dates)
    df = join_embeddings_with_outcomes(df_emb, df_outcomes, 'ip6')
    
    # Steps 1-6: Run each analysis step independently
    embedding_cols = identify_embedding_columns(df)
    corr_result = compute_embedding_correlations(df, 'ip6', embedding_cols)
    df_balanced = balance_dataset_for_clustering(df, 'ip6')
    df_viz, emb_2d, runtime = run_umap_projection(df_balanced, embedding_cols)
    sep_result = compute_cluster_separation(df_viz, 'ip6')
    visualize_clusters(df_viz, 'ip6')
    summary = generate_cluster_summary(corr_result, sep_result, 'my_model', 'IP Admission')

Owner: Pritha Ghosh (pritha.ghosh@cvshealth.com)
Team: Clinical & Social Determinants Intelligence (CSDI)
Created: January 2026
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Union
import warnings
import time
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import umap

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set default plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


# =============================================================================
# COLUMN EXCLUSION CONSTANTS
# =============================================================================

# Columns to EXCLUDE from tabular feature analysis
# These are identifiers, outcomes, eligibility flags, or known data leakage columns
EXCLUDE_COLUMNS = frozenset([
    # Keys and identifiers
    'individual_id', 'member_id', 'index_dt', 'birth_dt', 'feature_end_dt',
    
    # Outcome columns (target and related) - will be added dynamically
    'ip6', 'sum_ip6_admits', 'sum_ip6_los', 'sum_ip6_acu_days',
    'er6', 'sum_er6_visits',  # ER outcome variants
    
    # Eligibility/continuity flags
    'mon_3_include', 'mon_6_include', 'mon_12_include',
    'exclude_ip', 'include_post_6_status',
    
    # Split key
    'ind_id_last_digit',
    
    # Leakage columns (cost amounts - these are post-hoc outcomes)
    'clm_allowed_amt_1yr', 'clm_allowed_amt_2yr', 'clm_allowed_amt_3mo', 'clm_allowed_amt_6mo',
    'clm_paid_amt_1yr', 'clm_paid_amt_2yr', 'clm_paid_amt_3mo', 'clm_paid_amt_6mo',
    'clm_par_allowed_amt_1yr', 'clm_par_allowed_amt_2yr', 'clm_par_allowed_amt_3mo', 'clm_par_allowed_amt_6mo',
    'clm_par_paid_amt_1yr', 'clm_par_paid_amt_2yr', 'clm_par_paid_amt_3mo', 'clm_par_paid_amt_6mo',
    'clm_srv_copay_amt_1yr', 'clm_srv_copay_amt_3mo', 'clm_srv_copay_amt_6mo',
    
    # Outreach/intervention flags (post-hoc)
    'covid_19', 'hpd_major_flag', 'chronic',
    'txt_member', 'txt_referral', 'txt_1yr_outreach', 'talked',
    
    # Temporary/internal columns
    '_exp_name', 'index_dt_parsed', '_index_dt_parsed'
])


# =============================================================================
# DATA CLASSES - Configuration and Results
# =============================================================================

@dataclass
class UMAPConfig:
    """Configuration for UMAP dimensionality reduction."""
    n_neighbors: int = 100
    min_dist: float = 0.1
    n_components: int = 2
    metric: str = 'euclidean'
    random_state: int = 123
    verbose: bool = True


@dataclass
class ClusterConfig:
    """Configuration for cluster analysis."""
    balance_classes: bool = True
    correlation_threshold: float = 0.05
    imbalance_threshold: float = 0.20  # Downsample if minority < 20%
    random_state: int = 123


@dataclass
class CorrelationResult:
    """Result of correlation analysis between embeddings and outcome."""
    df_correlations: pd.DataFrame
    n_samples: int
    n_dimensions: int
    mean_abs_correlation: float
    max_abs_correlation: float
    top_predictor: str
    top_predictor_corr: float
    strong_predictors_count: int  # |r| > threshold
    very_strong_predictors_count: int  # |r| > 0.10


@dataclass
class ClusterSeparationResult:
    """Result of cluster separation analysis (Cohen's D)."""
    centroid_pos: np.ndarray
    centroid_neg: np.ndarray
    centroid_distance: float
    variance_pos: float
    variance_neg: float
    pooled_std: float
    cohens_d: float
    interpretation: str


@dataclass
class ClusterAnalysisResult:
    """Complete result of cluster analysis pipeline."""
    correlation_result: CorrelationResult
    separation_result: Optional[ClusterSeparationResult]
    df_balanced: pd.DataFrame
    df_viz: Optional[pd.DataFrame]  # With UMAP coordinates
    embedding_cols: List[str]
    outcome_column: str
    outcome_label: str
    embedding_name: str


# =============================================================================
# DATA INGESTION FUNCTIONS
# =============================================================================

def identify_tabular_feature_columns(
    df: pd.DataFrame,
    outcome_column: str = 'ip6',
    exclude_columns: Optional[frozenset] = None,
    verbose: bool = True
) -> List[str]:
    """
    Identify tabular (non-embedding) feature columns from a DataFrame.
    
    Excludes:
    - Embedding columns (embedding_*)
    - Outcome column
    - Identifier columns (individual_id, index_dt, etc.)
    - Known leakage columns (cost amounts, outreach flags)
    - Eligibility flags
    
    Args:
        df: Input dataframe
        outcome_column: Name of outcome column to exclude
        exclude_columns: Set of column names to exclude (default: EXCLUDE_COLUMNS)
        verbose: Whether to print summary
        
    Returns:
        Sorted list of tabular feature column names
        
    Example:
        tabular_cols = identify_tabular_feature_columns(df, 'ip6')
        print(f"Found {len(tabular_cols)} tabular features")
    """
    if exclude_columns is None:
        exclude_columns = EXCLUDE_COLUMNS
    
    all_cols = set(df.columns)
    
    # Identify embedding columns
    embedding_cols = {c for c in all_cols if c.startswith('embedding_')}
    
    # Build full exclusion set
    full_exclusion = exclude_columns | embedding_cols | {outcome_column}
    
    # Get tabular features
    tabular_features = sorted([
        c for c in all_cols 
        if c not in full_exclusion
    ])
    
    if verbose:
        print(f"📊 Identified {len(tabular_features)} tabular features")
        print(f"   (Excluded: {len(embedding_cols)} embedding cols, "
              f"{len(exclude_columns)} system cols, 1 outcome)")
        if tabular_features:
            print(f"   Sample features: {tabular_features[:5]}...")
    
    return tabular_features


def get_all_feature_columns(
    df: pd.DataFrame,
    outcome_column: str = 'ip6',
    exclude_columns: Optional[frozenset] = None,
    verbose: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Get both embedding and tabular feature columns from a DataFrame.
    
    Args:
        df: Input dataframe
        outcome_column: Name of outcome column to exclude
        exclude_columns: Set of column names to exclude
        verbose: Whether to print summary
        
    Returns:
        Tuple of (embedding_columns, tabular_columns)
        
    Example:
        emb_cols, tab_cols = get_all_feature_columns(df, 'ip6')
        print(f"Embeddings: {len(emb_cols)}, Tabular: {len(tab_cols)}")
    """
    embedding_cols = identify_embedding_columns(df)
    tabular_cols = identify_tabular_feature_columns(
        df, outcome_column, exclude_columns, verbose=False
    )
    
    if verbose:
        print(f"📊 Feature columns identified:")
        print(f"   Embedding features: {len(embedding_cols)}")
        print(f"   Tabular features: {len(tabular_cols)}")
        print(f"   Total: {len(embedding_cols) + len(tabular_cols)}")
    
    return embedding_cols, tabular_cols


def load_embeddings_from_local(
    embedding_path: str,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load embeddings from local NPZ file.
    
    Supports:
    - Direct path to .npz file
    - Directory containing embeddings_*.npz files (uses most recent)
    - Glob pattern (uses most recent match)
    
    Args:
        embedding_path: Path to NPZ file, directory, or glob pattern
        verbose: Whether to print loading info
        
    Returns:
        Tuple of (embeddings, individual_ids, index_dts)
        - embeddings: numpy array [num_members, embedding_dim]
        - individual_ids: numpy array of member IDs
        - index_dts: numpy array of index dates
        
    Example:
        embeddings, ids, dates = load_embeddings_from_local(
            "embedding_output/exp2b_flash_learned_pool/"
        )
    """
    if os.path.isdir(embedding_path):
        npz_files = glob.glob(os.path.join(embedding_path, "embeddings_*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No NPZ files found in {embedding_path}")
        npz_path = sorted(npz_files)[-1]  # Use most recent
        if verbose:
            print(f"📁 Found {len(npz_files)} NPZ files, using most recent: {os.path.basename(npz_path)}")
    elif '*' in embedding_path:
        npz_files = glob.glob(embedding_path)
        if not npz_files:
            raise FileNotFoundError(f"No files matching pattern: {embedding_path}")
        npz_path = sorted(npz_files)[-1]
        if verbose:
            print(f"📁 Found {len(npz_files)} matching files, using: {npz_path}")
    else:
        npz_path = embedding_path
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"File not found: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    embeddings = data['embeddings']
    individual_ids = data['individual_ids']
    index_dts = data['index_dts']
    
    if verbose:
        print(f"✅ Loaded embeddings from: {npz_path}")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Members: {len(individual_ids):,}")
    
    return embeddings, individual_ids, index_dts


def load_embeddings_from_bigquery(
    table_id: str,
    project_id: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load embeddings from BigQuery table.
    
    Args:
        table_id: Full table ID (project.dataset.table) or just table name
        project_id: GCP project ID (optional if table_id is fully qualified)
        verbose: Whether to print loading info
        
    Returns:
        DataFrame with individual_id, index_dt, and embedding_0...embedding_N columns
        
    Example:
        df = load_embeddings_from_bigquery(
            "edp-prod-storage.edp_ent_sdoheir_cns.a964286_commercial_embeddings"
        )
    """
    try:
        from google.cloud import bigquery
    except ImportError:
        raise ImportError("google-cloud-bigquery is required for BigQuery loading. "
                         "Install with: pip install google-cloud-bigquery")
    
    client = bigquery.Client(project=project_id) if project_id else bigquery.Client()
    
    query = f"SELECT * FROM `{table_id}`"
    
    if verbose:
        print(f"📥 Loading embeddings from BigQuery: {table_id}")
    
    df = client.query(query).to_dataframe()
    
    if verbose:
        embedding_cols = [c for c in df.columns if c.startswith('embedding_')]
        print(f"✅ Loaded {len(df):,} rows with {len(embedding_cols)} embedding dimensions")
    
    return df


def load_tabular_features_from_bigquery(
    table_id: str,
    project_id: Optional[str] = None,
    columns: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load tabular features from BigQuery table.
    
    Args:
        table_id: Full table ID (project.dataset.table)
        project_id: GCP project ID (optional if table_id is fully qualified)
        columns: Specific columns to load (None = all columns)
        verbose: Whether to print loading info
        
    Returns:
        DataFrame with features and outcome columns
        
    Example:
        df_features = load_tabular_features_from_bigquery(
            "edp-prod-storage.edp_ent_sdoheir_cns.a964286_commercial_ip_features"
        )
    """
    try:
        from google.cloud import bigquery
    except ImportError:
        raise ImportError("google-cloud-bigquery is required for BigQuery loading. "
                         "Install with: pip install google-cloud-bigquery")
    
    client = bigquery.Client(project=project_id) if project_id else bigquery.Client()
    
    if columns:
        col_str = ", ".join(columns)
        query = f"SELECT {col_str} FROM `{table_id}`"
    else:
        query = f"SELECT * FROM `{table_id}`"
    
    if verbose:
        print(f"📥 Loading features from BigQuery: {table_id}")
    
    df = client.query(query).to_dataframe()
    
    if verbose:
        print(f"✅ Loaded {len(df):,} rows with {len(df.columns)} columns")
    
    return df


def create_embedding_dataframe(
    embeddings: np.ndarray,
    individual_ids: np.ndarray,
    index_dts: np.ndarray
) -> pd.DataFrame:
    """
    Convert embedding arrays to a DataFrame.
    
    Args:
        embeddings: numpy array [num_members, embedding_dim]
        individual_ids: numpy array of member IDs
        index_dts: numpy array of index dates
        
    Returns:
        DataFrame with columns: individual_id, index_dt, embedding_0...embedding_N
        
    Example:
        embeddings, ids, dates = load_embeddings_from_local("path/to/embeddings/")
        df_emb = create_embedding_dataframe(embeddings, ids, dates)
    """
    embedding_dim = embeddings.shape[1]
    embedding_cols = [f'embedding_{i}' for i in range(embedding_dim)]
    
    df = pd.DataFrame({
        'individual_id': individual_ids,
        'index_dt': pd.to_datetime(index_dts).strftime('%Y-%m-%d')
    })
    
    embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)
    return pd.concat([df, embedding_df], axis=1)


def join_embeddings_with_outcomes(
    emb_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    outcome_column: str = 'ip6',
    id_columns: List[str] = ['individual_id', 'index_dt'],
    additional_columns: Optional[List[str]] = None,
    include_all_tabular: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Join embeddings with outcome data.
    
    Args:
        emb_df: DataFrame with embeddings (from create_embedding_dataframe or BigQuery)
        outcomes_df: DataFrame with outcomes and optional features
        outcome_column: Name of binary outcome column (0/1)
        id_columns: Columns to join on (default: individual_id, index_dt)
        additional_columns: Extra columns to keep from outcomes_df (ignored if include_all_tabular=True)
        include_all_tabular: If True, include ALL tabular features from outcomes_df
                             (excludes EXCLUDE_COLUMNS, embedding cols, and outcome)
        verbose: Whether to print join info
        
    Returns:
        Merged DataFrame with embeddings, outcome, and additional columns
        
    Example (specific columns):
        df_merged = join_embeddings_with_outcomes(
            emb_df=df_embeddings,
            outcomes_df=df_features,
            outcome_column='ip6',
            additional_columns=['age', 'gender_cd']
        )
        
    Example (all tabular features for cluster analysis):
        df_merged = join_embeddings_with_outcomes(
            emb_df=df_embeddings,
            outcomes_df=df_features,
            outcome_column='ip6',
            include_all_tabular=True
        )
    """
    # Standardize date format
    outcomes_df = outcomes_df.copy()
    emb_df = emb_df.copy()
    
    if 'index_dt' in id_columns:
        if 'index_dt' in outcomes_df.columns:
            outcomes_df['index_dt'] = pd.to_datetime(outcomes_df['index_dt']).dt.strftime('%Y-%m-%d')
        if 'index_dt' in emb_df.columns:
            emb_df['index_dt'] = pd.to_datetime(emb_df['index_dt']).dt.strftime('%Y-%m-%d')
    
    if 'individual_id' in id_columns:
        if 'individual_id' in outcomes_df.columns:
            outcomes_df['individual_id'] = outcomes_df['individual_id'].astype(str)
        if 'individual_id' in emb_df.columns:
            emb_df['individual_id'] = emb_df['individual_id'].astype(str)
    
    # Determine which columns to keep from outcomes_df
    if include_all_tabular:
        # Get all tabular features (excludes EXCLUDE_COLUMNS, embeddings, outcome)
        tabular_cols = identify_tabular_feature_columns(
            outcomes_df, outcome_column, verbose=False
        )
        keep_cols = list(id_columns) + [outcome_column] + tabular_cols
        if verbose:
            print(f"📋 Including {len(tabular_cols)} tabular features for analysis")
    else:
        keep_cols = list(id_columns) + [outcome_column]
        if additional_columns:
            keep_cols.extend([c for c in additional_columns if c not in keep_cols])
    
    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in outcomes_df.columns]
    outcomes_subset = outcomes_df[keep_cols]
    
    # Inner join
    df_merged = emb_df.merge(
        outcomes_subset,
        on=id_columns,
        how='inner'
    )
    
    # Remove duplicates
    df_merged = df_merged.drop_duplicates(
        subset=id_columns,
        keep='last'
    )
    
    if verbose:
        n_emb = len(emb_df)
        n_outcomes = len(outcomes_df)
        n_merged = len(df_merged)
        print(f"📊 Join results:")
        print(f"   Embeddings: {n_emb:,} rows")
        print(f"   Outcomes: {n_outcomes:,} rows")
        print(f"   Merged: {n_merged:,} rows ({n_merged/n_emb*100:.1f}% match rate)")
        print(f"   Outcome prevalence: {df_merged[outcome_column].mean()*100:.2f}%")
        if include_all_tabular:
            print(f"   Tabular features included: {len(tabular_cols)}")
    
    return df_merged


def load_and_prepare_data(
    embedding_source: str,
    outcome_source: Union[str, pd.DataFrame],
    outcome_column: str = 'ip6',
    from_bigquery: bool = False,
    additional_columns: Optional[List[str]] = None,
    include_all_tabular: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    High-level function to load embeddings and join with outcomes.
    
    This is a convenience function that combines:
    1. Loading embeddings (from local NPZ or BigQuery)
    2. Converting to DataFrame (if needed)
    3. Loading outcomes (from BigQuery or using provided DataFrame)
    4. Joining embeddings with outcomes
    
    Args:
        embedding_source: Path to local NPZ/directory OR BigQuery table ID
        outcome_source: BigQuery table ID OR pre-loaded DataFrame
        outcome_column: Name of binary outcome column (0/1)
        from_bigquery: Whether embedding_source is a BigQuery table
        additional_columns: Extra columns to keep from outcome table (ignored if include_all_tabular=True)
        include_all_tabular: If True, include ALL tabular features for cluster analysis
                             on non-embedding features
        verbose: Whether to print progress
        
    Returns:
        DataFrame ready for cluster analysis (embeddings + outcome + optional tabular features)
        
    Example (embeddings only):
        df = load_and_prepare_data(
            embedding_source="embedding_output/exp2b/",
            outcome_source="project.dataset.features_table",
            outcome_column='ip6'
        )
        
    Example (embeddings + all tabular features for comparison):
        df = load_and_prepare_data(
            embedding_source="embedding_output/exp2b/",
            outcome_source="project.dataset.features_table",
            outcome_column='ip6',
            include_all_tabular=True  # Include all tabular features
        )
    """
    if verbose:
        print("="*70)
        print("📥 LOADING AND PREPARING DATA")
        print("="*70 + "\n")
    
    # Step 1: Load embeddings
    if from_bigquery:
        if verbose:
            print("Step 1: Loading embeddings from BigQuery...")
        df_emb = load_embeddings_from_bigquery(embedding_source, verbose=verbose)
    else:
        if verbose:
            print("Step 1: Loading embeddings from local file...")
        embeddings, ids, dates = load_embeddings_from_local(embedding_source, verbose=verbose)
        df_emb = create_embedding_dataframe(embeddings, ids, dates)
    
    # Step 2: Load or use outcomes
    if isinstance(outcome_source, pd.DataFrame):
        if verbose:
            print("\nStep 2: Using provided outcomes DataFrame...")
        df_outcomes = outcome_source
    else:
        if verbose:
            print("\nStep 2: Loading outcomes from BigQuery...")
        df_outcomes = load_tabular_features_from_bigquery(outcome_source, verbose=verbose)
    
    # Step 3: Join
    if verbose:
        print("\nStep 3: Joining embeddings with outcomes...")
    df_merged = join_embeddings_with_outcomes(
        emb_df=df_emb,
        outcomes_df=df_outcomes,
        outcome_column=outcome_column,
        additional_columns=additional_columns,
        include_all_tabular=include_all_tabular,
        verbose=verbose
    )
    
    if verbose:
        print("\n" + "="*70)
        print("✅ DATA READY FOR CLUSTER ANALYSIS")
        print("="*70 + "\n")
    
    return df_merged


# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def identify_embedding_columns(
    df: pd.DataFrame,
    prefix: str = 'embedding_'
) -> List[str]:
    """
    Identify embedding columns in a dataframe.
    
    Args:
        df: Input dataframe
        prefix: Prefix for embedding columns (default: 'embedding_')
        
    Returns:
        List of embedding column names, sorted numerically if possible
    """
    embedding_cols = [col for col in df.columns if col.startswith(prefix)]
    
    if len(embedding_cols) == 0:
        raise ValueError(f"No embedding columns found with prefix '{prefix}'")
    
    # Try numeric sorting, fallback to lexicographic if suffixes aren't purely numeric
    def _sort_key(col_name: str):
        suffix = col_name.replace(prefix, '')
        try:
            return (0, int(suffix))  # Numeric sort
        except ValueError:
            return (1, suffix)  # Lexicographic fallback
    
    return sorted(embedding_cols, key=_sort_key)


def balance_dataset_for_clustering(
    df: pd.DataFrame,
    outcome_column: str,
    config: Optional[ClusterConfig] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Balance dataset by downsampling majority class.
    
    This addresses class imbalance for UMAP visualization by creating
    a 50/50 split between positive and negative outcome cases.
    
    Args:
        df: Input dataframe with outcome column
        outcome_column: Name of binary outcome column (0/1)
        config: ClusterConfig with balancing parameters
        verbose: Whether to print progress
        
    Returns:
        Balanced dataframe (50/50 split) or original if already balanced
        
    Raises:
        ValueError: If outcome_column is missing or dataframe is empty
    """
    if df is None or len(df) == 0:
        raise ValueError("Input dataframe is empty or None")
    
    if outcome_column not in df.columns:
        raise ValueError(f"Outcome column '{outcome_column}' not found in dataframe. "
                        f"Available columns: {list(df.columns[:10])}...")
    
    if config is None:
        config = ClusterConfig()
    
    if not config.balance_classes:
        if verbose:
            print("⚙️ Using FULL dataset (balanced sampling disabled)")
        return df.copy()
    
    outcome_prevalence = df[outcome_column].mean()
    
    if verbose:
        print("🎯 Preparing dataset for clustering analysis...\n")
        print(f"   Outcome prevalence: {outcome_prevalence*100:.1f}%")
    
    # Check if balancing is needed
    if config.imbalance_threshold <= outcome_prevalence <= (1 - config.imbalance_threshold):
        if verbose:
            print(f"   → Reasonably balanced (minority class >= {config.imbalance_threshold*100:.0f}%)")
            print("   → Using full dataset (no downsampling needed)")
        return df.copy()
    
    if verbose:
        print(f"   → Highly imbalanced! (Minority class < {config.imbalance_threshold*100:.0f}%)")
        print("\n💡 SOLUTION: Downsample majority class to create 50/50 balance.\n")
    
    # Separate classes
    df_pos = df[df[outcome_column] == 1]
    df_neg = df[df[outcome_column] == 0]
    
    # Validate we have both classes
    if len(df_pos) == 0:
        raise ValueError(f"No positive cases (outcome_column=1) found in dataset. "
                        f"Cannot perform cluster analysis.")
    if len(df_neg) == 0:
        raise ValueError(f"No negative cases (outcome_column=0) found in dataset. "
                        f"Cannot perform cluster analysis.")
    
    if verbose:
        print(f"📊 Original IMBALANCED distribution:")
        print(f"   Outcome=1 (minority): {len(df_pos):,} ({len(df_pos)/len(df)*100:.2f}%)")
        print(f"   Outcome=0 (majority): {len(df_neg):,} ({len(df_neg)/len(df)*100:.2f}%)")
        print(f"   Ratio: 1:{len(df_neg)/len(df_pos):.1f}")
    
    # Downsample to minority size
    minority_size = min(len(df_pos), len(df_neg))
    
    if verbose:
        print(f"\n⚙️ Downsampling to {minority_size:,} samples per class...")
    
    df_pos_balanced = df_pos.sample(
        n=minority_size, 
        random_state=config.random_state, 
        replace=False
    )
    df_neg_balanced = df_neg.sample(
        n=minority_size, 
        random_state=config.random_state, 
        replace=False
    )
    
    # Combine and shuffle
    df_balanced = pd.concat([df_pos_balanced, df_neg_balanced]).sample(
        frac=1, 
        random_state=config.random_state
    ).reset_index(drop=True)
    
    if verbose:
        print(f"\n✅ BALANCED dataset created:")
        print(f"   Outcome=1: {(df_balanced[outcome_column]==1).sum():,} (50.0%)")
        print(f"   Outcome=0: {(df_balanced[outcome_column]==0).sum():,} (50.0%)")
        print(f"   Total samples: {len(df_balanced):,}")
        print(f"\n📉 Dataset reduction: {len(df):,} → {len(df_balanced):,} "
              f"({len(df_balanced)/len(df)*100:.1f}% of original)")
    
    return df_balanced


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def compute_embedding_correlations(
    df: pd.DataFrame,
    outcome_column: str,
    embedding_cols: Optional[List[str]] = None,
    correlation_threshold: float = 0.05,
    verbose: bool = True
) -> CorrelationResult:
    """
    Compute point-biserial correlation between each embedding dimension and outcome.
    
    Args:
        df: Input dataframe with embedding columns and outcome
        outcome_column: Name of binary outcome column (0/1)
        embedding_cols: List of embedding column names (auto-detected if None)
        correlation_threshold: Threshold for "strong predictor" (default: 0.05)
        verbose: Whether to print progress
        
    Returns:
        CorrelationResult with correlation statistics
    """
    if embedding_cols is None:
        embedding_cols = identify_embedding_columns(df)
    
    if verbose:
        print(f"🔍 Analyzing correlation between embeddings and outcome...\n")
    
    y_outcome = df[outcome_column].values
    
    # Validate outcome is binary
    unique_outcomes = df[outcome_column].dropna().unique()
    if not set(unique_outcomes).issubset({0, 1}):
        raise ValueError(f"Outcome column '{outcome_column}' must be binary (0/1). "
                        f"Found values: {unique_outcomes}")
    
    # Calculate point-biserial correlation for each dimension
    correlations = []
    for i, col in enumerate(embedding_cols):
        x_values = df[col].values
        
        # Handle NaN/constant values gracefully
        valid_mask = ~(np.isnan(x_values) | np.isnan(y_outcome))
        if valid_mask.sum() < 2 or np.std(x_values[valid_mask]) == 0:
            # Constant or insufficient data - correlation is undefined
            corr, pval = 0.0, 1.0
        else:
            try:
                corr, pval = stats.pointbiserialr(y_outcome[valid_mask], x_values[valid_mask])
                # Handle NaN result from scipy
                if np.isnan(corr):
                    corr, pval = 0.0, 1.0
            except Exception:
                corr, pval = 0.0, 1.0
        
        correlations.append({
            'embedding': col,
            'dimension': i,
            'correlation': corr,
            'abs_correlation': abs(corr),
            'pval': pval
        })
    
    df_corr = pd.DataFrame(correlations).sort_values(
        'abs_correlation', 
        ascending=False
    ).reset_index(drop=True)
    
    # Compute summary statistics
    result = CorrelationResult(
        df_correlations=df_corr,
        n_samples=len(df),
        n_dimensions=len(embedding_cols),
        mean_abs_correlation=df_corr['abs_correlation'].mean(),
        max_abs_correlation=df_corr['abs_correlation'].max(),
        top_predictor=df_corr.iloc[0]['embedding'],
        top_predictor_corr=df_corr.iloc[0]['correlation'],
        strong_predictors_count=(df_corr['abs_correlation'] > correlation_threshold).sum(),
        very_strong_predictors_count=(df_corr['abs_correlation'] > 0.10).sum()
    )
    
    if verbose:
        print(f"📊 Correlation Summary:")
        print(f"   Sample size: {result.n_samples:,}")
        print(f"   Embedding dimensions: {result.n_dimensions}")
        print(f"   Top predictor: {result.top_predictor} (r={result.top_predictor_corr:.4f})")
        print(f"   Mean absolute correlation: {result.mean_abs_correlation:.4f}")
        print(f"   Strong predictors (|r| > {correlation_threshold}): "
              f"{result.strong_predictors_count} / {result.n_dimensions}")
        print(f"   Very strong predictors (|r| > 0.10): "
              f"{result.very_strong_predictors_count} / {result.n_dimensions}")
    
    return result


def visualize_correlation_analysis(
    correlation_result: CorrelationResult,
    outcome_label: str = "Outcome",
    top_n: int = 20,
    figsize: Tuple[int, int] = (14, 6),
    show_plot: bool = True
) -> plt.Figure:
    """
    Visualize correlation analysis results.
    
    Args:
        correlation_result: Result from compute_embedding_correlations
        outcome_label: Human-readable label for outcome
        top_n: Number of top dimensions to show in bar chart
        figsize: Figure size
        show_plot: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    df_corr = correlation_result.df_correlations
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Top N embedding dimensions
    ax = axes[0]
    top_embs = df_corr.head(top_n)
    colors = ['red' if x < 0 else 'green' for x in top_embs['correlation']]
    ax.barh(range(len(top_embs)), top_embs['correlation'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_embs)))
    ax.set_yticklabels(top_embs['embedding'])
    ax.set_xlabel(f'Correlation with {outcome_label}')
    ax.set_title(f'Top {top_n} Embedding Dimensions by {outcome_label} Correlation')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(alpha=0.3)
    
    # Plot 2: Distribution of correlations
    ax = axes[1]
    ax.hist(df_corr['correlation'], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel(f'Correlation with {outcome_label}')
    ax.set_ylabel('Number of Embeddings')
    ax.set_title('Distribution of Embedding-Outcome Correlations')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    return fig


def visualize_effect_size_thresholds(
    correlation_result: CorrelationResult,
    outcome_label: str = "Outcome",
    figsize: Tuple[int, int] = (16, 5),
    show_plot: bool = True,
    verbose: bool = True
) -> plt.Figure:
    """
    Visualize effect size thresholds for correlation analysis.
    
    Args:
        correlation_result: Result from compute_embedding_correlations
        outcome_label: Human-readable label for outcome
        figsize: Figure size
        show_plot: Whether to display the plot
        verbose: Whether to print threshold analysis
        
    Returns:
        matplotlib Figure object
    """
    df_corr = correlation_result.df_correlations
    n_dims = correlation_result.n_dimensions
    
    if verbose:
        print("🔬 Analyzing correlation thresholds...\n")
        
        # Effect size interpretation
        thresholds = {
            'Negligible': 0.01,
            'Very Small': 0.02,
            'Small': 0.05,
            'Small-Medium': 0.10,
            'Medium': 0.20,
            'Large': 0.30,
            'Very Large': 0.50
        }
        
        print(f"📏 Effect Size Distribution:")
        for label, thresh in thresholds.items():
            count = (df_corr['abs_correlation'] >= thresh).sum()
            pct = count / n_dims * 100
            print(f"   |r| ≥ {thresh:.2f} ({label:>12}): {count:3d} / {n_dims} ({pct:5.1f}%)")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax = axes[0]
    ax.hist(df_corr['abs_correlation'], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0.01, color='gray', linestyle='--', linewidth=2, 
               label='Negligible (0.01)', alpha=0.5)
    ax.axvline(x=0.05, color='orange', linestyle='--', linewidth=2, label='Small (0.05)')
    ax.axvline(x=0.10, color='red', linestyle='--', linewidth=2, label='Medium (0.10)')
    ax.axvline(x=0.20, color='darkred', linestyle='--', linewidth=3, label='Large (0.20)')
    ax.set_xlabel(f'Absolute Correlation with {outcome_label}')
    ax.set_ylabel('Number of Embedding Dimensions')
    ax.set_title(f'Distribution of Embedding-{outcome_label} Correlations\n'
                 '(Multiple Effect Size Thresholds)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1]
    sorted_corr = np.sort(df_corr['abs_correlation'].values)[::-1]
    ax.plot(range(1, n_dims + 1), sorted_corr, linewidth=2, color='blue')
    ax.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='Negligible')
    ax.axhline(y=0.05, color='orange', linestyle='--', label='Small')
    ax.axhline(y=0.10, color='red', linestyle='--', label='Medium')
    ax.axhline(y=0.20, color='darkred', linestyle='--', linewidth=2, label='Large')
    ax.set_xlabel('Embedding Dimension Rank (Best to Worst)')
    ax.set_ylabel('Absolute Correlation')
    ax.set_title('Ranked Embedding Correlations\n(How Many Dimensions Are "Strong"?)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    return fig


# =============================================================================
# UMAP PROJECTION
# =============================================================================

def run_umap_projection(
    df: pd.DataFrame,
    embedding_cols: List[str],
    config: Optional[UMAPConfig] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, float]:
    """
    Run UMAP dimensionality reduction (256D → 2D).
    
    Args:
        df: Input dataframe with embedding columns
        embedding_cols: List of embedding column names
        config: UMAPConfig with UMAP parameters
        verbose: Whether to print progress
        
    Returns:
        Tuple of (dataframe with umap_x/umap_y columns, 2D embedding array, runtime in seconds)
        
    Raises:
        ValueError: If embedding columns are missing or dataframe is empty
    """
    if df is None or len(df) == 0:
        raise ValueError("Input dataframe is empty or None")
    
    if not embedding_cols:
        raise ValueError("embedding_cols list is empty")
    
    missing_cols = [col for col in embedding_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing embedding columns in dataframe: {missing_cols[:5]}... "
                        f"(showing first 5 of {len(missing_cols)})")
    
    if config is None:
        config = UMAPConfig()
    
    if verbose:
        print("🗺️ Running UMAP dimensionality reduction (256D → 2D)...")
        print(f"   Samples: {len(df):,}")
        print(f"   Parameters: n_neighbors={config.n_neighbors}, min_dist={config.min_dist}")
        print(f"   Estimated time: {len(df)//50000 + 1}-{len(df)//20000 + 2} minutes\n")
    
    X = df[embedding_cols].values
    
    start_time = time.time()
    
    reducer = umap.UMAP(
        n_components=config.n_components,
        n_neighbors=config.n_neighbors,
        min_dist=config.min_dist,
        metric=config.metric,
        random_state=config.random_state,
        verbose=config.verbose
    )
    
    embedding_2d = reducer.fit_transform(X)
    umap_time = time.time() - start_time
    
    # Add UMAP coordinates to dataframe
    df_result = df.copy()
    df_result['umap_x'] = embedding_2d[:, 0]
    df_result['umap_y'] = embedding_2d[:, 1]
    
    if verbose:
        print(f"\n✅ UMAP complete in {umap_time/60:.1f} minutes ({umap_time:.1f} seconds)")
        print(f"   2D embedding shape: {embedding_2d.shape}")
    
    return df_result, embedding_2d, umap_time


# =============================================================================
# CLUSTER SEPARATION QUANTIFICATION
# =============================================================================

def compute_cluster_separation(
    df: pd.DataFrame,
    outcome_column: str,
    umap_x_col: str = 'umap_x',
    umap_y_col: str = 'umap_y',
    outcome_label: str = "Outcome",
    verbose: bool = True
) -> ClusterSeparationResult:
    """
    Compute cluster separation metrics (Cohen's D) from UMAP projection.
    
    Args:
        df: Dataframe with UMAP coordinates and outcome
        outcome_column: Name of binary outcome column (0/1)
        umap_x_col: Name of UMAP x coordinate column
        umap_y_col: Name of UMAP y coordinate column
        outcome_label: Human-readable label for outcome
        verbose: Whether to print progress
        
    Returns:
        ClusterSeparationResult with separation metrics
        
    Raises:
        ValueError: If required columns are missing
    """
    # Validate required columns exist
    required_cols = [outcome_column, umap_x_col, umap_y_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. "
                        f"Ensure UMAP has been run first.")
    
    if verbose:
        print(f"📊 Quantifying cluster separation for {outcome_label}...\n")
    
    # Split by outcome
    df_neg = df[df[outcome_column] == 0]
    df_pos = df[df[outcome_column] == 1]
    
    # Calculate centroids
    centroid_neg = df_neg[[umap_x_col, umap_y_col]].mean().values
    centroid_pos = df_pos[[umap_x_col, umap_y_col]].mean().values
    centroid_distance = np.linalg.norm(centroid_pos - centroid_neg)
    
    # Within-group variance
    var_neg = df_neg[[umap_x_col, umap_y_col]].var().mean()
    var_pos = df_pos[[umap_x_col, umap_y_col]].var().mean()
    
    # Cohen's D
    pooled_std = np.sqrt((var_neg + var_pos) / 2)
    cohens_d = centroid_distance / pooled_std if pooled_std > 0 else 0.0
    
    # Interpretation
    if cohens_d < 0.2:
        interpretation = "Negligible (no clustering)"
    elif cohens_d < 0.5:
        interpretation = "Small effect"
    elif cohens_d < 0.8:
        interpretation = "Medium effect (noticeable separation)"
    else:
        interpretation = "Large effect (strong clustering)"
    
    result = ClusterSeparationResult(
        centroid_pos=centroid_pos,
        centroid_neg=centroid_neg,
        centroid_distance=centroid_distance,
        variance_pos=var_pos,
        variance_neg=var_neg,
        pooled_std=pooled_std,
        cohens_d=cohens_d,
        interpretation=interpretation
    )
    
    if verbose:
        print(f"🎯 Cluster Centroids:")
        print(f"   {outcome_label}=0 Centroid: ({centroid_neg[0]:.3f}, {centroid_neg[1]:.3f})")
        print(f"   {outcome_label}=1 Centroid: ({centroid_pos[0]:.3f}, {centroid_pos[1]:.3f})")
        print(f"   Euclidean Distance: {centroid_distance:.3f}")
        
        print(f"\n📏 Within-Group Variance:")
        print(f"   {outcome_label}=0 Variance: {var_neg:.3f}")
        print(f"   {outcome_label}=1 Variance: {var_pos:.3f}")
        
        print(f"\n📈 Effect Size (Cohen's D): {cohens_d:.3f}")
        print(f"   Interpretation: {interpretation}")
        
        print(f"\n💡 What this means:")
        if cohens_d >= 0.5:
            print(f"   ✅ Embeddings meaningfully separate {outcome_label} groups!")
            print(f"   ✅ Transformer captures clinical patterns relevant to {outcome_label}")
        else:
            print(f"   ⚠️ Limited separation between {outcome_label} groups")
            print(f"   ⚠️ Embeddings may not strongly capture this specific outcome")
    
    return result


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_clusters(
    df: pd.DataFrame,
    outcome_column: str,
    outcome_label: str = "Outcome",
    umap_x_col: str = 'umap_x',
    umap_y_col: str = 'umap_y',
    figsize: Tuple[int, int] = (16, 12),
    show_plot: bool = True
) -> plt.Figure:
    """
    Create 4-panel cluster visualization.
    
    Panels:
    - Top-Left: All points (gradient by outcome)
    - Top-Right: Negative cases only (blue)
    - Bottom-Left: Positive cases only (red)
    - Bottom-Right: Risk heatmap
    
    Args:
        df: Dataframe with UMAP coordinates and outcome
        outcome_column: Name of binary outcome column (0/1)
        outcome_label: Human-readable label for outcome
        umap_x_col: Name of UMAP x coordinate column
        umap_y_col: Name of UMAP y coordinate column
        figsize: Figure size
        show_plot: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    print("🎨 Creating visualizations...\n")
    
    # Validate required columns exist
    required_cols = [outcome_column, umap_x_col, umap_y_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Split by outcome
    df_neg = df[df[outcome_column] == 0]
    df_pos = df[df[outcome_column] == 1]
    
    if len(df_neg) == 0 or len(df_pos) == 0:
        raise ValueError(f"Cannot visualize: one or both outcome classes are empty. "
                        f"Positive: {len(df_pos)}, Negative: {len(df_neg)}")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: All points colored by outcome (gradient)
    ax = axes[0, 0]
    scatter = ax.scatter(
        df[umap_x_col], 
        df[umap_y_col],
        c=df[outcome_column],
        cmap='RdYlGn_r',  # Red = high risk (1), Green = low risk (0)
        alpha=0.2,
        s=5,
        edgecolors='none'
    )
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_title(f'All Points: Gradient View\nColored by {outcome_label}')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f'{outcome_label} (0=No, 1=Yes)')
    ax.grid(alpha=0.3)
    
    # Plot 2: Negative cases only
    ax = axes[0, 1]
    ax.scatter(df_neg[umap_x_col], df_neg[umap_y_col], 
               c='blue', alpha=0.3, s=3, edgecolors='none')
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_title(f'No {outcome_label} Cases Only\n(n={len(df_neg):,})')
    ax.grid(alpha=0.3)
    
    # Plot 3: Positive cases only
    ax = axes[1, 0]
    ax.scatter(df_pos[umap_x_col], df_pos[umap_y_col], 
               c='red', alpha=0.3, s=3, edgecolors='none')
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_title(f'{outcome_label} Cases Only\n(n={len(df_pos):,})')
    ax.grid(alpha=0.3)
    
    # Plot 4: Risk heatmap
    ax = axes[1, 1]
    
    # Create 2D histogram
    h, xedges, yedges = np.histogram2d(
        df[umap_x_col], 
        df[umap_y_col], 
        bins=50
    )
    
    h_pos, _, _ = np.histogram2d(
        df_pos[umap_x_col], 
        df_pos[umap_y_col], 
        bins=[xedges, yedges]
    )
    
    # Calculate outcome prevalence per bin
    with np.errstate(divide='ignore', invalid='ignore'):
        prevalence = h_pos / h
        prevalence = np.nan_to_num(prevalence, nan=0)
    
    im = ax.imshow(
        prevalence.T, 
        origin='lower',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap='RdYlGn_r',
        aspect='auto',
        alpha=0.8,
        vmin=0,
        vmax=1
    )
    
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_title(f'{outcome_label} Risk Heatmap\n(Red = High Risk, Green = Low Risk)')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{outcome_label} Prevalence (0-100%)')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    print("✅ Visualizations complete")
    print(f"\n💡 How to Interpret the 4-Panel View:")
    print(f"\n   📊 TOP LEFT (Gradient): Overall distribution of both classes")
    print(f"   📊 TOP RIGHT (Blue): Where No-{outcome_label} cases cluster")
    print(f"   📊 BOTTOM LEFT (Red): Where {outcome_label} cases cluster")
    print(f"   📊 BOTTOM RIGHT (Heatmap): Risk 'hot zones' (red) vs 'safe zones' (green)")
    print(f"\n   ✅ GOOD CLUSTERING: Blue and Red plots show DIFFERENT dense regions")
    print(f"   ❌ POOR CLUSTERING: Blue and Red plots look identical")
    
    return fig


def visualize_alternative_comparison(
    df_viz: pd.DataFrame,
    outcome_column: str,
    cohens_d: float,
    embedding_name: str,
    outcome_label: str = "Outcome",
    umap_x_col: str = 'umap_x',
    umap_y_col: str = 'umap_y',
    figsize: Tuple[int, int] = (16, 6),
    show_plot: bool = True
) -> plt.Figure:
    """
    Visualize alternative embedding with scatter and heatmap.
    
    Args:
        df_viz: Dataframe with UMAP coordinates and outcome
        outcome_column: Name of binary outcome column (0/1)
        cohens_d: Cohen's D value for this embedding
        embedding_name: Name of embedding source
        outcome_label: Human-readable label for outcome
        umap_x_col: Name of UMAP x coordinate column
        umap_y_col: Name of UMAP y coordinate column
        figsize: Figure size
        show_plot: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    print(f"🎨 Creating visualizations for {embedding_name}...\n")
    
    # Validate required columns exist
    required_cols = [outcome_column, umap_x_col, umap_y_col]
    missing_cols = [col for col in required_cols if col not in df_viz.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df_neg = df_viz[df_viz[outcome_column] == 0]
    df_pos = df_viz[df_viz[outcome_column] == 1]
    
    if len(df_neg) == 0 or len(df_pos) == 0:
        raise ValueError(f"Cannot visualize: one or both outcome classes are empty. "
                        f"Positive: {len(df_pos)}, Negative: {len(df_neg)}")
    
    centroid_neg = df_neg[[umap_x_col, umap_y_col]].mean().values
    centroid_pos = df_pos[[umap_x_col, umap_y_col]].mean().values
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Scatter with centroids
    ax = axes[0]
    ax.scatter(df_neg[umap_x_col], df_neg[umap_y_col], 
               c='lightblue', alpha=0.3, s=10, 
               label=f'No {outcome_label} (n={len(df_neg):,})', edgecolors='none')
    ax.scatter(df_pos[umap_x_col], df_pos[umap_y_col], 
               c='red', alpha=0.6, s=20, 
               label=f'{outcome_label} (n={len(df_pos):,})', 
               edgecolors='darkred', linewidths=0.5)
    ax.scatter(*centroid_neg, marker='X', s=300, c='blue', 
               edgecolors='black', linewidths=2, label='Neg Centroid', zorder=10)
    ax.scatter(*centroid_pos, marker='X', s=300, c='darkred', 
               edgecolors='black', linewidths=2, label='Pos Centroid', zorder=10)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_title(f'{embedding_name} Embeddings\nCohen\'s D = {cohens_d:.3f}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Heatmap
    ax = axes[1]
    h, xedges, yedges = np.histogram2d(
        df_viz[umap_x_col], df_viz[umap_y_col], bins=50
    )
    h_pos, _, _ = np.histogram2d(
        df_pos[umap_x_col], df_pos[umap_y_col], bins=[xedges, yedges]
    )
    
    with np.errstate(divide='ignore', invalid='ignore'):
        prevalence = h_pos / h
        prevalence = np.nan_to_num(prevalence, nan=0)
    
    im = ax.imshow(prevalence.T, origin='lower',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap='RdYlGn_r', aspect='auto', alpha=0.8)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_title(f'{outcome_label} Risk Density Map\n({embedding_name})')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{outcome_label} Prevalence')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    print("✅ Visualizations complete")
    
    return fig


# =============================================================================
# SUMMARY REPORTING
# =============================================================================

def generate_cluster_summary(
    correlation_result: CorrelationResult,
    separation_result: ClusterSeparationResult,
    embedding_name: str,
    outcome_label: str = "Outcome",
    correlation_threshold: float = 0.05,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generate summary of cluster analysis findings.
    
    Args:
        correlation_result: Result from compute_embedding_correlations
        separation_result: Result from compute_cluster_separation
        embedding_name: Name of embedding source
        outcome_label: Human-readable label for outcome
        correlation_threshold: Threshold for strong predictor count
        verbose: Whether to print summary
        
    Returns:
        Dictionary with summary statistics
    """
    cohens_d = separation_result.cohens_d
    interpretation = separation_result.interpretation
    
    summary = {
        'embedding_name': embedding_name,
        'outcome_label': outcome_label,
        'n_samples': correlation_result.n_samples,
        'n_dimensions': correlation_result.n_dimensions,
        
        # Correlation metrics
        'top_predictor': correlation_result.top_predictor,
        'top_predictor_corr': correlation_result.top_predictor_corr,
        'mean_abs_correlation': correlation_result.mean_abs_correlation,
        'strong_predictors_count': correlation_result.strong_predictors_count,
        'very_strong_predictors_count': correlation_result.very_strong_predictors_count,
        
        # Cluster separation metrics
        'centroid_distance': separation_result.centroid_distance,
        'cohens_d': cohens_d,
        'interpretation': interpretation,
        
        # Business implication
        'recommendation': _get_business_recommendation(cohens_d, outcome_label)
    }
    
    if verbose:
        print("="*80)
        print(f"🎯 KEY FINDINGS: Do Embeddings Cluster by {outcome_label}?")
        print("="*80)
        
        print(f"\n1️⃣ CLUSTER SEPARATION (UMAP 2D):")
        print(f"   - Embedding Source: {embedding_name}")
        print(f"   - Centroid Distance: {separation_result.centroid_distance:.3f}")
        print(f"   - Cohen's D Effect Size: {cohens_d:.3f} ({interpretation})")
        
        sep_msg = ('✅ Embeddings separate outcome groups well' if cohens_d > 0.5 
                   else '⚠️ Limited separation' if cohens_d > 0.3 
                   else '❌ Minimal/no separation')
        print(f"   - Interpretation: {sep_msg}")
        
        print(f"\n2️⃣ PREDICTIVE EMBEDDING DIMENSIONS:")
        print(f"   - Most Predictive: {correlation_result.top_predictor} "
              f"(r={correlation_result.top_predictor_corr:.4f})")
        print(f"   - Mean Abs Correlation: {correlation_result.mean_abs_correlation:.4f}")
        print(f"   - Strong Predictors (|r| > {correlation_threshold}): "
              f"{correlation_result.strong_predictors_count} / {correlation_result.n_dimensions}")
        print(f"   - Very Strong Predictors (|r| > 0.10): "
              f"{correlation_result.very_strong_predictors_count} / {correlation_result.n_dimensions}")
        
        print(f"\n3️⃣ BUSINESS IMPLICATIONS:")
        for line in summary['recommendation']:
            print(f"   {line}")
        
        print(f"\n{'='*80}")
    
    return summary


def _get_business_recommendation(cohens_d: float, outcome_label: str) -> List[str]:
    """Get business recommendation based on Cohen's D."""
    if cohens_d >= 0.8:
        return [
            f"✅✅ Embeddings are EXCEPTIONALLY VALUABLE for {outcome_label} prediction",
            "✅ Very clear visual separation between outcome groups",
            "✅ Embeddings likely outperform baseline features alone",
            "✅ Strong candidate for standalone risk stratification"
        ]
    elif cohens_d >= 0.5:
        return [
            f"✅ Embeddings are HIGHLY VALUABLE for {outcome_label} prediction",
            "✅ Clear visual separation between outcome groups",
            "✅ Consider using embeddings + baseline features (hybrid model)",
            "✅ Embeddings add meaningful signal beyond demographics"
        ]
    elif cohens_d >= 0.3:
        return [
            f"⚠️ Embeddings provide MODERATE value for {outcome_label} prediction",
            "⚠️ Some visual separation, but not dramatic",
            "⚠️ Best used in combination with baseline features",
            "⚠️ May improve model performance incrementally"
        ]
    else:
        return [
            f"❌ Embeddings show LIMITED clustering by {outcome_label}",
            "❌ Minimal visual separation between outcome groups",
            "❌ May capture other clinical patterns, but not this specific outcome",
            "❌ Consider: (1) Different outcome? (2) Different embedding architecture?"
        ]


def compare_embeddings(
    result_a: ClusterAnalysisResult,
    result_b: ClusterAnalysisResult,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare two embedding sources.
    
    Args:
        result_a: First embedding analysis result (must have run UMAP)
        result_b: Second embedding analysis result (must have run UMAP)
        verbose: Whether to print comparison
        
    Returns:
        Dictionary with comparison metrics
        
    Raises:
        ValueError: If either result lacks separation_result (UMAP not run)
    """
    if result_a.separation_result is None:
        raise ValueError(f"result_a ({result_a.embedding_name}) has no separation_result. "
                        "Run with run_umap=True to enable comparison.")
    if result_b.separation_result is None:
        raise ValueError(f"result_b ({result_b.embedding_name}) has no separation_result. "
                        "Run with run_umap=True to enable comparison.")
    
    cohens_d_a = result_a.separation_result.cohens_d
    cohens_d_b = result_b.separation_result.cohens_d
    
    mean_corr_a = result_a.correlation_result.mean_abs_correlation
    mean_corr_b = result_b.correlation_result.mean_abs_correlation
    
    diff_cohens = cohens_d_b - cohens_d_a
    diff_corr = mean_corr_b - mean_corr_a
    
    # Determine winner
    if diff_cohens > 0.05:
        winner = result_b.embedding_name
        winner_msg = f"✅ {result_b.embedding_name} WINS! Better clustering"
    elif diff_cohens < -0.05:
        winner = result_a.embedding_name
        winner_msg = f"✅ {result_a.embedding_name} WINS! Better clustering"
    else:
        winner = "TIE"
        winner_msg = "➖ TIE: Similar clustering performance"
    
    comparison = {
        'embedding_a': result_a.embedding_name,
        'embedding_b': result_b.embedding_name,
        'cohens_d_a': cohens_d_a,
        'cohens_d_b': cohens_d_b,
        'cohens_d_diff': diff_cohens,
        'mean_corr_a': mean_corr_a,
        'mean_corr_b': mean_corr_b,
        'mean_corr_diff': diff_corr,
        'winner': winner,
        'winner_msg': winner_msg
    }
    
    if verbose:
        print("="*80)
        print(f"🔄 COMPARISON: {result_a.embedding_name} vs {result_b.embedding_name}")
        print("="*80)
        
        print(f"\n📊 Centroid Distance:")
        print(f"   {result_a.embedding_name}: {result_a.separation_result.centroid_distance:.3f}")
        print(f"   {result_b.embedding_name}: {result_b.separation_result.centroid_distance:.3f}")
        
        print(f"\n📊 Mean Absolute Correlation:")
        print(f"   {result_a.embedding_name}: {mean_corr_a:.4f}")
        print(f"   {result_b.embedding_name}: {mean_corr_b:.4f}")
        print(f"   Difference: {diff_corr:+.4f} "
              f"{'✅ BETTER' if diff_corr > 0 else '❌ WORSE' if diff_corr < 0 else '➖ SAME'}")
        
        print(f"\n📊 Cohen's D (KEY METRIC):")
        print(f"   {result_a.embedding_name}: {cohens_d_a:.3f} ({result_a.separation_result.interpretation})")
        print(f"   {result_b.embedding_name}: {cohens_d_b:.3f} ({result_b.separation_result.interpretation})")
        pct_change = (diff_cohens / cohens_d_a) * 100 if cohens_d_a > 0 else 0
        print(f"   Difference: {diff_cohens:+.3f} ({pct_change:+.1f}%)")
        
        print(f"\n🏆 WINNER: {winner_msg}")
        print("="*80)
    
    return comparison


# =============================================================================
# HIGH-LEVEL PIPELINE FUNCTION
# =============================================================================

def run_cluster_analysis(
    df: pd.DataFrame,
    outcome_column: str,
    embedding_cols: Optional[List[str]] = None,
    outcome_label: str = "Outcome",
    embedding_name: str = "embeddings",
    cluster_config: Optional[ClusterConfig] = None,
    umap_config: Optional[UMAPConfig] = None,
    run_umap: bool = True,
    show_plots: bool = True,
    verbose: bool = True
) -> ClusterAnalysisResult:
    """
    Run complete cluster analysis pipeline.
    
    This is the main entry point for the analysis. It orchestrates:
    1. Embedding column identification
    2. Correlation analysis
    3. Dataset balancing
    4. UMAP projection (optional)
    5. Cluster separation quantification
    6. Visualization
    7. Summary reporting
    
    Args:
        df: Input dataframe with embedding columns and outcome
        outcome_column: Name of binary outcome column (0/1)
        embedding_cols: List of embedding column names (auto-detected if None)
        outcome_label: Human-readable label for outcome (e.g., "IP Admission")
        embedding_name: Name of embedding source for reporting
        cluster_config: ClusterConfig with analysis parameters
        umap_config: UMAPConfig with UMAP parameters
        run_umap: Whether to run UMAP projection (set False for faster correlation-only analysis)
        show_plots: Whether to display plots
        verbose: Whether to print progress
        
    Returns:
        ClusterAnalysisResult with all analysis results
    """
    if cluster_config is None:
        cluster_config = ClusterConfig()
    if umap_config is None:
        umap_config = UMAPConfig()
    
    # Auto-detect embedding columns
    if embedding_cols is None:
        embedding_cols = identify_embedding_columns(df)
    
    if verbose:
        print("="*80)
        print(f"🧬 EMBEDDING CLUSTER ANALYSIS")
        print("="*80)
        print(f"\n📊 Dataset: {len(df):,} samples, {len(embedding_cols)} embedding dimensions")
        print(f"🎯 Outcome: {outcome_label} ({outcome_column})")
        print(f"📦 Embedding: {embedding_name}")
        print(f"\n📈 Outcome Distribution:")
        print(df[outcome_column].value_counts().to_dict())
        print(f"   Prevalence: {df[outcome_column].mean()*100:.2f}%")
        print("="*80 + "\n")
    
    # Step 1: Correlation Analysis
    correlation_result = compute_embedding_correlations(
        df=df,
        outcome_column=outcome_column,
        embedding_cols=embedding_cols,
        correlation_threshold=cluster_config.correlation_threshold,
        verbose=verbose
    )
    
    if show_plots:
        visualize_correlation_analysis(
            correlation_result,
            outcome_label=outcome_label,
            show_plot=True
        )
        visualize_effect_size_thresholds(
            correlation_result,
            outcome_label=outcome_label,
            show_plot=True,
            verbose=verbose
        )
    
    # Step 2: Balance dataset
    df_balanced = balance_dataset_for_clustering(
        df=df,
        outcome_column=outcome_column,
        config=cluster_config,
        verbose=verbose
    )
    
    # Step 3: UMAP + Cluster Separation (optional)
    df_viz = None
    separation_result = None
    
    if run_umap:
        df_viz, _, _ = run_umap_projection(
            df=df_balanced,
            embedding_cols=embedding_cols,
            config=umap_config,
            verbose=verbose
        )
        
        separation_result = compute_cluster_separation(
            df=df_viz,
            outcome_column=outcome_column,
            outcome_label=outcome_label,
            verbose=verbose
        )
        
        if show_plots:
            visualize_clusters(
                df=df_viz,
                outcome_column=outcome_column,
                outcome_label=outcome_label,
                show_plot=True
            )
        
        # Generate summary
        generate_cluster_summary(
            correlation_result=correlation_result,
            separation_result=separation_result,
            embedding_name=embedding_name,
            outcome_label=outcome_label,
            correlation_threshold=cluster_config.correlation_threshold,
            verbose=verbose
        )
    
    return ClusterAnalysisResult(
        correlation_result=correlation_result,
        separation_result=separation_result,
        df_balanced=df_balanced,
        df_viz=df_viz,
        embedding_cols=embedding_cols,
        outcome_column=outcome_column,
        outcome_label=outcome_label,
        embedding_name=embedding_name
    )


def run_tabular_cluster_analysis(
    df: pd.DataFrame,
    outcome_column: str,
    tabular_cols: Optional[List[str]] = None,
    outcome_label: str = "Outcome",
    feature_set_name: str = "tabular_features",
    cluster_config: Optional[ClusterConfig] = None,
    umap_config: Optional[UMAPConfig] = None,
    run_umap: bool = True,
    show_plots: bool = True,
    verbose: bool = True
) -> ClusterAnalysisResult:
    """
    Run cluster analysis on TABULAR (non-embedding) features.
    
    This is useful for comparing how well embeddings cluster compared to
    traditional tabular features like age, chronic conditions, utilization counts, etc.
    
    Args:
        df: Input dataframe with tabular feature columns and outcome
        outcome_column: Name of binary outcome column (0/1)
        tabular_cols: List of tabular feature column names (auto-detected if None)
        outcome_label: Human-readable label for outcome (e.g., "IP Admission")
        feature_set_name: Name for reporting (e.g., "baseline_features")
        cluster_config: ClusterConfig with analysis parameters
        umap_config: UMAPConfig with UMAP parameters
        run_umap: Whether to run UMAP projection
        show_plots: Whether to display plots
        verbose: Whether to print progress
        
    Returns:
        ClusterAnalysisResult with all analysis results
        
    Example:
        # Compare embeddings vs tabular features
        result_emb = run_cluster_analysis(df, 'ip6', outcome_label='IP Admission')
        result_tab = run_tabular_cluster_analysis(df, 'ip6', outcome_label='IP Admission')
        
        print(f"Embeddings Cohen's D: {result_emb.separation_result.cohens_d:.3f}")
        print(f"Tabular Cohen's D: {result_tab.separation_result.cohens_d:.3f}")
    """
    # Auto-detect tabular columns if not provided
    if tabular_cols is None:
        tabular_cols = identify_tabular_feature_columns(df, outcome_column, verbose=verbose)
    
    if len(tabular_cols) == 0:
        raise ValueError("No tabular feature columns found. Ensure dataframe contains "
                        "features beyond embedding_, outcome, and excluded columns.")
    
    if verbose:
        print("="*80)
        print(f"🔢 TABULAR FEATURE CLUSTER ANALYSIS")
        print("="*80)
        print(f"\n📊 Dataset: {len(df):,} samples, {len(tabular_cols)} tabular features")
        print(f"🎯 Outcome: {outcome_label} ({outcome_column})")
        print(f"📦 Feature set: {feature_set_name}")
        print(f"\n📋 Sample features: {tabular_cols[:10]}...")
        print("="*80 + "\n")
    
    # Prepare numeric features only (UMAP requires numeric)
    df_numeric = df.copy()
    
    # Convert tabular features to numeric, filling NaN
    for col in tabular_cols:
        if df_numeric[col].dtype == 'object' or df_numeric[col].dtype.name == 'category':
            # Encode categorical as numeric
            df_numeric[col] = pd.factorize(df_numeric[col])[0]
        else:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce').fillna(0)
    
    # Run the standard cluster analysis using tabular features
    return run_cluster_analysis(
        df=df_numeric,
        outcome_column=outcome_column,
        embedding_cols=tabular_cols,  # Use tabular cols instead of embedding cols
        outcome_label=outcome_label,
        embedding_name=feature_set_name,
        cluster_config=cluster_config,
        umap_config=umap_config,
        run_umap=run_umap,
        show_plots=show_plots,
        verbose=verbose
    )


# =============================================================================
# PROCEDURAL WORKFLOW (Step-by-Step)
# =============================================================================
# 
# Use the functions below to run each analysis step independently.
# This allows you to inspect intermediate results and customize the workflow.
#
# STEP-BY-STEP PROCEDURE:
# -----------------------
# 
# # Step 0a: Load embeddings from local NPZ file
# embeddings, individual_ids, index_dts = load_embeddings_from_local(
#     "embedding_output/exp2b_flash_learned_pool/"
# )
# df_emb = create_embedding_dataframe(embeddings, individual_ids, index_dts)
#
# # Step 0b: Load outcomes from BigQuery (or use pre-loaded DataFrame)
# df_outcomes = load_tabular_features_from_bigquery(
#     "project.dataset.a964286_commercial_ip_features"
# )
#
# # Step 0c: Join embeddings with outcomes
# df = join_embeddings_with_outcomes(
#     emb_df=df_emb,
#     outcomes_df=df_outcomes,
#     outcome_column='ip6'
# )
#
# # OR use the convenience function:
# # df = load_and_prepare_data(
# #     embedding_source="embedding_output/exp2b_flash_learned_pool/",
# #     outcome_source="project.dataset.features_table",
# #     outcome_column='ip6'
# # )
#
# outcome_column = 'ip6'
# outcome_label = 'IP Admission'
# embedding_name = 'transformer_embeddings'
#
# # Step 1: Identify embedding columns
# embedding_cols = identify_embedding_columns(df, prefix='embedding_')
# print(f"Found {len(embedding_cols)} embedding dimensions")
#
# # Step 2: Correlation analysis (fast, no UMAP needed)
# correlation_result = compute_embedding_correlations(
#     df=df,
#     outcome_column=outcome_column,
#     embedding_cols=embedding_cols,
#     verbose=True
# )
# # Inspect correlation results
# print(correlation_result.df_correlations.head(20))
# print(f"Top predictor: {correlation_result.top_predictor}")
# print(f"Mean |r|: {correlation_result.mean_abs_correlation:.4f}")
#
# # Step 2b: Visualize correlations (optional)
# fig = visualize_correlation_analysis(correlation_result, outcome_label=outcome_label)
# fig = visualize_effect_size_thresholds(correlation_result, outcome_label=outcome_label)
#
# # Step 3: Balance dataset for clustering
# cluster_config = ClusterConfig(balance_classes=True, random_state=123)
# df_balanced = balance_dataset_for_clustering(
#     df=df,
#     outcome_column=outcome_column,
#     config=cluster_config,
#     verbose=True
# )
# print(f"Balanced dataset: {len(df_balanced):,} samples")
#
# # Step 4: UMAP projection (slow, 2-5+ min)
# umap_config = UMAPConfig(n_neighbors=100, min_dist=0.1, random_state=123)
# df_viz, embedding_2d, umap_time = run_umap_projection(
#     df=df_balanced,
#     embedding_cols=embedding_cols,
#     config=umap_config,
#     verbose=True
# )
# print(f"UMAP completed in {umap_time:.1f}s")
# print(df_viz[['umap_x', 'umap_y']].describe())
#
# # Step 5: Compute cluster separation (Cohen's D)
# separation_result = compute_cluster_separation(
#     df=df_viz,
#     outcome_column=outcome_column,
#     outcome_label=outcome_label,
#     verbose=True
# )
# print(f"Cohen's D: {separation_result.cohens_d:.3f}")
# print(f"Interpretation: {separation_result.interpretation}")
#
# # Step 5b: Visualize clusters (optional)
# fig = visualize_clusters(
#     df=df_viz,
#     outcome_column=outcome_column,
#     outcome_label=outcome_label,
#     show_plot=True
# )
#
# # Step 6: Generate summary
# summary = generate_cluster_summary(
#     correlation_result=correlation_result,
#     separation_result=separation_result,
#     embedding_name=embedding_name,
#     outcome_label=outcome_label,
#     verbose=True
# )
#
# # Step 7 (optional): Compare two embedding sources
# # result_a = ... (run above steps for embedding A)
# # result_b = ... (run above steps for embedding B)
# # comparison = compare_embeddings(result_a, result_b)
#
# =============================================================================


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🧬 Embedding Cluster Analysis Module")
    print("="*80)
    print("""
OPTION 1: Full Pipeline with Data Loading (Recommended)
--------------------------------------------------------
from embedding_cluster_analysis import (
    load_and_prepare_data,
    run_cluster_analysis
)

# Load embeddings from local + outcomes from BigQuery
df = load_and_prepare_data(
    embedding_source="embedding_output/exp2b_flash_learned_pool/",
    outcome_source="project.dataset.features_table",
    outcome_column='ip6'
)

# Run full analysis
result = run_cluster_analysis(
    df=df,
    outcome_column='ip6',
    outcome_label='IP Admission',
    embedding_name='exp2b_flash',
    show_plots=True
)

# Access results
print(f"Cohen's D: {result.separation_result.cohens_d:.3f}")


OPTION 2: Data Loading Only (for downstream model training)
------------------------------------------------------------
from embedding_cluster_analysis import (
    load_embeddings_from_local,
    load_embeddings_from_bigquery,
    load_tabular_features_from_bigquery,
    create_embedding_dataframe,
    join_embeddings_with_outcomes
)

# Load embeddings from local NPZ
embeddings, ids, dates = load_embeddings_from_local(
    "embedding_output/exp2b_flash_learned_pool/"
)
df_emb = create_embedding_dataframe(embeddings, ids, dates)

# OR load embeddings from BigQuery
df_emb = load_embeddings_from_bigquery(
    "project.dataset.a964286_commercial_embeddings"
)

# Load tabular features
df_features = load_tabular_features_from_bigquery(
    "project.dataset.a964286_commercial_ip_features"
)

# Join embeddings with outcomes
df = join_embeddings_with_outcomes(
    emb_df=df_emb,
    outcomes_df=df_features,
    outcome_column='ip6',
    additional_columns=['age', 'gender_cd', 'chronic_count']
)


OPTION 3: Procedural (Step-by-Step Analysis)
--------------------------------------------
from embedding_cluster_analysis import (
    identify_embedding_columns,
    compute_embedding_correlations,
    visualize_correlation_analysis,
    visualize_effect_size_thresholds,
    balance_dataset_for_clustering,
    run_umap_projection,
    compute_cluster_separation,
    visualize_clusters,
    generate_cluster_summary,
    ClusterConfig,
    UMAPConfig
)

outcome_column = 'ip6'
outcome_label = 'IP Admission'
embedding_name = 'transformer_embeddings'

# Step 1: Identify embedding columns
embedding_cols = identify_embedding_columns(df)
print(f"Found {len(embedding_cols)} dimensions")

# Step 2: Correlation analysis (fast)
corr_result = compute_embedding_correlations(df, outcome_column, embedding_cols)
print(corr_result.df_correlations.head(10))

# Step 3: Visualize correlations
visualize_correlation_analysis(corr_result, outcome_label)
visualize_effect_size_thresholds(corr_result, outcome_label)

# Step 4: Balance dataset
config = ClusterConfig(balance_classes=True)
df_balanced = balance_dataset_for_clustering(df, outcome_column, config)

# Step 5: UMAP projection (slow)
umap_cfg = UMAPConfig(n_neighbors=100, min_dist=0.1)
df_viz, emb_2d, runtime = run_umap_projection(df_balanced, embedding_cols, umap_cfg)

# Step 6: Cluster separation
sep_result = compute_cluster_separation(df_viz, outcome_column, outcome_label=outcome_label)
print(f"Cohen's D: {sep_result.cohens_d:.3f}")

# Step 7: Visualize clusters
visualize_clusters(df_viz, outcome_column, outcome_label)

# Step 8: Summary
summary = generate_cluster_summary(corr_result, sep_result, embedding_name, outcome_label)


Key Benefits:
- Load from local NPZ files OR BigQuery tables
- Procedural steps allow inspecting intermediate results
- Skip expensive UMAP for quick correlation checks
- Save/load intermediate dataframes between sessions
""")

