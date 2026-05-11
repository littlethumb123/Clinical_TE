# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Clinical Transformer Embeddings (Clinical TE) — a hierarchical clinical transformer that encodes member health claims into 256-dimensional embeddings for downstream healthcare ML models. CVS Health / Aetna CSDI project.

**Two-level architecture:**
1. **Daily Encoder**: Encodes up to 80 medical codes per day (FlashAttention or LearnedAttentionPooling) → one 256-d vector per day
2. **Temporal Encoder**: Encodes up to 200 days of daily representations (Flash Attention, RoPE, SwiGLU FFN or MoE, causal masking) → final 256-d member embedding

**Pre-training objective**: Multi-label next-code prediction (BCEWithLogitsLoss) over ~6,297 grouped target codes from ~75,516 input codes. Dual-vocabulary design: granular input, collapsed output.

**Current best model**: `exp2b_flash_learned_pool` (Flash Attention + Learned Attention Pooling, no MoE).

## Repository Structure

```
dev/moe/                        # Active development — model code and training notebooks
  moe_flashattn_4_core.py       # Canonical importable module (~11K lines): all configs, models, training, eval
  moe_flashattn_4.py            # Training script counterpart (~18K lines)
  moe_flashattn_5.ipynb         # Latest training notebook (GCP Vertex AI)
  exp_round10_*.ipynb           # Formal training and evaluation notebooks
  requirements.txt              # Python dependencies (torch, sklearn, bigquery, xgboost, lightgbm)

dev/downstream/                 # Downstream evaluation scripts (embedding → probe classifiers)
  moe_flashattn_3_cm_me_downstream.py    # Commercial + Medicare downstream eval
  moe_flashattn_3_medicaid_downstream.py # Medicaid downstream eval

data_ingestion/                 # SQL pipelines for BigQuery data preparation
  TE_pretraining_data_ingestion/         # Pretraining data: per-LOB SQL + combine + w2ind_target
  Formal_training_full_downstream/       # Formal downstream evaluation SQL
  Medicaid_ip/, Com_ip/, Medicare_ip/    # Per-LOB outcome and feature SQL
  Legacy/                                # Legacy production SQL pipelines

docs/                           # Architecture docs, progress logs, PSS notes, plans
expe_logs/                      # Training run outputs (JSON metrics, logs; checkpoints gitignored)
expe_analysis/                  # Experiment analysis notebooks
```

## Key Code Locations in `moe_flashattn_4_core.py`

**Config dataclasses**: `BaseConfig`, `FlashAttentionConfig`, `MoEConfig`, `OptimizeConfig`, `DownstreamConfig`
- Key defaults: `len_dy=200, len_cd=80, cd_cnt=75516, target_cd_cnt=6297, embedding_size=256, nlayers=6`

**Model classes**: `BaselineTransformer` (~line 2120), `FlashAttentionTransformer` (~line 2292), `FlashMoETransformer` (~line 2505)

**Architecture components**: `RotaryPositionEmbedding` (~1126), `SwiGLU` (~1210), `FlashAttentionLayer` (~1246), `LearnedAttentionPooling` (~1481), `ExpertLayer` (~1671), `MoELayer` (~1702)

**Training**: `train_epoch()` (~5437), `evaluate()` (~7013), `run_single_experiment()` (~10811), `prepare_data_once()` (~10039)

**Loss functions**: `AsymmetricLoss` (~4495), `FocalLoss` (~4596), `LossTracker` (~4788)

**Data**: `ClinicalDataset` (eager), `ClinicalDatasetLazy` (lazy parsing via DataLoader workers), `TierAwareBatchSampler` (~5922), `DensityTierAwareBatchSampler` (~6500)

**Experiment configs**: `get_experiment_configs()` returns mapping of experiment names (exp1 through exp6) to config dicts

## Environment and Commands

**No build system** (no Makefile, pyproject.toml, or setup.py at root). This is a research workspace.

```bash
# Install dependencies
pip install -r dev/moe/requirements.txt

# Run inline smoke tests (ad-hoc, no pytest runner)
python3 -c "from dev.moe.moe_flashattn_4_core import test_prepare_tensor_and_multihot; test_prepare_tensor_and_multihot()"

# Training is done via Jupyter notebooks on GCP Vertex AI
# Primary training notebook: dev/moe/moe_flashattn_5.ipynb
```

**Data source**: BigQuery `edp-prod-storage.edp_ent_sdoheir_cns` — no local data files (all gitignored). Training data is ~11M members across 3 LOBs (Commercial, Medicare, Medicaid).

**Checkpoints**: `.pt` files are gitignored. Experiment metrics saved as JSON under `expe_logs/`.

## Development Principles (from Cursor rules)

**Priority hierarchy** — before suggesting architecture changes, verify this order:
1. Data — understood, quality verified, distribution analyzed?
2. Loss/Objective — aligned with evaluation metric? Gradient distribution appropriate?
3. Training Dynamics — gradients healthy, LR schedule, stability?
4. Architecture — only after eliminating 1-3 as bottlenecks

**Experiment hygiene**:
- Every experiment needs a hypothesis, single variable change, expected outcome, and refutation criterion
- If a diagnostic costs <4 GPU-hours, run it before consulting experts
- Establish simple baselines before complex model experiments

**Downstream evaluation discipline**:
- Prevent member-level leakage (same member in train/test)
- Prevent temporal leakage (no future claims in embeddings)
- Prevent pipeline leakage (fit transforms on training only)
- Primary metrics: AUC, AUPRC, lift-at-percentage
- Probe classifiers: LogisticRegression, XGBoost, LightGBM

**Code extraction** (`_core.py` files): When extracting from notebooks, copy code exactly — do not refactor, rename, or add docstrings. Exclude test/debug/script code. Verify all dependency chains.

**Evidence-based reasoning**: No hallucination, no guessing. State assumptions explicitly. Use calibrated language when uncertain. Prefer production-proven methods; flag emerging methods explicitly.
