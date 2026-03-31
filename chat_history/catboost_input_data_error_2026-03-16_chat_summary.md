# CatBoost input data error

## Metadata

- **Conversation ID:** 5b14ab70-4172-4e8c-a2bd-c24ad581e47c
- **Mode:** agent
- **Created:** 2026-03-16 23:43:03
- **Last Updated:** 2026-03-17 00:28:06
- **Branch:** main
- **Files Changed:** 1
- **Lines Added:** 60351
- **Lines Removed:** 60434
- **Summary:** Edited moe_flashattn_3_lob3_downstream_running.ipynb

---

## Conversation

### 🧑 User

I got the following error when I try to run results_df_exp_round5_legacy_dbcheck = evaluate_all_experiments(experiment_configs, df_ip_features, downsample_ratio=10.0)
Inspect the root cause 
The embedding tables looks like this 
{
  "emb0": "-0.69362062215805054",
  "emb1": "-1.1289623975753784",
  "emb2": "0.22302982211112976",
  "emb3": "0.20118197798728943",
  "emb4": "-1.4758185148239136",
  "emb5": "0.90361928939819336",
  "emb6": "-0.21049349009990692",
 ...... 
"emb256": 0.12,
  "individual_id": "1109380834",
  "index_dt": "2023-01-16T00:00:00"
}

Here is the cell output: 
Preparing data for: feature_set=embedding_only, downsample=10.0:1, path=edp-prod-storage.edp_ent_sdoheir_cns.a964286_te4ex...

 Loading and preparing data...
  Loading embeddings from BigQuery: edp-prod-storage.edp_ent_sdoheir_cns.a964286_te4exp_3lob_exp_round5_exp1_legacy_1_5m_commercial_30pc_sample_embedding
  Loading embeddings from BigQuery: edp-prod-storage.edp_ent_sdoheir_cns.a964286_te4exp_3lob_exp_round5_exp1_legacy_1_5m_commercial_30pc_sample_embedding
  Loaded 2,862,176 rows with 256 embedding dimensions
  Joining embeddings with features...
  Step 1 complete (123.25s)

[Step 2/6] Creating data splits...
Data splits created:
  train: 843,670 rows, 7,635 positives (0.90%)
  val: 105,259 rows, 927 positives (0.88%)
  test: 105,057 rows, 1,011 positives (0.96%)
  oot: 758,134 rows, 7,026 positives (0.93%)
  oot_strict: 75,490 rows, 718 positives (0.95%)
  Step 2 complete (6.07s)

[Step 4/6] Preparing feature matrices for each split...
  Step 4 complete (0.01s)

[Step 5/6] Rebalance the training dataset with a ratio of 10.0...
  Downsampling: 836035:7635 (109.5:1) → 76350:7635 (10.0:1)
Finish data preparation, total time: 129.4831235408783
  Evaluating: exp1_legacy_dbcheck_round5_3lobs_1-5M_catboost_emb_only (CatBoostClassifier)
---------------------------------------------------------------------------
CatBoostError                             Traceback (most recent call last)
Cell In[61], line 1
----> 1 results_df_exp_round5_legacy_dbcheck = evaluate_all_experiments(experiment_configs, df_ip_features, downsample_ratio=10.0)

Cell In[58], line 120
    116         apply_scaling = config.get('apply_scaling', False)
    118         print(f"  Evaluating: {exp_name} ({type(model).__name__})")
--> 120         result = evaluate_with_prepared_data(
    121             prepared_data=prepared_data,
    122             ml_model_object=model,
    123             exp_name=exp_name,
    124             apply_scaling=apply_scaling
    125         )
    126         results.append(result)
    128 return pd.DataFrame(results)

Cell In[58], line 29
     23 use_cat_features = (
     24     prepared_data.feature_set != 'embedding_only' and 
     25     len(prepared_data.cat_feature_indices) > 0
     26 )
     28 # Evaluate model
---> 29 split_results = evaluate_model_on_splits(
     30     model=ml_model_object,
...
    958     raise CatBoostError("Input data has invalid shape: {}. Must be 2 or 3 dimensional".format(data_shape))
    959 if data_shape[1] == 0:
--> 960     raise CatBoostError("Input data must have at least one feature")

CatBoostError: Input data must have at least one feature

---

### 🤖 Assistant

Let me investigate the root cause of this error by examining the notebook and the relevant functions.

---

### 🤖 Assistant

The notebook is very large. Let me search for the key functions - the `evaluate_model_on_splits` function and the data preparation code where the feature matrix is being constructed.

---

### 🤖 Assistant

The error originates from the notebook, but the function definitions are in `.py` files. Let me find the actual cell definitions in the notebook and also look at the data preparation step where feature matrices are built.

---

### 🤖 Assistant

The error is from Cell In[58] and Cell In[61]. Let me find those cells. The function at line 8376 and 8449 is likely in Cell 58. Let me read the relevant sections.

---

### 🤖 Assistant

Now let me look at the data preparation function - specifically the "Step 4" where feature matrices are created, and the `evaluate_all_experiments` function that calls everything.

---

### 🤖 Assistant

There are two sets of functions - one earlier (line ~2553) and one later (line ~8155). The error traceback says Cell In[58]. Let me check which cell is actually being used. Let me read the `evaluate_all_experiments` function and the `prepare_evaluation_data` / `PreparedData` class near line 8155, and also the one near line 2553.

---

### 🤖 Assistant

Now I need to see the `evaluate_all_experiments` function that the error originates from (Cell In[58]), and more importantly the `prepare_data` function it calls. Let me look at the earlier set of functions around line 2553 (Cell In[58]) since the error trace shows line 120 in Cell In[58].

---

### 🤖 Assistant

Now I need to understand what happens in the feature preparation step. The key issue is that `prepare_evaluation_data` calls `identify_feature_columns` and then `prepare_features`. Let me find those functions.

---

### 🤖 Assistant

Found it. Here's the root cause. Look at the `identify_feature_columns` function:

```python
embedding_features = sorted([c for c in all_cols if c.startswith('embedding_')])
```

It's looking for columns that start with `'embedding_'`, but your BigQuery embedding table has columns named `emb0`, `emb1`, `emb2`, ..., `emb256` — they start with `'emb'`, **not** `'embedding_'`.

So `identify_feature_columns` returns an **empty list** for `embedding_features`. When `feature_set='embedding_only'`, the code sets `feature_cols = embedding_features` (the empty list), which means `X_splits` DataFrames end up with **zero columns**. CatBoost then rightfully complains: `"Input data must have at least one feature"`.

Let me verify this by checking if there's any column renaming step in the loading or joining functions.

---

### 🤖 Assistant

Now let me look at the `load_embeddings_from_bigquery` function to see whether it renames columns.

---

### 🤖 Assistant

This confirms the issue. The `load_embeddings_from_bigquery` function:

1. Reads all columns starting with `'emb'` from BigQuery
2. Returns them **as-is** — so the column names remain `emb0`, `emb1`, ..., `emb256`
3. The docstring says it returns `embedding_0...embedding_N` columns, but **that's only true when the table was created by `save_embeddings_to_bigquery()`** which uses `embedding_` prefix

Then `identify_feature_columns` looks for columns starting with `'embedding_'` and finds **zero matches** because the legacy table uses `emb0`, `emb1`, etc.

## Root Cause

**Column naming mismatch between the legacy BigQuery embedding table and the `identify_feature_columns` function.**

| Component | Expected prefix | Actual prefix (legacy table) |
|---|---|---|
| `identify_feature_columns()` | `embedding_` | - |
| Legacy BQ table | - | `emb` (e.g., `emb0`, `emb1`, ..., `emb256`) |
| Tables from `save_embeddings_to_bigquery()` | - | `embedding_0`, `embedding_1`, ... |

Because the legacy table columns are named `emb0`–`emb256` (no underscore, no `embedding_` prefix), the filter `c.startswith('embedding_')` returns an empty list. With `feature_set='embedding_only'`, `feature_cols` becomes `[]`, so CatBoost receives a zero-column DataFrame and raises `"Input data must have at least one feature"`.

## Fix

The fix should be in `load_embeddings_from_bigquery` — rename the columns to the canonical `embedding_N` format before returning, so downstream code works regardless of the source table's naming convention. Let me apply this.

---

### 🤖 Assistant

Now I need to identify the exact cell index so I can edit the notebook properly. Let me find which cell contains `load_embeddings_from_bigquery`.

---

### 🤖 Assistant

I need to find the notebook cell index. Let me look for cells structure around line 2240.

---

### 🤖 Assistant

Let me search backward to find the cell boundary.

---

### 🤖 Assistant

This is `execution_count: 60`. I need to find the actual cell index. Let me search for cell boundaries near this area to count cells.

Let me take a different approach and use a Grep to find the cell IDs.

---

### 🤖 Assistant

I need to find the cell index. Let me count cells up to this point.

---

### 🤖 Assistant

The cell at line 2220 is the one I need (cell_type: "code" containing `load_embeddings_from_bigquery`). Counting from 0, it's the 42nd entry in the list (0-indexed: lines 4, 29, 50, 110, 121, 145, 168, 192, 313, 324, 443, 462, 474, 485, 656, 733, 743, 1153, 1163, 1239, 1350, 1361, 1376, 1429, 1437, 1445, 1453, 1637, 1647, 1682, 1705, 1732, 1743, 1824, 1952, 1962, 1992, 2002, 2042, 2124, 2149, 2220). That's cell index 41.

Let me verify by reading the cell content.

---

### 🤖 Assistant

I can see the issue clearly. The function detects columns starting with `'emb'` and sorts them if they have an underscore (`embedding_0`, `embedding_1`, ...), but does NOT handle the `emb0`, `emb1` naming convention (no underscore). And critically, it does NOT rename them to `embedding_N` format.

Let me apply the fix. I need to add a renaming step to normalize `emb{N}` columns to `embedding_{N}`.

---

### 🤖 Assistant

Now let me verify the edit looks correct.

---

### 🤖 Assistant

The fix is applied. Let me also check if there's a second copy of `load_embeddings_from_bigquery` at line 8539 that might also need fixing.

---

### 🤖 Assistant

This second copy is from a different section (the `MedicaidPreparedData` path) and returns the raw DataFrame without column selection. It's a different version of the function that's used by `prepare_medicaid_evaluation_data()` — but the error you're hitting is from the Cell In[58] path which uses the first copy. The second copy isn't being called in your failing execution.

## Summary

**Root cause:** The legacy BigQuery embedding table uses column names `emb0`, `emb1`, ..., `emb256`, but `identify_feature_columns()` filters for columns starting with `embedding_` (with underscore). This returns zero embedding feature columns, and with `feature_set='embedding_only'`, CatBoost receives an empty feature matrix.

**The fix** (applied to `load_embeddings_from_bigquery` in the data preparation cell): When columns use the `emb{N}` naming convention (no underscore), they are now automatically renamed to `embedding_{N}` to match what the rest of the pipeline expects. This handles both conventions:
- `embedding_0`, `embedding_1`, ... (from `save_embeddings_to_bigquery`) — sorted as before
- `emb0`, `emb1`, ... (legacy tables) — sorted and renamed to `embedding_0`, `embedding_1`, ...

Re-run the cell defining `load_embeddings_from_bigquery` (execution_count 60), then re-run your experiment call.

