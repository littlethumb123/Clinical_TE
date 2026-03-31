# Experiment results comparison and validation

## Metadata

- **Conversation ID:** 60c3236c-2321-4372-aad1-f85177239331
- **Mode:** agent
- **Created:** 2026-03-19 07:33:52
- **Last Updated:** 2026-03-19 08:10:56
- **Branch:** main
- **Files Changed:** 0
- **Lines Added:** 0
- **Lines Removed:** 0
- **Summary:** Edited verify_new_completeness.py, add_rounds_6_to_9.py, verify_comparison.py, build_comparison_df.py, verify_completeness.py

---

## Conversation

### 🧑 User

summarize the results @expe_logs/exp_round10/ and compare it to all results in @expe_logs/exp_round5/ and @expe_logs/exp_round5_1_lr_plateau/; create a json file that I can turn it to a dataframe where index is the experiment name (directory + file name that are distinct for each result, such as exp_round5_lr_plateau_exp2_v2_bce_weighted35, exp_round5_lr_plateau_exp2_v3_bce_weighted200, exp_round5_lr_plateau_exp2_v5_asymm_focalloss_dense_sampler_final_results.... etc) for the results, the columns I would like to have are full set of configurations (from all "config.json") and performance metrics, efficiency and resources metrics (from files with "final_results.json")
Output the json file; after that, rigorously and closelu examine the correctness of every single value against the original final_results.json for each experiment. DO not halluicnate, you wanted to make sure no values are missing and all values are correct.

---

### 🤖 Assistant

I'll start by exploring the directory structures to understand what experiments and result files exist across all three directories.

---

### 🤖 Assistant

Good, I have a clear picture of all directories. Now I need to identify which experiments have both config and final_results files. Let me also check the exp_round5 legacy results format (the `_results.json` files). Let me read all the config and results files systematically.

---

### 🤖 Assistant

Let me read all the config and final_results files. Starting with exp_round10 and exp_round5_1_lr_plateau (which have clear pairings), then exp_round5.

---

### 🤖 Assistant

Now let me read the remaining files - the v5 from lr_plateau, and the exp_round5 experiments.

---

### 🤖 Assistant

Now I have all the data. I've identified 8 experiments with paired config + final_results:

1. **exp_round10** (1 experiment)
2. **exp_round5_1_lr_plateau/exp2** (4 experiments: v2, v3, v4, v5)
3. **exp_round5** (3 experiments: exp1_dbcheck/epoch1, exp1_dbcheck/epoch2-3, exp1_opt)

Note that exp_round5/exp1_dbcheck and exp1_dbcheck/epoch2-3 use a different result schema (legacy format). Let me now build the comprehensive JSON programmatically with a Python script to ensure no values are missed or hallucinated.

---

### 🤖 Assistant

Good, the JSON was generated. Now I need to do the most critical step - rigorously verify every value. Let me read the generated JSON and then write a verification script that checks every single value against the source files.

---

### 🤖 Assistant

The JSON is being generated correctly. Now let me write the rigorous verification script that checks every value against the original source files.

---

### 🤖 Assistant

All 1,108 checks pass with 0 errors. Now let me also run a second-pass check that goes the other direction - verifying that no values from the source files were missed (i.e., checking for completeness).

---

### 🤖 Assistant

The completeness check found some missing fields in the legacy experiments. Let me add those to the extraction. The gaps are:
- Legacy config fields: `cd_cnt`, `target_cd_cnt`, `len_dy`, `len_cd`, `val_split`, `training_table`, `train_samples`, `val_samples`, `lob_vocab`, `checkpoint_path`
- Legacy efficiency fields: `data_load_percent`, `forward_percent`, `backward_percent`

---

### 🤖 Assistant

Now update the verification script to include the same new fields.

---

### 🤖 Assistant

Now rebuild and re-verify.

---

### 🤖 Assistant

All checks pass. Let me clean up the helper scripts and update the todo.

---

### 🤖 Assistant

Now let me delete the helper scripts since they served their purpose.

---

### 🤖 Assistant

The output file is at `expe_logs/experiment_comparison.json`. Here's a summary of what was done and the results:

---

## Output: `expe_logs/experiment_comparison.json`

**8 experiments** across 3 experiment rounds, with the JSON structured as `{experiment_name: {column: value}}` ready for `pd.DataFrame.from_dict(data, orient='index')`.

### Experiments Included

| Experiment Name | Source | Columns |
|---|---|---|
| `exp_round10` | exp_round10/ | 158 |
| `exp_round5_1_lr_plateau_exp2_v2_bce_weighted35` | exp_round5_1_lr_plateau/exp2/ | 148 |
| `exp_round5_1_lr_plateau_exp2_v3_bce_weighted200` | exp_round5_1_lr_plateau/exp2/ | 148 |
| `exp_round5_1_lr_plateau_exp2_v4_asymm_focalloss` | exp_round5_1_lr_plateau/exp2/ | 160 |
| `exp_round5_1_lr_plateau_exp2_v5_asymm_focalloss_dense_sampler` | exp_round5_1_lr_plateau/exp2/ | 160 |
| `exp_round5_exp1_opt` | exp_round5/exp1_opt/ | 145 |
| `exp_round5_exp1_dbcheck_epoch1` | exp_round5/exp1_dbcheck/epoch1/ | 101 |
| `exp_round5_exp1_dbcheck_epoch2_3` | exp_round5/exp1_dbcheck/epoch2-3/ | 114 |

### Column Categories

- **`config_*`** - All configuration parameters (architecture, optimizer, loss, batching, scheduler, etc.)
- **`results_*`** - Top-level training results (train/val loss, recall, generalization gap, etc.)
- **`perf_*`** - Full evaluation performance metrics (recall@k, precision@k, ndcg@k, f1@k, micro_recall@k, AUROC, AUPRC, Brier, tier accuracy, etc.)
- **`eff_*`** - Efficiency metrics (time per epoch, samples/sec, tokens/sec, etc.)
- **`res_*`** - Resource metrics (GPU memory per device, model params, FLOPS, MFU, cost, etc.)
- **`grad_tier_*`** - Gradient tier analysis (where available)

### Verification Results

Two independent verification passes confirmed:
1. **Value correctness**: 1,134 individual value checks against original source JSONs - **0 errors**
2. **Completeness**: All 1,083 scalar values from the 16 source files (8 config + 8 results) are present - **0 missing**

### Usage

```python
import pandas as pd, json
with open("expe_logs/experiment_comparison.json") as f:
    data = json.load(f)
df = pd.DataFrame.from_dict(data, orient="index")
```

### Key Findings at a Glance

The **exp_round10** experiment (full 11M dataset training) achieved the best performance: **recall@10 = 0.8528, ndcg@10 = 0.4292, micro_recall@10 = 0.5628** with the standard BCE + pos_weight_max=200 configuration. The lr_plateau variants v4 (ASL) and v5 (ASL + dense sampler) achieved notably better MRR (0.471 and 0.496 respectively) and positive_brier (0.313 and 0.308) at the cost of lower recall@10 (0.828 and 0.833).

---

### 🧑 User

Ok great, keep including @expe_logs/exp_round6/  @expe_logs/exp_round7_512dim/ @expe_logs/exp_round8/ @expe_logs/exp_round9/ into the json file; identifcal requirements; and integrate it into the same json file @expe_logs/exp5_exp10_result_comparison.json

---

### 🤖 Assistant

Let me explore these four new directories and read the existing comparison JSON to understand the current state.

---

### 🤖 Assistant

Good. Now I have the full picture of the 4 new directories. Let me identify all config+results pairs:

- **exp_round6**: 3 experiments (training_6-8M, training_3-4M/exp2, training_3-4M/exp2 doublesize)
- **exp_round7_512dim**: 1 experiment (exp2)
- **exp_round8**: 1 experiment (exp2b_512dim_6-8M)
- **exp_round9**: 2 experiments with results (exp2b_256dim_v1, exp2b_256dim_v2; v0 has no final_results)

That's 7 new experiments. Let me read all their config and results files, plus the existing comparison JSON.

---

### 🤖 Assistant

Now let me read the remaining files: exp_round8, exp_round9 v1 and v2.

---

### 🤖 Assistant

Now I have all the data. I notice the exp_round6 training_3-4M/exp2 has a `config_doublesize.json` + `final_results_doublesize.json` pair AND a `config.json` + `final_results.json` pair — but the config and results for the non-doublesize pair are actually identical to the doublesize pair (same content). Let me verify this quickly, then build the full script.

---

### 🤖 Assistant

The doublesize files are identical to the base files in training_3-4M/exp2, so we only need one entry. Now let me write the comprehensive script to add all 7 new experiments and merge them into the existing JSON file.

---

### 🤖 Assistant

All 1,092 checks pass with 0 errors. Now let me also run the completeness check on the new experiments to make sure no source values were missed.

---

### 🤖 Assistant

All 1,092 source scalars accounted for with 0 missing. Let me clean up the helper scripts.

---

### 🤖 Assistant

Let me also delete the old `experiment_comparison.json` that was created in the previous turn since everything is now consolidated into `exp5_exp10_result_comparison.json`.

---

### 🤖 Assistant

Already gone. Let me verify the final state of the output file.

---

### 🤖 Assistant

Done. Here's the summary:

## Updated: `expe_logs/exp5_exp10_result_comparison.json`

**15 total experiments** now consolidated in a single JSON file, ready for `pd.DataFrame.from_dict(data, orient='index')`.

### 7 New Experiments Added

| Experiment | Source | Columns | Notes |
|---|---|---|---|
| `exp_round6_training_6_8M` | exp_round6/training_6-8M/ | 158 | 256dim, 6-8M samples, pos_weight=200 |
| `exp_round6_training_3_4M_exp2` | exp_round6/training_3-4M/exp2/ | 145 | 256dim, 3-4M samples, pos_weight=35 |
| `exp_round6_training_3_4M_exp2_doublesize` | exp_round6/training_3-4M/exp2/ | 145 | Identical to above (config/results match) |
| `exp_round7_512dim_exp2` | exp_round7_512dim/exp2/ | 158 | **512dim**, 1.5M samples, pos_weight=200 |
| `exp_round8_exp2b_512dim_6_8M` | exp_round8/exp2b_512dim_6-8M/ | 158 | **512dim**, 6-8M samples, pos_weight=200 |
| `exp_round9_exp2b_256dim_v1` | exp_round9/exp2b_256dim_v1/ | 164 | 256dim, decoupled training, pretrained_emb=**false**, stage2 |
| `exp_round9_exp2b_256dim_v2` | exp_round9/exp2b_256dim_v2/ | 164 | 256dim, decoupled training, pretrained_emb=**true**, stage2 |

### Verification

- **Value correctness**: 1,092 checks against source JSONs -- **0 errors**
- **Completeness**: 1,092/1,092 source scalars present -- **0 missing**

Note: `exp_round9_exp2b_256dim_v0` was excluded because it has a config.json but no final_results.json (incomplete run).

