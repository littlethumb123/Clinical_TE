# Feature importance evaluation and embedding tables

## Metadata

- **Conversation ID:** df26bee0-fe38-4b54-b05d-b19d545c98ad
- **Mode:** agent
- **Created:** 2026-03-15 09:07:37
- **Last Updated:** 2026-03-15 11:43:40
- **Branch:** main
- **Files Changed:** 0
- **Lines Added:** 0
- **Lines Removed:** 0
- **Summary:** Edited moe_flashattn_3_lob3_downstream_running.py, 2026-03-15-feature-importance-and-multi-model-embedding-generation.md

---

## Conversation

### 🧑 User

**File: /Users/a964286/Documents/Projects/Clinical_TE/Clinical_TE/.cursor/rules/downstream_classification.mdc**
```

     8|You are a expert seasoned Staff-level Data Scientist / ML Engineer / Applied Researcher specializing in:
     9|- Representation learning with transformers on longitudinal healthcare/claims data
    10|- Reliable downstream evaluation of learned embeddings as features
    11|- Industry-grade tabular modeling (e.g., LogReg, XGBoost, LightGBM, CatBoost) and rigorous validation
    12|- Reproducible, audit-friendly experimentation (data/version control, leakage prevention, statistical comparison)
    13|
    14|You behave like a *careful scientific engineer*: precise, skeptical, and explicit about assumptions.

```


<manually_attached_skills>
The user has manually attached the following skills to their message.
These skills contain specific instructions or workflows that the user wants you to follow for this request.
Only read the files if needed, the full skill content is inlined here.

Skill Name: writing-plans
Path: /Users/a964286/.cursor/skills/writing-plans/SKILL.md
SKILL.md content:
# Writing Plans

## Overview

Write comprehensive implementation plans assuming the engineer has zero context for our codebase and questionable taste. Document everything they need to know: which files to touch for each task, code, testing, docs they might need to check, how to test it. Give them the whole plan as bite-sized tasks. DRY. YAGNI. TDD. Frequent commits.

Assume they are a skilled developer, but know almost nothing about our toolset or problem domain. Assume they don't know good test design very well.

**Announce at start:** "I'm using the writing-plans skill to create the implementation plan."

**Context:** This should be run in a dedicated worktree (created by brainstorming skill).

**Save plans to:** `docs/plans/YYYY-MM-DD-<feature-name>.md`

## Bite-Sized Task Granularity

**Each step is one action (2-5 minutes):**
- "Write the failing test" - step
- "Run it to make sure it fails" - step
- "Implement the minimal code to make the test pass" - step
- "Run the tests and make sure they pass" - step
- "Commit" - step

## Plan Document Header

**Every plan MUST start with this header:**

```markdown
# [Feature Name] Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** [One sentence describing what this builds]

**Architecture:** [2-3 sentences about approach]

**Tech Stack:** [Key technologies/libraries]

---
```

## Task Structure

````markdown
### Task N: [Component Name]

**Files:**
- Create: `exact/path/to/file.py`
- Modify: `exact/path/to/existing.py:123-145`
- Test: `tests/exact/path/to/test.py`

**Step 1: Write the failing test**

```python
def test_specific_behavior():
    result = function(input)
    assert result == expected
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/path/test.py::test_name -v`
Expected: FAIL with "function not defined"

**Step 3: Write minimal implementation**

```python
def function(input):
    return expected
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/path/test.py::test_name -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/path/test.py src/path/file.py
git commit -m "feat: add specific feature"
```
````

## Remember
- Exact file paths always
- Complete code in plan (not "add validation")
- Exact commands with expected output
- Reference relevant skills with @ syntax
- DRY, YAGNI, TDD, frequent commits

## Execution Handoff

After saving the plan, offer execution choice:

**"Plan complete and saved to `docs/plans/<filename>.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?"**

**If Subagent-Driven chosen:**
- **REQUIRED SUB-SKILL:** Use superpowers:subagent-driven-development
- Stay in this session
- Fresh subagent per task + code review

**If Parallel Session chosen:**
- Guide them to open new session in worktree
- **REQUIRED SUB-SKILL:** New session uses superpowers:executing-plans
</manually_attached_skills>
@downstream_classification.mdc (8-14) /writing-plans  i would like to complete the following tasks to the @dev/downstream/moe_flashattn_3_lob3_downstream_running.py for both commercial and medicaid; 
1) add feature importance evaluation section using shapley values and analyze what proportion of features in the top10, top 20 and top50 are embedding features; the feature importance module should be model agnostic, which means the it can take any models (logistic regression, cataboost, xgboost, lightgbm) Also this module will be shared by both commercial and medicaid; so it will not repeat the codes; (context, the goal is to demonstrate the additional values of embeddings)
2) For both medicare and medicaid; I would like to generate embedding tables based on the following models and ingest table to GCP with the name conventions as it is in commercial but customized to medicaid and medicare; 
'logs/exp_round10_3lobs_formal_training/exp2b_flash_learned_pool/saved_models/exp_round10_3lobs_formal_training_exp2b_flash_learned_pool_bs128_ep1_d256_20260312_095916_final.pt' 
'logs/exp_round7_3lobs_1-5M_pretrain_multi_gpu_test_v3_512dim/exp2b_flash_learned_poolexp_round7_3lobs_1-5M_pretrain_multi_gpu_test_v3_512dim_exp2b_flash_learned_pool_bs128_ep1_d512_20260303_023717_final/saved_models/exp_round7_3lobs_1-5M_pretrain_multi_gpu_test_v3_512dim_exp2b_flash_learned_pool_bs128_ep1_d512_20260303_023717_final.pt'
'logs/exp_round9_3lobs_1-5M_decoupled_training_embedding_v4_256dim/exp2b_flash_learned_pool_v2/saved_models/exp_round9_3lobs_1-5M_decoupled_training_embedding_v4_256dim_exp2b_flash_learned_pool_bs128_ep1_d256_20260310_123547_final.pt'

---

### 🤖 Assistant

I'm using the writing-plans skill to create the implementation plan.

Let me start by thoroughly understanding the existing codebase, conventions, and the current downstream pipeline before writing the plan.

---

### 🤖 Assistant

The downstream file is very large. Let me read it in sections to understand the full structure.

---

### 🤖 Assistant

Now let me read the rest of the file, focusing on the evaluate functions, the Medicaid section, and the embedding generation for Medicaid.

---

### 🤖 Assistant

Now let me read the Medicaid embedding generation section, the medicaid evaluation section, and the end of the file for the execution cells.

---

### 🤖 Assistant

Now I have a thorough understanding of the entire file. Let me also check the naming conventions for Medicare embedding tables and whether the `save_medicaid_embeddings_to_bigquery` function exists for Medicare.

---

### 🤖 Assistant

Let me look at the existing plan for reference and check the end of the file to understand the Medicare embedding table naming convention used in the BQ save call.

---

### 🤖 Assistant

Now I have comprehensive understanding. Let me also quickly check the naming conventions for the 3 model paths provided by the user, to understand the experiment round naming.

---

### 🤖 Assistant

Now I have full context. Let me create the implementation plan.

---

### 🤖 Assistant

Plan complete and saved to `docs/plans/2026-03-15-feature-importance-and-multi-model-embedding-generation.md`.

Here's a summary of the 7-task plan:

**Task 1** — Add a shared, model-agnostic SHAP feature importance module (`compute_shap_feature_importance` + `run_shap_for_all_feature_sets`) placed once before both Commercial and Medicaid sections. Supports LogReg, CatBoost, XGBoost, LightGBM via automatic SHAP backend selection. Outputs a ranked SHAP summary DataFrame and an embedding proportion analysis at top-10/20/50 cutoffs.

**Task 2** — Add SHAP evaluation cells for Commercial (trains CatBoost hybrid, runs SHAP, exports to Excel).

**Task 3** — Add SHAP evaluation cells for Medicaid (same approach, exports to Excel).

**Task 4** — Add Commercial embedding generation for the 3 new model checkpoints (rounds 10, 7, 9) with GCP table upload following existing naming conventions.

**Task 5** — Add Medicare embedding generation for the same 3 models (uses `individual_id`, `lob_value=None`, consistent naming).

**Task 6** — Add Medicaid embedding generation for the same 3 models (uses `asdb_member_key`, `lob_value='Medicaid'`, Medicaid-specific naming).

**Task 7** — Final syntax check, linter verification, and cleanup commit.

Two execution options:

**1. Subagent-Driven (this session)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Parallel Session (separate)** — Open a new session in a worktree with the executing-plans skill for batch execution with checkpoints.

Which approach?

---

### 🧑 User

<manually_attached_skills>
The user has manually attached the following skills to their message.
These skills contain specific instructions or workflows that the user wants you to follow for this request.
Only read the files if needed, the full skill content is inlined here.

Skill Name: executing-plans
Path: /Users/a964286/.cursor/skills/executing-plans/SKILL.md
SKILL.md content:
# Executing Plans

## Overview

Load plan, review critically, execute tasks in batches, report for review between batches.

**Core principle:** Batch execution with checkpoints for architect review.

**Announce at start:** "I'm using the executing-plans skill to implement this plan."

## The Process

### Step 1: Load and Review Plan
1. Read plan file
2. Review critically - identify any questions or concerns about the plan
3. If concerns: Raise them with your human partner before starting
4. If no concerns: Create TodoWrite and proceed

### Step 2: Execute Batch
**Default: First 3 tasks**

For each task:
1. Mark as in_progress
2. Follow each step exactly (plan has bite-sized steps)
3. Run verifications as specified
4. Mark as completed

### Step 3: Report
When batch complete:
- Show what was implemented
- Show verification output
- Say: "Ready for feedback."

### Step 4: Continue
Based on feedback:
- Apply changes if needed
- Execute next batch
- Repeat until complete

### Step 5: Complete Development

After all tasks complete and verified:
- Announce: "I'm using the finishing-a-development-branch skill to complete this work."
- **REQUIRED SUB-SKILL:** Use superpowers:finishing-a-development-branch
- Follow that skill to verify tests, present options, execute choice

## When to Stop and Ask for Help

**STOP executing immediately when:**
- Hit a blocker mid-batch (missing dependency, test fails, instruction unclear)
- Plan has critical gaps preventing starting
- You don't understand an instruction
- Verification fails repeatedly

**Ask for clarification rather than guessing.**

## When to Revisit Earlier Steps

**Return to Review (Step 1) when:**
- Partner updates the plan based on your feedback
- Fundamental approach needs rethinking

**Don't force through blockers** - stop and ask.

## Remember
- Review plan critically first
- Follow plan steps exactly
- Don't skip verifications
- Reference skills when plan says to
- Between batches: just report and wait
- Stop when blocked, don't guess
- Never start implementation on main/master branch without explicit user consent

## Integration

**Required workflow skills:**
- **superpowers:using-git-worktrees** - REQUIRED: Set up isolated workspace before starting
- **superpowers:writing-plans** - Creates the plan this skill executes
- **superpowers:finishing-a-development-branch** - Complete development after all tasks
</manually_attached_skills>
Ok here is my feedback and follow up task
1) for embedding generation, only medicaid and medicare are needed; commercial was already done; revise the plan correspondingly; 
2) /executing-plans execute the revised plan on @dev/downstream/moe_flashattn_3_lob3_downstream_running.py 
3) after excute the plan, verify and review the code carefully making sure they are bug-free
3) at the same time, in the chat session; explicitly explain which codes are added to where, becuase I am going to add them manually to remote jupyter notebook; so making sure give instruction like search for waht types of keywrods and add what code after the line with "xxxxxxx", the exact precise place;

---

### 🤖 Assistant

I'm using the executing-plans skill to implement this plan.

Let me first revise the plan to remove the Commercial embedding generation task, then execute it.

---

### 🤖 Assistant

Let me first update the plan to remove Commercial embedding generation, then re-read the exact lines I'll be inserting at for precision.

---

### 🤖 Assistant

Good. I have precise anchor lines for all 5 insertion points. Now let me execute all tasks. I'll insert the code blocks one by one.

**Task 1: Insert the shared SHAP module.** It goes right before line 1386 (`# In[75]:`) which precedes the METRIC FUNCTIONS section.

---

### 🤖 Assistant

**Task 2: Add SHAP evaluation cells for Commercial.** This goes after the commercial excel export line. Let me find the exact anchor now (line numbers shifted due to the Task 1 insertion).

---

### 🤖 Assistant

I'll insert the commercial SHAP cells after line 2527 (after `df_commercial_downstream.columns`) and before the `# ### Medicare embedding generation` section.

---

### 🤖 Assistant

**Task 3: Add SHAP evaluation cells for Medicaid.** This goes at the very end of the file after the last line.

---

### 🤖 Assistant

**Task 4: Add Medicare embedding generation for 3 new models.** This goes after the existing Medicare embedding loop. Let me find the exact anchor.

---

### 🤖 Assistant

I'll insert the new Medicare embedding generation cells between the existing Medicare loop and the `import sklearn` line. The anchor is the line `import sklearn` / `print("The scikit-learn version is"...`.

---

### 🤖 Assistant

**Task 5: Add Medicaid embedding generation for 3 new models.** This goes after the existing Medicaid embedding generation loop. Let me find the exact anchor.

---

### 🤖 Assistant

**Task 6: Syntax check + verification.** Let me run a syntax compile and verify all inserted blocks.

---

### 🤖 Assistant

Good, syntax passes. Now let me verify each insertion is correctly placed by reading the exact boundaries.

---

### 🤖 Assistant

All 5 blocks inserted at correct locations. Now let me do a careful code review by reading each inserted section.

---

### 🤖 Assistant

SHAP module looks correct. Let me check the other blocks.

---

### 🤖 Assistant

All code verified. Let me also check linter output.

---

### 🤖 Assistant

All code is verified, syntax passes, no linter errors. Here are the precise insertion instructions for your remote Jupyter notebook.

---

## Precise Insertion Instructions for Remote Jupyter

### Insertion 1: Shared SHAP Feature Importance Module (2 functions)

**Where to insert:** Search for the line `# METRIC FUNCTIONS` in the Commercial section. This is inside a comment block:
```
# =============================================================================
# METRIC FUNCTIONS
# =============================================================================
```
**Add a NEW CELL BEFORE that cell.** The new cell goes right after the cell that contains:
```
# 5	5: 6-12 months	100909	1.44	...
```

**Code to add as a new cell:**

```python
# =============================================================================
# SHARED FEATURE IMPORTANCE MODULE (SHAP — Model-Agnostic)
# =============================================================================
# Used by both Commercial and Medicaid sections.
# Goal: quantify what proportion of top-N important features are embeddings.

import shap

def compute_shap_feature_importance(
    fitted_model,
    X_eval: pd.DataFrame,
    feature_cols: List[str],
    embedding_features: List[str],
    top_k_list: List[int] = [10, 20, 50],
    max_samples: int = 2000,
    random_state: int = 42,
    model_name: str = "",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute SHAP feature importance and analyze embedding feature proportions.

    Works with any model that has a predict_proba method (LogisticRegression,
    CatBoost, XGBoost, LightGBM, etc.).

    Args:
        fitted_model: A TRAINED model with predict_proba().
        X_eval: Evaluation DataFrame (use val or test split, NOT train).
        feature_cols: Ordered list of feature column names matching X_eval columns.
        embedding_features: List of embedding column names (subset of feature_cols).
        top_k_list: List of top-K cutoffs for proportion analysis (default [10, 20, 50]).
        max_samples: Cap on background/eval samples for SHAP speed (default 2000).
        random_state: Seed for sampling reproducibility.
        model_name: Label for output (e.g. "CatBoost_hybrid").
        verbose: Print progress.

    Returns:
        shap_summary_df: DataFrame with columns [feature, mean_abs_shap, rank, is_embedding]
                         sorted by mean_abs_shap descending.
        proportion_df:   DataFrame with columns [model_name, top_k, n_embedding_in_top_k,
                         proportion_embedding, n_tabular_in_top_k, proportion_tabular]
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"SHAP FEATURE IMPORTANCE: {model_name or type(fitted_model).__name__}")
        print(f"{'='*70}")

    X_sample = X_eval
    if len(X_eval) > max_samples:
        X_sample = X_eval.sample(n=max_samples, random_state=random_state)
        if verbose:
            print(f"  Sampled {max_samples} rows from {len(X_eval)} for SHAP computation")

    model_type = type(fitted_model).__name__

    if model_type in ('CatBoostClassifier',):
        explainer = shap.TreeExplainer(fitted_model)
        shap_values = explainer.shap_values(X_sample)
    elif model_type in ('XGBClassifier', 'LGBMClassifier'):
        explainer = shap.TreeExplainer(fitted_model)
        shap_values = explainer.shap_values(X_sample)
    elif model_type == 'LogisticRegression':
        background = shap.sample(X_sample, min(100, len(X_sample)))
        explainer = shap.LinearExplainer(fitted_model, background)
        shap_values = explainer.shap_values(X_sample)
    else:
        background = shap.sample(X_sample, min(100, len(X_sample)))
        explainer = shap.KernelExplainer(
            fitted_model.predict_proba, background
        )
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    embedding_set = set(embedding_features)
    shap_summary_df = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_abs_shap,
        'is_embedding': [f in embedding_set for f in feature_cols],
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    shap_summary_df['rank'] = range(1, len(shap_summary_df) + 1)

    proportion_rows = []
    for k in top_k_list:
        k_actual = min(k, len(shap_summary_df))
        top_k_df = shap_summary_df.head(k_actual)
        n_emb = int(top_k_df['is_embedding'].sum())
        n_tab = k_actual - n_emb
        proportion_rows.append({
            'model_name': model_name or model_type,
            'top_k': k,
            'n_embedding_in_top_k': n_emb,
            'proportion_embedding': round(n_emb / k_actual, 4),
            'n_tabular_in_top_k': n_tab,
            'proportion_tabular': round(n_tab / k_actual, 4),
        })

    proportion_df = pd.DataFrame(proportion_rows)

    if verbose:
        print(f"\n  Top 20 features by mean |SHAP|:")
        print(shap_summary_df[['rank', 'feature', 'mean_abs_shap', 'is_embedding']].head(20).to_string(index=False))
        print(f"\n  Embedding Proportion Analysis:")
        print(proportion_df.to_string(index=False))

    return shap_summary_df, proportion_df

def run_shap_for_all_feature_sets(
    fitted_models: Dict[str, Any],
    prepared_data_dict: Dict[str, Any],
    top_k_list: List[int] = [10, 20, 50],
    max_samples: int = 2000,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run SHAP analysis across multiple (model, feature_set) combos.

    Args:
        fitted_models: Dict mapping label -> fitted model (must already be trained).
        prepared_data_dict: Dict mapping same labels -> prepared data objects
                            (PreparedData or MedicaidPreparedData).
        top_k_list: Top-K cutoffs.
        max_samples: SHAP sample cap.
        verbose: Print progress.

    Returns:
        all_shap_df: Concatenated SHAP summaries with 'experiment' column.
        all_proportion_df: Concatenated proportion summaries.
    """
    all_shap = []
    all_proportion = []

    for label, model in fitted_models.items():
        pd_obj = prepared_data_dict[label]

        if hasattr(pd_obj, 'X_splits'):
            X_eval = pd_obj.X_splits.get('test', pd_obj.X_splits.get('val'))
        elif hasattr(pd_obj, 'X_test'):
            X_eval = pd_obj.X_test
        else:
            raise ValueError(f"Cannot find evaluation data in {type(pd_obj)}")

        feature_cols = pd_obj.feature_cols
        embedding_features = pd_obj.embedding_features

        shap_df, prop_df = compute_shap_feature_importance(
            fitted_model=model,
            X_eval=X_eval,
            feature_cols=feature_cols,
            embedding_features=embedding_features,
            top_k_list=top_k_list,
            max_samples=max_samples,
            model_name=label,
            verbose=verbose,
        )
        shap_df['experiment'] = label
        all_shap.append(shap_df)
        all_proportion.append(prop_df)

    return pd.concat(all_shap, ignore_index=True), pd.concat(all_proportion, ignore_index=True)
```

---

### Insertion 2: Commercial SHAP Evaluation (2 cells)

**Where to insert:** Search for `df_commercial_downstream.columns`. Add **2 new cells AFTER** the cell that contains that line, and **BEFORE** the markdown header `### Medicare embedding generation`.

**Cell 1 — Train CatBoost for SHAP:**

```python
# =============================================================================
# COMMERCIAL: SHAP Feature Importance Analysis
# =============================================================================
# Demonstrates the additional value of embeddings via SHAP
# Uses the hybrid feature set to see embedding vs tabular importance

embedding_path_shap = 'edp-prod-storage.edp_ent_sdoheir_cns.a964286_te4exp_3lob_exp_round5_v2_exp2b_flash_learned_pool_asym_focalloss_densesampler_commercial_all_sample_embedding'

prepared_hybrid_commercial = prepare_evaluation_data(
    df_features=df_ip_features,
    embedding_location_path=embedding_path_shap,
    feature_set='hybrid',
    downsample_ratio=10.0
)

# Train CatBoost on hybrid and capture the fitted model
from sklearn.base import clone as sk_clone

catboost_shap = sk_clone(catboost_model)
cat_indices = prepared_hybrid_commercial.cat_feature_indices if prepared_hybrid_commercial.cat_feature_indices else []
from catboost import Pool
train_pool_shap = Pool(
    prepared_hybrid_commercial.X_splits['train'],
    prepared_hybrid_commercial.y_splits['train'],
    cat_features=cat_indices,
)
val_pool_shap = Pool(
    prepared_hybrid_commercial.X_splits['val'],
    prepared_hybrid_commercial.y_splits['val'],
    cat_features=cat_indices,
)
catboost_shap.fit(train_pool_shap, eval_set=val_pool_shap, verbose=0)
print("Commercial CatBoost (hybrid) trained for SHAP analysis")
```

**Cell 2 — Run SHAP and save:**

```python
# Run SHAP
commercial_shap_df, commercial_proportion_df = compute_shap_feature_importance(
    fitted_model=catboost_shap,
    X_eval=prepared_hybrid_commercial.X_splits['test'],
    feature_cols=prepared_hybrid_commercial.feature_cols,
    embedding_features=prepared_hybrid_commercial.embedding_features,
    top_k_list=[10, 20, 50],
    max_samples=2000,
    model_name="commercial_catboost_hybrid",
    verbose=True,
)

# Save results
commercial_shap_df.to_excel("experiment_logs/commercial_shap_feature_importance.xlsx", index=False)
commercial_proportion_df.to_excel("experiment_logs/commercial_shap_embedding_proportions.xlsx", index=False)
print("\nSHAP results saved to experiment_logs/")
```

---

### Insertion 3: Medicare Embedding Generation (1 cell)

**Where to insert:** Search for `import sklearn` / `print("The scikit-learn version is"`. Add **1 new cell BEFORE** that cell, which is right after the existing Medicare embedding generation loop (the one that ends with `torch.cuda.empty_cache()`).

**Code to add as a new cell:**

```python
# =============================================================================
# MEDICARE: Embedding Generation for Round 10, 9, and 7 Models
# =============================================================================

MODEL_PATHS_NEW_MEDICARE = {
    'exp_round10_formal_exp2b_flash_learned_pool_d256':
        'logs/exp_round10_3lobs_formal_training/exp2b_flash_learned_pool/saved_models/'
        'exp_round10_3lobs_formal_training_exp2b_flash_learned_pool_bs128_ep1_d256_20260312_095916_final.pt',

    'exp_round7_exp2b_flash_learned_pool_d512':
        'logs/exp_round7_3lobs_1-5M_pretrain_multi_gpu_test_v3_512dim/'
        'exp2b_flash_learned_poolexp_round7_3lobs_1-5M_pretrain_multi_gpu_test_v3_512dim_exp2b_flash_learned_pool_bs128_ep1_d512_20260303_023717_final/'
        'saved_models/'
        'exp_round7_3lobs_1-5M_pretrain_multi_gpu_test_v3_512dim_exp2b_flash_learned_pool_bs128_ep1_d512_20260303_023717_final.pt',

    'exp_round9_decoupled_exp2b_flash_learned_pool_v2_d256':
        'logs/exp_round9_3lobs_1-5M_decoupled_training_embedding_v4_256dim/exp2b_flash_learned_pool_v2/saved_models/'
        'exp_round9_3lobs_1-5M_decoupled_training_embedding_v4_256dim_exp2b_flash_learned_pool_bs128_ep1_d256_20260310_123547_final.pt',
}

results_new_medicare = {}
batch_size = 64
PROJECT_ID = "edp-prod-storage"
DATASET_ID = "edp_ent_sdoheir_cns"
LOB = 'medicare'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for exp_name, model_path in tqdm(MODEL_PATHS_NEW_MEDICARE.items(), desc="Medicare embedding generation"):
    cleanup_gpu_memory(verbose=False)
    model, config, moe_config, use_mixed_precision, model_type = load_model_from_checkpoint(
        model_path=model_path,
        device=device,
        verbose=True
    )

    inference_start_time = time.time()
    embeddings, individual_ids, index_dts = generate_embeddings(
        model=model,
        config=config,
        data=df_me_sample,
        device=device,
        id_column='individual_id',
        lob_value=None,
        desc_prefix='Medicare',
        batch_size=batch_size,
        use_mixed_precision=use_mixed_precision,
        verbose=True,
        multi_gpu=True,
        moe_config=moe_config,
    )
    inference_duration = time.time() - inference_start_time
    print(f"Inference duration for {exp_name}: {round(inference_duration/3600, 2):.2f} hr")

    safe_exp_name = exp_name.replace('-', '_').replace('.', '_')
    table_name = f"a964286_te4exp_3lob_{safe_exp_name}_{LOB}_all_sample_embedding"
    bq_table_path = save_embeddings_to_bigquery(
        embeddings=embeddings,
        individual_ids=individual_ids,
        index_dts=index_dts,
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_name=table_name,
        exp_name=exp_name,
        model_type=model_type,
        if_exists="replace"
    )
    results_new_medicare[exp_name] = {
        'bq_table_path': bq_table_path,
        'embedding_shape': embeddings.shape,
        'model_type': model_type,
        'model_path': model_path,
        'inference_duration_hr': round(inference_duration / 3600, 2),
        'status': 'success'
    }

    del model
    del embeddings
    torch.cuda.empty_cache()

print("\n=== Medicare Embedding Generation Summary ===")
for exp_name, result in results_new_medicare.items():
    print(f"  {exp_name}: {result['embedding_shape']} -> {result['bq_table_path']}")
```

---

### Insertion 4: Medicaid Embedding Generation (1 cell)

**Where to insert:** In the Medicaid section, search for the existing embedding generation loop that ends with:
```python
        print(f"  {exp_name}: {result['embedding_shape']} ({result['inference_duration_hr']:.2f}hr)")
```
This is followed by `#### Model training`. Add **1 new cell AFTER** that print loop and **BEFORE** the `#### Model training` / `MEDICAID_EMBEDDING_TABLES` cell.

**Code to add as a new cell:**

```python
# =============================================================================
# MEDICAID: Embedding Generation for Round 10, 9, and 7 Models
# =============================================================================

MODEL_PATHS_NEW_MEDICAID = {
    'exp_round10_formal_exp2b_flash_learned_pool_d256':
        'logs/exp_round10_3lobs_formal_training/exp2b_flash_learned_pool/saved_models/'
        'exp_round10_3lobs_formal_training_exp2b_flash_learned_pool_bs128_ep1_d256_20260312_095916_final.pt',

    'exp_round7_exp2b_flash_learned_pool_d512':
        'logs/exp_round7_3lobs_1-5M_pretrain_multi_gpu_test_v3_512dim/'
        'exp2b_flash_learned_poolexp_round7_3lobs_1-5M_pretrain_multi_gpu_test_v3_512dim_exp2b_flash_learned_pool_bs128_ep1_d512_20260303_023717_final/'
        'saved_models/'
        'exp_round7_3lobs_1-5M_pretrain_multi_gpu_test_v3_512dim_exp2b_flash_learned_pool_bs128_ep1_d512_20260303_023717_final.pt',

    'exp_round9_decoupled_exp2b_flash_learned_pool_v2_d256':
        'logs/exp_round9_3lobs_1-5M_decoupled_training_embedding_v4_256dim/exp2b_flash_learned_pool_v2/saved_models/'
        'exp_round9_3lobs_1-5M_decoupled_training_embedding_v4_256dim_exp2b_flash_learned_pool_bs128_ep1_d256_20260310_123547_final.pt',
}

results_new_medicaid = {}
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for exp_name, model_path in tqdm(MODEL_PATHS_NEW_MEDICAID.items(), desc="Medicaid embedding generation"):
    cleanup_gpu_memory(verbose=False)
    model, config, moe_config, use_mixed_precision, model_type = load_model_from_checkpoint(
        model_path=model_path,
        device=device,
        verbose=True
    )

    inference_start_time = time.time()
    embeddings, member_keys, index_dts = generate_embeddings(
        model=model,
        config=config,
        data=df_te_input,
        device=device,
        id_column='asdb_member_key',
        lob_value='Medicaid',
        desc_prefix='Medicaid',
        batch_size=batch_size,
        use_mixed_precision=use_mixed_precision,
        verbose=True,
        multi_gpu=True,
        moe_config=moe_config,
    )
    inference_duration = time.time() - inference_start_time
    print(f"Inference duration for {exp_name}: {round(inference_duration/3600, 2):.2f} hr")

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
    results_new_medicaid[exp_name] = {
        'bq_table_path': bq_table_path,
        'embedding_shape': embeddings.shape,
        'model_type': model_type,
        'model_path': model_path,
        'inference_duration_hr': round(inference_duration / 3600, 2),
        'status': 'success'
    }

    del model
    del embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n=== Medicaid Embedding Generation Summary ===")
for exp_name, result in results_new_medicaid.items():
    print(f"  {exp_name}: {result['embedding_shape']} -> {result['bq_table_path']}")
```

---

### Insertion 5: Medicaid SHAP Evaluation (2 cells)

**Where to insert:** At the **very end of the notebook**, after the last cell which contains:
```python
df_comparison.to_excel("exp_round5_3lob_1-5M_1epoch_128batch_dim256_medicaid_ip_downstream_eval.xlsx")
```

Add **2 new cells** after that.

**Cell 1 — Prepare and train:**

```python
# =============================================================================
# MEDICAID: SHAP Feature Importance Analysis
# =============================================================================
# Uses the hybrid feature set to quantify embedding vs tabular importance.
# Reuses df_merged from the Medicaid evaluation loop above.

prepared_medicaid_hybrid = prepare_medicaid_evaluation_data(
    df=df_merged,
    feature_set='hybrid',
    apply_downsampling=True,
    downsample_ratio=CATBOOST_UNDERSAMPLE_RATIO,
    split_random_state=RANDOM_STATE,
    undersample_random_state=UNDERSAMPLE_RANDOM_STATE,
    verbose=True,
)

# Train CatBoost for SHAP
catboost_medicaid_shap = CatBoostClassifier(**CATBOOST_TUNED_PARAMS)
train_pool_md_shap = Pool(
    prepared_medicaid_hybrid.X_train,
    prepared_medicaid_hybrid.y_train,
    cat_features=prepared_medicaid_hybrid.cat_feature_indices if prepared_medicaid_hybrid.cat_feature_indices else None,
)
val_pool_md_shap = Pool(
    prepared_medicaid_hybrid.X_val,
    prepared_medicaid_hybrid.y_val,
    cat_features=prepared_medicaid_hybrid.cat_feature_indices if prepared_medicaid_hybrid.cat_feature_indices else None,
)
catboost_medicaid_shap.fit(train_pool_md_shap, eval_set=val_pool_md_shap, verbose=0)
print("Medicaid CatBoost (hybrid) trained for SHAP analysis")
```

**Cell 2 — Run SHAP and save:**

```python
# Run SHAP
medicaid_shap_df, medicaid_proportion_df = compute_shap_feature_importance(
    fitted_model=catboost_medicaid_shap,
    X_eval=prepared_medicaid_hybrid.X_test,
    feature_cols=prepared_medicaid_hybrid.feature_cols,
    embedding_features=prepared_medicaid_hybrid.embedding_features,
    top_k_list=[10, 20, 50],
    max_samples=2000,
    model_name="medicaid_catboost_hybrid",
    verbose=True,
)

# Save results
medicaid_shap_df.to_excel("experiment_logs/medicaid_shap_feature_importance.xlsx", index=False)
medicaid_proportion_df.to_excel("experiment_logs/medicaid_shap_embedding_proportions.xlsx", index=False)
print("\nMedicaid SHAP results saved to experiment_logs/")
```

---

## Quick Reference Summary

| # | What | Search for this keyword | Insert position |
|---|---|---|---|
| 1 | SHAP module (shared) | `# METRIC FUNCTIONS` | **New cell BEFORE** that cell |
| 2 | Commercial SHAP (2 cells) | `df_commercial_downstream.columns` | **2 new cells AFTER** that cell, BEFORE `### Medicare embedding generation` |
| 3 | Medicare embedding gen (1 cell) | `import sklearn` / `print("The scikit-learn version is"` | **New cell BEFORE** that cell |
| 4 | Medicaid embedding gen (1 cell) | `{result['embedding_shape']} ({result['inference_duration_hr']:.2f}hr)` (in the Medicaid section) | **New cell AFTER** that cell, BEFORE `#### Model training` / `MEDICAID_EMBEDDING_TABLES` |
| 5 | Medicaid SHAP (2 cells) | `df_comparison.to_excel("exp_round5_3lob_1-5M...medicaid_ip_downstream_eval.xlsx")` | **2 new cells AFTER** that cell (end of notebook) |

**Key differences between Medicare vs Medicaid embedding generation:**
- **Medicare**: uses `id_column='individual_id'`, `lob_value=None`, `data=df_me_sample`, `save_embeddings_to_bigquery()`, table pattern `a964286_te4exp_3lob_{name}_medicare_all_sample_embedding`
- **Medicaid**: uses `id_column='asdb_member_key'`, `lob_value='Medicaid'`, `data=df_te_input`, `save_medicaid_embeddings_to_bigquery()`, table pattern `a964286_te4exp_{name}_medicaid_heldout_embedding`

---

### 🤖 Assistant

All tasks complete. The file grew from 4,247 lines to 4,711 lines (464 lines added). Syntax verified clean, zero linter errors. Ready for feedback.

