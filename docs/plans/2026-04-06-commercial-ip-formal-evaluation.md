# Commercial IP Formal Evaluation Implementation Plan

> **For Copilot or other skill-enabled agents:** REQUIRED SUB-SKILL: Use `/executing-plans` to implement this plan task-by-task.

**Goal:** Create a cloned Commercial IP downstream SQL pipeline for formal evaluation using the raw embedding feature cohort and isolated table names.

**Architecture:** Start from the existing Commercial IP outcome-generation SQL, create a parallel copy under the formal-training commercial path, then apply a minimal set of source-table and output-table renames while preserving all feature and outcome logic. Use systematic replacements for the table family rename and targeted edits for source-specific comments and cohort references.

**Tech Stack:** BigQuery SQL, repository file management, patch-based edits.

---

### Task 1: Create the design artifact

**Files:**
- Create: `docs/plans/2026-04-06-commercial-ip-formal-evaluation-design.md`

**Step 1: Write the design summary**

Document the source swap, naming convention, output location, and verification criteria.

**Step 2: Verify the file exists**

Run a file listing for `docs/plans/` and confirm the new document is present.

### Task 2: Clone the Commercial IP SQL script

**Files:**
- Read: `data_ingestion/Com_ip/commercial_ip_outcome_generation.sql`
- Create: `data_ingestion/Formal_training_full_downstream/commercial/commercial_ip_outcome_generation_formal_evaluation_20241120_20250930.sql`

**Step 1: Copy the source script into the target path**

Use a bulk clone approach so the new file starts as an exact duplicate.

**Step 2: Verify the cloned file exists**

List the target directory and confirm the SQL file is present.

### Task 3: Apply systematic naming changes

**Files:**
- Modify: `data_ingestion/Formal_training_full_downstream/commercial/commercial_ip_outcome_generation_formal_evaluation_20241120_20250930.sql`

**Step 1: Rename the created Commercial table family**

Replace `a834793_Commercial_*_4_te_experiment` with the corresponding `a834793_Commercial_*_4_te_formal_evaluation_20241120_20250930` names.

**Step 2: Change the final dataset name**

Confirm the final table create statement uses `a834793_Commercial_final_dataset_4_te_formal_evaluation_20241120_20250930`.

**Step 3: Verify no old created-table names remain**

Search the cloned file for old `_4_te_experiment` table names in created-table references.

### Task 4: Swap the cohort source

**Files:**
- Modify: `data_ingestion/Formal_training_full_downstream/commercial/commercial_ip_outcome_generation_formal_evaluation_20241120_20250930.sql`

**Step 1: Change the Step 0 source table**

Replace the `FROM` source for the base cohort with `edp-prod-storage.edp_ent_sdoheir_cns.a964286_commercial_embedding_raw_features_20241120_20250930`.

**Step 2: Update source comments for accuracy**

Adjust the header and Step 0 notes so they describe the formal raw-feature cohort instead of the original transformer-training table.

**Step 3: Verify the new source is referenced**

Search the cloned file for the raw-feature table name and confirm the old source table is no longer used for Step 0.

### Task 5: Validate the generated SQL artifact

**Files:**
- Verify: `data_ingestion/Formal_training_full_downstream/commercial/commercial_ip_outcome_generation_formal_evaluation_20241120_20250930.sql`

**Step 1: Run workspace error check on the new file**

Use the editor diagnostics tool to catch any syntax-level issues.

**Step 2: Spot-check the critical statements**

Confirm the file contains:
- the new final table name
- the new base source table
- renamed intermediate tables

**Step 3: Summarize the delivered artifact**

Report the new file path and the two functional changes: cohort source and renamed table family.