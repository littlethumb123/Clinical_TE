# Commercial IP Formal Evaluation Design

## Goal

Create a formal-evaluation variant of the Commercial IP tabular feature and outcome generation SQL by cloning the existing Commercial TE experiment pipeline and changing only the cohort source and table naming.

## Scope

- Clone the existing Commercial IP outcome-generation script from `data_ingestion/Com_ip/commercial_ip_outcome_generation.sql`.
- Change the Step 0 base cohort source from `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_member_o3_train_ending` to `edp-prod-storage.edp_ent_sdoheir_cns.a964286_commercial_embedding_raw_features_20241120_20250930`.
- Rename the final output table to `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Commercial_final_dataset_4_te_formal_evaluation_20241120_20250930`.
- Rename intermediate tables consistently so they do not collide with the existing `4_te_experiment` pipeline.
- Save the cloned SQL under `data_ingestion/Formal_training_full_downstream/commercial/`.

## Constraints

- Keep the feature engineering, outcome logic, joins, exclusions, and validation queries otherwise identical.
- Preserve the existing SQL structure so the new script is easy to diff against the original.
- Do not modify the original TE experiment script.

## Naming Strategy

Use the suffix `_4_te_formal_evaluation_20241120_20250930` for intermediate and final Commercial tables created by the cloned script.

## Verification

- Confirm the cloned SQL file exists in the target folder.
- Confirm the new file references the formal raw-feature source table.
- Confirm the new file creates the requested final table name.
- Confirm no created-table references remain on the old `_4_te_experiment` naming path inside the cloned script.