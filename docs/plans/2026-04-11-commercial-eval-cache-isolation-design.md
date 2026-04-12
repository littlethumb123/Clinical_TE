# Commercial Eval Cache Isolation Design

Date: 2026-04-11

## Goal

Separate modeling-table loading and prepared-feature caching from downstream model fitting and evaluation in the Commercial formal downstream notebook.

## Approved Approach

Use a minimal split in the notebook:

1. Add explicit cache-prep functions for dataset loading and prepared feature construction.
2. Change downstream evaluation functions so they consume prebuilt caches only.
3. Split the run workflow into separate notebook cells:
   - cache preparation
   - single experiment run
   - full sequential run

## Design Details

### Cache Preparation Boundary

- `build_dataset_cache(...)` loads the required modeling tables once.
- `build_prepared_cache(...)` converts dataset cache entries into `PreparedData` objects keyed by `(dataset_name, feature_set, downsample_ratio)`.
- `prepare_evaluation_caches(...)` orchestrates both steps and returns `(dataset_cache, prepared_cache)`.

### Downstream Modeling Boundary

- `evaluate_single_experiment(...)` no longer loads tables or prepares data.
- It resolves the prepared-cache key for the requested experiment and runs model fit / predict / metrics only.
- `run_experiments_sequentially(...)` loops through experiment configs and calls `evaluate_single_experiment(...)` using the prebuilt prepared cache.

### Notebook UX

- Section 11 becomes cache preparation.
- A new cell runs one selected experiment from cache.
- A new cell runs all experiments sequentially from cache.

## Rationale

This preserves the existing lazy-loading and cache reuse behavior, while making the boundary explicit:

- data access and preprocessing happen once in a dedicated stage
- downstream modeling cells become easier to rerun without reloading data
- single-experiment debugging becomes cleaner and more predictable

## Notes

- SHAP cells remain separate for now.
- No git commit is included in this step.