# Session Progress Report - Stage 2 Training Progress Tracking & Configuration
**Date**: 2026-03-26
**Status**: Added comprehensive progress tracking to `train_stage2` and `_build_code_index` across both experiment notebooks; upgraded Stage 2 config from conservative (v0) to aggressive (v1/v2) parameters.

## 1. Executive Summary

This session addressed a critical observability gap in the Stage 2 decoder re-training pipeline. When running `train_stage2` on the 11M dataset, the function appeared to "hang" for extended periods with no output — particularly during the CPU-bound `_build_code_index` step that iterates over all 11M samples to build a code-to-patient mapping. Progress tracking with time estimates (ETA) was added to `_build_code_index`, the sampler creation phase, and the training loop (per-batch and per-epoch) in both `moe_flashattn_5.ipynb` and `moe_flashattn_5_1.ipynb`. Additionally, the Stage 2 configuration was upgraded from the conservative v0 settings (lr=5e-5, 3 epochs, no focused loss) to the aggressive v1/v2-proven settings (lr=0.005, 20 epochs, focused loss enabled), based on exp_round9 results.

## 2. Planned vs. Executed
**Original Plan**: User reported `train_stage2` "took forever" with no way to tell if it was still running. Task was to add progress tracking.
**What Got Done**:
- [x] Diagnosed root cause of apparent hang (CPU-bound `_build_code_index` on 11M samples)
- [x] Added time-based progress reporting to `_build_code_index` (every 30s or 500K samples)
- [x] Added sampler creation timing to `train_stage2`
- [x] Added per-batch progress with throughput, ETA-epoch, ETA-total
- [x] Added per-epoch summary with epoch duration, total elapsed, ETA-remaining
- [x] Added training completion banner with total wall-clock time
- [x] Added timing fields to return dict (`total_time_sec`, `avg_epoch_time_sec`, `epoch_time_sec`)
- [x] Applied all changes to both `moe_flashattn_5.ipynb` and `moe_flashattn_5_1.ipynb`
- [x] Upgraded Stage 2 config from conservative to aggressive (prior session, included in diff)

**Alignment Notes**: Execution matched plan exactly. Both notebooks updated symmetrically.

## 3. Key Decisions & Rationale

### Decision: Time-based printing vs fixed-interval printing for `_build_code_index`
**Context**: The old code printed every 500K samples, which on 11M data meant only ~22 prints across a 30-60 minute indexing phase — potentially 2+ minutes of silence between prints.
**Options Considered**: (A) Decrease sample interval to 100K (more prints but still variable timing) vs. (B) Time-based interval every 30 seconds with sample fallback.
**Chosen**: Option B — **Rationale**: Guarantees user sees output at least every 30 seconds regardless of processing speed, while the 500K fallback catches cases where `time()` calls might be expensive. The 30s interval provides enough updates without flooding output.
**Trade-offs**: Minimal — one `time.time()` call per iteration adds negligible overhead (~50ns per call vs ~microseconds per sample processed).

### Decision: ETA calculation strategy for training loop
**Context**: Need to provide accurate time estimates across epochs of varying duration.
**Chosen**: Running average of completed epoch times for cross-epoch ETA, linear extrapolation within an epoch from elapsed batches. First epoch uses its own elapsed time as the estimate since no history exists yet.
**Trade-offs**: First-epoch ETA may be inaccurate since it extrapolates from early batches. Accuracy improves rapidly after epoch 1.

## 4. Technical Changes

### 4.1 Files Modified
- `dev/moe/moe_flashattn_5.ipynb` — 134 lines changed (cell 89: `train_stage2`, cell 92: `CodeBalancedBatchSampler`)
  - `_build_code_index()`: Replaced fixed-interval printing with time-based progress (%, rate, elapsed, ETA)
  - `train_stage2()`: Added `import time`, sampler build timing, training start/complete banners, per-batch ETA, per-epoch timing summary, enriched return dict
  - Stage 2 config cells (314-320): Updated from v0-conservative to v1/v2-aggressive parameters

- `dev/moe/moe_flashattn_5_1.ipynb` — 105 lines changed (cell 89: `train_stage2`, cell 92: `CodeBalancedBatchSampler`)
  - Identical progress tracking changes as `moe_flashattn_5.ipynb`

### 4.2 Configuration Updates
- `Stage2Config` in experiment cells:
  - `learning_rate`: 5e-5 → 0.005 (100x increase, matching v1/v2 proven config)
  - `epochs`: 3 → 20 (v1/v2 needed 20 epochs for convergence)
  - `warmup_fraction`: 0.1 → 0.05 (matching v2 shorter warmup)
  - `use_focused_loss`: already True (CRITICAL for preventing gradient dilution)
  - `EXPERIMENT_ROUND`: `exp_round10_stage2_only_11M` → `exp_round10_stage2_aggressive_11M`

## 5. Discussions & Reasoning

### Topic: Why `train_stage2` appeared to hang
**Question**: User ran `train_stage2` and it "took forever" with no visible output.
**Analysis**:
1. `train_stage2` calls `CodeBalancedBatchSampler.__init__()` which calls `_build_code_index()`
2. `_build_code_index` iterates over ALL samples to build a code→patient mapping
3. On 11M data, this pure-Python loop takes 30-60+ minutes
4. Old code only printed every 500K samples — ~2-3 minute gaps between prints
5. Before the first print (samples 0-499K), there was complete silence
6. Additionally, no timing existed for the sampler build phase or training batches

**Conclusion**: The function was working correctly but lacked observability. The fix adds time-based progress reporting at every phase: indexing, sampler build, batch training, epoch summary, and total completion.

**Citations**: `8078:8115:dev/moe/moe_flashattn_5.ipynb` (old `_build_code_index`), `expe_logs/exp_round9/exp2b_256dim_v2/training.log` (real epoch timing reference)

### Topic: Expected timing for 11M Stage 2
**Analysis**: From `exp_round9` training logs on 1.5M data, each Stage 2 epoch took ~11.5-12 minutes. The `CodeBalancedBatchSampler` generates `(active_codes // codes_per_batch) * 3` batches per epoch, which depends on active code count (not dataset size). With ~6,200 active codes and `codes_per_batch=16`, that's ~1,161 batches/epoch. The number of batches stays roughly the same for 11M data since it depends on code count, but each batch's forward/backward pass processes 128 samples through a larger memory footprint. Expected: ~15-25 min/epoch on 11M (batch processing overhead increases but batch count stays similar).

## 6. Verification & Quality Checks

**Automated Verification**: Python script checked all 10 critical markers in both notebooks:
- `import time as _time`, `t_sampler_start`, `t_sampler_elapsed`, `epoch_times`, `t_train_start`, `t_epoch_start`, `ETA epoch`, `ETA total`, `TRAINING COMPLETE`, `total_time_sec` — all present in both files.

**Manual Validation**: Git diff reviewed (196 insertions, 43 deletions across 2 files). Changes are symmetric between notebooks.

## 7. Blockers & Issues

**Resolved**: Apparent training hang → root cause: CPU-bound indexing with insufficient progress output. Fixed with comprehensive time-based reporting.

**Outstanding**: None blocking. The actual Stage 2 training on 11M data has not yet been run with the new progress tracking — user will run it in their next GPU session.

## 8. Next Session Plan

**Immediate Priorities** (ranked):
1. **Run Stage 2 with aggressive config on 11M data** — Now with full progress visibility. Expected total time: ~5-8 hours (indexing + 20 epochs).
2. **Monitor training trajectory** — Compare loss curve and logit diagnostics against exp_round9 v1/v2 baselines. Key metrics: S2 loss convergence, tail_pos_logit_mean, tail margin.
3. **Post-Stage 2 evaluation** — Run `compute_stage2_diagnostics()` and `evaluate()` to measure impact on tail code performance.

**Open Questions**:
- How will the 11M encoder (stronger than 1.5M) interact with aggressive Stage 2? The encoder representations may be more robust, potentially requiring fewer epochs or different LR.
- If tail_top10_acc remains at 0% despite positive tail margin (as happened in exp_round9 v2), consider Option B (embedding surgery) or Option C (co-occurrence embedding + fine-tune) as follow-ups.

---
**Session Duration**: ~45 minutes
**Files Modified**: 2 (`moe_flashattn_5.ipynb`, `moe_flashattn_5_1.ipynb`)
**Commits**: 0 (changes unstaged)
**Environment**: macOS, Cursor IDE, Python notebooks targeting GPU execution
