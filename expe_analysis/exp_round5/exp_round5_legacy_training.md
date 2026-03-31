## 1) `exp1_dbcheck/epoch2` — plateaus?
- March 17, 2026
### Background
- notebook: `dev/legacy/legacy_full_training.ipynb`
- log directory: `expe_logs/exp_round5/exp1_dbcheck/`
- Why: double check if the previous exp1 is correct and also the previous TE really works (as baseline)

### A. Validation metric used for early stopping (`micro_recall@20`)

From `epoch2/early_stopping_summary.json`:

| Step | `micro_recall@20` |
|------|-------------------|
| 500  | 0.3602 |
| 1000 | 0.3671 |
| 1500 | 0.3748 |
| 2000 | **0.3788** (logged as best in `training.log`) |
| 2500 | 0.3788 |
| 3000 | 0.3788 |

Same story in `training.log`: new bests at 500 → 1000 → 1500 → 2000; at 2500 and 3000 the log shows **no new best** (`patience=1/5`, `patience=2/5`) with `micro_recall@20=0.3788` still tied to step 2000.

**Conclusion (evidence):** On this **sparse validation schedule** (every 500 optimizer steps), **`micro_recall@20` is effectively flat from step 2000 through 3000** (increments on the order of **~10⁻⁴** in JSON: 0.3787619 → 0.3788212 → 0.3788240). That is a **validation plateau on the early-stopping metric** over the last 1000 steps checked.

`stopped_early` is **false**; `checks_without_improvement` is **2** (below patience 5).

### B. End-of-epoch validation vs mid-training checkpoints

`epoch_metrics.json` (full epoch) reports **`micro_recall@20`: 0.379127** and **`val_loss`: 0.011632**, whereas the last early-stop checkpoint at step 3000 had **`micro_recall@20` ≈ 0.378824**. So the **final epoch evaluation is slightly better** than the step-3000 checkpoint on that metric—only compare like with like (full val vs partial-step val).

### C. Training loss (batch-level, every 100 batches)

Parsed from `epoch2/batch_metrics.json` (494 points, batches 0 … 49300):

| Region | Mean loss (evidence) |
|--------|----------------------|
| First ~5% of samples | ~0.0341 |
| ~25% through epoch | ~0.0254 |
| ~50% | ~0.0197 |
| ~75% | ~0.0163 |
| Last ~5% (batches 47000–49300) | ~0.0141 |
| First logged batch | **0.034863** |
| Last logged batch | **0.013646** |

Chunk means over 50 consecutive log points each: **0.0328 → 0.0285 → … → 0.0151 → 0.0143** (monotonic decrease across all 10 chunks).

**Conclusion (evidence):** **No flat training-loss plateau over the epoch**—loss keeps falling, but the **step size shrinks** (e.g. last chunk mean ~5.6% below the previous chunk vs larger drops earlier). That is **slowing improvement**, not a horizontal train-loss plateau.

---

## 2) `epoch2` dbcheck vs `exp1/` — similar dynamics?

### What the tree actually contains

- **`exp1/`** only has artifacts for **a single epoch** (`bs128_ep1_...` in filenames; log says **“Training for 1 epochs…”**). There is **no epoch-2 run** under `exp1/` to compare to **dbcheck epoch2**.

So:

- **You cannot** claim similarity of **“epoch 2”** between the two folders—**only epoch 1 exists for exp1** in the repo snapshot.

### Different setups (from logs/config)

| Source | Evidence |
|--------|----------|
| **exp1** | `exp1_dense_baseline`, **effective batch 128**, log shows **d_model=256, nhid=1024**, **26,455,961** params, **CosineWarmup** peak **8e-4** |
| **dbcheck** | `legacy_replication`, **effective batch 512**, `epoch1/config.json`: **nhid=512**, **~24.88M** params, **CosineAnnealingLR T_max=1**, **lr=0.01** |

So **numerical loss curves and metrics are not directly comparable** as “same training” — batch size, hidden size, scheduler, and code path differ.

### Where numbers *do* line up (same order of magnitude)

- **Cold start first loss ~0.804:** exp1 log first line **Loss: 0.8045**; dbcheck epoch1 `final_results.json` has **`train_loss_first`: 0.80468** — consistent **random-init** starting scale.

### Where they diverge (first epoch, documented)

| Metric | exp1 `..._results.json` (epoch 1 end) | dbcheck epoch1 `final_results.json` | dbcheck epoch2 `epoch_metrics.json` |
|--------|--------------------------------------|--------------------------------------|-------------------------------------|
| `train_loss_mean` | 0.0645 | 0.191 | 0.0212 |
| `train_loss_first` / cold start | 0.8045 | 0.8047 | 0.0351 (warm) |
| Val `micro_recall@20` | **0.3172** | **0.3433** | **0.3791** |

**Conclusion (evidence):** **First-epoch validation `micro_recall@20` is not the same** (0.317 vs 0.343). **Training loss means differ a lot** (0.064 vs 0.191 for “epoch 1”) because **pipelines and batching differ**; only the **very first batch loss** aligns in scale. **Dynamics are not “the same”** in the sense of matching curves or end metrics; the closest honest statement is **both show large initial loss drop from ~0.8 when starting from scratch** (exp1 and dbcheck epoch1 only).

---

## Summary

1. **Plateau (dbcheck epoch2):** **Yes on validation `micro_recall@20` between steps 2000–3000** per `early_stopping_summary.json` and `training.log`. **No on full-epoch training loss**—it still decreases, but **more slowly** at the end (`batch_metrics.json` chunk means).
2. **vs exp1:** **No epoch2 in exp1**; setups differ; **end metrics and train-loss means differ**; **only cold-start first-batch loss ~0.804** is clearly aligned across exp1 and dbcheck epoch1.

**Next step for your continued run:** Expect **train loss** to keep creeping down if the pattern repeats; **validation on `micro_recall@20`** may **gain little** if you’re already near the step-2000–3000 plateau unless LR/schedule/data order changes—worth monitoring the same JSON fields after the new epoch.