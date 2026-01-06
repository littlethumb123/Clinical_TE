## thoughts to finetune the trnasformer to mimick the legacy finetune results
1/4/2026

Got it. Plan: I’ll (1) inspect `dev/moe/moe_flashattn_3.py` and `dev/moe/moe_flashattn_3_core.py` to see how each model is built/trained and where the “head” lives, (2) map how your current downstream script uses embeddings + external ML (CatBoost/LR), and then (3) outline the minimal code changes needed to add **true end-to-end fine-tuning** (all weights) for exp1/exp2b/exp6 using the same dataset/target logic you already use for downstream (`ip6`).

[3 tools called]

Nice—your current `moe_flashattn_3_downstream.py` is purely **embedding extraction + sklearn/catboost**, so to “mimic the old method” (end-to-end fine-tune) we’ll need to hook into the training code in `moe_flashattn_3.py`/`_core.py` where the PyTorch optimizer/loops already exist. Next I’ll pinpoint: where the model’s output head is defined, and whether there’s already a downstream fine-tune path or we need to add one.

[3 tools called]

Aha: `moe_flashattn_3.py` already contains a full pretraining training loop (multi-label daily target prediction) and even explicitly references `min_transformer_finetune.py`. Next I’ll extract the exact model API (what `forward()` returns, how loss is computed), then show precisely how to swap the head + loss for `ip6` and run end-to-end fine-tuning for exp1/exp2b/exp6.

[12 tools called]

### What you have today (based on the code)
- **`moe_flashattn_3_downstream.py`**: loads a pretrained checkpoint, **extracts patient embeddings** (last valid day via `dt_cnt`), then trains **external ML models** (CatBoost/LR/etc). That is **not** end-to-end fine-tuning.
- **`moe_flashattn_3.py`**: contains the real multi-GPU training loop (`train_epoch`) but it is wired for **pretraining** (daily multi-label target codes) via `DataParallelWrapper`, which expects `target_multihot` and computes BCE loss inside `forward()`.

Key blocker for “full-weight IP fine-tune”: your existing `EmbeddingExtractor` **explicitly `.detach()`’s** the captured embeddings, so it cannot be used for training end-to-end as-is.

---

### What “exactly same full-weights finetuning” should mean here
To mimic `min_transformer_finetune.py`’s *pattern* (replace head + update all weights), for **each** pretrained model (exp1 dense, exp2b flash, exp6 MoE):
1. **Load pretrained checkpoint**
2. **Replace the head** (your models expose `decoder_cd` in all 3 variants)
3. **Ensure all parameters are trainable** (`requires_grad=True`), i.e. full fine-tune
4. Train on a downstream label (`ip6`) using a patient-level prediction (recommended: last valid day embedding), and evaluate with the same metrics you already compute.

Important difference you should be aware of (no guessing):  
Your current commercial downstream pipeline uses a **member-level** label `ip6`. The legacy fine-tune script’s `ip_6m` appears to be a **per-day sequence label string** (`'*'`-separated) and trains on all valid days. Unless you have an equivalent per-day target column, the closest “apples-to-apples” with your current setup is **patient-level (last valid day) fine-tuning**.

---

### Minimal, code-consistent way to implement full fine-tuning (exp1/exp2b/exp6)

#### 1) Build a fine-tune dataset that includes raw transformer inputs + `ip6`
Because fine-tuning needs raw inputs (`age_in_months`, `gender_cd`, `lob`, `cd`, `dt_cnt`), use the **transformer input table** + join to an outcomes table that contains `ip6` by `(individual_id, index_dt)` (same keys you already use for embedding joins).

Then split using the exact same logic you already have in downstream code:
- in-time train/val/test by `ind_id_last_digit`
- OOT by `index_dt > OOT_CUTOFF_DATE`

#### 2) Add a trainable patient-embedding hook (same idea as `EmbeddingExtractor`, but without `.detach()`)
This wrapper captures the “pre-decoder” sequence embeddings and applies a binary head on the **last valid day**.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrainablePatientHead(nn.Module):
    """
    End-to-end finetune wrapper:
    - runs base transformer forward (dense/flash/moe)
    - captures pre-decoder temporal embeddings via a forward hook
    - selects last valid day using dt_cnt
    - applies a binary classifier head
    """
    def __init__(self, base_model: nn.Module, embedding_size: int, is_baseline: bool, use_moe_losses: bool = False, moe_config=None):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(embedding_size, 1)  # binary logit
        self._last_seq_emb = None
        self.use_moe_losses = use_moe_losses
        self.moe_config = moe_config

        # register hook to capture embeddings BEFORE decoder
        if is_baseline:
            target_layer = self.base_model.transformer_encoder_dy
            def hook_fn(module, inputs, output):
                # output: [len_dy, batch, d] -> store, keep grad
                self._last_seq_emb = output
            target_layer.register_forward_hook(hook_fn)
        else:
            target_layer = self.base_model.norm
            def hook_fn(module, inputs, output):
                # inputs[0]: [batch, len_dy, d] -> store, keep grad
                self._last_seq_emb = inputs[0]
            target_layer.register_forward_hook(hook_fn)

    def forward(self, x: torch.Tensor, dt_cnt: torch.Tensor, y: torch.Tensor | None = None):
        self._last_seq_emb = None

        moe_losses = {}
        if self.use_moe_losses:
            _ = self.base_model(x, return_moe_losses=True)  # returns (logits, moe_losses)
            _, moe_losses = _
        else:
            _ = self.base_model(x)

        if self._last_seq_emb is None:
            raise RuntimeError("Embedding hook did not fire; check hook registration / model type.")

        seq_emb = self._last_seq_emb
        # normalize to [batch, len_dy, d]
        if seq_emb.dim() == 3 and seq_emb.shape[0] == x.shape[1]:  # baseline: [len_dy, batch, d]
            seq_emb = seq_emb.permute(1, 0, 2)

        # last valid day embedding
        # dt_cnt: [batch], each value in [1..len_dy]
        idx = torch.clamp(dt_cnt.long() - 1, min=0, max=seq_emb.shape[1] - 1)
        patient_emb = seq_emb[torch.arange(seq_emb.shape[0], device=seq_emb.device), idx]  # [batch, d]

        logit = self.classifier(patient_emb).squeeze(-1)  # [batch]
        if y is None:
            return logit

        # BCE loss (y should be 0/1 float)
        pred_loss = F.binary_cross_entropy_with_logits(logit, y.float())

        # Optional: include MoE stability losses (mirrors your pretraining wrapper pattern)
        if self.moe_config is not None and moe_losses:
            z_loss = moe_losses.get("z_loss", torch.tensor(0.0, device=logit.device))
            aux_loss = moe_losses.get("aux_loss", torch.tensor(0.0, device=logit.device))
            total = pred_loss + self.moe_config.z_loss_weight * z_loss
            if self.moe_config.load_balance_strategy == "switch":
                total = total + self.moe_config.aux_loss_weight * aux_loss
            return total, {"pred_loss": pred_loss, "logit": logit, "moe_losses": moe_losses}

        return pred_loss, {"logit": logit}
```

#### 3) Load each experiment checkpoint and fine-tune all weights
- **exp1**: baseline dense
- **exp2b**: flash attention
- **exp6_v3**: flash + MoE

Use your existing checkpoint loader (you already have two variants: one in `moe_flashattn_3_downstream.py` that infers MoE `d_ff`, and one in `moe_flashattn_3_core.py`). Either is fine; the key is:
- replace head by using the wrapper above (no need to mutate `decoder_cd` at all)
- ensure full unfreeze:

```python
for p in model.parameters():
    p.requires_grad = True
```

Then train with:
- AdamW (or SGD) like your training utilities already do
- AMP (`torch.cuda.amp.autocast` + GradScaler) for flash / MoE variants (consistent with your code)

#### 4) Evaluate exactly like your current downstream metrics
In eval:
- compute `prob = sigmoid(logit)`
- feed to your existing `compute_split_metrics()` (AUC ROC, AUC PR, lift@1%, etc.)

This gives you exactly what you want to compare:
1) **Does exp1 fine-tuned end-to-end match the legacy baseline performance?**  
2) **Do exp2b / exp6 beat exp1 when all are end-to-end fine-tuned with the same splits and metrics?**

---

### What I need you to confirm / check (so we don’t guess)
- **Which table actually contains the raw transformer inputs for commercial IP fine-tuning?**  
  Your current eval table is `...commercial_ip_heldout_transformer_matched_final_dataset...` (great for CatBoost because it has `ip6`), but for end-to-end finetune you must have the string columns `age_in_months`, `gender_cd`, `lob`, `cd`, plus `dt_cnt`.
- If that matched table does *not* have them, you’ll need to **join** it with the transformer-input table (`...commercial_heldout_transformer_input_...`) to get those columns.

If you switch to **Agent mode**, I can implement this cleanly (a small new fine-tune module + a runnable driver that loops exp1/exp2b/exp6, trains, and prints the same metric table you already produce).