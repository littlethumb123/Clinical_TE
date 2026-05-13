# Data Transformation Visualization: Step-by-Step Architecture Walk-Through

## Purpose

Create a side-by-side visual showing how raw clinical data transforms through each architecture stage into a 256-d embedding. This document provides:
1. **Schematic mockups** you can recreate in PowerPoint/Excalidraw right now (no code needed)
2. **A runnable notebook cell** for GCP Vertex AI to capture real intermediate activations from a trained checkpoint

---

## The Visual Concept: "Data Transformation Strip"

Place this as a **horizontal strip below or beside the architecture diagram**. Each stage shows the data's shape and content as a heatmap/matrix, getting progressively compressed. The audience watches 80 sparse codes become 1 dense embedding.

```
STAGE 1          STAGE 2           STAGE 3         STAGE 4            STAGE 5          STAGE 6
Raw Codes        Embedding         After LAP       After Demo +       After 6          Final
(one day)        Lookup            (one day)       All Days Stacked   Temporal Layers  Embedding

┌──────────┐    ┌──────────┐     ┌──────────┐    ┌──────────┐      ┌──────────┐     ┌────┐
│ E11.9    │    │▓▓▒░▓▒▓▒▓▒│     │          │    │▓▓▒░▓▒▓▒▓▒│      │▓▓▒▓▓▒▓▒▓▒│     │▓▒▓▒│
│ I10      │    │▒▓░▓▒▓▒▓░▓│     │▓▓▒▓▓▒▓▒▓▒│    │▒▓░▓▒▓▒▓░▓│      │▒▓▒▓▒▓▒▓▒▓│     │▒▓▒▓│
│ Z79.4    │    │░▒▓▒░▓▒░▓▒│     │          │    │░▒▓▒░▓▒░▓▒│      │▓▒▓▒▓▒▓▒▓▒│     │▓▒▓▒│
│ 99213    │    │▓░▒▓▓▒░▓▒▓│     │ 1 vector │    │▓░▒▓▓▒░▓▒▓│      │▒▓▒▓▒▓▒▓▒▓│     │▒▓▒▓│
│ (pad)    │    │░░░░░░░░░░│     │          │    │          │      │          │     │    │
│ (pad)    │    │░░░░░░░░░░│     └──────────┘    │   ...    │      │   ...    │     │    │
│ ...      │    │   ...    │     ← 80 rows       │          │      │          │     │ 1  │
│ (pad)    │    │░░░░░░░░░░│       collapse       │▓▒▓▒▓▒▓▒▓▒│      │▓▒▓▒▓▒▓▒▓▒│     │vec │
└──────────┘    └──────────┘       to 1           └──────────┘      └──────────┘     └────┘

 5 codes         [80, 256]         [1, 256]        [200, 256]        [200, 256]       [256]
 + padding       sparse rows       dense           200 days          richer           member
                 (pads=zero)       weighted sum    of daily vecs     patterns         embedding
```

---

## Stage-by-Stage Mockup Specifications

### Stage 1: Raw Clinical Codes (One Day)

**What to show**: A simple text list — the actual codes a member had on one day.

```
┌─────────────────────────────┐
│  Day 47 — Member 12345      │
│                             │
│  E11.9   Type 2 Diabetes    │
│  I10     Hypertension       │
│  Z79.4   Insulin Rx         │
│  99213   Office Visit       │
│  Z23     Immunization       │
│                             │
│  (75 positions padded)      │
│                             │
│  5 real codes / 80 slots    │
└─────────────────────────────┘
```

**PowerPoint tip**: Use a simple table or text box. Color the real codes in blue/teal, padding rows in light gray. This is the most human-readable stage — the audience should recognize actual clinical codes.

---

### Stage 2: After Embedding Lookup — [80, 256] Matrix

**What to show**: A heatmap matrix. 80 rows (one per code slot), 256 columns (embedding dimensions). The 5 real codes have distinct color patterns; the 75 padding rows are uniform gray/zero.

```
                    ← 256 embedding dimensions →
              ┌──────────────────────────────────────┐
   E11.9  → 1 │▓▓▒░▓▒▓▒▓▒▓▓▒░▓▒▓▒▓▒▓▓▒░▓▒▓▒▓▒▓▓▒░▓│  distinct pattern
   I10    → 2 │▒▓░▓▒▓▒▓░▓▒▓░▓▒▓▒▓░▓▒▓░▓▒▓▒▓░▓▒▓░▓▒│  distinct pattern
   Z79.4  → 3 │░▒▓▒░▓▒░▓▒░▒▓▒░▓▒░▓▒░▒▓▒░▓▒░▓▒░▒▓▒░│  distinct pattern
   99213  → 4 │▓░▒▓▓▒░▓▒▓▓░▒▓▓▒░▓▒▓▓░▒▓▓▒░▓▒▓▓░▒▓▓│  distinct pattern
   Z23    → 5 │▒░▓▒▒▓░▒▓▒▒░▓▒▒▓░▒▓▒▒░▓▒▒▓░▒▓▒▒░▓▒▒│  distinct pattern
  (pad)   → 6 │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  zero (gray)
   ...        │         ...                          │
  (pad)  → 80 │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  zero (gray)
              └──────────────────────────────────────┘
```

**PowerPoint tip**: Create a colored rectangle grid (5 rows with varied colors, 75 rows in light gray). Annotate: "Each code becomes a 256-d vector. Padding rows are zeros. The model must learn which codes matter."

**Key visual point**: The rows look very different from each other — each code has its own learned embedding pattern. But most of the matrix is empty (padding).

---

### Stage 3: After Learned Attention Pooling — [1, 256] Vector

**What to show**: The 80 rows collapse into a single row. Show attention weights as a side annotation.

```
   Attention Weights              Weighted Sum
   (which codes matter?)          (one vector per day)

   E11.9  ████████░░  0.35       ┌──────────────────────────────────────┐
   I10    ██████░░░░  0.28    →  │▓▓▒▓▒▓▒▓▒▓▓▒▓▒▓▒▓▒▓▓▒▓▒▓▒▓▒▓▓▒▓▒▓▒│  [1, 256]
   Z79.4  █████░░░░░  0.22       └──────────────────────────────────────┘
   99213  ███░░░░░░░  0.10        dense — every dimension active
   Z23    █░░░░░░░░░  0.05
   (pads) ░░░░░░░░░░  0.00
```

**PowerPoint tip**: Draw a horizontal bar chart of attention weights on the left, with an arrow pointing to the single output vector on the right. The bar chart tells a clinical story: "The model decided diabetes and hypertension were the most important signals on this day."

**Key visual point**: 80 rows → 1 row. Massive compression, but it's a *learned* weighted sum, not a hard max. The output vector is dense (no zero regions) because it blends information from all codes.

---

### Stage 4: After Demographic Injection — [200, 256] Matrix (All Days)

**What to show**: Stack 200 days of pooled vectors. Each row is one day. Add demographic shading.

```
                    ← 256 dimensions →
              ┌──────────────────────────────────────┐
   Day 1    1 │▒▓▒▓▒░▓▒▓▒▓▒▓▒░▓▒▓▒▓▒▓▒▓▒░▓▒▓▒▓▒▓▒▓│  + age=456mo, M, Commercial
   Day 2    2 │▓▒░▓▒▓▒▓░▓▒▓▒░▓▒▓▒▓░▓▒▓▒░▓▒▓▒▓░▓▒▓▒│  + age=456mo, M, Commercial
   Day 3    3 │░▓▒▓░▒▓▒▓░▓▒▓░▒▓▒▓░▓▒▓░▒▓▒▓░▓▒▓░▒▓▒│  + age=456mo, M, Commercial
   ...        │                                      │
   Day 47   . │▓▓▒▓▒▓▒▓▒▓▓▒▓▒▓▒▓▒▓▓▒▓▒▓▒▓▒▓▓▒▓▒▓▒│  ← the day from Stage 1-3
   ...        │                                      │
   Day 147  . │▒░▓▒▒▓░▒▓▒▒░▓▒▒▓░▒▓▒▒░▓▒▒▓░▒▓▒▒░▓▒│
   (pad)    . │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
   ...        │         ...                          │
   (pad)  200 │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
              └──────────────────────────────────────┘

              Input to the Temporal Encoder
              Each row = one day's clinical summary + demographics
              Gray rows = padding (member had fewer than 200 days)
```

**PowerPoint tip**: Use a tall, narrow heatmap. Mark "Day 47" with a highlight to connect back to Stages 1-3. Add a right-side annotation showing the causal mask: "Day t sees only rows 1...t".

**Key visual point**: The temporal encoder's input. Each row is independently encoded — no cross-day information yet. The temporal encoder's job is to make each row aware of all preceding rows.

---

### Stage 5: After 6 Temporal Layers — [200, 256] Matrix (Enriched)

**What to show**: Same shape as Stage 4, but the patterns are smoother, more structured, with visible correlations between nearby rows.

```
                    ← 256 dimensions →
              ┌──────────────────────────────────────┐
   Day 1    1 │▒▓▒▓▒▓▓▒▓▒▓▒▓▒▓▓▒▓▒▓▒▓▒▓▒▓▓▒▓▒▓▒▓▒▓│
   Day 2    2 │▒▓▒▓▒▓▓▒▓▒▓▒▓▒▓▓▒▓▒▓▒▓▒▓▒▓▓▒▓▒▓▒▓▒▓│  ← similar to day 1
   Day 3    3 │▒▓▒▓▒░▓▒▓▒▓▒▓▒░▓▒▓▒▓▒▓▒▓▒░▓▒▓▒▓▒▓▒▓│     (temporal smoothing)
   ...        │                                      │
   Day 47   . │▓▓▒▓▒▓▒▓▒▓▒▓▒▓▒▓▓▓▒▓▒▓▒▓▒▓▒▓▒▓▒▓▓▒│
   ...        │                                      │
   Day 147  . │▓▒▓▒▓▒▓▒▓▒▓▒▓▒▓▓▒▓▒▓▒▓▒▓▒▓▓▒▓▒▓▒▓▒│
   (pad)    . │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
              └──────────────────────────────────────┘

              After 6 layers of causal attention + SwiGLU
              Each row now encodes its ENTIRE preceding history
              Nearby rows show correlated patterns (temporal continuity)
              Later rows are richer (more history to draw from)
```

**Visual contrast with Stage 4**: Side-by-side, Stage 4 rows look independent and noisy. Stage 5 rows look smoother, with visible vertical "stripes" (shared features across time) and gradual transitions. This difference IS what the temporal encoder learned.

**PowerPoint tip**: If using real data (from the GCP notebook below), the contrast is striking. If using mockups, make Stage 4 rows visually random/independent and Stage 5 rows with visible vertical coherence (adjacent rows share similar color patterns). Add annotation: "Each row now 'knows' all preceding days — 6 layers of causal attention."

---

### Stage 6: Final Member Embedding — [256] Vector

**What to show**: Extract the last valid row. One horizontal bar.

```
              ┌──────────────────────────────────────┐
   Day 147  → │▓▒▓▒▓▒▓▒▓▒▓▒▓▒▓▓▒▓▒▓▒▓▒▓▒▓▓▒▓▒▓▒▓▒│  [256]
              └──────────────────────────────────────┘

              The member's complete clinical trajectory
              compressed into 256 numbers.

              This vector:
              • Replaces 533 hand-engineered features
              • Achieves 0.810 AUC on IP risk (vs 0.838 production)
              • +110% Lift@1% over PCA(256) baseline
```

**PowerPoint tip**: Draw an arrow from the last valid row in Stage 5 to this isolated vector. Add the downstream performance stats below it. This is the payoff moment — "all that machinery produces this one vector, and it's powerful."

---

## Recommended Slide Layout: "The Data Transformation Strip"

### Option A: Horizontal strip below the architecture diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                    ARCHITECTURE DIAGRAM (top 60%)                            │
│                    [existing bottom-up diagram]                              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  "How Data Transforms" (bottom 40%)                                         │
│                                                                             │
│  Stage 1    →    Stage 2     →    Stage 3    →   Stage 4    → Stage 5 → S6  │
│  ┌──────┐     ┌──────────┐    ┌──────────┐   ┌────────┐   ┌────────┐ ┌──┐ │
│  │E11.9 │     │▓▓▒░▓▒▓▒▓▒│    │          │   │▓▓▒░▓▒▓▒│   │▓▓▒▓▓▒▓▒│ │▓▒│ │
│  │I10   │     │▒▓░▓▒▓▒▓░▓│    │▓▓▒▓▓▒▓▒▓▒│   │▒▓░▓▒▓▒▓│   │▒▓▒▓▒▓▒▓│ │▒▓│ │
│  │Z79.4 │     │░▒▓▒░▓▒░▓▒│    │          │   │░▒▓▒░▓▒░│   │▓▒▓▒▓▒▓▒│ │▓▒│ │
│  │99213 │     │▓░▒▓▓▒░▓▒▓│    └──────────┘   │  ...   │   │  ...   │ │  │ │
│  │(pad) │     │░░░░░░░░░░│                    │▓▒▓▒▓▒▓▒│   │▓▒▓▒▓▒▓▒│ │  │ │
│  └──────┘     └──────────┘                    └────────┘   └────────┘ └──┘ │
│  5 codes       [80, 256]       [1, 256]       [200, 256]   [200, 256] [256]│
│  "what          "each code      "pool into     "stack all   "temporal   "one│
│   happened"     has a vector"    one vector"    200 days"    context"   vec"│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Option B: Animated build (recommended for live presentation)

Build the strip left-to-right as you narrate each architecture section. Each click reveals the next stage. This creates a "watch the data transform" effect.

| Click | What appears | You say |
|-------|-------------|---------|
| 1 | Stage 1 (raw codes) | "We start with raw clinical codes — here's one day for one member" |
| 2 | Stage 2 (embedding matrix) | "Each code becomes a 256-d vector. Padding rows are zeros." |
| 3 | Stage 3 (LAP output) | "Learned Attention Pooling compresses 80 rows into 1 — weighted by what matters clinically" |
| 4 | Stage 4 (200-day stack) | "Repeat for all 200 days, add demographics — this is the temporal encoder's input" |
| 5 | Stage 5 (enriched matrix) | "After 6 layers of causal attention, each row knows its full history — notice the smoother patterns" |
| 6 | Stage 6 (final vector) | "Extract the last valid row — 256 numbers encoding the member's entire clinical trajectory" |

---

## GCP Notebook: Capture Real Intermediate Activations

Run this on your Vertex AI environment where the trained checkpoint is available. It hooks into the model's forward pass and captures tensors at each stage.

```python
# ============================================================
# Visualization: Data Transformation Through Architecture
# Run on GCP Vertex AI with a trained exp2b checkpoint
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

# --- 1. Load model and get a single member's data ---
# Adjust paths to your checkpoint and data
checkpoint_path = "path/to/exp2b_round10_best.pt"
# Load model (use your existing loading code from the training notebook)
# model = load_model(checkpoint_path)
# model.eval()

# Get one member from the dataset
# sample = dataset[0]  # or pick a specific member
# cd_tensor = sample['cd'].unsqueeze(0)  # [1, 200, 80]
# dt_cnt = sample['dt_cnt']

# --- 2. Hook-based activation capture ---
activations = {}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            activations[name] = output[0].detach().cpu()
        else:
            activations[name] = output.detach().cpu()
    return hook

# Register hooks at key points
# Adjust attribute names to match your model's structure
hooks = []

# After embedding lookup (before pooling)
hooks.append(model.code_embedding.register_forward_hook(make_hook('stage2_embeddings')))

# After learned attention pooling
if hasattr(model, 'daily_pooling'):
    hooks.append(model.daily_pooling.register_forward_hook(make_hook('stage3_after_lap')))

# After demographic injection (before temporal encoder)
# This requires hooking the right point — typically after fuse_embedding
# hooks.append(model.fuse_embedding.register_forward_hook(make_hook('stage4_after_demo')))

# After each temporal layer
for i, layer in enumerate(model.temporal_layers):
    hooks.append(layer.register_forward_hook(make_hook(f'stage5_layer_{i}')))

# --- 3. Forward pass ---
with torch.no_grad():
    # output = model(cd_tensor, ...)
    pass

# Remove hooks
for h in hooks:
    h.remove()

# --- 4. Visualization ---
fig = plt.figure(figsize=(24, 10))
gs = gridspec.GridSpec(2, 6, height_ratios=[1, 3], hspace=0.3, wspace=0.3)

# Helper
def plot_heatmap(ax, data, title, ylabel='', xlabel='256 dims', aspect='auto', cmap='RdBu_r'):
    if data.dim() == 3:
        data = data[0]  # remove batch dim
    d = data.numpy()
    norm = TwoSlopeNorm(vmin=d.min(), vcenter=0, vmax=d.max())
    ax.imshow(d, aspect=aspect, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=6)

# Stage 2: Embedding lookup [80, 256] — show first day
ax1 = fig.add_subplot(gs[:, 0])
day_idx = 0  # or pick a day with many codes
emb_data = activations['stage2_embeddings'][0, day_idx]  # [80, 256]
plot_heatmap(ax1, emb_data, f'Stage 2: Embeddings\n[80, 256] (Day {day_idx})', ylabel='80 code slots')

# Stage 3: After LAP [1, 256]
ax2 = fig.add_subplot(gs[0, 1])
lap_data = activations['stage3_after_lap'][0, day_idx:day_idx+1]  # [1, 256]
plot_heatmap(ax2, lap_data, 'Stage 3: After LAP\n[1, 256]', ylabel='1 day')

# Stage 4: All days stacked [200, 256]
ax3 = fig.add_subplot(gs[:, 2])
# Use stage3 data for all days if available, otherwise stage4
demo_data = activations.get('stage4_after_demo', activations['stage3_after_lap'])
plot_heatmap(ax3, demo_data[0], 'Stage 4: All Days\n[200, 256]', ylabel='200 days')

# Stage 5a: After Layer 0
ax4 = fig.add_subplot(gs[:, 3])
plot_heatmap(ax4, activations['stage5_layer_0'][0], 'Stage 5a: After Layer 0\n[200, 256]', ylabel='200 days')

# Stage 5b: After Layer 5 (final)
ax5 = fig.add_subplot(gs[:, 4])
plot_heatmap(ax5, activations['stage5_layer_5'][0], 'Stage 5b: After Layer 5\n[200, 256]', ylabel='200 days')

# Stage 6: Final embedding [256]
ax6 = fig.add_subplot(gs[:, 5])
# Extract last valid day
last_day = dt_cnt - 1  # adjust based on your data
final_emb = activations['stage5_layer_5'][0, last_day:last_day+1]  # [1, 256]
plot_heatmap(ax6, final_emb, f'Stage 6: Member Embedding\n[1, 256] (Day {last_day})', ylabel='')

# Add arrows between stages
for i in range(5):
    fig.text(0.1 + i * 0.155, 0.5, '→', fontsize=20, ha='center', va='center',
             fontweight='bold', color='#333333')

plt.suptitle('Data Transformation Through Clinical TE Architecture', fontsize=14, fontweight='bold', y=0.98)
plt.savefig('data_transformation_strip.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.show()

print("Saved: data_transformation_strip.png")
```

### What to look for in the real visualizations

| Stage | Expected visual pattern |
|-------|------------------------|
| Stage 2 (embeddings) | 5-10 distinct colored rows at top, rest is flat gray (padding zeros) |
| Stage 3 (after LAP) | Single dense row — no zero regions, blended pattern |
| Stage 4 (200 days) | Each row independent — no vertical correlation between adjacent rows |
| Stage 5 (after L5) | Vertical stripes emerge — shared features across time; nearby rows look similar; gradient from top (sparse context) to bottom (rich context) |
| Stage 6 (final) | One dense vector — the most "processed" looking, smoothest activation pattern |

### The Stage 4 → Stage 5 contrast is the star visual

This is the most impactful comparison. Stage 4 looks like "random rows stacked." Stage 5 looks like "a structured, coherent representation." The difference IS what 6 layers of causal attention learned — temporal patterns, disease progressions, comorbidity relationships.

If you can only show two stages side-by-side, show these two.

---

## Fallback: PowerPoint Mockup Without Real Data

If you can't run the notebook before the presentation, create a schematic version:

### Color scheme
- **Blue/teal tones**: Active, information-carrying values
- **Light gray**: Zero/padding
- **Red accents**: Attention weights or highlighted features

### Creating the heatmaps in PowerPoint
1. **Stage 2**: Create a table with 80 rows × ~20 visible columns. Fill top 5 rows with varied blue/teal cell colors. Fill remaining rows with light gray.
2. **Stage 3**: Single row table, all cells filled with blended blue/teal (no gray).
3. **Stage 4**: 200-row × 20-column table. Each row has different blue/teal patterns. No vertical coherence — rows look independent.
4. **Stage 5**: Same dimensions, but add vertical color continuity — adjacent rows share similar colors. Add subtle vertical "stripes."
5. **Stage 6**: Single row extracted from Stage 5 bottom, with a highlight border.

### Critical annotations to add
- **Between Stage 2→3**: "Learned Attention Pooling: 80 codes → 1 vector"
- **Between Stage 3→4**: "Repeat for 200 days + demographics"
- **Between Stage 4→5**: "6 layers of causal attention — each day learns from its full history"
- **At Stage 6**: Arrow pointing to "AUC 0.810 | +110% over PCA"

---

## Alternative Visualization: The "Compression Funnel"

If the horizontal strip doesn't fit, use a vertical funnel that matches the architecture diagram's vertical flow:

```
        75,516 codes per member
        ┌────────────────────────────────────┐
        │  Day 1: E11.9, I10, Z79.4, 99213  │
        │  Day 2: J06.9, Z23               │
        │  ...                              │
        │  Day 147: Z00.00                  │
        └────────────────────────────────────┘
                        ↓ embedding lookup
        ┌────────────────────────────────────┐
        │        [200 × 80 × 256]            │
        │        3D tensor of embeddings     │
        └────────────────────────────────────┘
                        ↓ LAP (80→1 per day)
              ┌──────────────────────┐
              │    [200 × 256]       │
              │    daily summaries   │
              └──────────────────────┘
                        ↓ + demographics
              ┌──────────────────────┐
              │    [200 × 256]       │
              │    context-ready     │
              └──────────────────────┘
                        ↓ 6 temporal layers
              ┌──────────────────────┐
              │    [200 × 256]       │
              │    history-aware     │
              └──────────────────────┘
                        ↓ extract last day
                  ┌────────────┐
                  │   [256]    │
                  │  embedding │
                  └────────────┘

    Compression: 200 × 80 × 75,516 one-hot → 256 floats
                 = 1.2 billion → 256 (4.7 million : 1 compression)
```

**Key stat to highlight**: The model compresses ~1.2 billion possible input combinations into 256 numbers — a 4.7 million to 1 compression ratio. This is a visceral way to communicate what the embedding does.
