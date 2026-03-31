# Data Information Saturation Analysis — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Quantify whether useful learning signal (information content, diversity, co-occurrence richness) saturates or degrades as (a) member history length increases and (b) dataset size scales from 1.5M → 11M — stratified by frequency tier, age group, and line of business.

**Architecture:** A standalone analysis notebook (`dev/downstream/data_information_saturation_analysis.ipynb`) that loads raw data from BigQuery at multiple scales, computes information-theoretic and distributional metrics at the member-level (within-member temporal saturation) and population-level (cross-member scaling saturation), then produces a structured JSON results file and a markdown report under `downstream_eval/te_training_information_saturation`.

**Tech Stack:** Python 3.10+, pandas, numpy, scipy (stats, spatial, sparse), google-cloud-bigquery, matplotlib, seaborn. All CPU-only, no GPU required.

---

## Preamble: Critical Review of the Original Proposal (Lines 882–954)

### What the original proposal gets right

1. **Bigram counting** (Test 1a) is a valid first-order proxy for temporal transition diversity.
2. **Co-occurrence entropy** (Test 1b) correctly captures distributional richness beyond frequency.
3. **Per-tier coverage at threshold K** (Test 1c) is well-targeted at the core question.
4. **Pre-registered interpretations** are well-structured with clear decision boundaries.

### What the original proposal gets wrong or misses

| Gap | Problem | Fix in this plan |
|-----|---------|-----------------|
| **No within-member analysis** | All 3 tests measure *population*-level metrics. The user's question (a) asks: "does information decrease over time *within a member's history*?" — which requires per-day-position analysis of code novelty within each patient | Task 2: Within-Member Temporal Saturation |
| **Bigrams are too coarse** | Bigrams count (code_A on day t, code_B on day t+1). But the TE model processes *all codes on a day simultaneously* via attention. Same-day co-occurrence and higher-order patterns (trigrams, skip-grams) matter more than adjacent-day bigrams | Task 3: Co-occurrence analysis uses same-day pairs and temporal skip-grams |
| **Clustering is computationally intractable** | Test 1d proposes k-means on patient trajectories. At 11M members × 200 days × 80 codes, this is prohibitively expensive and the results depend heavily on k | Replaced with information-theoretic measures (conditional entropy, mutual information) that are more principled and scalable |
| **No stratification** | None of the 3 tests stratify by age, LOB, or frequency tier. The user explicitly requested these stratifications | Every analysis in this plan is stratified by tier, age bucket, and LOB |
| **Test 2 is trivially true for same-data epochs** | "Fresh information per epoch" on the same dataset will always show near-zero novelty in epochs 2-3 by construction — this is a tautology, not a diagnostic. The interesting question is: what fraction of *within-epoch* information is novel at each batch position? | Replaced with cumulative novelty curve *within* an epoch and marginal information per additional member |
| **Test 3 uses vague metrics** | `mean_codes_per_day` and `mean_unique_codes_per_member` don't capture what the model sees (padded sequences, target codes, co-occurrence structure). Zipf exponent fitting is fragile on truncated discrete distributions | Replaced with KL-divergence between scales, Jensen-Shannon divergence on code distributions, and marginal entropy gain |
| **No "what model actually sees" perspective** | The original proposal treats data as a static object. But the model sees *target* codes (not input codes), and the gradient signal depends on the *loss-weighted* target distribution. Need to analyze targets, not just inputs | Task 4 analyzes target code distributions specifically |
| **No conditional information analysis** | Frequency alone is insufficient. Two codes can have the same frequency but completely different conditional distributions (code A always co-occurs with B; code C occurs independently). The model can learn A from seeing B, but must independently learn C. Need conditional entropy analysis | Task 5: Conditional entropy and mutual information |
| **Missing: marginal member contribution** | The question "does adding more members help?" is best answered by measuring the *marginal information* of each additional member, not by comparing endpoints. This requires a cumulative information curve | Task 6: Marginal information scaling curve |

---

## Data Access Reference

**BigQuery tables:**

| Scale | Table | dt_cnt filter | Approx members |
|-------|-------|--------------|----------------|
| Full (15.4M) | `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_ending` | varies | 15.4M |
| 1.5M (R5) | `...a834793_Combined_All_LOB_o3_train_10pct_sample` | >= 10 | ~1.58M |
| 3.4M (R6) | `...a834793_Combined_All_LOB_o3_train_20pct_sample` | >= 5 | ~3.4M |
| 6.8M (R7) | `...a834793_Combined_All_LOB_o3_train_40pct_sample` | >= 5 | ~6.8M |

**Schema:** `individual_id, lob, index_dt, gender_cd, age_in_months, cd, target, dt_cnt`

- `cd`: Input codes, `*`-separated days, `,`-separated codes per day (~84k vocab)
- `target`: Target codes, same format (~6,297 vocab, grouped from w2ind)
- `age_in_months`: `*`-separated per-day ages (0-1439)
- `lob`: `'Commercial'`, `'Medicare'`, `'Medicaid'`

**Existing reusable code:**
- `dev/transformer_training_pipeline.py:206-222` — `_parse_codes()`, `_parse_target()` string parsing
- `dev/moe/moe_flashattn_4.py:18027-18107` — `analyze_code_frequency_distribution()`, Gini computation
- `dev/moe/moe_flashattn_4.py:12349-12388` — `_compute_code_frequencies_from_strings()` target parsing

**Tier definitions (quartile-based on non-zero target code frequencies):**
- Common: >= 75th percentile frequency (1,420 codes, 98.8% of occurrences)
- Medium: 50th–75th percentile (1,421 codes, 1.1%)
- Rare: 25th–50th percentile (1,422 codes, 0.1%)
- Tail: < 25th percentile (1,414 codes, ~0.0%)

**Age buckets (derived from `age_in_months` at index_dt):**
- Pediatric: 0–215 months (0–17 years)
- Young Adult: 216–479 months (18–39)
- Middle Adult: 480–779 months (40–64)
- Senior: 780+ months (65+)

---

## Task 1: Project Setup and Data Loading Utilities

**Files:**
- Create: `dev/analysis/data_information_saturation_analysis.ipynb`
- Reference: `dev/transformer_training_pipeline.py:206-222` (parsing logic)
- Reference: `dev/moe/moe_flashattn_4.py:18027-18088` (Gini, frequency analysis)

### Step 1: Create notebook with imports and configuration

```python
import numpy as np
import pandas as pd
import json
import time
from collections import Counter, defaultdict
from scipy import stats, sparse
from scipy.spatial.distance import jensenshannon
from google.cloud import bigquery
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# === Configuration ===
GCP_PROJECT = 'edp-prod-storage'
DATASET = 'edp-prod-storage.edp_ent_sdoheir_cns'

TABLES = {
    '1.5M': f'{DATASET}.a834793_Combined_All_LOB_o3_train_10pct_sample',
    'full': f'{DATASET}.a834793_Combined_All_LOB_o3_train_ending',
}

TARGET_CD_CNT = 6297
LEN_DY = 200
LEN_CD = 80

AGE_BUCKETS = {
    'pediatric': (0, 215),
    'young_adult': (216, 479),
    'middle_adult': (480, 779),
    'senior': (780, 1439),
}

RESULTS_DIR = Path('../../expe_analysis/exp_round5/data_saturation/')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

results = {}
```

### Step 2: Write data loading utilities

```python
def load_sample_from_bigquery(table_name, sample_frac=None, min_dt_cnt=10, 
                               max_rows=None, seed=42):
    """Load data from BigQuery with optional sampling."""
    client = bigquery.Client(project=GCP_PROJECT)
    
    where_clause = f"WHERE dt_cnt >= {min_dt_cnt}"
    if sample_frac and sample_frac < 1.0:
        where_clause += (
            f" AND MOD(ABS(FARM_FINGERPRINT("
            f"CAST(individual_id AS STRING))), 1000) < {int(sample_frac * 1000)}"
        )
    
    limit_clause = f"LIMIT {max_rows}" if max_rows else ""
    
    query = f"""
    SELECT individual_id, lob, age_in_months, cd, target, dt_cnt
    FROM `{table_name}`
    {where_clause}
    {limit_clause}
    """
    
    print(f"Loading from {table_name.split('.')[-1]}...")
    t0 = time.time()
    df = client.query(query).to_dataframe()
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


def parse_target_string(target_str):
    """Parse target string into list of lists of ints.
    '45,67*89*12,34' -> [[45,67], [89], [12,34]]
    """
    if not target_str or pd.isna(target_str):
        return []
    days = target_str.split('*')[:LEN_DY]
    result = []
    for day_str in days:
        if not day_str:
            result.append([])
            continue
        codes = []
        for c in day_str.split(','):
            try:
                v = int(c)
                if 0 < v <= TARGET_CD_CNT:
                    codes.append(v)
            except (ValueError, TypeError):
                pass
        result.append(codes)
    return result


def parse_cd_string(cd_str):
    """Parse input code string into list of lists of ints."""
    if not cd_str or pd.isna(cd_str):
        return []
    days = cd_str.split('*')[:LEN_DY]
    result = []
    for day_str in days:
        if not day_str:
            result.append([])
            continue
        codes = []
        for c in day_str.split(','):
            try:
                v = int(c)
                if v > 0:
                    codes.append(v)
            except (ValueError, TypeError):
                pass
        result.append(codes)
    return result


def get_index_age(age_str):
    """Extract the age at index date (last non-zero value in age sequence)."""
    if not age_str or pd.isna(age_str):
        return 0
    parts = age_str.split('*')
    for p in reversed(parts):
        try:
            v = int(p)
            if v > 0:
                return min(v, 1439)
        except (ValueError, TypeError):
            continue
    return 0


def assign_age_bucket(age_months):
    for bucket, (lo, hi) in AGE_BUCKETS.items():
        if lo <= age_months <= hi:
            return bucket
    return 'unknown'


def compute_tier_boundaries(code_frequencies):
    """Compute tier boundaries from code frequency array."""
    freq_nz = code_frequencies[code_frequencies > 0]
    thresholds = np.percentile(freq_nz, [75, 50, 25])
    return {
        'common': thresholds[0],
        'medium': thresholds[1],
        'rare': thresholds[2],
    }


def assign_code_tier(code_idx, code_frequencies, tier_bounds):
    """Assign a code index (1-based) to its frequency tier."""
    if code_idx <= 0 or code_idx > len(code_frequencies):
        return 'unknown'
    freq = code_frequencies[code_idx - 1]
    if freq == 0:
        return 'zero'
    if freq >= tier_bounds['common']:
        return 'common'
    elif freq >= tier_bounds['medium']:
        return 'medium'
    elif freq >= tier_bounds['rare']:
        return 'rare'
    else:
        return 'tail'
```

### Step 3: Run — verify data loads correctly

Run the first two cells. Verify the DataFrame has columns `[individual_id, lob, age_in_months, cd, target, dt_cnt]` and row count ~1.58M for the 1.5M table.

### Step 4: Commit

```bash
git add dev/analysis/data_information_saturation_analysis.ipynb
git commit -m "feat: scaffold data information saturation analysis notebook"
```

---

## Task 2: Within-Member Temporal Information Saturation

**Goal:** For each member, measure how much *new information* each successive day of their history contributes. Does the N-th day add as much novel signal as the 10th day?

**Files:**
- Modify: `dev/analysis/data_information_saturation_analysis.ipynb` (add cells)

### Step 1: Write the within-member novelty computation

```python
def compute_within_member_saturation(df, code_frequencies, tier_bounds, 
                                       min_days=10, sample_n=50000):
    """
    For each member, walk through their daily target codes chronologically.
    At each day position d, measure:
      - cumulative unique target codes seen so far
      - number of NEW codes on day d (not seen on days 1..d-1)
      - cumulative unique same-day co-occurrence PAIRS seen so far
      - number of NEW co-occurrence pairs on day d
    
    Stratify by: tier of the new codes, age bucket, LOB.
    
    Returns DataFrame with one row per (member, day_position).
    """
    if sample_n and len(df) > sample_n:
        df_sample = df.sample(n=sample_n, random_state=42)
    else:
        df_sample = df
    
    records = []
    
    for idx, row in df_sample.iterrows():
        target_days = parse_target_string(row['target'])
        dt_cnt = int(row['dt_cnt'])
        n_days = min(dt_cnt, len(target_days))
        
        if n_days < min_days:
            continue
        
        age_months = get_index_age(row['age_in_months'])
        age_bucket = assign_age_bucket(age_months)
        lob = row['lob']
        
        seen_codes = set()
        seen_pairs = set()
        
        tier_new_counts = {t: 0 for t in ['common', 'medium', 'rare', 'tail']}
        tier_cumul_counts = {t: 0 for t in ['common', 'medium', 'rare', 'tail']}
        
        for d in range(n_days):
            day_codes = target_days[d]
            if not day_codes:
                continue
            
            new_codes = [c for c in day_codes if c not in seen_codes]
            
            day_pairs = set()
            sorted_dc = sorted(set(day_codes))
            for i in range(len(sorted_dc)):
                for j in range(i + 1, len(sorted_dc)):
                    day_pairs.add((sorted_dc[i], sorted_dc[j]))
            new_pairs = day_pairs - seen_pairs
            
            tier_new = defaultdict(int)
            for c in new_codes:
                tier = assign_code_tier(c, code_frequencies, tier_bounds)
                tier_new[tier] += 1
            
            seen_codes.update(day_codes)
            seen_pairs.update(day_pairs)
            
            tier_cumul = defaultdict(int)
            for c in seen_codes:
                tier = assign_code_tier(c, code_frequencies, tier_bounds)
                tier_cumul[tier] += 1
            
            records.append({
                'day_position': d,
                'dt_cnt': n_days,
                'lob': lob,
                'age_bucket': age_bucket,
                'n_codes_today': len(day_codes),
                'n_new_codes': len(new_codes),
                'cumul_unique_codes': len(seen_codes),
                'novelty_rate': len(new_codes) / max(len(day_codes), 1),
                'n_pairs_today': len(day_pairs),
                'n_new_pairs': len(new_pairs),
                'cumul_unique_pairs': len(seen_pairs),
                'pair_novelty_rate': len(new_pairs) / max(len(day_pairs), 1),
                'new_common': tier_new.get('common', 0),
                'new_medium': tier_new.get('medium', 0),
                'new_rare': tier_new.get('rare', 0),
                'new_tail': tier_new.get('tail', 0),
                'cumul_common': tier_cumul.get('common', 0),
                'cumul_medium': tier_cumul.get('medium', 0),
                'cumul_rare': tier_cumul.get('rare', 0),
                'cumul_tail': tier_cumul.get('tail', 0),
            })
    
    return pd.DataFrame(records)
```

### Step 2: Run computation on 1.5M dataset

```python
print("Loading 1.5M dataset...")
df_1_5m = load_sample_from_bigquery(TABLES['1.5M'], min_dt_cnt=10)

print("Computing target code frequencies...")
code_freq = np.zeros(TARGET_CD_CNT, dtype=np.int64)
for target_str in df_1_5m['target']:
    for day_str in (target_str or '').split('*')[:LEN_DY]:
        if not day_str:
            continue
        for c_str in day_str.split(','):
            try:
                v = int(c_str)
                if 0 < v <= TARGET_CD_CNT:
                    code_freq[v - 1] += 1
            except:
                pass

tier_bounds = compute_tier_boundaries(code_freq)
print(f"Tier boundaries: {tier_bounds}")

print("Computing within-member saturation (50k sample)...")
within_member_df = compute_within_member_saturation(
    df_1_5m, code_freq, tier_bounds, min_days=10, sample_n=50000
)
print(f"Result: {len(within_member_df):,} rows")
```

### Step 3: Generate saturation curves and save results

```python
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Within-Member Temporal Information Saturation', fontsize=14)

# (a) Overall novelty rate by day position
agg = within_member_df.groupby('day_position').agg(
    mean_novelty=('novelty_rate', 'mean'),
    mean_pair_novelty=('pair_novelty_rate', 'mean'),
    mean_cumul=('cumul_unique_codes', 'mean'),
).reset_index()

ax = axes[0, 0]
ax.plot(agg['day_position'], agg['mean_novelty'], label='Code novelty rate')
ax.plot(agg['day_position'], agg['mean_pair_novelty'], label='Pair novelty rate', linestyle='--')
ax.set_xlabel('Day position in history')
ax.set_ylabel('Novelty rate (fraction new)')
ax.set_title('Overall novelty rate by day position')
ax.legend()
ax.set_xlim(0, 200)

# (b) Cumulative unique codes by day
ax = axes[0, 1]
ax.plot(agg['day_position'], agg['mean_cumul'])
ax.set_xlabel('Day position')
ax.set_ylabel('Mean cumulative unique codes')
ax.set_title('Cumulative code discovery')

# (c) Novelty rate by tier
ax = axes[0, 2]
for tier in ['common', 'medium', 'rare', 'tail']:
    within_member_df[f'novelty_{tier}'] = (
        within_member_df[f'new_{tier}'] / 
        within_member_df['n_codes_today'].clip(lower=1)
    )
    tier_agg = within_member_df.groupby('day_position')[f'novelty_{tier}'].mean()
    ax.plot(tier_agg.index, tier_agg.values, label=tier)
ax.set_xlabel('Day position')
ax.set_ylabel('Tier-specific novelty rate')
ax.set_title('Per-tier novelty decay')
ax.legend()

# (d) By LOB
ax = axes[1, 0]
for lob in ['Commercial', 'Medicare', 'Medicaid']:
    lob_df = within_member_df[within_member_df['lob'] == lob]
    if len(lob_df) == 0:
        continue
    lob_agg = lob_df.groupby('day_position')['novelty_rate'].mean()
    ax.plot(lob_agg.index, lob_agg.values, label=lob)
ax.set_xlabel('Day position')
ax.set_ylabel('Novelty rate')
ax.set_title('Novelty rate by LOB')
ax.legend()

# (e) By age bucket
ax = axes[1, 1]
for bucket in ['pediatric', 'young_adult', 'middle_adult', 'senior']:
    b_df = within_member_df[within_member_df['age_bucket'] == bucket]
    if len(b_df) == 0:
        continue
    b_agg = b_df.groupby('day_position')['novelty_rate'].mean()
    ax.plot(b_agg.index, b_agg.values, label=bucket)
ax.set_xlabel('Day position')
ax.set_ylabel('Novelty rate')
ax.set_title('Novelty rate by age group')
ax.legend()

# (f) Pair novelty by tier
ax = axes[1, 2]
ax.plot(agg['day_position'], agg['mean_pair_novelty'])
ax.set_xlabel('Day position')
ax.set_ylabel('Co-occurrence pair novelty rate')
ax.set_title('Same-day pair novelty decay')

plt.tight_layout()
plt.savefig(str(RESULTS_DIR / 'within_member_saturation.png'), dpi=150, bbox_inches='tight')
plt.show()

# Save numeric results
within_member_summary = {
    'metric': 'within_member_temporal_saturation',
    'sample_size': len(within_member_df['dt_cnt'].unique()),
    'novelty_rate_day_10': float(agg.loc[agg['day_position']==10, 'mean_novelty'].iloc[0]) 
        if 10 in agg['day_position'].values else None,
    'novelty_rate_day_50': float(agg.loc[agg['day_position']==50, 'mean_novelty'].iloc[0])
        if 50 in agg['day_position'].values else None,
    'novelty_rate_day_100': float(agg.loc[agg['day_position']==100, 'mean_novelty'].iloc[0])
        if 100 in agg['day_position'].values else None,
    'novelty_rate_day_200': float(agg.loc[agg['day_position']==199, 'mean_novelty'].iloc[0])
        if 199 in agg['day_position'].values else None,
}
results['within_member'] = within_member_summary
print(json.dumps(within_member_summary, indent=2))
```

### Step 4: Interpret results

**Pre-registered interpretation:**
- If novelty_rate at day 100 < 0.10 → by the halfway mark, <10% of daily codes are new to that member → individual history saturates rapidly
- If tail-tier novelty decays faster than common-tier → rare conditions front-load in member history, later days are redundant
- If Medicare novelty decays faster than Commercial → older populations have more repetitive encounter patterns
- If pair novelty decays faster than code novelty → co-occurrence patterns are even more redundant than individual codes

### Step 5: Commit

```bash
git add dev/analysis/data_information_saturation_analysis.ipynb
git commit -m "feat: add within-member temporal saturation analysis"
```

---

## Task 3: Same-Day Co-occurrence and Temporal Transition Analysis

**Goal:** Measure co-occurrence pattern diversity (same-day code pairs + temporal skip-grams) and how it scales with data size. This goes beyond simple code frequencies to capture the relational structure the transformer is supposed to learn.

**Files:**
- Modify: `dev/analysis/data_information_saturation_analysis.ipynb`

### Step 1: Write co-occurrence diversity computation

```python
def compute_cooccurrence_diversity(df, code_freq, tier_bounds, sample_n=50000):
    """
    Compute same-day co-occurrence pair statistics and temporal skip-gram diversity.
    
    Same-day pairs: (code_A, code_B) appearing on the same day for a member
    Temporal skip-grams: (code_A on day d, code_B on day d+k) for k in [1,2,3]
    
    Returns dict with diversity metrics.
    """
    if sample_n and len(df) > sample_n:
        df_sample = df.sample(n=sample_n, random_state=42)
    else:
        df_sample = df

    same_day_pairs = Counter()
    temporal_bigrams = Counter()       # (A day d, B day d+1)
    temporal_skipgrams_2 = Counter()   # (A day d, B day d+2)
    temporal_skipgrams_3 = Counter()   # (A day d, B day d+3)
    
    tier_pair_counts = defaultdict(Counter)  # tier_pair -> count of unique pairs
    lob_pair_diversity = defaultdict(set)
    age_pair_diversity = defaultdict(set)
    
    for _, row in df_sample.iterrows():
        target_days = parse_target_string(row['target'])
        dt_cnt = min(int(row['dt_cnt']), len(target_days))
        lob = row['lob']
        age_bucket = assign_age_bucket(get_index_age(row['age_in_months']))
        
        for d in range(dt_cnt):
            day_codes = sorted(set(target_days[d])) if d < len(target_days) else []
            
            # Same-day co-occurrence pairs
            for i in range(len(day_codes)):
                for j in range(i + 1, len(day_codes)):
                    pair = (day_codes[i], day_codes[j])
                    same_day_pairs[pair] += 1
                    lob_pair_diversity[lob].add(pair)
                    age_pair_diversity[age_bucket].add(pair)
                    
                    tier_a = assign_code_tier(day_codes[i], code_freq, tier_bounds)
                    tier_b = assign_code_tier(day_codes[j], code_freq, tier_bounds)
                    tier_key = tuple(sorted([tier_a, tier_b]))
                    tier_pair_counts[tier_key][pair] += 1
            
            # Temporal transitions
            for skip, counter in [(1, temporal_bigrams), 
                                   (2, temporal_skipgrams_2),
                                   (3, temporal_skipgrams_3)]:
                if d + skip < dt_cnt and d + skip < len(target_days):
                    next_codes = set(target_days[d + skip])
                    for a in day_codes:
                        for b in next_codes:
                            if a > 0 and b > 0:
                                counter[(a, b)] += 1
    
    # Compute entropy of pair distributions
    def distribution_entropy(counter):
        total = sum(counter.values())
        if total == 0:
            return 0.0
        probs = np.array(list(counter.values()), dtype=np.float64) / total
        return float(stats.entropy(probs, base=2))
    
    return {
        'n_unique_same_day_pairs': len(same_day_pairs),
        'total_same_day_pair_occurrences': sum(same_day_pairs.values()),
        'same_day_pair_entropy_bits': distribution_entropy(same_day_pairs),
        'same_day_pair_gini': float(
            _gini(np.array(list(same_day_pairs.values())))
        ) if same_day_pairs else 0,
        
        'n_unique_temporal_bigrams': len(temporal_bigrams),
        'temporal_bigram_entropy_bits': distribution_entropy(temporal_bigrams),
        
        'n_unique_skip2grams': len(temporal_skipgrams_2),
        'skip2gram_entropy_bits': distribution_entropy(temporal_skipgrams_2),
        
        'n_unique_skip3grams': len(temporal_skipgrams_3),
        'skip3gram_entropy_bits': distribution_entropy(temporal_skipgrams_3),
        
        'tier_pair_diversity': {
            str(k): len(v) for k, v in tier_pair_counts.items()
        },
        'lob_pair_diversity': {
            k: len(v) for k, v in lob_pair_diversity.items()
        },
        'age_pair_diversity': {
            k: len(v) for k, v in age_pair_diversity.items()
        },
        'same_day_pair_concentration': {
            'top_10_pct_share': float(
                sum(sorted(same_day_pairs.values(), reverse=True)
                    [:max(1, len(same_day_pairs)//10)]) 
                / max(1, sum(same_day_pairs.values()))
            ),
            'top_1_pct_share': float(
                sum(sorted(same_day_pairs.values(), reverse=True)
                    [:max(1, len(same_day_pairs)//100)]) 
                / max(1, sum(same_day_pairs.values()))
            ),
        }
    }


def _gini(values):
    """Compute Gini coefficient."""
    sorted_v = np.sort(values)
    n = len(sorted_v)
    if n == 0 or sorted_v.sum() == 0:
        return 0.0
    return (2 * np.sum((np.arange(1, n+1) * sorted_v))) / (n * sorted_v.sum()) - (n + 1) / n
```

### Step 2: Run on 1.5M dataset

```python
print("Computing co-occurrence diversity on 1.5M (50k sample)...")
cooccurrence_results_1_5m = compute_cooccurrence_diversity(
    df_1_5m, code_freq, tier_bounds, sample_n=50000
)
results['cooccurrence_1_5m'] = cooccurrence_results_1_5m
print(json.dumps(cooccurrence_results_1_5m, indent=2, default=str))
```

### Step 3: Commit

```bash
git add dev/analysis/data_information_saturation_analysis.ipynb
git commit -m "feat: add co-occurrence and temporal transition diversity analysis"
```

---

## Task 4: Target Code Distribution Shift Across Scales

**Goal:** Quantify how the *target* code distribution changes as dataset size grows, using incremental subsets of the full table. This measures what the model's loss function actually "sees."

**Files:**
- Modify: `dev/analysis/data_information_saturation_analysis.ipynb`

### Step 1: Write incremental scale analysis function

```python
def compute_target_distribution_at_scale(table_name, sample_fracs, min_dt_cnt=10):
    """
    Load incrementally larger subsets from a single table using deterministic
    FARM_FINGERPRINT sampling. Compute target code frequency distributions,
    entropy, Gini, and tier coverage at each scale.
    
    Uses nested sampling: the 10% sample is a strict SUBSET of the 20% sample, etc.
    This ensures monotonic inclusion and clean marginal analysis.
    """
    client = bigquery.Client(project=GCP_PROJECT)
    scale_results = []
    
    for frac in sample_fracs:
        frac_int = int(frac * 1000)
        
        query = f"""
        WITH sampled AS (
            SELECT target, dt_cnt, lob, age_in_months
            FROM `{table_name}`
            WHERE dt_cnt >= {min_dt_cnt}
              AND MOD(ABS(FARM_FINGERPRINT(
                  CAST(individual_id AS STRING))), 1000) < {frac_int}
        )
        SELECT 
            COUNT(*) as n_members,
            COUNT(DISTINCT lob) as n_lobs
        FROM sampled
        """
        count_df = client.query(query).to_dataframe()
        n_members = int(count_df['n_members'].iloc[0])
        
        # Load actual data for this scale
        data_query = f"""
        SELECT target, lob, age_in_months, dt_cnt
        FROM `{table_name}`
        WHERE dt_cnt >= {min_dt_cnt}
          AND MOD(ABS(FARM_FINGERPRINT(
              CAST(individual_id AS STRING))), 1000) < {frac_int}
        """
        df_scale = client.query(data_query).to_dataframe()
        
        # Compute target code frequencies
        freq = np.zeros(TARGET_CD_CNT, dtype=np.int64)
        for target_str in df_scale['target']:
            for day_str in (target_str or '').split('*')[:LEN_DY]:
                if not day_str:
                    continue
                for c_str in day_str.split(','):
                    try:
                        v = int(c_str)
                        if 0 < v <= TARGET_CD_CNT:
                            freq[v - 1] += 1
                    except:
                        pass
        
        freq_nz = freq[freq > 0]
        n_nonzero = len(freq_nz)
        
        # Compute metrics
        sorted_freq = np.sort(freq_nz)
        n = len(sorted_freq)
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_freq))) / (n * sorted_freq.sum()) - (n + 1) / n
        
        probs = freq_nz / freq_nz.sum()
        entropy = float(stats.entropy(probs, base=2))
        
        # Tier coverage at multiple thresholds
        tier_b = compute_tier_boundaries(freq)
        tier_coverage = {}
        for K in [10, 50, 100, 500]:
            for tier_name, threshold in [
                ('common', tier_b['common']),
                ('medium', tier_b['medium']),
                ('rare', tier_b['rare']),
                ('tail', 0),
            ]:
                if tier_name == 'common':
                    mask = freq >= threshold
                elif tier_name == 'medium':
                    mask = (freq >= tier_b['medium']) & (freq < tier_b['common'])
                elif tier_name == 'rare':
                    mask = (freq >= tier_b['rare']) & (freq < tier_b['medium'])
                else:
                    mask = (freq > 0) & (freq < tier_b['rare'])
                
                n_tier = mask.sum()
                n_above_K = ((freq >= K) & mask).sum()
                tier_coverage[f'{tier_name}_above_{K}'] = (
                    float(n_above_K / n_tier) if n_tier > 0 else 0
                )
        
        # LOB breakdown
        lob_counts = df_scale['lob'].value_counts().to_dict()
        
        # Age bucket breakdown
        df_scale['age_idx'] = df_scale['age_in_months'].apply(get_index_age)
        df_scale['age_bucket'] = df_scale['age_idx'].apply(assign_age_bucket)
        age_counts = df_scale['age_bucket'].value_counts().to_dict()
        
        scale_results.append({
            'sample_frac': frac,
            'n_members': n_members,
            'n_nonzero_codes': int(n_nonzero),
            'total_occurrences': int(freq.sum()),
            'gini': float(gini),
            'entropy_bits': entropy,
            'max_frequency': int(freq.max()),
            'median_frequency': float(np.median(freq_nz)),
            'tier_coverage': tier_coverage,
            'lob_counts': lob_counts,
            'age_counts': age_counts,
            'freq_array': freq,  # keep for cross-scale comparison
        })
        
        print(f"  Scale {frac:.1%}: {n_members:,} members, "
              f"Gini={gini:.4f}, Entropy={entropy:.2f} bits, "
              f"nonzero={n_nonzero}")
    
    return scale_results
```

### Step 2: Run across scales

```python
sample_fracs = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]

print("Computing target distributions across scales...")
print("(Using nested FARM_FINGERPRINT sampling for monotonic inclusion)")
scale_results = compute_target_distribution_at_scale(
    TABLES['full'], sample_fracs, min_dt_cnt=5
)
results['cross_scale'] = [
    {k: v for k, v in r.items() if k != 'freq_array'} 
    for r in scale_results
]
```

### Step 3: Compute cross-scale divergence measures

```python
def compute_cross_scale_divergences(scale_results):
    """
    For each pair of adjacent scales, compute:
    - KL-divergence of target code distributions
    - Jensen-Shannon divergence
    - Marginal entropy gain (entropy at scale N+1 minus entropy at scale N)
    - Per-tier distribution shift
    """
    divergences = []
    for i in range(1, len(scale_results)):
        prev = scale_results[i-1]
        curr = scale_results[i]
        
        # Normalize to probability distributions (non-zero only aligned)
        p = prev['freq_array'].astype(np.float64)
        q = curr['freq_array'].astype(np.float64)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        p_norm = (p + eps) / (p + eps).sum()
        q_norm = (q + eps) / (q + eps).sum()
        
        kl_pq = float(stats.entropy(p_norm, q_norm, base=2))
        js = float(jensenshannon(p_norm, q_norm, base=2) ** 2)
        
        marginal_entropy = curr['entropy_bits'] - prev['entropy_bits']
        marginal_members = curr['n_members'] - prev['n_members']
        entropy_per_member = (
            marginal_entropy / marginal_members if marginal_members > 0 else 0
        )
        
        divergences.append({
            'from_frac': prev['sample_frac'],
            'to_frac': curr['sample_frac'],
            'from_members': prev['n_members'],
            'to_members': curr['n_members'],
            'marginal_members': marginal_members,
            'kl_divergence_bits': kl_pq,
            'js_divergence_bits': js,
            'marginal_entropy_gain_bits': marginal_entropy,
            'entropy_per_marginal_member': entropy_per_member,
            'gini_delta': curr['gini'] - prev['gini'],
        })
    
    return divergences

divergences = compute_cross_scale_divergences(scale_results)
results['cross_scale_divergences'] = divergences

for d in divergences:
    print(f"  {d['from_frac']:.0%} → {d['to_frac']:.0%}: "
          f"JSD={d['js_divergence_bits']:.6f}, "
          f"Δentropy={d['marginal_entropy_gain_bits']:.4f}, "
          f"ΔGini={d['gini_delta']:.6f}")
```

### Step 4: Plot scaling curves

```python
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Target Code Distribution Across Data Scales', fontsize=14)

members = [r['n_members'] for r in scale_results]

# (a) Gini vs scale
ax = axes[0, 0]
ax.plot(members, [r['gini'] for r in scale_results], 'o-')
ax.set_xlabel('Number of members')
ax.set_ylabel('Gini coefficient')
ax.set_title('Code frequency concentration vs scale')
ax.set_xscale('log')

# (b) Entropy vs scale
ax = axes[0, 1]
ax.plot(members, [r['entropy_bits'] for r in scale_results], 'o-')
ax.set_xlabel('Number of members')
ax.set_ylabel('Shannon entropy (bits)')
ax.set_title('Target distribution entropy vs scale')
ax.set_xscale('log')

# (c) Nonzero codes vs scale
ax = axes[0, 2]
ax.plot(members, [r['n_nonzero_codes'] for r in scale_results], 'o-')
ax.set_xlabel('Number of members')
ax.set_ylabel('Non-zero target codes')
ax.set_title('Code coverage vs scale')
ax.set_xscale('log')
ax.axhline(y=TARGET_CD_CNT, color='r', linestyle='--', alpha=0.5, label=f'Max={TARGET_CD_CNT}')
ax.legend()

# (d) JS divergence (marginal)
ax = axes[1, 0]
marg_members = [(d['from_members'] + d['to_members'])/2 for d in divergences]
ax.plot(marg_members, [d['js_divergence_bits'] for d in divergences], 'o-')
ax.set_xlabel('Approx members (midpoint)')
ax.set_ylabel('JS divergence (bits)')
ax.set_title('Distribution shift between adjacent scales')
ax.set_xscale('log')

# (e) Marginal entropy gain per member
ax = axes[1, 1]
ax.plot(marg_members, [d['entropy_per_marginal_member'] for d in divergences], 'o-')
ax.set_xlabel('Approx members')
ax.set_ylabel('Entropy gain per marginal member')
ax.set_title('Diminishing information returns')
ax.set_xscale('log')
ax.set_yscale('log')

# (f) Tier coverage at K=100 vs scale
ax = axes[1, 2]
for tier in ['common', 'medium', 'rare', 'tail']:
    cov = [r['tier_coverage'][f'{tier}_above_100'] for r in scale_results]
    ax.plot(members, cov, 'o-', label=tier)
ax.set_xlabel('Number of members')
ax.set_ylabel(f'Fraction of codes with freq >= 100')
ax.set_title('Tier coverage (K=100) vs scale')
ax.set_xscale('log')
ax.legend()

plt.tight_layout()
plt.savefig(str(RESULTS_DIR / 'cross_scale_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()
```

### Step 5: Commit

```bash
git add dev/analysis/data_information_saturation_analysis.ipynb
git commit -m "feat: add cross-scale target distribution and divergence analysis"
```

---

## Task 5: Conditional Entropy and Mutual Information Analysis

**Goal:** Frequency alone is insufficient. Two codes can have the same marginal frequency but completely different *conditional* distributions. Measure how much information about code B is gained by knowing code A is present (mutual information), and how this scales with data.

**Files:**
- Modify: `dev/analysis/data_information_saturation_analysis.ipynb`

### Step 1: Write conditional information analysis

```python
def compute_conditional_information(df, code_freq, tier_bounds, 
                                     sample_n=30000, top_k_codes=500):
    """
    Compute conditional entropy and mutual information between code pairs.
    
    For the top_k most frequent codes, measure:
    - H(B|A) = conditional entropy of code B given code A is present
    - I(A;B) = mutual information between code A and code B
    - How these metrics differ across tiers
    
    Uses same-day co-occurrence as the conditioning event.
    """
    if sample_n and len(df) > sample_n:
        df_sample = df.sample(n=sample_n, random_state=42)
    else:
        df_sample = df
    
    # Select top_k codes by frequency for tractability
    top_codes = np.argsort(code_freq)[-top_k_codes:][::-1] + 1  # 1-indexed
    top_code_set = set(top_codes.tolist())
    
    n_members = len(df_sample)
    
    # Build member-level binary presence vectors for top codes
    # presence[i, j] = 1 if member i has code top_codes[j] on ANY day
    code_to_idx = {c: i for i, c in enumerate(top_codes)}
    presence = np.zeros((n_members, top_k_codes), dtype=np.int8)
    
    for mem_idx, (_, row) in enumerate(df_sample.iterrows()):
        target_days = parse_target_string(row['target'])
        for day_codes in target_days:
            for c in day_codes:
                if c in code_to_idx:
                    presence[mem_idx, code_to_idx[c]] = 1
    
    # Compute pairwise mutual information for a sample of pairs
    # Use vectorized computation
    p_a = presence.mean(axis=0)  # P(A=1) for each code
    
    # Sample pairs across tiers
    tier_assignments = {}
    for c in top_codes:
        tier_assignments[c] = assign_code_tier(c, code_freq, tier_bounds)
    
    mi_results = []
    pair_types = [
        ('common', 'common'), ('common', 'medium'), ('common', 'rare'),
        ('common', 'tail'), ('medium', 'medium'), ('medium', 'rare'),
        ('rare', 'rare'), ('rare', 'tail'), ('tail', 'tail'),
    ]
    
    for tier_a, tier_b in pair_types:
        codes_a = [c for c in top_codes if tier_assignments[c] == tier_a]
        codes_b = [c for c in top_codes if tier_assignments[c] == tier_b]
        
        if not codes_a or not codes_b:
            continue
        
        # Sample up to 100 pairs per tier combination
        n_pairs = min(100, len(codes_a) * len(codes_b))
        sampled_pairs = []
        for _ in range(n_pairs):
            a = codes_a[np.random.randint(len(codes_a))]
            b = codes_b[np.random.randint(len(codes_b))]
            if a != b:
                sampled_pairs.append((a, b))
        
        for a, b in sampled_pairs:
            idx_a = code_to_idx[a]
            idx_b = code_to_idx[b]
            
            # Joint distribution
            p_11 = float(((presence[:, idx_a] == 1) & (presence[:, idx_b] == 1)).mean())
            p_10 = float(((presence[:, idx_a] == 1) & (presence[:, idx_b] == 0)).mean())
            p_01 = float(((presence[:, idx_a] == 0) & (presence[:, idx_b] == 1)).mean())
            p_00 = float(((presence[:, idx_a] == 0) & (presence[:, idx_b] == 0)).mean())
            
            # Mutual information
            mi = 0.0
            for p_joint, p_marg_a, p_marg_b in [
                (p_11, p_a[idx_a], p_a[idx_b]),
                (p_10, p_a[idx_a], 1-p_a[idx_b]),
                (p_01, 1-p_a[idx_a], p_a[idx_b]),
                (p_00, 1-p_a[idx_a], 1-p_a[idx_b]),
            ]:
                if p_joint > 0 and p_marg_a > 0 and p_marg_b > 0:
                    mi += p_joint * np.log2(p_joint / (p_marg_a * p_marg_b))
            
            mi_results.append({
                'tier_a': tier_a, 'tier_b': tier_b,
                'code_a': int(a), 'code_b': int(b),
                'mutual_info_bits': mi,
                'p_cooccur': p_11,
            })
    
    mi_df = pd.DataFrame(mi_results)
    
    # Aggregate by tier pair
    tier_mi_summary = mi_df.groupby(['tier_a', 'tier_b']).agg(
        mean_mi=('mutual_info_bits', 'mean'),
        median_mi=('mutual_info_bits', 'median'),
        max_mi=('mutual_info_bits', 'max'),
        mean_cooccur=('p_cooccur', 'mean'),
        n_pairs=('mutual_info_bits', 'count'),
    ).reset_index()
    
    return {
        'tier_mi_summary': tier_mi_summary.to_dict('records'),
        'overall_mean_mi': float(mi_df['mutual_info_bits'].mean()),
        'overall_median_mi': float(mi_df['mutual_info_bits'].median()),
        'n_pairs_analyzed': len(mi_df),
    }
```

### Step 2: Run and interpret

```python
print("Computing conditional information analysis (30k sample, top 500 codes)...")
cond_info = compute_conditional_information(
    df_1_5m, code_freq, tier_bounds, sample_n=30000, top_k_codes=500
)
results['conditional_information'] = cond_info

print("\nMutual Information by Tier Pair:")
print(f"{'Tier A':<12} {'Tier B':<12} {'Mean MI':<12} {'Med MI':<12} {'Mean CoOcc':<12}")
for r in cond_info['tier_mi_summary']:
    print(f"{r['tier_a']:<12} {r['tier_b']:<12} {r['mean_mi']:<12.6f} "
          f"{r['median_mi']:<12.6f} {r['mean_cooccur']:<12.4f}")
```

**Pre-registered interpretation:**
- If common-common MI >> rare-rare MI → common codes carry rich inter-code information; rare codes are informationally isolated → the model learns common-code structure because that's where the information is
- If common-rare MI is low → rare codes don't co-occur with common codes → shared encoder can't transfer common-code knowledge to rare codes
- If all MI values are low (< 0.01 bits) → codes are nearly independent → the multi-label BCE objective is inherently limited because there are no meaningful code interactions to learn

### Step 3: Commit

```bash
git add dev/analysis/data_information_saturation_analysis.ipynb
git commit -m "feat: add conditional entropy and mutual information analysis"
```

---

## Task 6: Marginal Member Information Contribution

**Goal:** Directly answer: "As we add each additional member, how much new information does the dataset gain?" This is the most direct test of data saturation.

**Files:**
- Modify: `dev/analysis/data_information_saturation_analysis.ipynb`

### Step 1: Write marginal information computation

```python
def compute_marginal_member_information(df, sample_n=100000, 
                                         n_increments=50):
    """
    Process members one at a time (in random order) and measure cumulative
    information metrics. Produces a curve of information vs. number of members.
    
    Metrics at each increment:
    - Cumulative unique target codes seen
    - Cumulative unique same-day pairs seen
    - Cumulative unique temporal bigrams seen
    - Shannon entropy of the cumulative frequency distribution
    - Per-tier cumulative coverage
    
    Stratified by LOB and age bucket.
    """
    if sample_n and len(df) > sample_n:
        df_sample = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    else:
        df_sample = df.reset_index(drop=True)
    
    n_total = len(df_sample)
    increment_size = max(1, n_total // n_increments)
    checkpoints = list(range(increment_size, n_total + 1, increment_size))
    if checkpoints[-1] != n_total:
        checkpoints.append(n_total)
    
    cumul_code_freq = np.zeros(TARGET_CD_CNT, dtype=np.int64)
    cumul_codes_seen = set()
    cumul_pairs_seen = set()
    cumul_bigrams_seen = set()
    
    # Per-LOB / per-age tracking
    lob_codes = defaultdict(set)
    age_codes = defaultdict(set)
    
    curve = []
    checkpoint_idx = 0
    
    for mem_idx in range(n_total):
        row = df_sample.iloc[mem_idx]
        target_days = parse_target_string(row['target'])
        dt_cnt = min(int(row['dt_cnt']), len(target_days))
        lob = row['lob']
        age_bucket = assign_age_bucket(get_index_age(row['age_in_months']))
        
        member_codes = set()
        
        for d in range(dt_cnt):
            day_codes = target_days[d] if d < len(target_days) else []
            
            for c in day_codes:
                if 0 < c <= TARGET_CD_CNT:
                    cumul_code_freq[c - 1] += 1
                    cumul_codes_seen.add(c)
                    member_codes.add(c)
                    lob_codes[lob].add(c)
                    age_codes[age_bucket].add(c)
            
            # Same-day pairs
            sorted_dc = sorted(set(day_codes))
            for i in range(len(sorted_dc)):
                for j in range(i + 1, len(sorted_dc)):
                    if sorted_dc[i] > 0 and sorted_dc[j] > 0:
                        cumul_pairs_seen.add((sorted_dc[i], sorted_dc[j]))
            
            # Temporal bigrams
            if d + 1 < dt_cnt and d + 1 < len(target_days):
                next_codes = target_days[d + 1]
                for a in day_codes:
                    for b in next_codes:
                        if a > 0 and b > 0:
                            cumul_bigrams_seen.add((a, b))
        
        if checkpoint_idx < len(checkpoints) and mem_idx + 1 >= checkpoints[checkpoint_idx]:
            freq_nz = cumul_code_freq[cumul_code_freq > 0]
            entropy = float(stats.entropy(
                freq_nz / freq_nz.sum(), base=2
            )) if len(freq_nz) > 0 else 0
            
            curve.append({
                'n_members': mem_idx + 1,
                'unique_codes': len(cumul_codes_seen),
                'unique_pairs': len(cumul_pairs_seen),
                'unique_bigrams': len(cumul_bigrams_seen),
                'entropy_bits': entropy,
                'nonzero_codes': int(len(freq_nz)),
                'lob_code_counts': {k: len(v) for k, v in lob_codes.items()},
                'age_code_counts': {k: len(v) for k, v in age_codes.items()},
            })
            checkpoint_idx += 1
            
            if (mem_idx + 1) % (n_total // 5) == 0:
                print(f"  {mem_idx+1:,}/{n_total:,}: "
                      f"codes={len(cumul_codes_seen)}, "
                      f"pairs={len(cumul_pairs_seen):,}, "
                      f"bigrams={len(cumul_bigrams_seen):,}, "
                      f"H={entropy:.2f}")
    
    return curve
```

### Step 2: Run and plot

```python
print("Computing marginal member information curve (100k members)...")
marginal_curve = compute_marginal_member_information(
    df_1_5m, sample_n=100000, n_increments=50
)
results['marginal_member_curve'] = marginal_curve

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Marginal Member Information Contribution', fontsize=14)

members = [p['n_members'] for p in marginal_curve]

ax = axes[0, 0]
ax.plot(members, [p['unique_codes'] for p in marginal_curve], 'o-')
ax.set_xlabel('Members processed')
ax.set_ylabel('Cumulative unique codes')
ax.set_title('Code discovery curve')

ax = axes[0, 1]
ax.plot(members, [p['unique_pairs'] for p in marginal_curve], 'o-')
ax.set_xlabel('Members processed')
ax.set_ylabel('Cumulative unique co-occurrence pairs')
ax.set_title('Pair discovery curve')

ax = axes[0, 2]
ax.plot(members, [p['unique_bigrams'] for p in marginal_curve], 'o-')
ax.set_xlabel('Members processed')
ax.set_ylabel('Cumulative unique temporal bigrams')
ax.set_title('Temporal bigram discovery curve')

ax = axes[1, 0]
ax.plot(members, [p['entropy_bits'] for p in marginal_curve], 'o-')
ax.set_xlabel('Members processed')
ax.set_ylabel('Shannon entropy (bits)')
ax.set_title('Entropy accumulation curve')

# Marginal gain (derivative)
ax = axes[1, 1]
if len(marginal_curve) > 1:
    marginal_codes = [
        marginal_curve[i]['unique_codes'] - marginal_curve[i-1]['unique_codes']
        for i in range(1, len(marginal_curve))
    ]
    marginal_members_inc = [
        marginal_curve[i]['n_members'] - marginal_curve[i-1]['n_members']
        for i in range(1, len(marginal_curve))
    ]
    codes_per_member = [
        c / m if m > 0 else 0 
        for c, m in zip(marginal_codes, marginal_members_inc)
    ]
    ax.plot(members[1:], codes_per_member, 'o-')
ax.set_xlabel('Members processed')
ax.set_ylabel('New codes per marginal member')
ax.set_title('Marginal information gain (derivative)')

# LOB breakdown
ax = axes[1, 2]
lobs_present = set()
for p in marginal_curve:
    lobs_present.update(p['lob_code_counts'].keys())
for lob in sorted(lobs_present):
    vals = [p['lob_code_counts'].get(lob, 0) for p in marginal_curve]
    ax.plot(members, vals, 'o-', label=lob)
ax.set_xlabel('Members processed')
ax.set_ylabel('Unique codes from LOB')
ax.set_title('Code discovery by LOB')
ax.legend()

plt.tight_layout()
plt.savefig(str(RESULTS_DIR / 'marginal_member_information.png'), dpi=150, bbox_inches='tight')
plt.show()
```

### Step 3: Compute saturation point

```python
# Fit a logarithmic saturation model: y = a * log(x) + b
from scipy.optimize import curve_fit

def log_model(x, a, b):
    return a * np.log(x) + b

x = np.array([p['n_members'] for p in marginal_curve], dtype=np.float64)
y_codes = np.array([p['unique_codes'] for p in marginal_curve], dtype=np.float64)
y_pairs = np.array([p['unique_pairs'] for p in marginal_curve], dtype=np.float64)
y_entropy = np.array([p['entropy_bits'] for p in marginal_curve], dtype=np.float64)

for name, y in [('codes', y_codes), ('pairs', y_pairs), ('entropy', y_entropy)]:
    try:
        popt, _ = curve_fit(log_model, x, y)
        # At what member count does marginal gain drop below 1% of initial rate?
        initial_rate = popt[0] / x[0]
        saturation_point = popt[0] / (initial_rate * 0.01) if initial_rate > 0 else float('inf')
        print(f"{name}: y = {popt[0]:.2f} * log(x) + {popt[1]:.2f}")
        print(f"  Estimated 99% saturation at ~{saturation_point:,.0f} members")
    except Exception as e:
        print(f"{name}: fit failed: {e}")

results['saturation_estimates'] = {
    'method': 'logarithmic_fit',
    'note': 'Members at which marginal gain drops below 1% of initial rate'
}
```

### Step 4: Commit

```bash
git add dev/analysis/data_information_saturation_analysis.ipynb
git commit -m "feat: add marginal member information and saturation analysis"
```

---

## Task 7: LOB-Stratified and Age-Stratified Deep Dive

**Goal:** The overall metrics may mask important subgroup differences. Medicare patients may saturate differently than Commercial. Pediatric patients may have fundamentally different temporal patterns.

**Files:**
- Modify: `dev/analysis/data_information_saturation_analysis.ipynb`

### Step 1: Write stratified analysis

```python
def compute_stratified_saturation(df, code_freq, tier_bounds, 
                                    stratify_col, sample_per_stratum=10000):
    """
    Run within-member saturation and co-occurrence diversity analysis 
    separately for each stratum (LOB or age bucket).
    """
    if stratify_col == 'age_bucket':
        df = df.copy()
        df['age_bucket'] = df['age_in_months'].apply(
            lambda x: assign_age_bucket(get_index_age(x))
        )
    
    strata = df[stratify_col].unique()
    stratum_results = {}
    
    for s in sorted(strata):
        df_s = df[df[stratify_col] == s]
        n_s = len(df_s)
        print(f"\n  Stratum '{s}': {n_s:,} members")
        
        if n_s < 100:
            print(f"    Skipping (too few members)")
            continue
        
        # Within-member saturation
        wm = compute_within_member_saturation(
            df_s, code_freq, tier_bounds, 
            min_days=10, sample_n=min(sample_per_stratum, n_s)
        )
        
        if len(wm) == 0:
            continue
        
        wm_agg = wm.groupby('day_position').agg(
            mean_novelty=('novelty_rate', 'mean'),
            mean_pair_novelty=('pair_novelty_rate', 'mean'),
            mean_cumul=('cumul_unique_codes', 'mean'),
        ).reset_index()
        
        # Co-occurrence diversity
        cooc = compute_cooccurrence_diversity(
            df_s, code_freq, tier_bounds, 
            sample_n=min(sample_per_stratum, n_s)
        )
        
        # Stratum-specific code frequency distribution
        s_freq = np.zeros(TARGET_CD_CNT, dtype=np.int64)
        for target_str in df_s.head(sample_per_stratum)['target']:
            for day_str in (target_str or '').split('*')[:LEN_DY]:
                if not day_str:
                    continue
                for c_str in day_str.split(','):
                    try:
                        v = int(c_str)
                        if 0 < v <= TARGET_CD_CNT:
                            s_freq[v - 1] += 1
                    except:
                        pass
        
        s_freq_nz = s_freq[s_freq > 0]
        s_gini = _gini(s_freq_nz) if len(s_freq_nz) > 0 else 0
        s_entropy = float(stats.entropy(
            s_freq_nz / s_freq_nz.sum(), base=2
        )) if len(s_freq_nz) > 0 else 0
        
        novelty_50 = float(
            wm_agg.loc[wm_agg['day_position']==50, 'mean_novelty'].iloc[0]
        ) if 50 in wm_agg['day_position'].values else None
        
        stratum_results[str(s)] = {
            'n_members': n_s,
            'n_nonzero_codes': int(len(s_freq_nz)),
            'gini': float(s_gini),
            'entropy_bits': s_entropy,
            'novelty_rate_day_10': float(
                wm_agg.loc[wm_agg['day_position']==10, 'mean_novelty'].iloc[0]
            ) if 10 in wm_agg['day_position'].values else None,
            'novelty_rate_day_50': novelty_50,
            'cooccurrence': {
                k: v for k, v in cooc.items() 
                if k not in ['same_day_pair_concentration']
            },
        }
    
    return stratum_results
```

### Step 2: Run LOB and age stratification

```python
print("=" * 60)
print("LOB-STRATIFIED ANALYSIS")
print("=" * 60)
lob_results = compute_stratified_saturation(
    df_1_5m, code_freq, tier_bounds, 
    stratify_col='lob', sample_per_stratum=10000
)
results['lob_stratified'] = lob_results

print("\n" + "=" * 60)
print("AGE-STRATIFIED ANALYSIS")
print("=" * 60)
age_results = compute_stratified_saturation(
    df_1_5m, code_freq, tier_bounds, 
    stratify_col='age_bucket', sample_per_stratum=10000
)
results['age_stratified'] = age_results

# Summary table
print("\n\nSTRATIFIED SUMMARY")
print(f"{'Stratum':<18} {'Members':>10} {'NonZero':>8} {'Gini':>8} "
      f"{'Entropy':>10} {'Nov@10':>8} {'Nov@50':>8}")
print("-" * 80)
for name, res_dict in [('LOB', lob_results), ('Age', age_results)]:
    for s, r in sorted(res_dict.items()):
        print(f"{s:<18} {r['n_members']:>10,} {r['n_nonzero_codes']:>8} "
              f"{r['gini']:>8.4f} {r['entropy_bits']:>10.2f} "
              f"{r.get('novelty_rate_day_10', 'N/A'):>8} "
              f"{r.get('novelty_rate_day_50', 'N/A'):>8}")
```

### Step 3: Commit

```bash
git add dev/analysis/data_information_saturation_analysis.ipynb
git commit -m "feat: add LOB and age-stratified saturation analysis"
```

---

## Task 8: Marginal Member Characterization (Who Are the Extra Members?)

**Goal:** Directly compare the members in the 1.5M sample vs. the "marginal" members in the full dataset. Are marginal members less informationally diverse?

**Files:**
- Modify: `dev/analysis/data_information_saturation_analysis.ipynb`

### Step 1: Write marginal member comparison

```python
def compare_core_vs_marginal_members(table_full, core_frac=0.10, 
                                      min_dt_cnt=5, sample_n=50000):
    """
    Compare 'core' members (in the smallest sample) vs. 'marginal' members 
    (those added when scaling up). Measures whether marginal members are
    systematically less diverse.
    """
    client = bigquery.Client(project=GCP_PROJECT)
    
    core_threshold = int(core_frac * 1000)
    
    # Load core members
    core_query = f"""
    SELECT target, lob, age_in_months, dt_cnt
    FROM `{table_full}`
    WHERE dt_cnt >= {min_dt_cnt}
      AND MOD(ABS(FARM_FINGERPRINT(CAST(individual_id AS STRING))), 1000) < {core_threshold}
    """
    
    # Load marginal members (NOT in core)
    marginal_query = f"""
    SELECT target, lob, age_in_months, dt_cnt
    FROM `{table_full}`
    WHERE dt_cnt >= {min_dt_cnt}
      AND MOD(ABS(FARM_FINGERPRINT(CAST(individual_id AS STRING))), 1000) >= {core_threshold}
    """
    
    print("Loading core members...")
    df_core = client.query(core_query).to_dataframe()
    if len(df_core) > sample_n:
        df_core = df_core.sample(n=sample_n, random_state=42)
    
    print(f"  Core: {len(df_core):,} members")
    
    print("Loading marginal members...")
    df_marginal = client.query(marginal_query).to_dataframe()
    if len(df_marginal) > sample_n:
        df_marginal = df_marginal.sample(n=sample_n, random_state=42)
    
    print(f"  Marginal: {len(df_marginal):,} members")
    
    def member_stats(df):
        stats_list = []
        for _, row in df.iterrows():
            target_days = parse_target_string(row['target'])
            dt_cnt = min(int(row['dt_cnt']), len(target_days))
            
            all_codes = set()
            all_pairs = set()
            total_codes = 0
            
            for d in range(dt_cnt):
                day_codes = target_days[d] if d < len(target_days) else []
                valid = [c for c in day_codes if 0 < c <= TARGET_CD_CNT]
                all_codes.update(valid)
                total_codes += len(valid)
                
                sorted_v = sorted(set(valid))
                for i in range(len(sorted_v)):
                    for j in range(i + 1, len(sorted_v)):
                        all_pairs.add((sorted_v[i], sorted_v[j]))
            
            stats_list.append({
                'dt_cnt': dt_cnt,
                'unique_codes': len(all_codes),
                'unique_pairs': len(all_pairs),
                'total_occurrences': total_codes,
                'codes_per_day': total_codes / max(dt_cnt, 1),
                'code_density': len(all_codes) / max(dt_cnt, 1),
                'lob': row['lob'],
                'age_bucket': assign_age_bucket(get_index_age(row['age_in_months'])),
            })
        return pd.DataFrame(stats_list)
    
    print("Computing core member stats...")
    core_stats = member_stats(df_core)
    print("Computing marginal member stats...")
    marginal_stats = member_stats(df_marginal)
    
    comparison = {}
    for col in ['dt_cnt', 'unique_codes', 'unique_pairs', 'total_occurrences',
                'codes_per_day', 'code_density']:
        mw_stat, mw_p = stats.mannwhitneyu(
            core_stats[col], marginal_stats[col], alternative='two-sided'
        )
        comparison[col] = {
            'core_mean': float(core_stats[col].mean()),
            'core_median': float(core_stats[col].median()),
            'marginal_mean': float(marginal_stats[col].mean()),
            'marginal_median': float(marginal_stats[col].median()),
            'mannwhitney_p': float(mw_p),
            'effect_direction': 'core > marginal' if core_stats[col].mean() > marginal_stats[col].mean() else 'marginal > core',
        }
    
    # LOB composition
    comparison['lob_composition'] = {
        'core': core_stats['lob'].value_counts(normalize=True).to_dict(),
        'marginal': marginal_stats['lob'].value_counts(normalize=True).to_dict(),
    }
    comparison['age_composition'] = {
        'core': core_stats['age_bucket'].value_counts(normalize=True).to_dict(),
        'marginal': marginal_stats['age_bucket'].value_counts(normalize=True).to_dict(),
    }
    
    return comparison
```

### Step 2: Run and interpret

```python
print("Comparing core vs marginal members...")
core_vs_marginal = compare_core_vs_marginal_members(
    TABLES['full'], core_frac=0.10, min_dt_cnt=5, sample_n=30000
)
results['core_vs_marginal'] = core_vs_marginal

print("\nCORE vs MARGINAL MEMBER COMPARISON")
print(f"{'Metric':<22} {'Core Mean':>12} {'Marginal Mean':>14} {'Direction':>20} {'p-value':>12}")
print("-" * 82)
for col, v in core_vs_marginal.items():
    if isinstance(v, dict) and 'core_mean' in v:
        print(f"{col:<22} {v['core_mean']:>12.2f} {v['marginal_mean']:>14.2f} "
              f"{v['effect_direction']:>20} {v['mannwhitney_p']:>12.2e}")
```

**Pre-registered interpretation:**
- If `core_mean(unique_codes) > marginal_mean(unique_codes)` with p < 0.01 → marginal members ARE less diverse → confirms diminishing returns hypothesis
- If `core_mean(dt_cnt) > marginal_mean(dt_cnt)` → marginal members have shorter histories → temporal saturation mechanism confirmed
- If LOB composition shifts (e.g., more Medicaid in marginal) → scale changes population mix, not just member count
- If no significant differences → the random sampling is truly random and the saturation is due to distributional properties (Zipf), not member selection

### Step 3: Commit

```bash
git add dev/analysis/data_information_saturation_analysis.ipynb
git commit -m "feat: add core vs marginal member comparison analysis"
```

---

## Task 9: Save Results, Generate Report, and Final Synthesis

**Files:**
- Modify: `dev/analysis/data_information_saturation_analysis.ipynb`
- Create: `expe_analysis/exp_round5/data_saturation/data_information_saturation_results.json`
- Create: `expe_analysis/exp_round5/data_saturation/data_information_saturation_report.md`

### Step 1: Save all results to JSON

```python
import copy

# Remove non-serializable arrays
results_serializable = json.loads(json.dumps(results, default=str))

output_path = RESULTS_DIR / 'data_information_saturation_results.json'
with open(output_path, 'w') as f:
    json.dump(results_serializable, f, indent=2)
print(f"Results saved to {output_path}")
```

### Step 2: Generate markdown report

```python
report = []
report.append("# Data Information Saturation Analysis — Results Report\n")
report.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
report.append("**Question:** Does useful learning signal saturate or degrade as "
              "(a) member history length increases and (b) dataset size scales?\n")
report.append("---\n")

report.append("## 1. Within-Member Temporal Saturation\n")
wm = results.get('within_member', {})
report.append(f"- Novelty rate at day 10: **{wm.get('novelty_rate_day_10', 'N/A')}**\n")
report.append(f"- Novelty rate at day 50: **{wm.get('novelty_rate_day_50', 'N/A')}**\n")
report.append(f"- Novelty rate at day 100: **{wm.get('novelty_rate_day_100', 'N/A')}**\n")
report.append(f"- Novelty rate at day 200: **{wm.get('novelty_rate_day_200', 'N/A')}**\n")
report.append("\n**Interpretation:** [Fill after reviewing plots]\n")

report.append("\n## 2. Cross-Scale Distribution Analysis\n")
if 'cross_scale_divergences' in results:
    report.append("| From | To | JS Div | Δ Entropy | Δ Gini |\n")
    report.append("|------|----|---------|-----------|---------|\n")
    for d in results['cross_scale_divergences']:
        report.append(
            f"| {d['from_frac']:.0%} | {d['to_frac']:.0%} | "
            f"{d['js_divergence_bits']:.6f} | "
            f"{d['marginal_entropy_gain_bits']:.4f} | "
            f"{d['gini_delta']:.6f} |\n"
        )

report.append("\n## 3. LOB-Stratified Results\n")
if 'lob_stratified' in results:
    report.append("| LOB | Members | Gini | Entropy | Nov@10 |\n")
    report.append("|-----|---------|------|---------|--------|\n")
    for s, r in sorted(results.get('lob_stratified', {}).items()):
        report.append(
            f"| {s} | {r['n_members']:,} | {r['gini']:.4f} | "
            f"{r['entropy_bits']:.2f} | {r.get('novelty_rate_day_10', 'N/A')} |\n"
        )

report.append("\n## 4. Age-Stratified Results\n")
if 'age_stratified' in results:
    report.append("| Age Bucket | Members | Gini | Entropy | Nov@10 |\n")
    report.append("|------------|---------|------|---------|--------|\n")
    for s, r in sorted(results.get('age_stratified', {}).items()):
        report.append(
            f"| {s} | {r['n_members']:,} | {r['gini']:.4f} | "
            f"{r['entropy_bits']:.2f} | {r.get('novelty_rate_day_10', 'N/A')} |\n"
        )

report.append("\n## 5. Core vs Marginal Members\n")
if 'core_vs_marginal' in results:
    report.append("| Metric | Core Mean | Marginal Mean | Direction | p-value |\n")
    report.append("|--------|-----------|---------------|-----------|--------|\n")
    for col, v in results.get('core_vs_marginal', {}).items():
        if isinstance(v, dict) and 'core_mean' in v:
            report.append(
                f"| {col} | {v['core_mean']:.2f} | {v['marginal_mean']:.2f} | "
                f"{v['effect_direction']} | {v['mannwhitney_p']:.2e} |\n"
            )

report.append("\n## 6. Conditional Information (Mutual Information by Tier)\n")
if 'conditional_information' in results:
    ci = results['conditional_information']
    report.append(f"- Overall mean MI: **{ci['overall_mean_mi']:.6f}** bits\n")
    report.append(f"- Pairs analyzed: **{ci['n_pairs_analyzed']}**\n")

report.append("\n## 7. Conclusions\n")
report.append("\n[To be filled based on evidence above]\n")

report_text = ''.join(report)
report_path = RESULTS_DIR / 'data_information_saturation_report.md'
with open(report_path, 'w') as f:
    f.write(report_text)
print(f"Report saved to {report_path}")
```

### Step 3: Commit all results

```bash
git add dev/analysis/data_information_saturation_analysis.ipynb
git add expe_analysis/exp_round5/data_saturation/
git commit -m "feat: complete data information saturation analysis with report"
```

---

## Pre-Registered Interpretation Summary

| Analysis | Key Decision Threshold | If TRUE → | If FALSE → |
|----------|----------------------|-----------|------------|
| Within-member novelty rate at day 50 | < 0.10 | Individual histories saturate by mid-sequence; later days are mostly redundant | Patients continue encountering diverse codes throughout history |
| Tail-tier novelty decays faster than common | tail novelty@50 < 0.5 × common novelty@50 | Rare conditions front-load, confirming architectural monopolization matters more than data | Rare codes spread across history; data scaling COULD help tails if loss allowed |
| Cross-scale Gini increases | ΔGini > 0 at 11M vs 1.5M | More data concentrates the distribution further — counterproductive | Distribution stays similar; saturation is not from concentration but from finite support |
| Marginal entropy gain collapses | entropy/member at 11M < 10% of rate at 1.5M | Information saturated well before 11M | Dataset still informationally rich at scale |
| Core vs marginal unique codes | core_mean > marginal_mean (p < 0.01) | Marginal members are simpler cases | Random sampling is unbiased; saturation from Zipf, not member selection |
| Cross-tier MI | common-common MI >> rare-rare MI | Common codes carry relational structure; rare codes are isolated | Information uniformly distributed across tiers |
| Co-occurrence pair Gini | > 0.9 | Same-day pair structure is as concentrated as code frequencies | Pairs are more evenly distributed than codes (a form of diversity the model could exploit) |

---

## Estimated Runtime

| Task | Compute | Wall Time |
|------|---------|-----------|
| 1: Setup + data load | BigQuery | ~5 min |
| 2: Within-member saturation (50k) | CPU | ~15 min |
| 3: Co-occurrence diversity (50k) | CPU | ~20 min |
| 4: Cross-scale distributions (9 scales) | BigQuery + CPU | ~45 min |
| 5: Conditional information (30k, 500 codes) | CPU | ~10 min |
| 6: Marginal member curve (100k) | CPU | ~30 min |
| 7: Stratified analysis (LOB + age) | CPU | ~20 min |
| 8: Core vs marginal comparison | BigQuery + CPU | ~15 min |
| 9: Report generation | CPU | ~2 min |
| **Total** | | **~2.5 hours** |
