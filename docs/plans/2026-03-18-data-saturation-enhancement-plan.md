# Data Information Saturation Analysis Enhancement Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enhance the existing data information saturation notebook with five capabilities: (1) member trajectory analysis, (2) raw input code (cd column) integration, (3) front-matter metrics reference, (4) conditional entropy H(X_t | X_{t-1}, ..., X_1), and (5) all-pairs temporal conditional mutual information.

**Architecture:** Edits to existing notebook `dev/downstream/data_information_saturation_analysis.ipynb` (32 cells currently). New sections are inserted at specific cell positions. All new analysis reuses existing utility functions and parsed data structures where possible. Memory-efficient streaming/chunked computation is used throughout to avoid OOM on the ~1.5M member dataset.

**Tech Stack:** Python 3.10+, pandas, numpy, scipy, google-cloud-bigquery, matplotlib, seaborn, collections (Counter/defaultdict). CPU-only.

---

## Notebook Structure After Enhancement

| Cell Range | Section | Status |
|------------|---------|--------|
| 0 | Title + Goal (unchanged) | Existing |
| **1 (new)** | **R0: Metrics Technical Reference** | **NEW** |
| 2 (was 1) | Imports + Configuration | Existing (minor edit: add raw cd vocab size) |
| 3 (was 2) | Data Loading Utilities | Existing (minor edit: ensure parse_cd_string reusable) |
| 4 (was 3) | Load 1.5M + Compute Frequencies | Existing (edit: also compute raw cd frequencies) |
| 5-8 | Task 2: Within-Member Temporal Saturation | Existing (unchanged) |
| **9-11 (new)** | **R1: Member Trajectory Analysis** | **NEW** |
| 12-14 (was 8-10) | Task 3: Co-occurrence (edit: add cd-based co-occurrence) | Modified |
| 15-18 (was 11-15) | Task 4: Cross-Scale Distribution | Existing (unchanged) |
| **19-20 (new)** | **R2.3: Temporal Conditional Entropy H(X_t | past)** | **NEW** |
| **21-22 (new)** | **R2.4: All-Pairs Temporal Conditional MI** | **NEW** |
| 23-25 (was 16-18) | Task 5: MI Analysis (edit: expand to all tier pairs + cd) | Modified |
| 26-31 (was 19-28) | Tasks 6-8 | Existing (unchanged) |
| 32-33 (was 29-31) | Task 9: Save + Report (edit: include new results) | Modified |

**Note on cell numbering:** The plan uses a logical task ordering. In practice, tasks are implemented by editing/inserting cells at the correct positions. Cell indices shift as cells are inserted.

---

## Task 1: Add Metrics Technical Reference Section (Cell 1)

**Files:**
- Modify: `dev/downstream/data_information_saturation_analysis.ipynb` — insert new markdown cell after Cell 0

**Step 1: Insert the metrics reference markdown cell**

Insert a new markdown cell at index 1 with the following content:

```markdown
---
## R0: Technical Reference — Metrics Used in This Analysis

This section defines every metric used in the analysis, its mathematical formulation, why it was chosen, and how to interpret results.

### Shannon Entropy H(X)

**Formula:** $H(X) = -\sum_{x} p(x) \log_2 p(x)$

**What it measures:** The average number of bits needed to encode a draw from distribution $X$. Higher entropy = more uniform/diverse distribution; lower = more concentrated.

**Why chosen:** Entropy is the canonical information-theoretic measure of distributional diversity. Unlike variance, it is invariant to relabeling and naturally handles discrete categorical distributions (code frequencies). It directly quantifies "how much surprise" each observation carries.

**Interpretation:** For $N$ non-zero codes, $H_{max} = \log_2(N)$ (uniform). If $H \ll H_{max}$, the distribution is heavily concentrated on a few codes. In this analysis, flat entropy across data scales means the distributional shape is scale-invariant.

---

### Gini Coefficient

**Formula:** $G = \frac{\sum_{i=1}^{n} \sum_{j=1}^{n} |x_i - x_j|}{2n \sum_{i=1}^{n} x_i}$ (equivalently, from sorted values: $G = \frac{2 \sum_{i=1}^{n} i \cdot x_{(i)}}{n \sum x_{(i)}} - \frac{n+1}{n}$)

**What it measures:** Inequality in a non-negative distribution. $G=0$ means perfect equality (all codes equally frequent); $G=1$ means maximal inequality (one code dominates).

**Why chosen:** Gini is more sensitive to concentration in heavy-tailed distributions than entropy. Two distributions can have similar entropy but very different Gini coefficients if the tail shapes differ. Clinical code frequencies follow a power-law, making Gini the right complement to entropy.

**Interpretation:** $G > 0.9$ indicates extreme concentration — a small fraction of codes account for the vast majority of occurrences. Increasing Gini with scale means more data amplifies existing inequalities.

---

### Novelty Rate (Within-Member)

**Formula:** $\text{Novelty}(d) = \frac{|\{c \in \text{codes}(d) : c \notin \text{seen}(1..d{-}1)\}|}{|\text{codes}(d)|}$

**What it measures:** At day position $d$ in a member's history, what fraction of codes are genuinely new (never seen in days 1 through $d-1$)?

**Why chosen:** Directly answers "does later history add information?" for the transformer, which processes all 200 days via attention. If novelty drops to near-zero, later days are informationally redundant.

**Interpretation:** Novelty rate of 0.10 at day 50 means 90% of codes on day 50 are repeats of earlier days. The rate of decay indicates how quickly within-member information exhausts.

---

### Mutual Information I(A; B)

**Formula:** $I(A;B) = \sum_{a,b} p(a,b) \log_2 \frac{p(a,b)}{p(a) p(b)}$

**What it measures:** How much knowing code $A$'s presence/absence tells you about code $B$'s presence/absence. $I=0$ means independence; higher values mean stronger statistical association.

**Why chosen:** MI quantifies the relational structure between codes that the transformer's attention mechanism should exploit. If MI is near-zero between code pairs, codes are approximately conditionally independent — meaning a shared encoder has little relational structure to learn beyond marginal frequencies.

**Interpretation:** For binary variables (present/absent), max MI is ~1 bit (perfect correlation). Values $< 0.01$ bits indicate near-independence. The median MI across tier pairs indicates the "typical" relational signal available to the model.

---

### Conditional Entropy H(X_t | X_{t-1}, ..., X_1)

**Formula:** $H(X_t | X_{<t}) = H(X_1, ..., X_t) - H(X_1, ..., X_{t-1})$

**What it measures:** How much *new* information day $t$'s codes carry, given full knowledge of all previous days. This is the irreducible surprise at position $t$ — the information the model can extract from temporal sequencing.

**Why chosen:** Unlike novelty rate (which only measures set membership), conditional entropy captures the full distributional uncertainty reduction from temporal context. A code that appears 50% of the time on day 1 but 100% of the time after seeing the first 10 days has zero conditional entropy despite being "not novel."

**Interpretation:** Rapidly declining $H(X_t | X_{<t})$ means the temporal sequence becomes highly predictable early — the model learns less from later time steps. Flat conditional entropy means each day carries equal new information (ideal for sequence models).

---

### Jensen-Shannon Divergence JSD(P || Q)

**Formula:** $JSD(P \| Q) = \frac{1}{2} KL(P \| M) + \frac{1}{2} KL(Q \| M)$, where $M = \frac{P + Q}{2}$

**What it measures:** A symmetric, bounded measure of how different two probability distributions are. $JSD = 0$ means identical; $JSD = 1$ (in bits, base-2) means maximally different.

**Why chosen:** KL divergence is asymmetric and can be infinite. JSD is symmetric, always finite, and its square root is a proper metric. Used to compare code distributions across data scales.

**Interpretation:** $JSD < 0.001$ means distributions are effectively indistinguishable. In this analysis, JSD between adjacent data scales approaching zero proves the distribution shape is fixed regardless of dataset size.

---

### Co-occurrence Pair Entropy

**Formula:** Shannon entropy computed over the distribution of co-occurrence pair frequencies.

**What it measures:** How evenly distributed the co-occurrence signal is. High pair entropy = many diverse code-code relationships; low = signal concentrated on a few dominant pairs.

**Why chosen:** The transformer learns code relationships through attention. Pair entropy tells us whether the relational gradient signal is diverse (useful for learning rich representations) or concentrated (the model memorizes a few dominant co-occurrences).

**Interpretation:** Compare to $H_{max} = \log_2(\text{unique pairs})$. Ratio indicates effective relational diversity. Combined with pair Gini, identifies whether the model's attention is learning a broad or narrow set of relationships.

---

### Temporal Conditional Mutual Information I(A_t; B_t | history)

**Formula:** $I(A_t; B_t | C) = H(A_t | C) + H(B_t | C) - H(A_t, B_t | C)$, where $C$ represents the shared temporal history.

**What it measures:** Whether two codes on the same day share information *beyond* what the temporal history already predicts. This isolates the "fresh" relational signal at each time step.

**Why chosen:** Standard MI between code pairs conflates temporal autocorrelation with genuine cross-code dependence. Conditional MI removes the predictable component, revealing whether the model can learn something about code B from code A that it couldn't learn from the history alone.

**Interpretation:** If conditional MI $\approx$ unconditional MI, the history provides no shared context. If conditional MI $\ll$ unconditional MI, most apparent code association is explained by temporal patterns (chronic disease trajectories), not genuine code-code interaction.

---

### Trajectory Complexity Metrics

**Transition Entropy:** $H_{\text{trans}} = -\sum_{s'|s} p(s'|s) \log_2 p(s'|s)$ averaged over states $s$

**What it measures:** How predictable the next day's code set is given the current day's code set. High transition entropy = diverse trajectories; low = repetitive.

**Code Velocity:** $v(d) = |\text{codes}(d) \setminus \text{codes}(d{-}1)| + |\text{codes}(d{-}1) \setminus \text{codes}(d)|$

**What it measures:** The symmetric difference between consecutive days' code sets — how much the clinical picture changes day-to-day.

**Trajectory Entropy Rate:** $h = \lim_{d \to \infty} H(X_d | X_{d-1}, ..., X_1)$

**What it measures:** The asymptotic per-step information rate of the member's clinical trajectory. This is the fundamental limit on what a sequence model can learn per additional time step.
```

**Step 2: Verify the cell was inserted correctly**

Run: read cell 1 of the notebook and confirm it starts with "## R0: Technical Reference"

**Step 3: Commit**

```bash
git add dev/downstream/data_information_saturation_analysis.ipynb
git commit -m "feat: add technical metrics reference section R0 to saturation analysis"
```

---

## Task 2: Add Raw Code (cd column) Support to Configuration and Utilities

**Files:**
- Modify: `dev/downstream/data_information_saturation_analysis.ipynb` — edit Cells 1→2 (imports/config) and Cell 2→3 (utilities)

**Step 1: Add raw cd vocabulary size constant to configuration cell**

In the configuration cell (currently cell 1, will be cell 2 after Task 1), add after `LEN_CD = 80`:

```python
RAW_CD_VOCAB = 84_000  # approximate raw input code vocabulary size
```

**Step 2: Add `compute_cd_frequencies` utility to the utilities cell**

In the utilities cell (currently cell 2, will be cell 3), add after the existing `compute_code_frequencies` function:

```python
def compute_cd_frequencies(df, top_n=10000):
    """Compute raw input code (cd) frequency array.
    Returns (freq_counter, vocab_size) where freq_counter is a Counter
    of {code_int: count}. Uses Counter instead of dense array because
    the cd vocabulary (~84k) is sparse — most codes have zero frequency.
    """
    freq = Counter()
    for cd_str in df['cd']:
        if not cd_str or pd.isna(cd_str):
            continue
        for day_str in cd_str.split('*')[:LEN_DY]:
            if not day_str:
                continue
            for c_str in day_str.split(','):
                try:
                    v = int(c_str)
                    if v > 0:
                        freq[v] += 1
                except (ValueError, TypeError):
                    pass
    return freq


def compute_cd_tier_boundaries(cd_freq_counter, n_tiers=4):
    """Compute tier boundaries for raw cd codes using the same
    percentile approach as target codes."""
    freqs = np.array(list(cd_freq_counter.values()))
    if len(freqs) == 0:
        return {'p80': 0, 'p50': 0, 'p20': 0}
    percentiles = np.percentile(freqs, [20, 50, 80])
    return {'p80': percentiles[2], 'p50': percentiles[1], 'p20': percentiles[0]}


def assign_cd_tier(code, cd_freq_counter, cd_tier_bounds):
    """Assign a raw cd code to its frequency tier."""
    freq = cd_freq_counter.get(code, 0)
    if freq == 0:
        return 'zero'
    if freq > cd_tier_bounds['p80']:
        return 'common'
    elif freq > cd_tier_bounds['p50']:
        return 'medium'
    elif freq > cd_tier_bounds['p20']:
        return 'rare'
    return 'tail'
```

**Step 3: Add cd frequency computation to the data loading cell**

In the cell that loads 1.5M and computes frequencies (currently cell 3, will be cell 4), add after the target frequency computation:

```python
print("\nComputing raw cd code frequencies (streaming Counter)...")
cd_freq = compute_cd_frequencies(df_1_5m)
cd_tier_bounds = compute_cd_tier_boundaries(cd_freq)
cd_vocab_size = len(cd_freq)
print(f"Raw cd vocabulary: {cd_vocab_size:,} unique codes")
print(f"Raw cd tier boundaries: {cd_tier_bounds}")
print(f"Total cd occurrences: {sum(cd_freq.values()):,}")
```

**Step 4: Verify no errors**

Read the edited cells and confirm they are syntactically correct.

**Step 5: Commit**

```bash
git add dev/downstream/data_information_saturation_analysis.ipynb
git commit -m "feat: add raw cd code frequency computation and tier utilities"
```

---

## Task 3: Integrate Raw Codes into Co-occurrence Analysis (Cell 9)

**Files:**
- Modify: `dev/downstream/data_information_saturation_analysis.ipynb` — edit the `compute_cooccurrence_diversity` function cell and its invocation cell

**Step 1: Extend `compute_cooccurrence_diversity` to support both target and cd columns**

Replace the function signature and add cd-based pair tracking. The key design: a single pass through members computes both target and cd co-occurrences simultaneously. For cd codes, use a bounded pair generation (top-20 codes per day) to prevent combinatorial explosion with the larger vocabulary.

In the `compute_cooccurrence_diversity` function cell (currently cell 9), replace the entire function with:

```python
def compute_cooccurrence_diversity(df, code_freq, tier_bounds,
                                   cd_freq_counter=None, cd_tier_bounds=None,
                                   sample_n=50000):
    """
    Compute same-day co-occurrence pair statistics and temporal skip-gram diversity.
    When cd_freq_counter is provided, also computes co-occurrence stats for raw
    input codes (cd column) in the same pass to avoid redundant iteration.
    """
    if sample_n and len(df) > sample_n:
        df_sample = df.sample(n=sample_n, random_state=42)
    else:
        df_sample = df

    same_day_pairs = Counter()
    temporal_bigrams = Counter()
    temporal_skipgrams_2 = Counter()
    temporal_skipgrams_3 = Counter()

    lob_pair_count = defaultdict(int)
    age_pair_count = defaultdict(int)

    code_tier_map = {}
    for i in range(1, TARGET_CD_CNT + 1):
        code_tier_map[i] = assign_code_tier(i, code_freq, tier_bounds)
    tier_pair_unique = defaultdict(set)

    include_cd = cd_freq_counter is not None and cd_tier_bounds is not None
    cd_same_day_pairs = Counter() if include_cd else None
    cd_temporal_bigrams = Counter() if include_cd else None
    cd_tier_pair_unique = defaultdict(set) if include_cd else None

    t0 = time.time()
    processed = 0

    for _, row in df_sample.iterrows():
        target_days = parse_target_string(row['target'])
        dt_cnt = min(int(row['dt_cnt']), len(target_days))
        lob = row['lob']
        age_bucket = assign_age_bucket(get_index_age(row['age_in_months']))

        cd_days = parse_cd_string(row['cd']) if include_cd else None

        for d in range(dt_cnt):
            # --- Target code co-occurrence (existing logic) ---
            day_codes = sorted(set(target_days[d])) if d < len(target_days) else []
            if day_codes:
                n_dc = len(day_codes)
                for i in range(min(n_dc, LEN_CD)):
                    for j in range(i + 1, min(n_dc, LEN_CD)):
                        pair = (day_codes[i], day_codes[j])
                        same_day_pairs[pair] += 1
                        lob_pair_count[lob] += 1
                        age_pair_count[age_bucket] += 1

                        tier_a = code_tier_map.get(day_codes[i], 'unknown')
                        tier_b = code_tier_map.get(day_codes[j], 'unknown')
                        tier_key = tuple(sorted([tier_a, tier_b]))
                        tier_pair_unique[tier_key].add(pair)

                day_sample = day_codes[:10]
                for skip, counter in [(1, temporal_bigrams),
                                      (2, temporal_skipgrams_2),
                                      (3, temporal_skipgrams_3)]:
                    if d + skip < dt_cnt and d + skip < len(target_days):
                        next_codes = target_days[d + skip][:10]
                        for a in day_sample:
                            for b in next_codes:
                                if a > 0 and b > 0:
                                    counter[(a, b)] += 1

            # --- Raw cd co-occurrence (new, bounded) ---
            if include_cd and cd_days and d < len(cd_days):
                cd_day = sorted(set(cd_days[d]))[:30]
                n_cd = len(cd_day)
                for i in range(min(n_cd, 20)):
                    for j in range(i + 1, min(n_cd, 20)):
                        cd_same_day_pairs[(cd_day[i], cd_day[j])] += 1
                        tier_a = assign_cd_tier(cd_day[i], cd_freq_counter, cd_tier_bounds)
                        tier_b = assign_cd_tier(cd_day[j], cd_freq_counter, cd_tier_bounds)
                        cd_tier_pair_unique[tuple(sorted([tier_a, tier_b]))].add(
                            (cd_day[i], cd_day[j]))

                if d + 1 < dt_cnt and d + 1 < len(cd_days):
                    cd_src = cd_day[:10]
                    cd_tgt = sorted(set(cd_days[d + 1]))[:10]
                    for a in cd_src:
                        for b in cd_tgt:
                            if a > 0 and b > 0:
                                cd_temporal_bigrams[(a, b)] += 1

        processed += 1
        if processed % 10000 == 0:
            print(f"  Processed {processed:,} members ({time.time()-t0:.0f}s)")

    print(f"  Target: {len(same_day_pairs):,} unique same-day pairs")
    print(f"  Target: {len(temporal_bigrams):,} unique temporal bigrams")
    if include_cd:
        print(f"  Raw cd: {len(cd_same_day_pairs):,} unique same-day pairs")
        print(f"  Raw cd: {len(cd_temporal_bigrams):,} unique temporal bigrams")
    print(f"  Elapsed: {time.time()-t0:.0f}s")

    def distribution_entropy(counter):
        total = sum(counter.values())
        if total == 0:
            return 0.0
        probs = np.array(list(counter.values()), dtype=np.float64) / total
        return float(stats.entropy(probs, base=2))

    result = {
        'n_unique_same_day_pairs': len(same_day_pairs),
        'total_same_day_pair_occurrences': sum(same_day_pairs.values()),
        'same_day_pair_entropy_bits': distribution_entropy(same_day_pairs),
        'same_day_pair_gini': _gini(np.fromiter(
            same_day_pairs.values(), dtype=np.int64, count=len(same_day_pairs)
        )) if same_day_pairs else 0,
        'n_unique_temporal_bigrams': len(temporal_bigrams),
        'temporal_bigram_entropy_bits': distribution_entropy(temporal_bigrams),
        'n_unique_skip2grams': len(temporal_skipgrams_2),
        'skip2gram_entropy_bits': distribution_entropy(temporal_skipgrams_2),
        'n_unique_skip3grams': len(temporal_skipgrams_3),
        'skip3gram_entropy_bits': distribution_entropy(temporal_skipgrams_3),
        'tier_pair_diversity': {str(k): len(v) for k, v in tier_pair_unique.items()},
        'lob_pair_total_occurrences': dict(lob_pair_count),
        'age_pair_total_occurrences': dict(age_pair_count),
        'same_day_pair_concentration': {
            'top_10_pct_share': float(
                sum(sorted(same_day_pairs.values(), reverse=True)[:max(1, len(same_day_pairs) // 10)])
                / max(1, sum(same_day_pairs.values()))
            ),
            'top_1_pct_share': float(
                sum(sorted(same_day_pairs.values(), reverse=True)[:max(1, len(same_day_pairs) // 100)])
                / max(1, sum(same_day_pairs.values()))
            ),
        },
    }

    if include_cd:
        result['cd_cooccurrence'] = {
            'n_unique_same_day_pairs': len(cd_same_day_pairs),
            'total_same_day_pair_occurrences': sum(cd_same_day_pairs.values()),
            'same_day_pair_entropy_bits': distribution_entropy(cd_same_day_pairs),
            'same_day_pair_gini': _gini(np.fromiter(
                cd_same_day_pairs.values(), dtype=np.int64, count=len(cd_same_day_pairs)
            )) if cd_same_day_pairs else 0,
            'n_unique_temporal_bigrams': len(cd_temporal_bigrams),
            'temporal_bigram_entropy_bits': distribution_entropy(cd_temporal_bigrams),
            'tier_pair_diversity': {str(k): len(v) for k, v in cd_tier_pair_unique.items()},
            'cd_vocab_size': len(cd_freq_counter) if cd_freq_counter else 0,
        }

    return result
```

**Step 2: Update the invocation cell to pass cd parameters**

In the co-occurrence invocation cell (currently cell 10), replace with:

```python
print("Computing co-occurrence diversity on 1.5M (50k sample, target + raw cd)...")
cooccurrence_results_1_5m = compute_cooccurrence_diversity(
    df_1_5m, code_freq, tier_bounds,
    cd_freq_counter=cd_freq, cd_tier_bounds=cd_tier_bounds,
    sample_n=50000
)
results['cooccurrence_1_5m'] = cooccurrence_results_1_5m
print(json.dumps({k: v for k, v in cooccurrence_results_1_5m.items()
                  if k != 'cd_cooccurrence'}, indent=2, default=str))
if 'cd_cooccurrence' in cooccurrence_results_1_5m:
    print("\n--- Raw cd co-occurrence ---")
    print(json.dumps(cooccurrence_results_1_5m['cd_cooccurrence'], indent=2, default=str))
```

**Step 3: Verify edits**

Read the edited cells and confirm correctness.

**Step 4: Commit**

```bash
git add dev/downstream/data_information_saturation_analysis.ipynb
git commit -m "feat: integrate raw cd code co-occurrence analysis into Task 3"
```

---

## Task 4: Add Member Trajectory Analysis Section

**Files:**
- Modify: `dev/downstream/data_information_saturation_analysis.ipynb` — insert 3 new cells after the existing within-member saturation plots (after current cell 7)

**Step 1: Insert trajectory analysis markdown header**

Insert a new markdown cell after current cell 7 (within-member plots):

```markdown
---
## R1: Member Trajectory Analysis

Goes beyond point-in-time snapshots to characterize the *dynamics* of individual member clinical trajectories. Answers: How predictable are member trajectories? How much does the clinical picture change day-to-day? Do different trajectory types exist?
```

**Step 2: Insert trajectory computation cell**

Insert a code cell with the trajectory analysis function:

```python
def compute_member_trajectory_analysis(df, code_freq, tier_bounds,
                                       sample_n=20000, use_cd=False):
    """
    Analyze member-level trajectory dynamics:
    - Code velocity: symmetric difference between consecutive days
    - Trajectory entropy rate: H(X_d | X_{d-1}) estimated via transition counts
    - Persistence score: fraction of codes that persist across consecutive days
    - Trajectory type classification: stable/volatile/monotone based on velocity
    
    Operates on target codes by default; set use_cd=True for raw cd analysis.
    Memory-efficient: processes one member at a time, accumulates lightweight aggregates.
    """
    if sample_n and len(df) > sample_n:
        df_sample = df.sample(n=sample_n, random_state=42)
    else:
        df_sample = df

    parse_fn = parse_cd_string if use_cd else parse_target_string
    col = 'cd' if use_cd else 'target'
    label = 'cd' if use_cd else 'target'

    velocity_by_day = defaultdict(list)
    persistence_by_day = defaultdict(list)
    transition_counts = Counter()
    member_summaries = []

    t0 = time.time()
    processed = 0

    for _, row in df_sample.iterrows():
        days = parse_fn(row[col])
        dt_cnt = min(int(row['dt_cnt']), len(days))
        if dt_cnt < 5:
            continue

        lob = row['lob']
        age_bucket = assign_age_bucket(get_index_age(row['age_in_months']))

        velocities = []
        persistences = []
        prev_set = frozenset()

        for d in range(dt_cnt):
            curr_codes = days[d] if d < len(days) else []
            curr_set = frozenset(c for c in curr_codes if c > 0)

            if d > 0 and (prev_set or curr_set):
                added = len(curr_set - prev_set)
                removed = len(prev_set - curr_set)
                velocity = added + removed
                union_size = len(prev_set | curr_set)
                jaccard = len(prev_set & curr_set) / union_size if union_size > 0 else 1.0

                velocities.append(velocity)
                persistences.append(jaccard)
                velocity_by_day[d].append(velocity)
                persistence_by_day[d].append(jaccard)

                prev_hash = hash(prev_set) % 10000
                curr_hash = hash(curr_set) % 10000
                transition_counts[(prev_hash, curr_hash)] += 1

            prev_set = curr_set

        if velocities:
            mean_v = np.mean(velocities)
            std_v = np.std(velocities)
            mean_p = np.mean(persistences)

            if mean_v < 1.0:
                traj_type = 'stable'
            elif std_v / (mean_v + 1e-8) > 1.0:
                traj_type = 'volatile'
            elif mean_p > 0.8:
                traj_type = 'persistent'
            else:
                traj_type = 'dynamic'

            member_summaries.append({
                'lob': lob, 'age_bucket': age_bucket,
                'dt_cnt': dt_cnt,
                'mean_velocity': mean_v, 'std_velocity': std_v,
                'mean_persistence': mean_p,
                'trajectory_type': traj_type,
                'code_type': label,
            })

        processed += 1
        if processed % 5000 == 0:
            print(f"  [{label}] Processed {processed:,} ({time.time()-t0:.0f}s)")

    # Transition entropy
    total_trans = sum(transition_counts.values())
    trans_probs = np.array(list(transition_counts.values()), dtype=np.float64) / total_trans
    transition_entropy = float(stats.entropy(trans_probs, base=2))

    # Velocity by day position (mean curve)
    day_positions = sorted(velocity_by_day.keys())
    velocity_curve = [(d, np.mean(velocity_by_day[d]), np.std(velocity_by_day[d]))
                      for d in day_positions if len(velocity_by_day[d]) >= 50]
    persistence_curve = [(d, np.mean(persistence_by_day[d]), np.std(persistence_by_day[d]))
                         for d in day_positions if len(persistence_by_day[d]) >= 50]

    summary_df = pd.DataFrame(member_summaries)

    print(f"  [{label}] Done: {processed:,} members, "
          f"transition entropy={transition_entropy:.2f} bits ({time.time()-t0:.0f}s)")

    return {
        'transition_entropy_bits': transition_entropy,
        'n_unique_transitions': len(transition_counts),
        'velocity_curve': velocity_curve,
        'persistence_curve': persistence_curve,
        'summary_df': summary_df,
        'code_type': label,
    }
```

**Step 3: Insert trajectory execution and visualization cell**

```python
print("=" * 60)
print("MEMBER TRAJECTORY ANALYSIS")
print("=" * 60)

traj_target = compute_member_trajectory_analysis(
    df_1_5m, code_freq, tier_bounds, sample_n=20000, use_cd=False)
traj_cd = compute_member_trajectory_analysis(
    df_1_5m, code_freq, tier_bounds, sample_n=20000, use_cd=True)

fig, axes = plt.subplots(2, 4, figsize=(24, 10))
fig.suptitle('Member Trajectory Analysis', fontsize=14)

for col_offset, traj, label in [(0, traj_target, 'Target'), (2, traj_cd, 'Raw cd')]:
    sdf = traj['summary_df']

    # Velocity curve
    ax = axes[0, col_offset]
    vc = traj['velocity_curve']
    if vc:
        days_v, means_v, stds_v = zip(*vc)
        ax.plot(days_v, means_v, label='Mean velocity')
        ax.fill_between(days_v,
                        np.array(means_v) - np.array(stds_v),
                        np.array(means_v) + np.array(stds_v), alpha=0.2)
    ax.set_xlabel('Day position')
    ax.set_ylabel('Code velocity (sym. diff)')
    ax.set_title(f'{label}: Day-to-day code velocity')

    # Persistence curve
    ax = axes[0, col_offset + 1]
    pc = traj['persistence_curve']
    if pc:
        days_p, means_p, stds_p = zip(*pc)
        ax.plot(days_p, means_p, label='Mean Jaccard')
        ax.fill_between(days_p,
                        np.array(means_p) - np.array(stds_p),
                        np.array(means_p) + np.array(stds_p), alpha=0.2)
    ax.set_xlabel('Day position')
    ax.set_ylabel('Day-to-day Jaccard similarity')
    ax.set_title(f'{label}: Code persistence')

    # Trajectory type distribution
    ax = axes[1, col_offset]
    if len(sdf) > 0:
        type_counts = sdf['trajectory_type'].value_counts()
        type_counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'])
    ax.set_title(f'{label}: Trajectory type distribution')
    ax.set_ylabel('Count')

    # Velocity by LOB
    ax = axes[1, col_offset + 1]
    if len(sdf) > 0:
        for lob in ['Commercial', 'Medicare', 'Medicaid']:
            lob_df = sdf[sdf['lob'] == lob]
            if len(lob_df) > 0:
                ax.hist(lob_df['mean_velocity'], bins=50, alpha=0.5, label=lob, density=True)
    ax.set_xlabel('Mean code velocity')
    ax.set_ylabel('Density')
    ax.set_title(f'{label}: Velocity distribution by LOB')
    ax.legend()

plt.tight_layout()
plt.savefig(str(RESULTS_DIR / 'member_trajectory_analysis.png'), dpi=150, bbox_inches='tight')
plt.show()

# Save to results
for traj, key in [(traj_target, 'trajectory_target'), (traj_cd, 'trajectory_cd')]:
    sdf = traj['summary_df']
    results[key] = {
        'transition_entropy_bits': traj['transition_entropy_bits'],
        'n_unique_transitions': traj['n_unique_transitions'],
        'mean_velocity': float(sdf['mean_velocity'].mean()) if len(sdf) > 0 else 0,
        'mean_persistence': float(sdf['mean_persistence'].mean()) if len(sdf) > 0 else 0,
        'trajectory_type_distribution': sdf['trajectory_type'].value_counts().to_dict() if len(sdf) > 0 else {},
        'by_lob': {
            lob: {
                'mean_velocity': float(g['mean_velocity'].mean()),
                'mean_persistence': float(g['mean_persistence'].mean()),
                'n': len(g),
            }
            for lob, g in sdf.groupby('lob') if len(g) > 0
        } if len(sdf) > 0 else {},
    }

print("\nTrajectory Summary (Target):")
print(json.dumps({k: v for k, v in results['trajectory_target'].items()
                  if k != 'by_lob'}, indent=2, default=str))
print("\nTrajectory Summary (Raw cd):")
print(json.dumps({k: v for k, v in results['trajectory_cd'].items()
                  if k != 'by_lob'}, indent=2, default=str))
```

**Step 4: Verify the cells render correctly**

Read the inserted cells and confirm syntax.

**Step 5: Commit**

```bash
git add dev/downstream/data_information_saturation_analysis.ipynb
git commit -m "feat: add member trajectory analysis section R1 (target + raw cd)"
```

---

## Task 5: Add Temporal Conditional Entropy H(X_t | X_{<t}) Section

**Files:**
- Modify: `dev/downstream/data_information_saturation_analysis.ipynb` — insert 2 new cells before the existing MI analysis section

This is the R2.3 enhancement. It computes how much new information day `t` carries given the full prior history, using a hash-based state representation to keep memory bounded.

**Step 1: Insert conditional entropy markdown header**

Insert a markdown cell before the existing Task 5 (MI analysis) section:

```markdown
---
## R2.3: Temporal Conditional Entropy H(X_t | X_{t-1}, ..., X_1)

Measures how much *new* information each day position contributes given full knowledge of all prior days. This determines how much the model can actually learn from temporal sequences — the fundamental limit on per-step learning signal.

Unlike novelty rate (which measures set membership), conditional entropy captures the full distributional uncertainty reduction. A code that was uncertain on day 1 but becomes certain by day 10 has low conditional entropy even though it's "not novel."
```

**Step 2: Insert conditional entropy computation and visualization cell**

```python
def compute_temporal_conditional_entropy(df, sample_n=15000, max_days=100,
                                         use_cd=False, hash_bins=5000):
    """
    Estimate H(X_t | X_{<t}) for each day position t using discretized state
    representations.
    
    Approach: For each member, represent the "state at day d" as a hash of the
    cumulative code set up to day d. At each day position, build a conditional
    frequency table: P(today's codes | state). Then compute:
        H(X_t | X_{<t}) = H(X_1, ..., X_t) - H(X_1, ..., X_{t-1})
    
    The hash_bins parameter controls the state space discretization — higher
    values give more precise estimates but require more memory.
    
    Memory: O(max_days * hash_bins * avg_codes_per_day) — bounded and predictable.
    """
    if sample_n and len(df) > sample_n:
        df_sample = df.sample(n=sample_n, random_state=42)
    else:
        df_sample = df

    parse_fn = parse_cd_string if use_cd else parse_target_string
    col = 'cd' if use_cd else 'target'
    label = 'cd' if use_cd else 'target'

    # state_code_counts[d][(state_hash, code)] = count
    # We accumulate joint counts of (state, next_code) at each day position
    joint_counts = [Counter() for _ in range(max_days)]
    state_counts = [Counter() for _ in range(max_days)]
    marginal_counts = [Counter() for _ in range(max_days)]
    n_members_at_day = np.zeros(max_days, dtype=np.int64)

    t0 = time.time()
    processed = 0

    for _, row in df_sample.iterrows():
        days = parse_fn(row[col])
        dt_cnt = min(int(row['dt_cnt']), len(days), max_days)
        if dt_cnt < 3:
            continue

        cumul_codes = frozenset()
        for d in range(dt_cnt):
            curr_codes = tuple(sorted(set(c for c in (days[d] if d < len(days) else []) if c > 0)))
            if not curr_codes:
                cumul_codes = cumul_codes  # no update
                continue

            state_hash = hash(cumul_codes) % hash_bins
            code_hash = hash(curr_codes) % hash_bins

            joint_counts[d][(state_hash, code_hash)] += 1
            state_counts[d][state_hash] += 1
            marginal_counts[d][code_hash] += 1
            n_members_at_day[d] += 1

            cumul_codes = frozenset(cumul_codes | set(curr_codes))

        processed += 1
        if processed % 5000 == 0:
            print(f"  [{label}] {processed:,} members ({time.time()-t0:.0f}s)")

    # Compute conditional entropy at each day position
    cond_entropy = []
    for d in range(max_days):
        n = n_members_at_day[d]
        if n < 100:
            continue
        total = sum(joint_counts[d].values())
        if total == 0:
            continue

        # H(X_t | State_{t-1}) = H(X_t, State_{t-1}) - H(State_{t-1})
        joint_probs = np.array(list(joint_counts[d].values()), dtype=np.float64) / total
        h_joint = float(stats.entropy(joint_probs, base=2))

        state_probs = np.array(list(state_counts[d].values()), dtype=np.float64) / total
        h_state = float(stats.entropy(state_probs, base=2))

        h_cond = max(0.0, h_joint - h_state)

        marginal_probs = np.array(list(marginal_counts[d].values()), dtype=np.float64) / total
        h_marginal = float(stats.entropy(marginal_probs, base=2))

        cond_entropy.append({
            'day': d,
            'H_cond': h_cond,
            'H_marginal': h_marginal,
            'reduction_ratio': 1.0 - (h_cond / h_marginal) if h_marginal > 0 else 0.0,
            'n_members': int(n),
        })

    print(f"  [{label}] Done: {len(cond_entropy)} day positions ({time.time()-t0:.0f}s)")
    return cond_entropy


# Compute for both target and cd
print("Computing temporal conditional entropy...")
cond_ent_target = compute_temporal_conditional_entropy(
    df_1_5m, sample_n=15000, max_days=100, use_cd=False)
cond_ent_cd = compute_temporal_conditional_entropy(
    df_1_5m, sample_n=15000, max_days=100, use_cd=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Temporal Conditional Entropy H(X_t | X_{<t})', fontsize=14)

for cond_ent, label, color in [(cond_ent_target, 'Target', '#3498db'),
                                (cond_ent_cd, 'Raw cd', '#e74c3c')]:
    if not cond_ent:
        continue
    days = [r['day'] for r in cond_ent]
    h_cond = [r['H_cond'] for r in cond_ent]
    h_marg = [r['H_marginal'] for r in cond_ent]
    reduction = [r['reduction_ratio'] for r in cond_ent]

    axes[0].plot(days, h_cond, label=f'{label} H(X_t|past)', color=color)
    axes[0].plot(days, h_marg, label=f'{label} H(X_t)', color=color, linestyle='--', alpha=0.5)
    axes[1].plot(days, reduction, label=label, color=color)

axes[0].set_xlabel('Day position')
axes[0].set_ylabel('Entropy (bits)')
axes[0].set_title('Conditional vs marginal entropy')
axes[0].legend()
axes[1].set_xlabel('Day position')
axes[1].set_ylabel('Reduction ratio (1 - H_cond/H_marg)')
axes[1].set_title('Information reduction from temporal context')
axes[1].legend()

# Entropy rate estimation
if cond_ent_target:
    late_days = [r for r in cond_ent_target if r['day'] >= 50]
    if late_days:
        entropy_rate = np.mean([r['H_cond'] for r in late_days])
        axes[2].axhline(y=entropy_rate, color='#3498db', linestyle='--',
                        label=f'Target rate: {entropy_rate:.3f}')
if cond_ent_cd:
    late_days_cd = [r for r in cond_ent_cd if r['day'] >= 50]
    if late_days_cd:
        entropy_rate_cd = np.mean([r['H_cond'] for r in late_days_cd])
        axes[2].axhline(y=entropy_rate_cd, color='#e74c3c', linestyle='--',
                        label=f'cd rate: {entropy_rate_cd:.3f}')
    days_cd = [r['day'] for r in cond_ent_cd]
    h_cond_cd = [r['H_cond'] for r in cond_ent_cd]
    axes[2].plot(days_cd, h_cond_cd, color='#e74c3c', alpha=0.3)
if cond_ent_target:
    days_t = [r['day'] for r in cond_ent_target]
    h_cond_t = [r['H_cond'] for r in cond_ent_target]
    axes[2].plot(days_t, h_cond_t, color='#3498db', alpha=0.3)
axes[2].set_xlabel('Day position')
axes[2].set_ylabel('H(X_t | past) bits')
axes[2].set_title('Entropy rate convergence')
axes[2].legend()

plt.tight_layout()
plt.savefig(str(RESULTS_DIR / 'temporal_conditional_entropy.png'), dpi=150, bbox_inches='tight')
plt.show()

results['conditional_entropy_target'] = cond_ent_target
results['conditional_entropy_cd'] = cond_ent_cd
print(f"Target: {len(cond_ent_target)} day positions computed")
print(f"Raw cd: {len(cond_ent_cd)} day positions computed")
if cond_ent_target:
    print(f"Target entropy rate (days 50+): {np.mean([r['H_cond'] for r in cond_ent_target if r['day']>=50]):.4f} bits")
if cond_ent_cd:
    print(f"Raw cd entropy rate (days 50+): {np.mean([r['H_cond'] for r in cond_ent_cd if r['day']>=50]):.4f} bits")
```

**Step 3: Verify cells**

Read the inserted cells.

**Step 4: Commit**

```bash
git add dev/downstream/data_information_saturation_analysis.ipynb
git commit -m "feat: add temporal conditional entropy R2.3 for target and raw cd"
```

---

## Task 6: Expand Mutual Information to All Tier Pairs + Add Temporal Conditional MI

**Files:**
- Modify: `dev/downstream/data_information_saturation_analysis.ipynb` — (a) insert temporal conditional MI section, (b) fix existing MI function to analyze all tier pairs and add cd support

### Part A: Add All-Pairs Temporal Conditional MI Section

**Step 1: Insert R2.4 markdown header**

Insert a markdown cell after the new R2.3 cells:

```markdown
---
## R2.4: Temporal Conditional Mutual Information — All Tier Pairs

Standard MI conflates chronic disease autocorrelation with genuine code-code interaction. Temporal conditional MI isolates the "fresh" relational signal at each time step by conditioning on shared history.

Computes all tier pairs: common-common, common-medium, common-rare, common-tail, medium-medium, medium-rare, medium-tail, rare-rare, rare-tail, tail-tail. For both target and raw cd codes.
```

**Step 2: Insert temporal conditional MI computation cell**

```python
def compute_temporal_conditional_mi(df, code_freq, tier_bounds,
                                    cd_freq_counter=None, cd_tier_bounds=None,
                                    sample_n=15000, top_k=500):
    """
    Compute temporal conditional MI: I(A_t; B_t | history) for all tier pairs.
    
    Approach:
    1. Build per-member binary presence vectors for top_k codes at each day
    2. For each pair (A, B), compute MI at the current day conditioned on
       whether both were present in the prior window (last 5 days).
    3. Conditional MI = H(A_t | hist) + H(B_t | hist) - H(A_t, B_t | hist)
    
    The conditioning variable is a binary: "was the code present in days [d-5, d-1]?"
    This is a tractable approximation to full history conditioning.
    """
    if sample_n and len(df) > sample_n:
        df_sample = df.sample(n=sample_n, random_state=42)
    else:
        df_sample = df

    all_results = {}

    for code_type, parse_fn, col, freq_data, tb in [
        ('target', parse_target_string, 'target', code_freq, tier_bounds),
        ('cd', parse_cd_string, 'cd',
         cd_freq_counter, cd_tier_bounds) if cd_freq_counter else (None,)*5,
    ]:
        if code_type is None:
            continue

        if code_type == 'target':
            top_codes = np.argsort(freq_data)[-top_k:][::-1] + 1
            get_tier = lambda c: assign_code_tier(int(c), freq_data, tb)
        else:
            sorted_cd = sorted(freq_data.keys(), key=freq_data.get, reverse=True)[:top_k]
            top_codes = np.array(sorted_cd)
            get_tier = lambda c: assign_cd_tier(int(c), freq_data, tb)

        code_to_idx = {int(c): i for i, c in enumerate(top_codes)}
        tier_assignments = {int(c): get_tier(c) for c in top_codes}

        n_members = len(df_sample)
        WINDOW = 5

        # Accumulate conditional co-occurrence counts in a streaming fashion
        # For each pair, track: n_both_given_hist, n_a_given_hist, n_b_given_hist, n_given_hist
        # Grouped by tier pair, using sampled pairs per tier combination
        tier_pairs_list = [
            ('common', 'common'), ('common', 'medium'), ('common', 'rare'),
            ('common', 'tail'), ('medium', 'medium'), ('medium', 'rare'),
            ('medium', 'tail'), ('rare', 'rare'), ('rare', 'tail'), ('tail', 'tail'),
        ]

        rng = np.random.RandomState(42)
        sampled_pairs_by_tier = {}

        for tier_a, tier_b in tier_pairs_list:
            codes_a = [int(c) for c in top_codes if tier_assignments.get(int(c)) == tier_a]
            codes_b = [int(c) for c in top_codes if tier_assignments.get(int(c)) == tier_b]
            if not codes_a or not codes_b:
                continue

            n_sample_pairs = min(50, len(codes_a) * len(codes_b))
            pairs = set()
            attempts = 0
            while len(pairs) < n_sample_pairs and attempts < n_sample_pairs * 5:
                a = codes_a[rng.randint(len(codes_a))]
                b = codes_b[rng.randint(len(codes_b))]
                if a != b:
                    pairs.add((a, b))
                attempts += 1
            sampled_pairs_by_tier[(tier_a, tier_b)] = list(pairs)

        # For each sampled pair, accumulate conditional counts
        pair_stats = {}
        for tier_key, pairs in sampled_pairs_by_tier.items():
            for a, b in pairs:
                pair_stats[(a, b)] = {
                    'n_both': 0, 'n_a_only': 0, 'n_b_only': 0, 'n_neither': 0,
                    'n_both_cond': 0, 'n_a_cond': 0, 'n_b_cond': 0, 'n_neither_cond': 0,
                    'n_hist_present': 0, 'n_hist_absent': 0,
                }

        t0 = time.time()
        processed = 0

        for _, row in df_sample.iterrows():
            days = parse_fn(row[col])
            dt_cnt = min(int(row['dt_cnt']), len(days))
            if dt_cnt < WINDOW + 1:
                continue

            # Build per-day presence for tracked codes
            day_presence = []
            for d in range(dt_cnt):
                curr = set(days[d]) if d < len(days) else set()
                day_presence.append(curr)

            for d in range(WINDOW, dt_cnt):
                history = set()
                for hd in range(max(0, d - WINDOW), d):
                    history.update(day_presence[hd])

                curr = day_presence[d]

                for (a, b), ps in pair_stats.items():
                    a_present = a in curr
                    b_present = b in curr
                    hist_has_both = a in history and b in history

                    if a_present and b_present:
                        ps['n_both'] += 1
                    elif a_present:
                        ps['n_a_only'] += 1
                    elif b_present:
                        ps['n_b_only'] += 1
                    else:
                        ps['n_neither'] += 1

                    if hist_has_both:
                        ps['n_hist_present'] += 1
                        if a_present and b_present:
                            ps['n_both_cond'] += 1
                        elif a_present:
                            ps['n_a_cond'] += 1
                        elif b_present:
                            ps['n_b_cond'] += 1
                        else:
                            ps['n_neither_cond'] += 1
                    else:
                        ps['n_hist_absent'] += 1

            processed += 1
            if processed % 3000 == 0:
                print(f"  [{code_type}] {processed:,} members ({time.time()-t0:.0f}s)")

        def compute_mi_from_counts(n11, n10, n01, n00):
            total = n11 + n10 + n01 + n00
            if total == 0:
                return 0.0
            p11, p10, p01, p00 = n11/total, n10/total, n01/total, n00/total
            pa = p11 + p10
            pb = p11 + p01
            mi = 0.0
            for pj, pm_a, pm_b in [(p11, pa, pb), (p10, pa, 1-pb),
                                     (p01, 1-pa, pb), (p00, 1-pa, 1-pb)]:
                if pj > 0 and pm_a > 0 and pm_b > 0:
                    mi += pj * np.log2(pj / (pm_a * pm_b))
            return max(0.0, mi)

        tier_results = []
        for tier_key, pairs in sampled_pairs_by_tier.items():
            mi_vals, cmi_vals = [], []
            for a, b in pairs:
                ps = pair_stats[(a, b)]
                mi = compute_mi_from_counts(ps['n_both'], ps['n_a_only'],
                                            ps['n_b_only'], ps['n_neither'])
                cmi = compute_mi_from_counts(ps['n_both_cond'], ps['n_a_cond'],
                                             ps['n_b_cond'], ps['n_neither_cond'])
                mi_vals.append(mi)
                cmi_vals.append(cmi)

            if mi_vals:
                tier_results.append({
                    'tier_a': tier_key[0], 'tier_b': tier_key[1],
                    'mean_mi': float(np.mean(mi_vals)),
                    'median_mi': float(np.median(mi_vals)),
                    'mean_cond_mi': float(np.mean(cmi_vals)),
                    'median_cond_mi': float(np.median(cmi_vals)),
                    'mean_reduction': float(1 - np.mean(cmi_vals) / max(np.mean(mi_vals), 1e-10)),
                    'n_pairs': len(mi_vals),
                })

        all_results[code_type] = tier_results
        print(f"  [{code_type}] Completed: {len(tier_results)} tier pairs ({time.time()-t0:.0f}s)")

    return all_results


print("Computing temporal conditional MI (all tier pairs, target + cd)...")
temporal_cmi = compute_temporal_conditional_mi(
    df_1_5m, code_freq, tier_bounds,
    cd_freq_counter=cd_freq, cd_tier_bounds=cd_tier_bounds,
    sample_n=15000, top_k=500
)
results['temporal_conditional_mi'] = temporal_cmi

for code_type, tier_results in temporal_cmi.items():
    print(f"\n--- {code_type.upper()} Temporal Conditional MI ---")
    print(f"{'Tier A':<10} {'Tier B':<10} {'MI':>10} {'Cond MI':>10} {'Reduction':>10} {'Pairs':>6}")
    for r in tier_results:
        print(f"{r['tier_a']:<10} {r['tier_b']:<10} "
              f"{r['mean_mi']:>10.6f} {r['mean_cond_mi']:>10.6f} "
              f"{r['mean_reduction']:>10.2%} {r['n_pairs']:>6}")
```

### Part B: Expand Existing MI Function to All Tier Pairs

**Step 3: Fix `compute_conditional_information` to include all tier pairs**

The existing function (current cell 17) already has the `pair_types` list with all tier combinations, but the results JSON shows only common-common was actually computed. This is because the `top_k_codes=500` selection may not include enough rare/tail codes. Fix by explicitly including codes from each tier.

In the `compute_conditional_information` function, replace the line:

```python
    top_codes = np.argsort(code_freq)[-top_k_codes:][::-1] + 1  # 1-indexed
```

with:

```python
    # Include top codes from EACH tier to ensure all tier pairs are represented
    tier_codes = defaultdict(list)
    for i in range(1, TARGET_CD_CNT + 1):
        t = assign_code_tier(i, code_frequencies, tier_bounds)
        if t not in ('zero', 'unknown'):
            tier_codes[t].append((code_frequencies[i-1], i))
    
    selected = []
    per_tier = top_k_codes // 4
    for tier in ['common', 'medium', 'rare', 'tail']:
        tier_sorted = sorted(tier_codes[tier], reverse=True)[:per_tier]
        selected.extend([c for _, c in tier_sorted])
    
    # Fill remaining slots with highest-frequency codes not already selected
    selected_set = set(selected)
    remaining = top_k_codes - len(selected)
    if remaining > 0:
        all_sorted = np.argsort(code_freq)[::-1] + 1
        for c in all_sorted:
            if int(c) not in selected_set:
                selected.append(int(c))
                if len(selected) >= top_k_codes:
                    break
    
    top_codes = np.array(selected)
```

**Step 4: Verify the MI function now includes all tiers**

Read the edited cell.

**Step 5: Commit**

```bash
git add dev/downstream/data_information_saturation_analysis.ipynb
git commit -m "feat: add temporal conditional MI R2.4 and expand MI to all tier pairs"
```

---

## Task 7: Extend Within-Member Saturation to Raw Codes

**Files:**
- Modify: `dev/downstream/data_information_saturation_analysis.ipynb` — edit within-member saturation invocation and plot cells

**Step 1: Add cd-based within-member saturation computation**

In the cell that runs within-member saturation (current cell 6), add after the existing target computation:

```python
print("\nComputing within-member saturation for raw cd codes (20k sample)...")
within_member_cd_df = compute_within_member_saturation_generic(
    df_1_5m, sample_n=20000, use_cd=True
)
print(f"Raw cd result: {len(within_member_cd_df):,} rows")
```

But first, we need to make the saturation function work with cd codes. Modify `compute_within_member_saturation` to accept a `use_cd` flag.

In the function definition cell (current cell 5), replace the function signature and the inner loop where codes are parsed:

Replace the first line:
```python
def compute_within_member_saturation(df, code_frequencies, tier_bounds,
                                     min_days=10, sample_n=50000):
```
with:
```python
def compute_within_member_saturation(df, code_frequencies=None, tier_bounds=None,
                                     min_days=10, sample_n=50000, use_cd=False,
                                     cd_freq_counter=None, cd_tier_bounds=None):
```

And replace `target_days = parse_target_string(row['target'])` with:
```python
        if use_cd:
            target_days = parse_cd_string(row['cd'])
        else:
            target_days = parse_target_string(row['target'])
```

And replace the tier assignment line:
```python
    code_tier_map = np.empty(TARGET_CD_CNT + 1, dtype='U8')
    code_tier_map[0] = 'zero'
    for i in range(1, TARGET_CD_CNT + 1):
        code_tier_map[i] = assign_code_tier(i, code_frequencies, tier_bounds)
```
with:
```python
    if use_cd and cd_freq_counter is not None:
        code_tier_map = {}
        for c in cd_freq_counter:
            code_tier_map[c] = assign_cd_tier(c, cd_freq_counter, cd_tier_bounds)
    else:
        code_tier_map_arr = np.empty(TARGET_CD_CNT + 1, dtype='U8')
        code_tier_map_arr[0] = 'zero'
        for i in range(1, TARGET_CD_CNT + 1):
            code_tier_map_arr[i] = assign_code_tier(i, code_frequencies, tier_bounds)
        code_tier_map = {i: code_tier_map_arr[i] for i in range(TARGET_CD_CNT + 1)}
```

And update the tier lookup in the inner loop:
```python
            tier_new = defaultdict(int)
            for c in new_codes:
                tier = code_tier_map.get(c, 'unknown')
                if tier and tier != 'zero':
                    tier_new[tier] += 1
```

**Step 2: Add the cd invocation after existing target invocation**

In the invocation cell (cell 6), add:

```python
print("\nComputing within-member saturation for raw cd codes (20k sample)...")
within_member_cd_df = compute_within_member_saturation(
    df_1_5m, use_cd=True, cd_freq_counter=cd_freq, cd_tier_bounds=cd_tier_bounds,
    min_days=10, sample_n=20000
)
print(f"Raw cd result: {len(within_member_cd_df):,} rows")
```

**Step 3: Add cd overlay to the existing plots**

In the plot cell (cell 7), add cd novelty curve overlay to the first subplot:

After the line `ax.set_xlim(0, 200)` in the first subplot, add:

```python
if len(within_member_cd_df) > 0:
    cd_agg = within_member_cd_df.groupby('day_position')['novelty_rate'].mean()
    ax.plot(cd_agg.index, cd_agg.values, label='Code novelty (cd)', color='red', alpha=0.7)
    ax.legend()
```

**Step 4: Save cd novelty summary to results**

Add after the target summary saving:

```python
within_member_cd_summary = {'metric': 'within_member_temporal_saturation_cd'}
if len(within_member_cd_df) > 0:
    cd_agg_full = within_member_cd_df.groupby('day_position')['novelty_rate'].mean().reset_index()
    for day_val in [10, 50, 100, 199]:
        key = f'novelty_rate_day_{day_val}'
        row = cd_agg_full.loc[cd_agg_full['day_position'] == day_val, 'novelty_rate']
        within_member_cd_summary[key] = float(row.iloc[0]) if len(row) > 0 else None
results['within_member_cd'] = within_member_cd_summary
```

**Step 5: Commit**

```bash
git add dev/downstream/data_information_saturation_analysis.ipynb
git commit -m "feat: extend within-member saturation to raw cd codes"
```

---

## Task 8: Update Results Saving and Report Generation

**Files:**
- Modify: `dev/downstream/data_information_saturation_analysis.ipynb` — edit the results saving cell and report generation cell

**Step 1: Update the JSON results path**

In the results saving cell (currently cell 30), change the output path:

```python
output_path = Path('../../expe_logs/exp_round5/data_information_saturation_results.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
```

**Step 2: Add new sections to the report generation**

In the report cell (currently cell 31), add after the existing Section 6 (MI):

```python
report.append("\n## R1: Member Trajectory Analysis\n")
for key in ['trajectory_target', 'trajectory_cd']:
    if key in results:
        t = results[key]
        label = 'Target' if 'target' in key else 'Raw cd'
        report.append(f"\n### {label} Codes\n")
        report.append(f"- Transition entropy: **{t.get('transition_entropy_bits', 'N/A'):.2f}** bits\n")
        report.append(f"- Mean velocity: **{t.get('mean_velocity', 'N/A'):.2f}**\n")
        report.append(f"- Mean persistence (Jaccard): **{t.get('mean_persistence', 'N/A'):.3f}**\n")
        if 'trajectory_type_distribution' in t:
            report.append(f"- Trajectory types: {t['trajectory_type_distribution']}\n")

report.append("\n## R2.3: Temporal Conditional Entropy\n")
for key, label in [('conditional_entropy_target', 'Target'), ('conditional_entropy_cd', 'Raw cd')]:
    if key in results and results[key]:
        late = [r for r in results[key] if r['day'] >= 50]
        if late:
            rate = np.mean([r['H_cond'] for r in late])
            report.append(f"- **{label}** entropy rate (days 50+): **{rate:.4f}** bits\n")

report.append("\n## R2.4: Temporal Conditional MI (All Pairs)\n")
if 'temporal_conditional_mi' in results:
    for ct, tier_results in results['temporal_conditional_mi'].items():
        report.append(f"\n### {ct.upper()}\n")
        report.append("| Tier A | Tier B | MI | Cond MI | Reduction |\n")
        report.append("|--------|--------|-----|---------|----------|\n")
        for r in tier_results:
            report.append(f"| {r['tier_a']} | {r['tier_b']} | "
                          f"{r['mean_mi']:.6f} | {r['mean_cond_mi']:.6f} | "
                          f"{r['mean_reduction']:.2%} |\n")
```

**Step 3: Commit**

```bash
git add dev/downstream/data_information_saturation_analysis.ipynb
git commit -m "feat: update results saving and report to include all new sections"
```

---

## Implementation Order Summary

| Task | Description | Cells Affected | Dependencies |
|------|-------------|----------------|-------------|
| 1 | Metrics technical reference (R0) | Insert cell 1 | None |
| 2 | Raw cd utilities + config | Edit cells 2, 3, 4 | None |
| 3 | Cd co-occurrence integration | Edit cells 9, 10 | Task 2 |
| 4 | Member trajectory analysis (R1) | Insert 3 cells after cell 7 | Task 2 |
| 5 | Temporal conditional entropy (R2.3) | Insert 2 cells before MI section | Task 2 |
| 6 | All-pairs temporal conditional MI (R2.4) + MI fix | Insert 2 cells + edit MI function | Tasks 2, 5 |
| 7 | Extend within-member to cd | Edit cells 5, 6, 7 | Task 2 |
| 8 | Update results saving + report | Edit cells 30, 31 | All above |

**Total new cells:** ~8 (3 markdown headers, 5 code cells)
**Modified cells:** ~8 (config, utilities, loading, co-occurrence, MI function, within-member function + invocation + plots, results saving, report)
