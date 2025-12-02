
## Implementation: Linear Probe Evaluation for age, gender, and presence of ICD9 codes

### Section 1: Add Required Import (Near top of file, after existing imports)

```python
# Add these imports near the top of the file (around line 55-60)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
import warnings
```

---

### Section 2: Add Clinical Condition Detector Class (Place after `compute_moe_batch_metrics` function, around line 4727)

```python
# ============================================================================
# LINEAR PROBE EVALUATION FOR EMBEDDING QUALITY
# ============================================================================
# The gold standard for evaluating learned representations without downstream tasks.
# Uses synthetic classification tasks derived from available data.
# Reference: Radford et al. (2021) "CLIP" - Linear probe is the primary embedding metric
# ============================================================================

class ClinicalConditionDetector:
    """
    Defines synthetic clinical classification tasks from code mappings.
    
    Uses the code_meaning column to identify clinical conditions that can serve
    as proxy downstream tasks for embedding quality evaluation.
    
    Args:
        code_mapping_df: DataFrame with columns ['raw_cd', 'target_cd', 'code_meaning']
        
    Example:
        detector = ClinicalConditionDetector(code_mapping)
        has_diabetes = detector.detect_condition(patient_codes, 'diabetes')
    """
    
    # Define clinical condition patterns (case-insensitive regex patterns)
    # These map condition names to keywords in code_meaning
    CONDITION_PATTERNS = {
        'diabetes': r'diabetes|diabetic|dm\b|a1c|hyperglycemia',
        'hypertension': r'hypertension|hypertensive|htn\b|elevated blood pressure',
        'heart_disease': r'cardiac|heart|coronary|myocardial|angina|arrhythmia|chf\b|cad\b',
        'respiratory': r'copd|asthma|pneumonia|bronchitis|respiratory|pulmonary',
        'kidney_disease': r'renal|kidney|nephro|dialysis|ckd\b|esrd\b',
        'mental_health': r'depression|anxiety|bipolar|schizophrenia|psychiatric|mental',
        'cancer': r'cancer|malignant|neoplasm|tumor|carcinoma|oncology',
        'infection': r'infection|sepsis|bacterial|viral|abscess|cellulitis',
        'emergency': r'emergency|trauma|critical|acute|urgent',
        'chronic': r'chronic|long-term|persistent|ongoing',
    }
    
    def __init__(self, code_mapping_df: Optional[pd.DataFrame] = None):
        """
        Initialize with optional code mapping DataFrame.
        
        Args:
            code_mapping_df: DataFrame with columns ['raw_cd', 'target_cd', 'code_meaning']
                            If None, will use only code-based heuristics
        """
        self.code_mapping = code_mapping_df
        self.condition_code_sets = {}
        
        if code_mapping_df is not None:
            self._build_condition_code_sets()
    
    def _build_condition_code_sets(self):
        """Pre-compute sets of target_cd for each condition."""
        import re
        
        if self.code_mapping is None:
            return
            
        for condition, pattern in self.CONDITION_PATTERNS.items():
            # Find all codes whose meaning matches the pattern
            mask = self.code_mapping['code_meaning'].str.contains(
                pattern, case=False, na=False, regex=True
            )
            matching_codes = set(self.code_mapping.loc[mask, 'target_cd'].tolist())
            self.condition_code_sets[condition] = matching_codes
            
            # Log how many codes found
            if len(matching_codes) > 0:
                print(f"  ClinicalConditionDetector: Found {len(matching_codes)} codes for '{condition}'")
    
    def detect_condition(self, patient_codes: set, condition: str) -> bool:
        """
        Check if a patient has a specific condition based on their codes.
        
        Args:
            patient_codes: Set of target_cd codes for this patient
            condition: Condition name (key in CONDITION_PATTERNS)
            
        Returns:
            True if patient has any code matching the condition
        """
        if condition not in self.condition_code_sets:
            return False
        
        return bool(patient_codes & self.condition_code_sets[condition])
    
    def get_available_conditions(self) -> List[str]:
        """Return list of conditions that have at least some codes mapped."""
        return [c for c, codes in self.condition_code_sets.items() if len(codes) > 0]
    
    def get_condition_prevalence(self, all_patient_codes: List[set]) -> Dict[str, float]:
        """
        Calculate prevalence of each condition across patients.
        
        Args:
            all_patient_codes: List of sets, each containing target_cd for one patient
            
        Returns:
            Dict mapping condition name to prevalence (0-1)
        """
        prevalences = {}
        n_patients = len(all_patient_codes)
        
        for condition in self.get_available_conditions():
            count = sum(1 for codes in all_patient_codes if self.detect_condition(codes, condition))
            prevalences[condition] = count / n_patients if n_patients > 0 else 0.0
            
        return prevalences


def extract_patient_embeddings(
    model: nn.Module,
    val_data: pd.DataFrame,
    config: BaseConfig,
    device: torch.device,
    num_samples: int = 500,
    use_mixed_precision: bool = False
) -> Tuple[torch.Tensor, List[set], List[int], List[int]]:
    """
    Extract patient-level embeddings and metadata for linear probe evaluation.
    
    This function works with all model types (Baseline, Flash, MoE) by using
    the return_embeddings flag for Flash/MoE models.
    
    Args:
        model: Trained model (any type)
        val_data: Validation DataFrame with columns: age_in_months, gender_cd, cd, target_cd, dt_cnt
        config: Model configuration
        device: Torch device
        num_samples: Maximum number of patients to extract
        use_mixed_precision: Whether to use FP16
        
    Returns:
        Tuple of:
        - embeddings: Tensor [num_patients, embedding_dim]
        - all_codes: List of sets containing target_cd per patient
        - ages: List of age values (in months, last day)
        - genders: List of gender codes
    """
    model.eval()
    
    # Sample validation data
    sample_size = min(num_samples, len(val_data))
    val_sample = val_data.sample(sample_size, random_state=42)
    val_dataset = ClinicalDataset(val_sample, config)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=clinical_collate_fn
    )
    
    all_embeddings = []
    all_codes = []
    all_ages = []
    all_genders = []
    
    with torch.no_grad():
        for batch in val_loader:
            age = batch['age'].to(device)
            gender = batch['gender'].to(device)
            codes = batch['codes'].to(device)
            dt_cnt = batch['dt_cnt']
            y = batch['target']
            
            x = torch.cat([
                age.unsqueeze(-1),
                gender.unsqueeze(-1),
                codes
            ], dim=-1)
            
            batch_size_actual = x.shape[0]
            
            # ================================================================
            # EXTRACT EMBEDDINGS BASED ON MODEL TYPE
            # ================================================================
            if isinstance(model, BaselineTransformer):
                # Baseline: Manual forward pass to get embeddings before decoder
                age_in_months = model.embedding_age_in_months(x[:, :, 0].long())
                gender_cd = model.embedding_gender_cd(x[:, :, 1].long())
                cd = model.embedding_cd(x[:, :, 2:].long())
                cd_res = cd.sum(-2)
                
                # Daily encoding
                cd = cd.reshape(-1, config.len_cd, config.embedding_size)
                cd = torch.swapaxes(cd, 0, 1)
                cd = model.transformer_encoder_cd(cd)
                cd = cd.permute(1, 2, 0)
                cd = nn.MaxPool1d(config.len_cd)(cd)
                cd = cd.reshape(batch_size_actual, config.len_dy, config.embedding_size)
                
                # Combine
                cd = cd_res + cd + gender_cd + age_in_months
                cd = model.mm(cd)
                cd = model.norm(cd)
                cd = torch.swapaxes(cd, 0, 1)
                
                # Temporal encoding
                actual_len_dy = x.shape[1]
                mth_mask = model._generate_square_subsequent_mask(actual_len_dy).to(device)
                embeddings = model.transformer_encoder_dy(cd, mth_mask)
                embeddings = torch.swapaxes(embeddings, 0, 1)  # [batch, len_dy, d_model]
                
            else:
                # Flash/MoE models: Use return_embeddings flag (REQUIRES MODEL UPDATE)
                if use_mixed_precision:
                    dtype = getattr(config, 'dtype', torch.float16)
                    with torch.cuda.amp.autocast(dtype=dtype):
                        if hasattr(model, 'forward'):
                            # Check if model supports return_embeddings
                            if 'return_embeddings' in model.forward.__code__.co_varnames:
                                output, _, embeddings = model(x, return_moe_losses=False, return_embeddings=True)
                            elif 'return_moe_losses' in model.forward.__code__.co_varnames:
                                # Fallback: use output logits (suboptimal)
                                output, _ = model(x, return_moe_losses=False)
                                embeddings = output  # [batch, len_dy, target_cd_cnt]
                            else:
                                output = model(x)
                                embeddings = output
                else:
                    if hasattr(model, 'forward'):
                        if 'return_embeddings' in model.forward.__code__.co_varnames:
                            output, _, embeddings = model(x, return_moe_losses=False, return_embeddings=True)
                        elif 'return_moe_losses' in model.forward.__code__.co_varnames:
                            output, _ = model(x, return_moe_losses=False)
                            embeddings = output
                        else:
                            output = model(x)
                            embeddings = output
            
            # ================================================================
            # EXTRACT LAST VALID DAY EMBEDDING PER PATIENT
            # ================================================================
            for j in range(batch_size_actual):
                valid_days = min(int(dt_cnt[j]), embeddings.shape[1])
                if valid_days > 0:
                    # Get embedding at last valid day
                    patient_emb = embeddings[j, valid_days - 1, :].cpu()
                    all_embeddings.append(patient_emb)
                    
                    # Aggregate all target codes for this patient
                    patient_targets = y[j]
                    patient_codes = set()
                    for day_codes in patient_targets:
                        patient_codes.update([c for c in day_codes if c > 0])
                    all_codes.append(patient_codes)
                    
                    # Extract age and gender at last valid day
                    # Age is stored in months
                    last_age = age[j, valid_days - 1].item()
                    last_gender = gender[j, valid_days - 1].item()
                    all_ages.append(last_age)
                    all_genders.append(last_gender)
    
    if len(all_embeddings) == 0:
        return torch.tensor([]), [], [], []
    
    embeddings_tensor = torch.stack(all_embeddings)  # [num_patients, d_model or target_cd_cnt]
    
    return embeddings_tensor, all_codes, all_ages, all_genders


def compute_linear_probe_metrics(
    model: nn.Module,
    val_data: pd.DataFrame,
    config: BaseConfig,
    device: torch.device,
    code_mapping: Optional[pd.DataFrame] = None,
    num_samples: int = 500,
    use_mixed_precision: bool = False,
    n_cv_folds: int = 5
) -> Dict[str, float]:
    """
    Compute linear probe evaluation metrics for embedding quality.
    
    This is the gold standard for evaluating learned representations without
    access to actual downstream tasks. We create synthetic classification tasks:
    
    1. Age Group Prediction: Young (<40), Middle (40-65), Senior (65+)
    2. Gender Prediction: Binary classification
    3. Clinical Condition Presence: Multiple binary tasks (diabetes, heart disease, etc.)
    
    The intuition: If embeddings capture meaningful clinical information, a simple
    linear classifier should be able to predict these attributes without any
    fine-tuning of the embeddings.
    
    Args:
        model: Trained model (Baseline, Flash, or MoE)
        val_data: Validation DataFrame
        config: Model configuration
        device: Torch device
        code_mapping: Optional DataFrame with ['raw_cd', 'target_cd', 'code_meaning']
                     If provided, enables clinical condition probing
        num_samples: Number of patients to evaluate (recommend 300-1000)
        use_mixed_precision: Whether to use FP16 for inference
        n_cv_folds: Number of cross-validation folds
        
    Returns:
        Dict with probe accuracies:
        - 'probe_age_acc': Age group classification accuracy
        - 'probe_gender_acc': Gender classification accuracy
        - 'probe_diabetes_acc': Diabetes presence classification (if code_mapping provided)
        - 'probe_heart_disease_acc': Heart disease presence (if code_mapping provided)
        - ... (other conditions)
        - 'probe_mean_condition_acc': Average across all condition probes
        - 'probe_mean_overall_acc': Average across all probes (age, gender, conditions)
    """
    print("Computing linear probe metrics...")
    metrics = {}
    
    # ========================================================================
    # STEP 1: Extract patient embeddings
    # ========================================================================
    embeddings, all_codes, ages, genders = extract_patient_embeddings(
        model=model,
        val_data=val_data,
        config=config,
        device=device,
        num_samples=num_samples,
        use_mixed_precision=use_mixed_precision
    )
    
    if len(embeddings) < 50:
        print(f"  ⚠️ Too few samples ({len(embeddings)}) for reliable linear probe")
        return {
            'probe_age_acc': 0.0,
            'probe_gender_acc': 0.0,
            'probe_mean_overall_acc': 0.0
        }
    
    # Convert to numpy for sklearn
    X = embeddings.numpy()
    
    # Standardize features (important for logistic regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"  Extracted {len(X)} patient embeddings, dim={X.shape[1]}")
    
    # ========================================================================
    # STEP 2: Age Group Probe (3-class classification)
    # ========================================================================
    # Convert age in months to age groups
    # <480 months = <40 years (young)
    # 480-780 months = 40-65 years (middle)
    # >780 months = >65 years (senior)
    age_labels = []
    for age_months in ages:
        if age_months < 480:
            age_labels.append(0)  # Young
        elif age_months < 780:
            age_labels.append(1)  # Middle
        else:
            age_labels.append(2)  # Senior
    
    age_labels = np.array(age_labels)
    
    # Check if we have enough samples per class
    unique_ages, age_counts = np.unique(age_labels, return_counts=True)
    min_age_count = min(age_counts)
    
    if min_age_count >= n_cv_folds:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            try:
                age_probe = LogisticRegression(
                    max_iter=500, 
                    solver='lbfgs',
                    multi_class='multinomial',
                    random_state=42
                )
                cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
                age_scores = cross_val_score(age_probe, X_scaled, age_labels, cv=cv, scoring='accuracy')
                metrics['probe_age_acc'] = float(age_scores.mean())
                metrics['probe_age_std'] = float(age_scores.std())
                print(f"  Age Probe: {metrics['probe_age_acc']:.3f} ± {metrics['probe_age_std']:.3f}")
            except Exception as e:
                print(f"  ⚠️ Age probe failed: {e}")
                metrics['probe_age_acc'] = 0.0
    else:
        print(f"  ⚠️ Insufficient age diversity (min class count: {min_age_count})")
        metrics['probe_age_acc'] = 0.0
    
    # ========================================================================
    # STEP 3: Gender Probe (Binary classification)
    # ========================================================================
    gender_labels = np.array(genders)
    unique_genders = np.unique(gender_labels)
    
    if len(unique_genders) >= 2:
        # Binarize gender (in case of multiple categories)
        # Assume 0 = unknown, 1 = male, 2 = female
        # Map to binary: male (1) vs female (2)
        valid_gender_mask = (gender_labels == 1) | (gender_labels == 2)
        
        if valid_gender_mask.sum() >= 50:
            X_gender = X_scaled[valid_gender_mask]
            y_gender = (gender_labels[valid_gender_mask] == 2).astype(int)  # 1 = female, 0 = male
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                try:
                    gender_probe = LogisticRegression(max_iter=500, solver='lbfgs', random_state=42)
                    cv = StratifiedKFold(n_splits=min(n_cv_folds, len(y_gender) // 10), shuffle=True, random_state=42)
                    gender_scores = cross_val_score(gender_probe, X_gender, y_gender, cv=cv, scoring='accuracy')
                    metrics['probe_gender_acc'] = float(gender_scores.mean())
                    metrics['probe_gender_std'] = float(gender_scores.std())
                    print(f"  Gender Probe: {metrics['probe_gender_acc']:.3f} ± {metrics['probe_gender_std']:.3f}")
                except Exception as e:
                    print(f"  ⚠️ Gender probe failed: {e}")
                    metrics['probe_gender_acc'] = 0.0
        else:
            print(f"  ⚠️ Insufficient gender diversity")
            metrics['probe_gender_acc'] = 0.0
    else:
        metrics['probe_gender_acc'] = 0.0
    
    # ========================================================================
    # STEP 4: Clinical Condition Probes (Multiple binary classifications)
    # ========================================================================
    condition_accs = []
    
    if code_mapping is not None:
        print("  Building clinical condition detector...")
        detector = ClinicalConditionDetector(code_mapping)
        
        # Get available conditions and their prevalence
        prevalences = detector.get_condition_prevalence(all_codes)
        
        for condition in detector.get_available_conditions():
            prevalence = prevalences.get(condition, 0.0)
            
            # Skip conditions that are too rare or too common (uninformative)
            if prevalence < 0.05 or prevalence > 0.95:
                print(f"    Skipping '{condition}' (prevalence={prevalence:.2%})")
                continue
            
            # Create binary labels
            condition_labels = np.array([
                1 if detector.detect_condition(codes, condition) else 0 
                for codes in all_codes
            ])
            
            # Ensure sufficient positive/negative samples
            n_positive = condition_labels.sum()
            n_negative = len(condition_labels) - n_positive
            
            if min(n_positive, n_negative) < n_cv_folds:
                print(f"    Skipping '{condition}' (too few samples: {n_positive} pos, {n_negative} neg)")
                continue
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                try:
                    condition_probe = LogisticRegression(
                        max_iter=500, 
                        solver='lbfgs',
                        class_weight='balanced',  # Handle imbalance
                        random_state=42
                    )
                    cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
                    scores = cross_val_score(condition_probe, X_scaled, condition_labels, cv=cv, scoring='accuracy')
                    
                    acc = float(scores.mean())
                    std = float(scores.std())
                    metrics[f'probe_{condition}_acc'] = acc
                    metrics[f'probe_{condition}_std'] = std
                    condition_accs.append(acc)
                    
                    print(f"    {condition}: {acc:.3f} ± {std:.3f} (prevalence={prevalence:.1%})")
                    
                except Exception as e:
                    print(f"    ⚠️ {condition} probe failed: {e}")
    else:
        print("  ⚠️ No code_mapping provided, skipping clinical condition probes")
    
    # ========================================================================
    # STEP 5: Aggregate Metrics
    # ========================================================================
    # Mean across condition probes
    if condition_accs:
        metrics['probe_mean_condition_acc'] = float(np.mean(condition_accs))
    else:
        metrics['probe_mean_condition_acc'] = 0.0
    
    # Overall mean (age + gender + conditions)
    all_probe_accs = [
        metrics.get('probe_age_acc', 0.0),
        metrics.get('probe_gender_acc', 0.0)
    ] + condition_accs
    
    # Filter out zeros (failed probes)
    valid_accs = [a for a in all_probe_accs if a > 0]
    metrics['probe_mean_overall_acc'] = float(np.mean(valid_accs)) if valid_accs else 0.0
    metrics['probe_num_tasks'] = len(valid_accs)
    
    print(f"  Overall Probe Accuracy: {metrics['probe_mean_overall_acc']:.3f} ({metrics['probe_num_tasks']} tasks)")
    
    return metrics


def test_linear_probe_metrics():
    """Test linear probe evaluation with synthetic data."""
    print("Testing linear probe metrics...")
    
    # Create mock data
    n_samples = 100
    embed_dim = 256
    
    # Create embeddings with some structure
    embeddings = torch.randn(n_samples, embed_dim)
    
    # Create structured metadata
    ages = [np.random.choice([300, 600, 900]) for _ in range(n_samples)]  # 3 age groups
    genders = [np.random.choice([1, 2]) for _ in range(n_samples)]  # male/female
    all_codes = [set(np.random.choice(range(100), size=5)) for _ in range(n_samples)]
    
    # Test condition detector (without actual mapping)
    detector = ClinicalConditionDetector(None)
    assert detector.get_available_conditions() == []
    
    # Test with mock mapping
    mock_mapping = pd.DataFrame({
        'raw_cd': range(100),
        'target_cd': range(100),
        'code_meaning': ['diabetes mellitus'] * 10 + ['hypertension'] * 10 + ['other'] * 80
    })
    
    detector = ClinicalConditionDetector(mock_mapping)
    assert 'diabetes' in detector.get_available_conditions()
    assert 'hypertension' in detector.get_available_conditions()
    
    # Test detection
    test_codes = {0, 1, 2}  # Should have diabetes
    assert detector.detect_condition(test_codes, 'diabetes') == True
    assert detector.detect_condition(test_codes, 'cancer') == False
    
    print("Linear probe test ✔️")

# Uncomment to run test:
# test_linear_probe_metrics()
```

---

### Section 3: Integrate into `compute_embedding_quality_epoch` (Modify existing function around line 4518)

Update the existing `compute_embedding_quality_epoch` function to include linear probe metrics:

```python
def compute_embedding_quality_epoch(
    model: nn.Module,
    val_data: pd.DataFrame,
    config: BaseConfig,
    device: torch.device,
    num_samples: int = 200,
    use_mixed_precision: bool = False,
    code_mapping: Optional[pd.DataFrame] = None  # NEW PARAMETER
) -> Dict[str, float]:
    """
    Evaluate embedding quality at epoch end.
    
    Run this ONCE per epoch (expensive!) to check if embeddings are useful
    for downstream tasks.
    
    Metrics computed:
    1. Embedding std_mean - Detects embedding collapse (should be > 0.05)
    2. NN target overlap - Do similar embeddings have similar codes? (higher = better)
    3. Linear probe accuracies - Synthetic downstream task performance (NEW)
    
    Why these matter for downstream tasks:
    - If embeddings collapse (low std), they won't transfer to downstream classifiers
    - If NN overlap is low, embeddings don't capture clinical similarity
    - If linear probe accuracy is low, embeddings lack predictive structure
    
    Args:
        model: Trained model
        val_data: Validation DataFrame
        config: Model configuration
        device: Torch device
        num_samples: Number of samples for evaluation
        use_mixed_precision: Whether to use FP16
        code_mapping: DataFrame with ['raw_cd', 'target_cd', 'code_meaning'] for condition probes
    
    Returns:
        Dict with 'embedding_std_mean', 'nn_target_overlap', 'probe_*' metrics
    """
    model.eval()
    metrics = {}
    
    # ========================================================================
    # EXISTING: Extract embeddings and compute std/overlap metrics
    # ========================================================================
    # ... [Keep all existing code from lines 4546-4676] ...
    
    # ========================================================================
    # NEW: Linear Probe Evaluation
    # ========================================================================
    # Only run linear probe if we have enough samples
    if len(all_embeddings) >= 50:
        probe_metrics = compute_linear_probe_metrics(
            model=model,
            val_data=val_data,
            config=config,
            device=device,
            code_mapping=code_mapping,
            num_samples=num_samples,
            use_mixed_precision=use_mixed_precision,
            n_cv_folds=5
        )
        metrics.update(probe_metrics)
    else:
        metrics['probe_mean_overall_acc'] = 0.0
    
    return metrics
```

---

### Section 4: Update `run_single_experiment` to Pass Code Mapping (Around line 5972)

Add a new parameter to `run_single_experiment`:

```python
def run_single_experiment(
    exp_name: str,
    moe_config: Optional[MoEConfig],
    use_learnt_att_pool: bool,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    device: torch.device,
    epochs: int = 4,
    code_frequencies: Optional[np.ndarray] = None,
    log_dir: str = "logs",
    experiment_round: Optional[str] = None,
    check_embeddings_every: int = 2,
    log_metrics_every: int = 100,
    resume_from: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    embedding_size: Optional[int] = None,
    code_mapping: Optional[pd.DataFrame] = None  # NEW PARAMETER
) -> Dict[str, any]:
    """
    Run a SINGLE experiment.
    
    Args:
        ...existing args...
        code_mapping: DataFrame with ['raw_cd', 'target_cd', 'code_meaning'] 
                     for linear probe clinical condition detection
    """
    # ... existing code ...
```

---

### Section 5: Update Embedding Quality Call in Training Loop (Around line 6279-6292)

```python
        if epoch % check_embeddings_every == 0:
            logger.info("Computing embedding quality...")
            emb_metrics = compute_embedding_quality_epoch(
                model, 
                val_data,
                config, 
                device, 
                num_samples=200,
                use_mixed_precision=use_mixed_precision,
                code_mapping=code_mapping  # NEW: Pass code mapping
            )
            val_metrics.update(emb_metrics)

            logger.info(f"    Embedding std: {emb_metrics['embedding_std_mean']:.4f}")
            logger.info(f"    NN overlap: {emb_metrics['nn_target_overlap']:.3f}")
            # NEW: Log linear probe metrics
            if 'probe_mean_overall_acc' in emb_metrics:
                logger.info(f"    Probe Accuracy: {emb_metrics['probe_mean_overall_acc']:.3f}")
```

---

### Section 6: Update Final Results Dict (Around line 6418)

Add linear probe metrics to the results:

```python
    results = {
        'experiment': exp_name,
        'parameters': total_params,
        'use_learned_pooling': use_learnt_att_pool,
        'use_bucketing': use_bucketing,
        
        # ... existing metrics ...
        
        # NEW: Linear Probe Metrics (from last epoch with embedding evaluation)
        'probe_age_acc': final_metrics.get('probe_age_acc', 0.0),
        'probe_gender_acc': final_metrics.get('probe_gender_acc', 0.0),
        'probe_mean_condition_acc': final_metrics.get('probe_mean_condition_acc', 0.0),
        'probe_mean_overall_acc': final_metrics.get('probe_mean_overall_acc', 0.0),
        
        # ... existing metrics ...
        'full_evaluation': evaluation,
        'all_epochs': epoch_history
    }
```

---

### Section 7: Update `run_selected_experiments` (Around line 6458)

Add code_mapping parameter:

```python
def run_selected_experiments(
    experiment_names: List[str],
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    device: torch.device,
    epochs: int = 10,
    experiment_round: Optional[str] = None,
    code_mapping: Optional[pd.DataFrame] = None  # NEW PARAMETER
) -> pd.DataFrame:
    """..."""
    
    # ... existing code ...
    
    for exp_name in experiment_names:
        # ... existing cleanup code ...
        
        results = run_single_experiment(
            exp_name=exp_name,
            moe_config=moe_config,
            use_learnt_att_pool=use_learnt_att_pool,
            train_data=train_data,
            val_data=val_data,
            device=device,
            epochs=epochs,
            experiment_round=experiment_round,
            code_mapping=code_mapping  # NEW: Pass through
        )
        
        all_results.append(results)
    
    # ... rest of function ...
```

---

### Usage Example

```python
# Load your code mapping table
code_mapping = pd.read_csv('path/to/code_mapping.csv')
# Expected columns: raw_cd, target_cd, code_meaning

# Run experiments with linear probe evaluation
results = run_single_experiment(
    exp_name='exp3_standard_moe',
    moe_config=moe_config,
    use_learnt_att_pool=False,
    train_data=df_train,
    val_data=df_val,
    device=device,
    epochs=10,
    code_mapping=code_mapping  # Enable linear probe
)

# Results will include:
# - probe_age_acc: How well embeddings predict age group
# - probe_gender_acc: How well embeddings predict gender
# - probe_diabetes_acc: How well embeddings predict diabetes presence
# - probe_heart_disease_acc: How well embeddings predict heart disease
# - probe_mean_overall_acc: Average across all probe tasks
```

---

## Interpretation Guide

| Metric | Good Value | Interpretation |
|--------|------------|----------------|
| `probe_age_acc` | >0.5 | Better than random (0.33 for 3 classes) |
| `probe_gender_acc` | >0.6 | Better than random (0.5 for binary) |
| `probe_*_condition_acc` | >0.6 | Embeddings capture clinical information |
| `probe_mean_overall_acc` | >0.55 | Strong embedding quality signal |

**Key Insight**: If `probe_mean_overall_acc` is high but `tail_top10_acc` is low, your embeddings capture general patient attributes but fail on rare clinical events. If the reverse, your model predicts rare codes but doesn't learn generalizable representations.

---

You're in **ask mode**. To apply these changes, please copy the code sections above and add them to your `moe_flashattn_2.py` file at the indicated locations, or switch to **agent mode** for me to apply them directly.