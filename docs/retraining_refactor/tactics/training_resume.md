# **Comprehensive Checkpoint Management Strategy for LLM Training in GCP**

## **Executive Summary**

Based on your codebase analysis, I can see you're training hierarchical transformer models for healthcare claims prediction. Your current training has **minimal checkpoint management** (only saving best model based on validation loss). For multi-hour training with potential kernel failures, you need a **robust, industry-grade checkpoint system**.

---

## **🎯 Industry Best Practices Overview**

### **The Three Pillars of Robust Training**

1. **Checkpoint Management**: Save training state regularly
2. **Experiment Tracking**: Monitor metrics and system health
3. **Cloud Storage Integration**: Persist everything to GCS for durability

---

## **📋 Phase 1: Comprehensive Checkpoint Strategy**

### **What to Save in Checkpoints**

A complete checkpoint must include **ALL** state needed to resume training:

```python
def save_checkpoint(
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    train_metrics: Dict,
    val_metrics: Dict,
    config: Dict,
    rng_states: Dict,
    checkpoint_dir: str,
    is_best: bool = False
) -> str:
    """
    Save complete training state following industry best practices.
    
    References:
    - PyTorch official docs: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    - HuggingFace Trainer: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
    - DeepSpeed checkpointing: https://www.deepspeed.ai/tutorials/model-checkpointing/
    """
    
    checkpoint = {
        # ===== Model & Optimizer State (Required) =====
        'epoch': epoch,
        'global_step': global_step,  # Total batches processed across all epochs
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        
        # ===== Learning Rate Scheduler (If Used) =====
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        
        # ===== Training Metrics History =====
        'train_metrics': train_metrics,  # Full history: losses, accuracies per epoch
        'val_metrics': val_metrics,
        'best_val_loss': min([m['val_loss'] for m in val_metrics]) if val_metrics else float('inf'),
        
        # ===== Configuration =====
        'config': config,  # All hyperparameters, model architecture params
        
        # ===== Random State (CRITICAL for reproducibility) =====
        'rng_states': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        
        # ===== Metadata =====
        'timestamp': datetime.now(pytz.timezone("America/New_York")).isoformat(),
        'hostname': os.uname().nodename,
        'pytorch_version': torch.__version__,
        
        # ===== For MoE Models (Your Experiments) =====
        'moe_metrics': {
            'expert_usage_history': train_metrics.get('expert_usage_history', []),
            'router_entropy_history': train_metrics.get('router_entropy', []),
        } if 'expert_usage_history' in train_metrics else None,
    }
    
    # Save to local disk first (fast)
    local_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}_step{global_step}.pt')
    torch.save(checkpoint, local_path)
    
    # Save "latest" checkpoint (always overwrite)
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)
    
    # Save "best" checkpoint (if best validation loss)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, best_path)
        print(f"✅ New best model saved! Val loss: {val_metrics[-1]['val_loss']:.4f}")
    
    print(f"✅ Checkpoint saved: {local_path}")
    return local_path
```

---

### **Checkpoint Save Strategy**

**Three types of checkpoints** (industry standard from Google, OpenAI, Meta):

```python
# 1. REGULAR CHECKPOINTS: Every N epochs or M steps
#    Purpose: Resume training if interrupted
#    Retention: Keep last 3-5, delete older ones
#    Frequency: Every 25% of training (your docs suggest this)

# 2. BEST CHECKPOINT: Lowest validation loss
#    Purpose: Model selection for downstream tasks
#    Retention: Always keep
#    Update: Whenever val_loss improves

# 3. LATEST CHECKPOINT: Current state
#    Purpose: Quick resume
#    Retention: Always overwrite
#    Update: Every epoch or every K steps
```

**Example Implementation:**

```python
class CheckpointManager:
    """
    Manages checkpoint saving with automatic cleanup.
    
    Industry inspiration:
    - TensorFlow's CheckpointManager
    - PyTorch Lightning's ModelCheckpoint
    - HuggingFace's Trainer
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        gcs_bucket: str,
        gcs_prefix: str,
        max_to_keep: int = 3,
        save_frequency: str = 'epoch',  # 'epoch' or 'step'
        save_every_n: int = 1,
        upload_to_gcs: bool = True
    ):
        self.checkpoint_dir = checkpoint_dir
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        self.max_to_keep = max_to_keep
        self.save_frequency = save_frequency
        self.save_every_n = save_every_n
        self.upload_to_gcs = upload_to_gcs
        
        # Track checkpoint history
        self.checkpoint_history = []  # List of (path, metric, timestamp)
        self.best_metric = float('inf')
        
        # GCS client
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(gcs_bucket)
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def should_save(self, epoch: int, global_step: int) -> bool:
        """Determine if checkpoint should be saved."""
        if self.save_frequency == 'epoch':
            return epoch % self.save_every_n == 0
        else:  # 'step'
            return global_step % self.save_every_n == 0
    
    def save(
        self,
        epoch: int,
        global_step: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any],
        train_metrics: Dict,
        val_metrics: Dict,
        config: Dict
    ):
        """Save checkpoint and upload to GCS."""
        
        current_metric = val_metrics[-1]['val_loss']
        is_best = current_metric < self.best_metric
        
        if is_best:
            self.best_metric = current_metric
        
        # Save checkpoint locally
        local_path = save_checkpoint(
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            config=config,
            rng_states=self._get_rng_states(),
            checkpoint_dir=self.checkpoint_dir,
            is_best=is_best
        )
        
        # Upload to GCS (async in background)
        if self.upload_to_gcs:
            self._upload_to_gcs(local_path)
        
        # Track this checkpoint
        self.checkpoint_history.append({
            'path': local_path,
            'metric': current_metric,
            'timestamp': time.time(),
            'epoch': epoch,
            'step': global_step
        })
        
        # Cleanup old checkpoints (keep only max_to_keep)
        self._cleanup_old_checkpoints()
    
    def _upload_to_gcs(self, local_path: str):
        """Upload checkpoint to GCS."""
        filename = os.path.basename(local_path)
        gcs_path = f"{self.gcs_prefix}/{filename}"
        
        try:
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            print(f"☁️  Uploaded to GCS: gs://{self.gcs_bucket}/{gcs_path}")
        except Exception as e:
            print(f"⚠️  Failed to upload to GCS: {e}")
            print(f"   Checkpoint still saved locally: {local_path}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only max_to_keep."""
        if len(self.checkpoint_history) <= self.max_to_keep:
            return
        
        # Sort by timestamp, keep most recent
        sorted_checkpoints = sorted(
            self.checkpoint_history,
            key=lambda x: x['timestamp'],
            reverse=True
        )
        
        # Remove oldest checkpoints
        to_remove = sorted_checkpoints[self.max_to_keep:]
        for ckpt in to_remove:
            if 'best' not in ckpt['path'] and 'latest' not in ckpt['path']:
                try:
                    os.remove(ckpt['path'])
                    print(f"🗑️  Removed old checkpoint: {ckpt['path']}")
                except:
                    pass
        
        self.checkpoint_history = sorted_checkpoints[:self.max_to_keep]
    
    def _get_rng_states(self) -> Dict:
        """Capture all random states for reproducibility."""
        return {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module, 
                       optimizer: optim.Optimizer, scheduler: Optional[Any] = None) -> Dict:
        """
        Load checkpoint and restore ALL state.
        
        Returns:
            Dictionary with: epoch, global_step, train_metrics, val_metrics
        """
        print(f"📂 Loading checkpoint from: {checkpoint_path}")
        
        # If GCS path, download first
        if checkpoint_path.startswith('gs://'):
            checkpoint_path = self._download_from_gcs(checkpoint_path)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Restore model
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler (if exists)
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore RNG states (CRITICAL for reproducibility)
        if 'rng_states' in checkpoint:
            random.setstate(checkpoint['rng_states']['python'])
            np.random.set_state(checkpoint['rng_states']['numpy'])
            torch.set_rng_state(checkpoint['rng_states']['torch'])
            if torch.cuda.is_available() and checkpoint['rng_states']['torch_cuda']:
                torch.cuda.set_rng_state_all(checkpoint['rng_states']['torch_cuda'])
        
        print(f"✅ Checkpoint loaded successfully!")
        print(f"   Resuming from Epoch {checkpoint['epoch']}, Step {checkpoint['global_step']}")
        print(f"   Best val loss so far: {checkpoint.get('best_val_loss', 'N/A')}")
        
        return {
            'epoch': checkpoint['epoch'],
            'global_step': checkpoint['global_step'],
            'train_metrics': checkpoint.get('train_metrics', []),
            'val_metrics': checkpoint.get('val_metrics', []),
            'config': checkpoint.get('config', {}),
        }
    
    def _download_from_gcs(self, gcs_path: str) -> str:
        """Download checkpoint from GCS to local disk."""
        # Parse gs://bucket/path/to/file
        parts = gcs_path.replace('gs://', '').split('/', 1)
        bucket_name = parts[0]
        blob_path = parts[1]
        
        local_path = os.path.join(self.checkpoint_dir, os.path.basename(blob_path))
        
        blob = self.storage_client.bucket(bucket_name).blob(blob_path)
        blob.download_to_filename(local_path)
        
        print(f"☁️  Downloaded from GCS: {gcs_path}")
        return local_path
```

---

## **📊 Phase 2: Experiment Tracking Integration**

### **Why Experiment Tracking Matters**

When running multiple experiments over hours/days:
- Track metrics across experiments
- Visualize training curves in real-time
- Compare experiments side-by-side
- Never lose results even if notebook crashes

### **Recommended Tools for GCP**

#### **Option 1: Weights & Biases (W&B)** ⭐ **RECOMMENDED**

**Why W&B?**
- Industry standard (used by OpenAI, Google, Anthropic)
- Real-time visualization
- Free for academics/researchers
- Excellent integration with PyTorch
- Automatic system monitoring (GPU, memory)
- Survives notebook crashes (data logged to cloud)

**Setup (5 minutes):**

```python
import wandb

# Initialize at start of training
wandb.init(
    project="clinical-transformer-experiments",
    name=f"exp_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M')}",
    config={
        'batch_size': batch_size,
        'embedding_size': embedding_size,
        'nhead': nhead,
        'nhid': nhid,
        'nlayers': nlayers,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'model_type': 'moe' if moe_config else 'dense',
        'moe_config': moe_config.__dict__ if moe_config else None,
    },
    tags=['flash_attention', 'healthcare', 'moe'],
    notes="Experiment description here"
)

# Log during training (inside training loop)
wandb.log({
    'epoch': epoch,
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_nll': val_nll,
    'top_5_acc': top_5_acc,
    'top_10_acc': top_10_acc,
    'mrr': mrr,
    'learning_rate': optimizer.param_groups[0]['lr'],
    
    # MoE specific
    'expert_usage_std': expert_usage_std,
    'router_entropy': router_entropy,
    
    # System metrics (automatic)
    'gpu_memory_used': torch.cuda.memory_allocated() / 1e9,
}, step=global_step)

# Log checkpoints
wandb.save(checkpoint_path)  # Backs up checkpoint to W&B cloud

# Finish
wandb.finish()
```

**Benefits:**
- Dashboard updates in real-time (check from phone!)
- Compare all experiments in one view
- Checkpoint backup to W&B cloud
- Automatic alerts if training crashes

#### **Option 2: TensorBoard** (Free, GCP-native)

```python
from torch.utils.tensorboard import SummaryWriter

# Initialize
writer = SummaryWriter(log_dir=f'runs/{experiment_name}')

# Log during training
writer.add_scalar('Loss/train', train_loss, global_step)
writer.add_scalar('Loss/val', val_loss, global_step)
writer.add_scalar('Metrics/top_5_acc', top_5_acc, global_step)

# Upload logs to GCS for persistence
# tensorboard dev upload --logdir runs/
```

#### **Option 3: GCP Vertex AI TensorBoard** (Enterprise)

- Native GCP integration
- Managed service
- Best for production deployments

---

## **☁️ Phase 3: GCS Integration Strategy**

### **Why GCS for Checkpoints?**

1. **Durability**: GCS has 99.999999999% durability
2. **Notebook crashes**: Local disk lost, GCS persists
3. **Multi-session**: Resume on different VM/notebook
4. **Backup**: Checkpoints backed up automatically
5. **Cost-effective**: ~$0.02/GB/month (Standard storage)

### **GCS Checkpoint Architecture**

```
gs://your-bucket/experiments/
├── exp1_dense_baseline/
│   ├── checkpoints/
│   │   ├── checkpoint_epoch1.pt
│   │   ├── checkpoint_epoch2.pt
│   │   ├── checkpoint_epoch3.pt
│   │   ├── checkpoint_best.pt       ← Always keep
│   │   └── checkpoint_latest.pt     ← For resume
│   ├── config.yaml
│   ├── training_log.txt
│   └── metrics/
│       └── metrics_history.json
├── exp2_moe_switch/
│   └── ...
└── exp3_moe_deepseek/
    └── ...
```

### **Implementation**

```python
class GCSCheckpointSync:
    """
    Background sync of checkpoints to GCS.
    Ensures checkpoints are uploaded ASAP without blocking training.
    """
    
    def __init__(self, bucket_name: str, experiment_prefix: str):
        self.bucket_name = bucket_name
        self.experiment_prefix = experiment_prefix
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Thread pool for async uploads
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.upload_futures = []
    
    def upload_checkpoint_async(self, local_path: str):
        """Upload checkpoint in background thread."""
        future = self.executor.submit(self._upload, local_path)
        self.upload_futures.append(future)
    
    def _upload(self, local_path: str):
        """Actual upload logic."""
        filename = os.path.basename(local_path)
        gcs_path = f"{self.experiment_prefix}/checkpoints/{filename}"
        
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        
        print(f"☁️  Uploaded: gs://{self.bucket_name}/{gcs_path}")
        return gcs_path
    
    def wait_for_uploads(self):
        """Block until all uploads complete."""
        for future in self.upload_futures:
            future.result()
        self.upload_futures = []
    
    def download_checkpoint(self, gcs_path: str, local_path: str):
        """Download checkpoint from GCS."""
        blob = self.bucket.blob(gcs_path.replace(f"gs://{self.bucket_name}/", ""))
        blob.download_to_filename(local_path)
        print(f"📥 Downloaded: {local_path}")
```

---

## **🔄 Phase 4: Training Loop Integration**

### **Complete Training Loop with Checkpointing**

Here's how to integrate everything into your training:

```python
def train_with_checkpointing(
    model: nn.Module,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    config: Dict,
    resume_from: Optional[str] = None
):
    """
    Complete training loop with robust checkpointing.
    
    Args:
        resume_from: Path to checkpoint (local or gs://...) to resume from
    """
    
    # ===== Initialize =====
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs']
    )
    criterion = nn.NLLLoss()
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir='./checkpoints',
        gcs_bucket='your-bucket-name',
        gcs_prefix=f"experiments/{config['experiment_name']}",
        max_to_keep=3,
        save_frequency='epoch',
        save_every_n=1,  # Save every epoch
        upload_to_gcs=True
    )
    
    # W&B tracking
    wandb.init(
        project="clinical-transformer",
        name=config['experiment_name'],
        config=config
    )
    
    # ===== Resume from checkpoint if provided =====
    start_epoch = 0
    global_step = 0
    train_metrics_history = []
    val_metrics_history = []
    
    if resume_from:
        checkpoint_state = checkpoint_manager.load_checkpoint(
            resume_from, model, optimizer, scheduler
        )
        start_epoch = checkpoint_state['epoch'] + 1  # Resume from next epoch
        global_step = checkpoint_state['global_step']
        train_metrics_history = checkpoint_state['train_metrics']
        val_metrics_history = checkpoint_state['val_metrics']
        
        print(f"\n{'='*60}")
        print(f"🔄 RESUMING TRAINING")
        print(f"{'='*60}")
        print(f"Starting from epoch {start_epoch}")
        print(f"Global step: {global_step}")
        print(f"{'='*60}\n")
    
    # ===== Training Loop =====
    for epoch in range(start_epoch, config['num_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(
            model, train_data, optimizer, criterion, 
            device, config, epoch, global_step
        )
        
        # Update global step
        global_step = train_metrics['global_step']
        
        # Validate
        val_metrics = validate_epoch(
            model, val_data, criterion, device, config
        )
        
        # Learning rate step
        scheduler.step()
        
        # Log to W&B
        wandb.log({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'val_nll': val_metrics['nll'],
            'top_5_acc': val_metrics['top_5_acc'],
            'top_10_acc': val_metrics['top_10_acc'],
            'learning_rate': optimizer.param_groups[0]['lr'],
        }, step=global_step)
        
        # Store metrics
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)
        
        # ===== Save Checkpoint =====
        checkpoint_manager.save(
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_metrics=train_metrics_history,
            val_metrics=val_metrics_history,
            config=config
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}")
        print(f"  Top-5 Acc:  {val_metrics['top_5_acc']:.3f}")
        print(f"  Top-10 Acc: {val_metrics['top_10_acc']:.3f}")
        print(f"{'='*60}\n")
    
    # Wait for all uploads to complete
    checkpoint_manager._upload_to_gcs(checkpoint_manager.checkpoint_dir)
    
    wandb.finish()
    
    print("\n✅ Training completed successfully!")
    return model, train_metrics_history, val_metrics_history
```

### **Modified train_epoch to track global_step**

```python
def train_epoch(
    model: nn.Module,
    train_data: pd.DataFrame,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: Dict,
    epoch: int,
    global_step: int  # ← Pass in current global step
) -> Dict:
    """Train one epoch with step tracking."""
    
    model.train()
    batch_size = config['batch_size']
    nbatch = len(train_data) // batch_size
    
    total_loss = 0.0
    step_start = global_step
    
    for i in range(nbatch):
        if i % 100 == 0:
            print(f'  Batch {i}/{nbatch}, Step {global_step}')
        
        optimizer.zero_grad()
        
        # Prepare batch (your existing code)
        batch = train_data.iloc[i*batch_size:(i+1)*batch_size]
        dt_cnt, x, y = prepare_tensor(batch)
        
        # Forward
        opt = model(x)
        
        # Reshape (your existing logic)
        opt = opt.reshape(batch_size * 200, -1)
        y = [item for sublist in y for item in sublist]
        opt = torch.cat([opt[200*j:200*j+dt_cnt[j], :] for j in range(batch_size)], dim=0)
        y = torch.tensor(y).to(device)
        
        # Loss & backward
        loss = criterion(opt, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        global_step += 1
        
        # Cleanup
        del batch, x, y, opt, loss
        gc.collect()
        torch.cuda.empty_cache()
    
    avg_loss = total_loss / nbatch if nbatch > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'global_step': global_step,  # ← Return updated step
        'num_batches': nbatch
    }
```

---

## **🚀 Phase 5: Practical Usage Guide**

### **Scenario 1: Starting New Training**

```python
# Configuration
config = {
    'experiment_name': 'exp1_dense_baseline',
    'batch_size': 16,
    'embedding_size': 256,
    'nhead': 16,
    'nhid': 512,
    'nlayers': 6,
    'learning_rate': 1e-4,
    'num_epochs': 10,
}

# Start training
model = TransformerModel(...)
model = model.to(device)

train_with_checkpointing(
    model=model,
    train_data=train_df,
    val_data=val_df,
    config=config,
    resume_from=None  # ← Starting fresh
)
```

### **Scenario 2: Notebook Crashed - Resume Training**

```python
# Option A: Resume from local checkpoint
resume_from = './checkpoints/checkpoint_latest.pt'

# Option B: Resume from GCS (if local deleted)
resume_from = 'gs://your-bucket/experiments/exp1_dense_baseline/checkpoints/checkpoint_latest.pt'

# Resume training
model = TransformerModel(...)  # Recreate model architecture
model = model.to(device)

train_with_checkpointing(
    model=model,
    train_data=train_df,
    val_data=val_df,
    config=config,
    resume_from=resume_from  # ← Resume from here
)
```

### **Scenario 3: Load Best Model for Evaluation**

```python
# Load best checkpoint
best_checkpoint = './checkpoints/checkpoint_best.pt'
checkpoint = torch.load(best_checkpoint)

model = TransformerModel(...)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Extract embeddings
embeddings = score(model, test_data)
```

---

## **⚙️ Phase 6: GCP-Specific Optimizations**

### **1. Use Preemptible VMs with Auto-Resume**

Preemptible VMs cost 60-80% less but can be terminated. With checkpointing, you can auto-resume:

```python
# Create startup script that auto-resumes
startup_script = """
#!/bin/bash
cd /home/user/project
python train.py --resume_from gs://bucket/experiments/exp1/checkpoints/checkpoint_latest.pt
"""
```

### **2. GCS Performance Optimization**

```python
# For large checkpoints (>1GB), use multipart upload
blob = bucket.blob(gcs_path)
blob.chunk_size = 50 * 1024 * 1024  # 50MB chunks
blob.upload_from_filename(local_path)
```

### **3. Vertex AI Integration** (Optional)

For production training:

```python
from google.cloud import aiplatform

aiplatform.init(project='your-project', location='us-central1')

job = aiplatform.CustomTrainingJob(
    display_name='clinical-transformer-training',
    script_path='train.py',
    container_uri='gcr.io/deeplearning-platform-release/pytorch-gpu.1-13',
    requirements=['wandb', 'pandas', 'numpy'],
)

job.run(
    replica_count=1,
    machine_type='n1-highmem-8',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,
    args=['--experiment', 'exp1_dense_baseline'],
)
```

---

## **🎓 Best Practices Summary**

### **DO:**
✅ Save checkpoints every epoch (or every N steps for very long epochs)
✅ Upload to GCS immediately after saving
✅ Track global_step (total batches across all epochs)
✅ Save RNG states for reproducibility
✅ Use experiment tracking (W&B or TensorBoard)
✅ Keep "latest", "best", and last N checkpoints
✅ Test checkpoint loading before long training runs
✅ Monitor GCS upload completion before shutting down

### **DON'T:**
❌ Rely only on notebook's local disk
❌ Save only model weights (missing optimizer state → training diverges)
❌ Forget to save scheduler state
❌ Skip RNG state (non-reproducible results)
❌ Save checkpoints too infrequently (>1 hour of training lost)
❌ Keep all checkpoints forever (storage costs add up)

---

## **📝 Quick Start Checklist**

For your next training run:

```python
# 1. Install dependencies
# pip install wandb google-cloud-storage

# 2. Setup W&B (one time)
# wandb login

# 3. Create checkpoint manager
checkpoint_mgr = CheckpointManager(
    checkpoint_dir='./checkpoints',
    gcs_bucket='clin-analytics-data-hcb-dev',  # Your bucket
    gcs_prefix='transformer_experiments/exp1',
    max_to_keep=3,
    save_every_n=1
)

# 4. Initialize W&B
wandb.init(project='clinical-transformer', name='exp1_dense')

# 5. Training loop with checkpointing (see Phase 4)

# 6. If crash, resume:
# python train.py --resume gs://bucket/experiments/exp1/checkpoints/checkpoint_latest.pt
```

---

## **🔍 Monitoring During Training**

### **Terminal Output**

```
Epoch 3/10
========================================================
  Batch 0/1250, Step 2500
  Batch 100/1250, Step 2600
  ...
✅ Checkpoint saved: ./checkpoints/checkpoint_epoch3_step3750.pt
☁️  Uploaded to GCS: gs://bucket/experiments/exp1/checkpoints/checkpoint_epoch3_step3750.pt
✅ New best model saved! Val loss: 2.1234

========================================================
Epoch 3 Summary:
  Train Loss: 2.1500
  Val Loss:   2.1234
  Top-5 Acc:  0.651
  Top-10 Acc: 0.823
========================================================
```

### **W&B Dashboard**

- Real-time loss curves
- Compare multiple experiments
- Download checkpoints from UI
- Set alerts if loss spikes

---

This strategy follows industry best practices from Google (Vertex AI), OpenAI (model training infrastructure), Meta (LLaMA training), and Anthropic (Claude training). You're now equipped to handle multi-hour training runs with confidence! 🎯


# **Simple checkpoint implementation** 
## **CURRENT LOGGING SYSTEM ANALYSIS**

Your current system has:

1. **`setup_experiment_logging()`**: Creates Python logger with console + file handler
   - File handler opens in **'w' mode** (overwrites on each run)
   - Logs to `logs/{exp_name}/training.log`

2. **`MetricsLogger`**: Stores metrics in memory, saves to JSON at end
   - `epoch_metrics` list (in memory)
   - `batch_metrics` list (in memory)
   - `save()` writes to JSON files

3. **Usage**: Logger for text, MetricsLogger for structured metrics

---

## **MINIMAL INTEGRATION PLAN**

Make these changes to support resume without breaking existing code:

### **1. Modify `setup_experiment_logging()` to support resume**

```python
def setup_experiment_logging(
    exp_name: str,
    log_dir: str = "logs",
    resume: bool = False  # ← ADD THIS PARAMETER
) -> logging.Logger:
    """
    Set up comprehensive logging for experiment tracking.
    
    Args:
        resume: If True, append to existing log file instead of overwriting
    
    Creates:
    1. Console logger (INFO level)
    2. File logger (DEBUG level) - saves to logs/{exp_name}/training.log
    3. Metrics JSON logger - saves metrics to logs/{exp_name}/metrics.json
    """
    # Create log directory
    log_path = Path(log_dir) / exp_name
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear existing handlers
    
    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (DEBUG level) - CHANGE: Append mode if resuming
    file_mode = 'a' if resume else 'w'  # ← KEY CHANGE
    file_handler = logging.FileHandler(log_path / 'training.log', mode=file_mode)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Log resume marker if resuming
    if resume:
        logger.info(f"\n{'='*80}")
        logger.info(f"🔄 TRAINING RESUMED")
        logger.info(f"{'='*80}\n")
    else:
        logger.info(f"Starting experiment: {exp_name}")
    
    return logger
```

### **2. Enhance `MetricsLogger` to load existing metrics on resume**

```python
class MetricsLogger:
    """
    JSON-based metrics logger for structured experiment tracking.
    
    ENHANCED: Now supports resume - loads existing metrics if resuming.
    """
    
    def __init__(self, exp_name: str, log_dir: str = "logs", resume: bool = False):
        self.exp_name = exp_name
        self.log_path = Path(log_dir) / exp_name
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # ENHANCED: Load existing metrics if resuming
        if resume:
            # Try to load existing epoch metrics
            epoch_file = self.log_path / 'epoch_metrics.json'
            if epoch_file.exists():
                try:
                    with open(epoch_file, 'r') as f:
                        self.epoch_metrics = json.load(f)
                    print(f"📂 Loaded {len(self.epoch_metrics)} existing epochs from metrics")
                except:
                    self.epoch_metrics = []
            else:
                self.epoch_metrics = []
            
            # Try to load existing batch metrics
            batch_file = self.log_path / 'batch_metrics.json'
            if batch_file.exists():
                try:
                    with open(batch_file, 'r') as f:
                        self.batch_metrics = json.load(f)
                except:
                    self.batch_metrics = []
            else:
                self.batch_metrics = []
            
            # Log resume event to metrics
            self.epoch_metrics.append({
                'epoch': len(self.epoch_metrics),  # Next epoch number
                'resume_event': True,
                'resume_timestamp': time.time(),
                'previous_epochs': len(self.epoch_metrics) - 1
            })
        else:
            self.epoch_metrics = []
            self.batch_metrics = []
        
        self.config = {}
    
    def log_config(self, config: Dict):
        """Log experiment configuration."""
        self.config = config
        # Only save config if new experiment (not resuming)
        if not self.epoch_metrics or not any('resume_event' in m for m in self.epoch_metrics):
            config_path = self.log_path / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch-level metrics."""
        entry = {'epoch': epoch, **metrics}
        self.epoch_metrics.append(entry)
    
    def log_batch(self, epoch: int, batch: int, metrics: Dict[str, float]):
        """Log batch-level metrics (for real-time monitoring)."""
        entry = {'epoch': epoch, 'batch': batch, **metrics}
        self.batch_metrics.append(entry)
    
    def save(self):
        """Save all metrics to JSON files."""
        # Save epoch metrics
        with open(self.log_path / 'epoch_metrics.json', 'w') as f:
            json.dump(self.epoch_metrics, f, indent=2)
        
        # Save batch metrics
        with open(self.log_path / 'batch_metrics.json', 'w') as f:
            json.dump(self.batch_metrics, f, indent=2)
        
        # Save config (only if not already saved)
        if self.config and not (self.log_path / 'config.json').exists():
            with open(self.log_path / 'config.json', 'w') as f:
                json.dump(self.config, f, indent=2)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.epoch_metrics:
            return {}
        
        # Filter out resume events for summary
        real_epochs = [m for m in self.epoch_metrics if 'resume_event' not in m]
        if not real_epochs:
            return {}
        
        final_epoch = real_epochs[-1]
        best_val_loss_epoch = min(real_epochs, key=lambda x: x.get('val_loss', float('inf')))
        
        # Count resume events
        resume_count = sum(1 for m in self.epoch_metrics if 'resume_event' in m)
        
        return {
            'num_epochs': len(real_epochs),
            'final_train_loss': final_epoch.get('train_loss', 0),
            'final_val_loss': final_epoch.get('val_loss', 0),
            'best_val_loss': best_val_loss_epoch.get('val_loss', 0),
            'best_epoch': best_val_loss_epoch.get('epoch', 0),
            'resume_count': resume_count  # ← NEW: Track how many times resumed
        }
```

### **3. Add checkpoint functions (simple, PyTorch standard)**

```python
# ============================================================================
# CHECKPOINT FUNCTIONS (Simple PyTorch Standard Pattern)
# ============================================================================

def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    scaler: Optional[GradScaler],
    metrics: List[Dict],
    is_best: bool = False
) -> str:
    """
    Save checkpoint - PyTorch official pattern.
    
    Saves:
    - checkpoint_latest.pt (always, for resume)
    - checkpoint_epoch{N}.pt (every epoch)
    - checkpoint_best.pt (when val loss improves)
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': metrics,
        'timestamp': time.time()
    }
    
    # Save latest (for resume)
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)
    
    # Save epoch checkpoint
    epoch_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pt')
    torch.save(checkpoint, epoch_path)
    
    # Save best (when val loss improves)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, best_path)
    
    return latest_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any = None,
    scaler: Optional[GradScaler] = None,
    device: torch.device = None
) -> Dict:
    """
    Load checkpoint - PyTorch official pattern.
    
    Returns:
        Dict with: epoch, global_step, metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Restore states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'global_step': checkpoint['global_step'],
        'metrics': checkpoint.get('metrics', [])
    }
```

### **4. Modify `train_epoch()` to return `global_step`**

```python
def train_epoch(
    model: nn.Module,
    train_data: pd.DataFrame,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    criterion: nn.Module,
    config: BaseConfig,
    device: torch.device,
    use_mixed_precision: bool = False,
    moe_config: Optional[MoEConfig] = None,
    epoch: int = 1,
    global_step: int = 0,  # ← ADD THIS PARAMETER
    use_bucketing: bool = False,
    log_interval: int = 100 
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    ENHANCED: Now tracks global_step across epochs for resume support.
    """
    # ... existing code ...
    
    # In your training loop, increment global_step:
    for i in range(nbatch):
        # ... existing training code ...
        
        global_step += 1  # ← ADD THIS
        
        # ... rest of loop ...
    
    # Return global_step in metrics
    return {
        'train_loss': total_pred_loss / nbatch,
        'aux_loss': total_aux_loss / nbatch,
        'global_step': global_step,  # ← ADD THIS
        # ... other metrics ...
    }
```

### **5. Modify `run_single_experiment()` to support resume**

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
    check_embeddings_every: int = 2,
    log_metrics_every: int = 100,
    resume_from: Optional[str] = None,  # ← ADD THIS PARAMETER
    checkpoint_dir: Optional[str] = None  # ← ADD THIS PARAMETER
) -> Dict[str, any]:
    """
    Run a SINGLE experiment.
    
    ENHANCED: Now supports checkpoint/resume.
    
    Args:
        resume_from: Path to checkpoint.pt to resume from (e.g., './checkpoints/exp1/checkpoint_latest.pt')
        checkpoint_dir: Directory to save checkpoints (defaults to log_dir/checkpoints)
    """
    
    # Determine if resuming
    is_resume = resume_from is not None
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(log_dir, exp_name, 'checkpoints')
    
    # ============================================================
    # SETUP LOGGING (ENHANCED: Pass resume flag)
    # ============================================================
    
    logger = setup_experiment_logging(exp_name, log_dir, resume=is_resume)  # ← CHANGE
    metrics_logger = MetricsLogger(exp_name, log_dir, resume=is_resume)  # ← CHANGE
    
    if is_resume:
        logger.info(f"🔄 RESUMING from checkpoint: {resume_from}")
    else:
        logger.info(f"Starting experiment: {exp_name}")
    
    # ... existing model creation code ...
    
    # ============================================================
    # TRAINING SETUP
    # ============================================================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    criterion = nn.BCEWithLogitsLoss()
    
    # ============================================================
    # RESUME FROM CHECKPOINT (NEW)
    # ============================================================
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if is_resume:
        resumed_state = load_checkpoint(
            resume_from, model, optimizer, scheduler, scaler, device
        )
        start_epoch = resumed_state['epoch'] + 1
        global_step = resumed_state['global_step']
        
        # Get best val loss from history
        if resumed_state['metrics']:
            best_val_loss = min(m.get('val_loss', float('inf')) for m in resumed_state['metrics'])
        
        logger.info(f"Resumed from epoch {start_epoch}, global_step {global_step}")
        logger.info(f"Previous best val loss: {best_val_loss:.4f}")
    
    # ============================================================
    # TRAINING LOOP (MODIFIED: Start from start_epoch)
    # ============================================================
    logger.info(f"Training for {epochs} epochs...")
    epoch_history = []
    
    start_time = time.time()
    
    for epoch in range(start_epoch, epochs):  # ← CHANGE: Start from start_epoch
        logger.info(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Train (MODIFIED: Pass global_step)
        train_metrics = train_epoch(
            model=model,
            train_data=train_data,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            config=config,
            device=device,
            use_mixed_precision=use_mixed_precision,
            moe_config=moe_config,
            epoch=epoch,
            global_step=global_step,  # ← ADD THIS
            use_bucketing=use_bucketing,
            log_interval=log_metrics_every
        )
        
        # Update global_step
        global_step = train_metrics['global_step']  # ← ADD THIS
        
        # ... rest of existing code ...
        
        # Save checkpoint (NEW)
        is_best = epoch_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = epoch_metrics['val_loss']
        
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            metrics=epoch_history,
            is_best=is_best
        )
        logger.info(f"💾 Checkpoint saved (epoch {epoch+1}, step {global_step})")
    
    # ... rest of existing code ...
```

---

## **WHAT THIS INTEGRATION PROVIDES**

### **Backward Compatible:**
- Existing code works unchanged (resume_from=None by default)
- No breaking changes to function signatures (only adds optional params)
- Existing logs/metrics format unchanged

### **Resume Support:**
- Log file appends (doesn't overwrite) when resuming
- Metrics load from existing JSON files
- Resume events tracked in metrics
- Checkpoint saves all training state

### **What Gets Logged:**

**Text Log (`training.log`):**
```
2025-01-10 14:23:15 - exp1 - INFO - Starting experiment: exp1
2025-01-10 14:24:10 - exp1 - INFO - Epoch 1: Train Loss: 2.5
<crash>

2025-01-10 15:00:00 - exp1 - INFO - ================================================================================
2025-01-10 15:00:00 - exp1 - INFO - 🔄 TRAINING RESUMED
2025-01-10 15:00:00 - exp1 - INFO - ================================================================================
2025-01-10 15:00:01 - exp1 - INFO - Resumed from epoch 1, global_step 1250
2025-01-10 15:00:50 - exp1 - INFO - Epoch 2: Train Loss: 2.2
```

**Metrics JSON (`epoch_metrics.json`):**
```json
[
  {"epoch": 0, "train_loss": 2.5, "val_loss": 2.3, ...},
  {"epoch": 1, "resume_event": true, "resume_timestamp": 1736532000, "previous_epochs": 1},
  {"epoch": 2, "train_loss": 2.2, "val_loss": 2.1, ...}
]
```

---

## **USAGE EXAMPLE**

```python
# Start fresh
results = run_single_experiment(
    exp_name='exp1',
    moe_config=None,
    use_learnt_att_pool=False,
    train_data=df_train,
    val_data=df_val,
    device=device,
    epochs=10
)

# Resume after crash
results = run_single_experiment(
    exp_name='exp1',
    moe_config=None,
    use_learnt_att_pool=False,
    train_data=df_train,
    val_data=df_val,
    device=device,
    epochs=10,
    resume_from='./logs/exp1/checkpoints/checkpoint_latest.pt'  # ← Just add this
)
```

---

## **SUMMARY**

- Minimal changes: only 3 functions modified, 2 new functions added
- Backward compatible: existing code works unchanged
- Industry standard: PyTorch checkpoint pattern
- Integrated logging: resume events visible in logs and metrics
- No complexity: simple, maintainable code

Should I switch to agent mode and implement these changes directly in your `moe_flashattn_1.py`?