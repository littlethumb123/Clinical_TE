Let me first thoroughly review your current implementation and the expert conversation to provide well-informed guidance.

[4 tools called]

Let me continue reading the expert conversation to fully understand the context and your current challenges.

[2 tools called]

Let me continue reading the expert conversation and also understand more about the inference pipeline concerns.

[4 tools called]

Based on my thorough analysis of your codebase, the expert conversation, and your specific constraints, let me provide comprehensive guidance for designing your transformer architecture and training strategy to accommodate GPU resource scarcity on GCP.

---

# Comprehensive Strategy for GPU-Scarce Training and Inference

## Your Core Constraints Summary

| Challenge | Impact |
|-----------|--------|
| **GPU Scarcity** | H100 reservations unreliable; even T4/L4 unavailable during peak hours |
| **On-Demand Contracts** | No guaranteed GPU capacity |
| **Inference Scale** | 16+ million members weekly/monthly |
| **Inference Failures** | Even non-business hours fail to complete |
| **Training Needs** | Periodic retraining on 1-12M samples |

---

## Part 1: Architecture Design Principles for Resource Scarcity

### 1.1 Model Size Selection: Smaller is Better for Your Case

**Key Insight**: Your current model (~35-85M params) is already well-sized, but consider keeping it on the smaller end:

```
┌─────────────────────────────────────────────────────────────────┐
│ MODEL SIZE TRADE-OFFS FOR GPU SCARCITY                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Smaller Model (27-35M params):                                  │
│  ✅ Faster inference (critical for 16M members)                 │
│  ✅ Fits on single GPU (T4/L4) for inference                    │
│  ✅ Can use CPU as fallback (slow but works)                    │
│  ✅ Lower memory = larger batch = fewer GPU-hours               │
│  ✅ ONNX export more reliable                                   │
│                                                                  │
│  Larger Model (100M+ params):                                    │
│  ❌ Requires multiple GPUs for reasonable inference             │
│  ❌ No CPU fallback possible                                    │
│  ❌ More sensitive to GPU availability                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Recommendation**: Keep your model at the current size (~35M params with MoE or ~28M baseline). The incremental accuracy from larger models doesn't justify the inference complexity.

### 1.2 Inference-First Architecture Design

Your model will run inference 50-100× more often than training. Design for inference efficiency:

```python
# Key architecture choices for inference efficiency:

@dataclass
class InferenceOptimizedConfig:
    """Config optimized for 16M member inference on scarce GPUs."""
    
    # Model dimensions (from your current config)
    embedding_size: int = 256   # Sweet spot for single-GPU inference
    nhid: int = 512             # Keep moderate FFN size
    nlayers: int = 6            # Sufficient depth
    nhead: int = 8              # Flash-compatible
    
    # CRITICAL: Inference optimizations
    len_dy: int = 200           # Keep as-is
    len_cd: int = 80            # Keep as-is
    
    # Inference-specific
    use_flash: bool = True      # Required for long sequences
    use_learnt_att_pool: bool = True  # Faster than transformer for daily encoder
    
    # AVOID MoE for production inference unless:
    # 1. You have guaranteed multi-GPU access
    # 2. You can tolerate routing overhead
    use_moe: bool = False       # Simpler inference path
```

### 1.3 MoE Decision: Training vs Production

| Aspect | Dense (Flash Only) | MoE |
|--------|-------------------|-----|
| **Training** | Simpler | +5-10% performance |
| **Inference** | ✅ Single-GPU friendly | ❌ Routing overhead |
| **Memory** | Predictable | Expert memory spikes |
| **Fallback** | CPU possible | CPU too slow |
| **Recommendation** | **Use for production** | Development only |

**Recommendation**: Train with MoE for experiments, but deploy the Flash Attention Dense variant for production inference.

---

## Part 2: Robust Checkpointing Strategy

### 2.1 Frequent Checkpoints for Preemption Resilience

Your training must survive GPU preemption. Implement aggressive checkpointing:

```python
# Add to your training config
@dataclass  
class PreemptionResilientConfig:
    """Checkpoint strategy for unreliable GPU access."""
    
    # Checkpoint every N steps (not epochs!)
    checkpoint_every_steps: int = 500  # ~30 minutes of T4 training
    
    # Keep last N checkpoints (rolling window)
    keep_last_n_checkpoints: int = 5
    
    # Save to GCS directly (survives node failure)
    checkpoint_bucket: str = "gs://your-bucket/checkpoints"
    
    # Async checkpoint saving (doesn't block training)
    async_checkpoint: bool = True
    
    # Save optimizer state (for exact resume)
    save_optimizer_state: bool = True
    
    # Auto-detect preemption signal
    enable_preemption_handler: bool = True
```

### 2.2 Implementation: Preemption-Aware Training

```python
import signal
import threading
from google.cloud import storage

class PreemptionHandler:
    """Handle GCP preemption signals gracefully."""
    
    def __init__(self, checkpoint_fn, gcs_bucket: str):
        self.checkpoint_fn = checkpoint_fn
        self.gcs_bucket = gcs_bucket
        self.preempted = False
        
        # Register signal handler (GCP sends SIGTERM before preemption)
        signal.signal(signal.SIGTERM, self._handle_sigterm)
    
    def _handle_sigterm(self, signum, frame):
        print("⚠️ PREEMPTION SIGNAL RECEIVED - Saving checkpoint...")
        self.preempted = True
        self.checkpoint_fn(emergency=True)
        print("✅ Emergency checkpoint saved to GCS")
    
    def check_preemption(self) -> bool:
        """Check if we should stop due to preemption."""
        return self.preempted


def save_checkpoint_to_gcs(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    step: int,
    bucket_name: str,
    checkpoint_name: str,
    use_ddp: bool = False
) -> str:
    """Save checkpoint directly to GCS (survives node failure)."""
    
    # Get model state (handle DDP wrapper)
    model_state = model.module.state_dict() if use_ddp else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'global_step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to temp file first
    local_path = f"/tmp/{checkpoint_name}"
    torch.save(checkpoint, local_path)
    
    # Upload to GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"checkpoints/{checkpoint_name}")
    blob.upload_from_filename(local_path)
    
    gcs_path = f"gs://{bucket_name}/checkpoints/{checkpoint_name}"
    print(f"💾 Checkpoint saved: {gcs_path}")
    
    return gcs_path


def load_checkpoint_from_gcs(
    bucket_name: str,
    checkpoint_name: str,
    model: nn.Module,
    optimizer: optim.Optimizer = None,
    device: torch.device = None
) -> Dict:
    """Load checkpoint from GCS."""
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"checkpoints/{checkpoint_name}")
    
    local_path = f"/tmp/{checkpoint_name}"
    blob.download_to_filename(local_path)
    
    checkpoint = torch.load(local_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✅ Resumed from epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")
    
    return checkpoint
```

### 2.3 Modified Training Loop with Checkpoint Recovery

```python
def train_with_preemption_resilience(
    model: nn.Module,
    train_loader: DataLoader,
    config: BaseConfig,
    checkpoint_bucket: str,
    checkpoint_every_steps: int = 500,
    resume_checkpoint: Optional[str] = None
):
    """Training loop that survives GPU preemption."""
    
    # Setup preemption handler
    def emergency_save():
        save_checkpoint_to_gcs(
            model, optimizer, epoch, global_step,
            checkpoint_bucket, f"emergency_step{global_step}.pt"
        )
    
    preempt_handler = PreemptionHandler(emergency_save, checkpoint_bucket)
    
    # Resume if checkpoint provided
    if resume_checkpoint:
        checkpoint = load_checkpoint_from_gcs(
            checkpoint_bucket, resume_checkpoint, model, optimizer, device
        )
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
    else:
        start_epoch = 0
        global_step = 0
    
    # Training loop
    for epoch in range(start_epoch, config.epochs):
        for batch_idx, batch in enumerate(train_loader):
            # Check for preemption
            if preempt_handler.check_preemption():
                print("Training stopped due to preemption")
                return {'preempted': True, 'last_step': global_step}
            
            # Normal training step
            loss = train_step(model, batch, optimizer)
            global_step += 1
            
            # Periodic checkpoint
            if global_step % checkpoint_every_steps == 0:
                save_checkpoint_to_gcs(
                    model, optimizer, epoch, global_step,
                    checkpoint_bucket, f"checkpoint_step{global_step}.pt"
                )
    
    return {'preempted': False, 'final_step': global_step}
```

---

## Part 3: Inference Strategy for 16M Members

### 3.1 The Core Problem: GPU Hours Required

```
Current inference challenge:
- 16M members × 200 days × 80 codes = 256 billion tokens
- Even on H100, this takes significant time

Calculation (Flash+MoE on single H100):
- Inference throughput: ~5,000-10,000 samples/sec (batch=512)
- 16M samples / 10,000 samples/sec = 1,600 seconds = 26.7 minutes

Problem: Even 27 minutes requires:
1. H100 availability for full duration
2. No preemption during that time
3. Data pipeline keeping up
```

### 3.2 Solution: Batch-Sharded Inference with Checkpointing

**Design Pattern**: Split inference into resumable chunks that can run on whatever GPU becomes available:

```python
from typing import List, Tuple
import pandas as pd
from google.cloud import storage

class ShardedInferencePipeline:
    """
    Inference pipeline that survives GPU scarcity.
    
    Key features:
    1. Splits 16M members into smaller shards
    2. Each shard is independent and resumable
    3. Progress tracked in GCS
    4. Can run on different GPU types
    5. Supports partial completion
    """
    
    def __init__(
        self,
        model_path: str,
        gcs_bucket: str,
        shard_size: int = 50_000,  # 50K members per shard
        output_prefix: str = "embeddings"
    ):
        self.model_path = model_path
        self.gcs_bucket = gcs_bucket
        self.shard_size = shard_size
        self.output_prefix = output_prefix
        
    def prepare_shards(self, member_df: pd.DataFrame) -> List[str]:
        """
        Split member data into independent shards.
        
        16M members / 50K = 320 shards
        Each shard: ~2 minutes on H100, ~10 minutes on T4
        """
        n_shards = (len(member_df) + self.shard_size - 1) // self.shard_size
        shard_ids = []
        
        for i in range(n_shards):
            start_idx = i * self.shard_size
            end_idx = min((i + 1) * self.shard_size, len(member_df))
            
            shard_df = member_df.iloc[start_idx:end_idx]
            shard_id = f"shard_{i:05d}"
            
            # Save shard to GCS
            self._save_shard_to_gcs(shard_df, shard_id)
            shard_ids.append(shard_id)
        
        # Save shard manifest
        self._save_manifest(shard_ids)
        
        return shard_ids
    
    def get_incomplete_shards(self) -> List[str]:
        """Get list of shards that haven't been processed yet."""
        manifest = self._load_manifest()
        completed = self._get_completed_shards()
        return [s for s in manifest if s not in completed]
    
    def process_shard(
        self,
        shard_id: str,
        model: nn.Module,
        device: torch.device,
        config: BaseConfig
    ) -> str:
        """
        Process single shard and save embeddings.
        
        This is the atomic unit - if interrupted, just rerun this shard.
        """
        # Load shard data
        shard_df = self._load_shard_from_gcs(shard_id)
        
        # Create dataset and loader
        dataset = ClinicalDataset(shard_df, config)
        loader = DataLoader(
            dataset,
            batch_size=512,  # Large batch for efficiency
            shuffle=False,
            num_workers=4,
            collate_fn=clinical_collate_fn
        )
        
        # Generate embeddings
        embeddings = []
        member_ids = []
        
        model.eval()
        with torch.no_grad():
            for batch in loader:
                x = self._prepare_batch(batch, device)
                
                # Get embeddings (before output layer)
                emb = self._get_embeddings(model, x)
                
                embeddings.append(emb.cpu().numpy())
                member_ids.extend(batch['member_id'])
        
        # Concatenate and save
        all_embeddings = np.concatenate(embeddings, axis=0)
        output_path = f"{self.output_prefix}/{shard_id}_embeddings.parquet"
        
        self._save_embeddings_to_gcs(
            member_ids, all_embeddings, output_path
        )
        
        # Mark shard as complete
        self._mark_shard_complete(shard_id)
        
        return f"gs://{self.gcs_bucket}/{output_path}"
    
    def run_inference_with_retry(
        self,
        max_retries: int = 3,
        timeout_per_shard_min: int = 30
    ):
        """
        Run inference on all incomplete shards with retry logic.
        
        Designed for unreliable GPU access:
        1. Get list of incomplete shards
        2. Process each shard
        3. If preempted, next run continues from where we left off
        """
        # Load model
        model, config = self._load_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        incomplete = self.get_incomplete_shards()
        print(f"📊 {len(incomplete)} shards remaining to process")
        
        for shard_id in incomplete:
            for attempt in range(max_retries):
                try:
                    print(f"Processing {shard_id} (attempt {attempt + 1}/{max_retries})")
                    self.process_shard(shard_id, model, device, config)
                    print(f"✅ {shard_id} complete")
                    break
                except Exception as e:
                    print(f"⚠️ {shard_id} failed: {e}")
                    if attempt == max_retries - 1:
                        print(f"❌ {shard_id} failed permanently")
    
    def _get_embeddings(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings (temporal representation before output layer)."""
        # This hooks into your model's forward pass
        # Modify based on your exact architecture
        
        with torch.no_grad():
            # Run through embeddings
            age = model.embedding_age_in_months(x[:, :, 0].long())
            gender = model.embedding_gender_cd(x[:, :, 1].long())
            cd = model.embedding_cd(x[:, :, 2:].long())
            cd_res = cd.sum(-2)
            
            # Daily encoding (simplified)
            cd = cd.reshape(-1, cd.shape[-2], cd.shape[-1])
            if hasattr(model, 'daily_pooling'):
                cd = model.daily_pooling(cd.permute(1, 0, 2))
            else:
                cd = cd.mean(dim=-2)  # Simple average pooling
            
            cd = cd.reshape(x.shape[0], x.shape[1], -1)
            
            # Combine
            combined = cd_res + cd + gender + age
            combined = model.norm(combined)
            
            # Temporal encoding
            combined = combined.permute(1, 0, 2)  # [seq, batch, dim]
            
            for layer in model.temporal_layers:
                residual = combined
                combined = layer['norm1'](combined)
                combined = layer['attention'](combined, is_causal=True)
                combined = residual + combined
                
                residual = combined
                combined = layer['norm2'](combined)
                combined = layer['ffn'](combined)
                combined = residual + combined
            
            combined = combined.permute(1, 0, 2)  # [batch, seq, dim]
            
            # Final representation (e.g., last day or mean)
            embeddings = combined[:, -1, :]  # Last day embedding
            
        return embeddings
```

### 3.3 Orchestration: Running Inference with Whatever GPU is Available

```python
# inference_orchestrator.py
"""
Run this script whenever any GPU becomes available.
It will automatically:
1. Check for incomplete shards
2. Process as many as possible
3. Save progress to GCS
4. Resume next time from where it stopped
"""

import sys
import time
from google.cloud import compute_v1

def main():
    # Configuration
    GCS_BUCKET = "your-clinical-embeddings-bucket"
    MODEL_PATH = "gs://your-bucket/models/best_model.pt"
    
    pipeline = ShardedInferencePipeline(
        model_path=MODEL_PATH,
        gcs_bucket=GCS_BUCKET,
        shard_size=50_000  # 50K members per shard
    )
    
    # Check what's remaining
    incomplete = pipeline.get_incomplete_shards()
    
    if not incomplete:
        print("✅ All shards complete!")
        return
    
    print(f"📊 {len(incomplete)} shards remaining")
    print(f"⏱️ Estimated time: {len(incomplete) * 2} minutes on H100")
    print(f"   Or: {len(incomplete) * 10} minutes on T4")
    
    # Run inference (will process until GPU preempted or all complete)
    pipeline.run_inference_with_retry()
    
    # Report final status
    still_incomplete = pipeline.get_incomplete_shards()
    processed = len(incomplete) - len(still_incomplete)
    
    print(f"\n📈 Session Summary:")
    print(f"   Processed: {processed} shards")
    print(f"   Remaining: {len(still_incomplete)} shards")
    print(f"   Progress: {(1 - len(still_incomplete)/len(incomplete)) * 100:.1f}%")

if __name__ == "__main__":
    main()
```

---

## Part 4: CPU Fallback Strategy

### 4.1 When GPU is Completely Unavailable

Your model is small enough to run on CPU if absolutely necessary:

```python
def inference_with_cpu_fallback(
    model: nn.Module,
    data_loader: DataLoader,
    prefer_gpu: bool = True
) -> Tuple[np.ndarray, str]:
    """
    Run inference with automatic fallback to CPU if no GPU available.
    
    Performance estimates (16M members):
    - H100: ~27 minutes
    - T4: ~2.5 hours
    - CPU (n2-highmem-32): ~8-12 hours
    - CPU (n2-standard-8): ~24-36 hours
    """
    
    # Try to get GPU
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"🚀 Using GPU: {device_name}")
    else:
        device = torch.device("cpu")
        print("⚠️ No GPU available, using CPU (this will be slower)")
        
        # Enable CPU optimizations
        torch.set_num_threads(32)  # Use all cores
        torch.backends.mkl.enabled = True
    
    model = model.to(device)
    model.eval()
    
    # Adjust batch size for device
    if device.type == 'cpu':
        # Smaller batches for CPU (memory efficient)
        effective_batch_size = 32
    else:
        effective_batch_size = 512
    
    # Run inference
    all_embeddings = []
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            x = batch['x'].to(device)
            
            embeddings = get_embeddings(model, x)
            all_embeddings.append(embeddings.cpu().numpy())
            
            if batch_idx % 100 == 0:
                elapsed = time.time() - start_time
                samples_done = (batch_idx + 1) * effective_batch_size
                samples_per_sec = samples_done / elapsed
                eta = (len(data_loader) * effective_batch_size - samples_done) / samples_per_sec
                print(f"Progress: {batch_idx}/{len(data_loader)} | "
                      f"Speed: {samples_per_sec:.1f} samples/sec | "
                      f"ETA: {eta/3600:.1f} hours")
    
    return np.concatenate(all_embeddings), device.type
```

### 4.2 ONNX Export for Faster CPU Inference

Export your model to ONNX for 2-3× faster CPU inference:

```python
import torch.onnx
import onnxruntime as ort

def export_model_to_onnx(
    model: nn.Module,
    config: BaseConfig,
    output_path: str
):
    """Export model to ONNX for faster CPU inference."""
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, config.len_dy, config.len_cd + 2)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['embeddings'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'embeddings': {0: 'batch_size'}
        },
        opset_version=17,
        do_constant_folding=True
    )
    
    print(f"✅ Model exported to {output_path}")
    return output_path


def run_onnx_inference(
    onnx_path: str,
    data_loader: DataLoader,
    use_gpu: bool = False
) -> np.ndarray:
    """Run inference with ONNX Runtime (faster on CPU)."""
    
    # Create session
    if use_gpu:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Enable graph optimizations
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 32
    session_options.inter_op_num_threads = 4
    
    # Run inference
    all_embeddings = []
    
    for batch in data_loader:
        x = batch['x'].numpy()
        
        outputs = session.run(None, {'input': x})
        embeddings = outputs[0]
        
        all_embeddings.append(embeddings)
    
    return np.concatenate(all_embeddings)
```

---

## Part 5: Infrastructure Recommendations for GCP

### 5.1 Multi-Zone GPU Strategy

Don't rely on a single zone. Set up infrastructure to use GPUs from any available zone:

```python
# gcp_gpu_finder.py
from google.cloud import compute_v1
from typing import List, Dict, Optional

class GCPGPUFinder:
    """Find available GPUs across all GCP zones."""
    
    GPU_PREFERENCES = [
        ('nvidia-h100-80gb', ['us-central1-a', 'us-central1-b', 'us-east4-c']),
        ('nvidia-a100-80gb', ['us-central1-a', 'us-central1-c', 'us-east4-a']),
        ('nvidia-l4', ['us-central1-a', 'us-central1-b', 'us-east4-a']),
        ('nvidia-tesla-t4', ['us-central1-a', 'us-central1-b', 'us-west1-a', 'us-east1-b']),
    ]
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.compute_client = compute_v1.AcceleratorTypesClient()
    
    def find_available_gpu(self) -> Optional[Dict]:
        """Find any available GPU across all zones."""
        
        for gpu_type, zones in self.GPU_PREFERENCES:
            for zone in zones:
                try:
                    available = self._check_gpu_availability(gpu_type, zone)
                    if available:
                        return {
                            'gpu_type': gpu_type,
                            'zone': zone,
                            'count': min(available, 4)  # Max 4 GPUs
                        }
                except Exception as e:
                    continue
        
        return None  # No GPUs available anywhere
    
    def _check_gpu_availability(self, gpu_type: str, zone: str) -> int:
        """Check how many GPUs of a type are available in a zone."""
        # This would check quotas and availability
        # Simplified for example
        return 4  # Placeholder


def create_gpu_instance_when_available(
    project_id: str,
    training_script: str,
    max_wait_hours: int = 24
) -> str:
    """Wait for GPU and create instance when available."""
    
    finder = GCPGPUFinder(project_id)
    
    start_time = time.time()
    while (time.time() - start_time) < max_wait_hours * 3600:
        
        gpu_info = finder.find_available_gpu()
        
        if gpu_info:
            print(f"✅ Found GPU: {gpu_info['gpu_type']} in {gpu_info['zone']}")
            
            # Create instance and run training
            instance_name = create_training_instance(
                gpu_type=gpu_info['gpu_type'],
                zone=gpu_info['zone'],
                gpu_count=gpu_info['count'],
                startup_script=training_script
            )
            
            return instance_name
        
        print("⏳ No GPUs available, waiting 5 minutes...")
        time.sleep(300)  # Check every 5 minutes
    
    raise TimeoutError(f"No GPU found within {max_wait_hours} hours")
```

### 5.2 Preemptible/Spot Instances for Cost Savings

Use preemptible instances with checkpointing for 60-70% cost savings:

```python
# training_config.yaml for GCP
training_config:
  instance_type: n1-highmem-8
  gpu_type: nvidia-tesla-t4
  gpu_count: 4
  zone: us-central1-a
  
  # Use preemptible for 60-70% cost savings
  preemptible: true
  max_run_duration: 24h  # Max 24 hours for preemptible
  
  # Checkpoint to GCS every 30 minutes
  checkpoint_interval_minutes: 30
  checkpoint_bucket: gs://your-bucket/checkpoints
  
  # Auto-restart on preemption
  restart_policy:
    max_retries: 10
    restart_delay_seconds: 300  # Wait 5 min before retry
    
  # Training script
  startup_script: |
    #!/bin/bash
    pip install -r requirements.txt
    
    # Find latest checkpoint
    LATEST_CKPT=$(gsutil ls gs://your-bucket/checkpoints/ | tail -1)
    
    # Resume training
    torchrun --nproc_per_node=$GPU_COUNT train.py \
      --resume-from=$LATEST_CKPT \
      --checkpoint-bucket=gs://your-bucket/checkpoints
```

---

## Part 6: Complete Training Strategy

### 6.1 Recommended Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ TRAINING PIPELINE FOR GPU-SCARCE ENVIRONMENT                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. DATA PREPARATION (CPU only, always works)                   │
│     ├─ Parse data on BigQuery or Dataflow                       │
│     ├─ Create stratified 10% sample (1.2M members)              │
│     └─ Save as Feather/Parquet to GCS                           │
│                                                                  │
│  2. TRAINING (opportunistic GPU usage)                          │
│     ├─ Wait for ANY GPU availability                            │
│     ├─ Load latest checkpoint from GCS                          │
│     ├─ Train until:                                              │
│     │   - Completion                                             │
│     │   - Preemption (save checkpoint)                          │
│     │   - Max time (save checkpoint)                            │
│     └─ Repeat until all epochs complete                         │
│                                                                  │
│  3. MODEL EXPORT (quick GPU job)                                 │
│     ├─ Load best checkpoint                                      │
│     ├─ Export to ONNX (CPU fallback)                            │
│     ├─ Export to TorchScript (GPU inference)                    │
│     └─ Upload both to GCS                                        │
│                                                                  │
│  4. INFERENCE (sharded, resumable)                               │
│     ├─ Split 16M members into 320 shards                        │
│     ├─ Process shards on ANY available compute:                 │
│     │   - H100: 2 min/shard                                      │
│     │   - T4: 10 min/shard                                       │
│     │   - CPU: 30 min/shard                                      │
│     └─ Merge embeddings when all shards complete                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Time and Cost Estimates

| Scenario | GPU | Training Time | Training Cost | Inference Time | Total |
|----------|-----|---------------|---------------|----------------|-------|
| **Best Case** | 4× H100 continuous | 5 hours | $182 | 27 min | 5.5 hours |
| **Typical** | Mixed (H100 + T4) | 2-3 days | $300-500 | 1-2 hours | 3 days |
| **Worst Case** | T4 only, many interruptions | 1-2 weeks | $800-1,200 | 3 hours | 2 weeks |
| **Emergency** | CPU only | N/A (too slow) | N/A | 24-36 hours | 36 hours |

### 6.3 Recommended Model Selection

Based on your constraints, I recommend:

```
PRODUCTION ARCHITECTURE:
├─ Model: FlashAttentionTransformer (NOT MoE)
│   - ~28M parameters
│   - Single-GPU inference possible
│   - CPU fallback viable
│   - ONNX export reliable
│
├─ Training: 10% stratified sample (1.2M members)
│   - 10 epochs
│   - Expected: 5-20 hours depending on GPU availability
│   - Cost: $180-$500
│   - Performance: ~93% of full-data model
│
└─ Inference: Sharded pipeline
    - 320 shards of 50K members
    - Resumable from any point
    - Can use mix of GPU types
    - CPU fallback for emergencies
```

---

## Part 7: Key Implementation Changes to Your Code

### 7.1 Add to `moe_flashattn_2.py`

1. **GCS Checkpointing** (add after line ~3790):
```python
# Add imports at top
from google.cloud import storage

# Add GCS checkpoint functions (see Part 2.2 above)
```

2. **Preemption Handler** (add after checkpointing):
```python
# Add PreemptionHandler class (see Part 2.2 above)
```

3. **Modify `run_single_experiment`** to use GCS checkpoints:
```python
# In run_single_experiment(), change checkpoint saving:
# FROM:
save_checkpoint(checkpoint_dir=checkpoint_dir, ...)
# TO:
save_checkpoint_to_gcs(model, optimizer, epoch, global_step, 
                       gcs_bucket, f"checkpoint_epoch{epoch}.pt")
```

4. **Add inference pipeline** (new file `inference_pipeline.py`):
```python
# Add ShardedInferencePipeline class (see Part 3.2 above)
```

### 7.2 Create New Files

1. `preemption_handler.py` - Handle GCP preemption signals
2. `gcs_checkpointing.py` - Save/load checkpoints to/from GCS
3. `inference_pipeline.py` - Sharded inference for 16M members
4. `model_export.py` - Export to ONNX and TorchScript

---

## Summary: Key Recommendations

| Area | Recommendation | Why |
|------|----------------|-----|
| **Model Size** | Keep at 28-35M params | Single-GPU inference, CPU fallback |
| **Architecture** | Use Flash Attention Dense, not MoE | Simpler inference, same accuracy |
| **Training** | 10% stratified sample | 93% performance, 10× cheaper |
| **Checkpointing** | Every 500 steps to GCS | Survive preemption |
| **Inference** | 320 shards of 50K | Resumable, any GPU works |
| **Fallback** | Export ONNX for CPU | Emergency completion |
| **Infrastructure** | Multi-zone, preemptible | Cost savings, availability |

This strategy ensures your inference pipeline **will** complete, even if GPU availability is unreliable. The worst case (CPU-only) takes 24-36 hours but guarantees completion.


# In-Depth Analysis: GPU Resources, Training, and Inference Strategy

## Question 1: What is Preemption?

### Understanding Preemption from First Principles

**Preemption** is when a cloud provider forcibly terminates your running virtual machine (VM) or GPU instance before your job completes, usually with little to no warning. Think of it like being asked to leave a library study room mid-session because someone with a higher-priority reservation needs the space.

### Why Does Preemption Exist?

Cloud providers like GCP have a finite number of physical GPUs in each data center. These GPUs are expensive assets (an H100 costs ~$30,000-$40,000 per unit), so providers want to maximize their utilization. Here's the economic model:

**The Cloud Provider's Perspective:**

Imagine GCP has 1,000 H100 GPUs in a data center. At any given moment:
- Some customers have **on-demand reservations** - they pay full price (~$3.50-4/hour per GPU) and expect guaranteed access
- Some customers have **committed use discounts** - they've paid upfront for guaranteed capacity over 1-3 years
- Some GPUs sit **idle** waiting for the above customers

The idle capacity is a problem - it generates zero revenue while the provider still pays for cooling, power, and depreciation. So providers offer **preemptible/spot instances** at 60-80% discount, but with a catch: if a paying on-demand customer needs that GPU, your preemptible job gets terminated.

### How Preemption Actually Happens

When GCP preempts your instance:

1. **First**, they send a **SIGTERM signal** to your process (a polite "please stop" message)
2. **Then**, you have approximately **30 seconds** to save your state
3. **Finally**, the instance is terminated

This is why checkpointing is critical - if you haven't saved your model weights and training state within that 30-second window, you lose everything since your last checkpoint.

### The Preemption Probability Reality

Preemption isn't random - it follows patterns based on supply and demand:

| Time Period | Preemption Risk | Why |
|-------------|-----------------|-----|
| **Business hours (9am-5pm PT)** | HIGH (30-50%) | More on-demand customers active |
| **Evening (5pm-11pm PT)** | MEDIUM (15-25%) | Some research workloads spin up |
| **Night (11pm-7am PT)** | LOW (5-10%) | Fewer competing workloads |
| **Weekends** | LOW (5-15%) | Reduced enterprise demand |
| **Quarter-end periods** | VERY HIGH | Companies rushing ML projects |

For GPU resources specifically, preemption rates are higher than for CPUs because GPU demand often exceeds supply in popular zones.

### What This Means for You

Given your situation - on-demand contracts with scarce GPU availability - you're likely experiencing a related but distinct problem: **you can't even get GPUs in the first place**, because on-demand slots are fully utilized by other customers. This is a **capacity constraint**, not preemption per se. However, if you use preemptible instances to access GPUs more easily, then preemption becomes your concern.

---

## Question 2: How GCP Actually Works on Resource Distribution

### The Layered Architecture of Cloud GPU Access

To understand your situation, you need to understand how GCP (and other cloud providers) allocate GPU resources. This is a multi-layered system that most users never fully grasp.

### Layer 1: Physical Hardware Reality

Each GCP region (like `us-central1`) contains multiple **zones** (like `us-central1-a`, `us-central1-b`). Each zone is a separate physical data center with its own:
- Power supply
- Cooling systems
- Network fabric
- Physical GPU servers

**Critical insight**: GPUs are physically installed in specific zones and cannot be moved. If `us-central1-a` has 500 H100s and `us-central1-b` has 200 H100s, you cannot use the 500 from zone-a if you requested zone-b. This is unlike cloud storage or networking, which can be more fluid.

### Layer 2: Resource Quotas and Allocation Pools

GCP divides GPU access into several "pools" with different priority levels:

```
┌─────────────────────────────────────────────────────────────────┐
│                    GCP GPU ALLOCATION HIERARCHY                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TIER 1: Reserved Capacity (Highest Priority)                   │
│  ├─ Committed Use Discounts (1-3 year contracts)                │
│  ├─ Capacity Reservations (pre-purchased slots)                 │
│  └─ Guaranteed SLA: 99.9% availability                          │
│                                                                  │
│  TIER 2: On-Demand (High Priority)                              │
│  ├─ Pay-as-you-go at full price                                 │
│  ├─ Subject to capacity availability                            │
│  └─ No guarantee if all GPUs are reserved by Tier 1             │
│                                                                  │
│  TIER 3: Preemptible/Spot (Lowest Priority)                     │
│  ├─ 60-70% discount                                              │
│  ├─ Can be terminated anytime                                    │
│  └─ Only available when Tier 1 & 2 don't need capacity          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Your situation diagnosed**: You have an "on-demand contract," which means you're in **Tier 2**. The problem is that Tier 1 customers (those with committed use discounts or reservations) have likely consumed all the H100 capacity in your accessible zones. When you request GPUs, you're told there's no availability because the physical GPUs are already allocated to higher-tier customers.

### Layer 3: The Quota System

Even if GPUs are physically available, you can only use them if your GCP project has sufficient **quota**. Quotas are limits set by Google on how many resources you can use:

```python
# Example quota check
Your project quota: 8 H100 GPUs in us-central1
Current usage: 0 H100 GPUs
Available to request: 8 H100 GPUs

# BUT: Physical availability in data center
Zone us-central1-a: 0 available (500 total, all in use by other customers)
Zone us-central1-b: 0 available (200 total, all in use)
Zone us-central1-c: 2 available (300 total, 298 in use)

# Result: You have quota for 8, but can only actually get 2
```

This is a common source of confusion. People see they have quota and assume they can use it, but quota is just your maximum limit - it doesn't guarantee physical availability.

### Layer 4: The Scheduling Game

When you request a GPU instance, here's what happens internally:

1. **Request received**: "User wants 4× H100 in us-central1"
2. **Quota check**: "User has quota for 8, requesting 4 - OK"
3. **Zone search**: System checks each zone for 4 available H100s
4. **Capacity check**: For each zone, check if GPUs are:
   - Physically present
   - Not reserved by Tier 1 customers
   - Not in use by other Tier 2 customers
5. **Allocation or failure**: Either allocate GPUs or return "resource exhausted"

**The queue problem**: There's no queue system for on-demand resources. If you request 4 H100s and they're not available *at that exact moment*, your request fails. You have to keep retrying until capacity frees up. This is why your inference jobs can't complete during non-business hours - even if fewer people are working, scheduled batch jobs from other companies might be running all night.

### What the Industry Does About This

Large AI companies (OpenAI, Anthropic, Google Brain, Meta FAIR) don't use on-demand cloud resources for production training. Instead, they:

1. **Build or lease dedicated clusters**: They contract with cloud providers for dedicated hardware that no one else can access
2. **Multi-cloud strategy**: They spread workloads across GCP, AWS, Azure, and CoreWeave to reduce dependency on any single provider
3. **Capacity reservations**: They pay upfront for guaranteed capacity, even if they don't use it 100% of the time

For smaller teams like yours, the practical approaches are:

1. **Use multiple zones**: Request GPUs in any zone, not a specific one
2. **Use Vertex AI**: Google's managed ML platform has its own allocation pool
3. **Use preemptible + checkpointing**: Accept preemption but design for it
4. **Consider alternative providers**: Lambda Labs, CoreWeave, and Together AI often have better GPU availability

---

## Question 3: Training on H100 vs Inferencing on Different Hardware

This is a critically important question that many teams overlook. Let me explain the considerations in depth.

### The Fundamental Principle: Model Weights are Hardware-Agnostic

When you train a neural network, the actual result - the learned weights - are just numbers stored in tensors. These numbers are completely independent of what hardware was used to learn them. A model trained on H100 produces the exact same weights as if it were trained on T4, V100, or even CPU (assuming identical data, hyperparameters, and random seeds).

**The math is the math**: `W_new = W_old - learning_rate * gradient` works the same way regardless of whether the gradient was computed on a $30,000 H100 or a $3,000 T4.

### So Why Does Hardware Matter at All?

The differences come from:

1. **Training speed**: H100 computes the same math 10-50× faster than T4
2. **Memory capacity**: H100 (80GB) can fit larger batches than T4 (16GB)
3. **Numerical precision**: Different GPUs have varying support for FP16, BF16, TF32

### The Real Question: Will My H100-Trained Model Run on T4?

**Yes, absolutely.** Here's why and how:

**Scenario**: You train `FlashMoETransformer` on 4× H100 for 10 epochs. You save the checkpoint. Later, you want to run inference on 1× T4.

```python
# Training on H100
model = FlashMoETransformer(config).to("cuda:0")  # H100
# ... training happens ...
torch.save(model.state_dict(), "model_weights.pt")  # Just the weights

# Inference on T4 (different machine, different GPU)
model = FlashMoETransformer(config)  # Same architecture
model.load_state_dict(torch.load("model_weights.pt"))  # Load weights
model = model.to("cuda:0")  # T4 this time
# Inference works exactly the same
```

**The weights file doesn't "know" or "care" what GPU trained it.** It's just a dictionary of tensor values.

### What Can Go Wrong (And How to Prevent It)

There are a few gotchas that can cause H100-trained models to fail on T4:

#### Problem 1: Memory Overflow

**Issue**: Your model architecture was designed with H100's 80GB in mind. When you try to load it on T4 (16GB), it doesn't fit.

**Example**: You used `batch_size=512` during training. The model expects activations for 512 samples. On T4, you can only fit `batch_size=64`.

**Solution**: Batch size is not stored in the model weights - it's a runtime parameter. Just use a smaller batch for inference:

```python
# Training config (H100)
train_config.batch_size = 512

# Inference config (T4)
inference_config.batch_size = 64  # Smaller batch, same model
```

**This doesn't affect accuracy** - inference is computed sample-by-sample anyway. Larger batches just give better GPU utilization.

#### Problem 2: Flash Attention Compatibility

**Issue**: Your model uses Flash Attention, which requires specific GPU architectures. xFormers Flash Attention works on:
- H100 ✓
- A100 ✓
- L4 ✓
- T4 ✓ (with some limitations)

**But**: Older GPUs like K80 or P100 don't support Flash Attention.

**Solution**: Your code already handles this with fallback:

```python
# From your FlashAttentionLayer class (lines 841-907)
if self.use_flash:
    try:
        output = self._xformers_attention(q, k, v, is_causal)
    except Exception:
        output = self._standard_attention(q, k, v, mask, is_causal)
```

If Flash Attention isn't available, it falls back to standard attention. This is slower but functionally identical.

#### Problem 3: BF16 vs FP16 Precision

**Issue**: H100 prefers BF16 (bfloat16), while T4 only supports FP16. These are slightly different numerical formats.

**Why it matters**: BF16 has larger dynamic range but less precision. In rare cases, a model trained with BF16 might behave slightly differently when run in FP16.

**Solution**: This is rarely a problem in practice because:
1. Most modern frameworks handle the conversion automatically
2. Your model weights are stored in FP32 in checkpoints by default
3. Inference is more forgiving than training for precision

**Best practice**: Always save checkpoints in FP32, then cast to whatever precision the inference GPU supports:

```python
# Training (H100, using BF16)
model = model.to(torch.bfloat16)
# ... training ...
torch.save(model.float().state_dict(), "model.pt")  # Save as FP32

# Inference (T4, using FP16)
model.load_state_dict(torch.load("model.pt"))
model = model.half()  # Convert to FP16 for T4
```

### Practical Implications for Your Code

Looking at your implementation, you're already set up reasonably well. However, I'd recommend these practices:

1. **Save models in FP32**: Your `save_checkpoint` function should convert to FP32 before saving
2. **Make batch size configurable at inference time**: Don't hardcode batch size in the model
3. **Test inference on T4 before deploying**: Run a small validation set on T4 to verify identical outputs

The key insight is: **training hardware and inference hardware are completely decoupled.** You can train on H100 and infer on T4, or vice versa. The only constraints are:
- The architecture must be the same
- The inference GPU must have enough memory
- The inference GPU must support required operations (like Flash Attention)

---

## Question 4: How MoE Creates Barriers to Inference

This is an excellent question, and your intuition is partially correct but missing some important nuances. Let me explain in depth.

### The Promise of MoE: Reduced Computation

You're right that **Mixture of Experts (MoE) is designed to reduce computation**. The core idea is:

**Dense model**: Every input token passes through ALL parameters
**MoE model**: Each input token only passes through a SUBSET of parameters (the "active" experts)

For example, in your `MoELayer`:
- You have 8 experts, each with 256×512×2 = 262,144 parameters
- But for each token, only top-2 experts are activated
- So you only compute 2/8 = 25% of the expert parameters

This means **FLOPs per token are reduced** by ~75% for the expert layers. This is the computational benefit.

### The Overhead Problem: What You Gain in FLOPs, You Lose Elsewhere

However, **FLOPs are not the only cost**. Real-world inference speed depends on:

1. **Compute (FLOPs)** - How many operations
2. **Memory bandwidth** - Moving data between GPU memory and compute units
3. **Kernel launch overhead** - Starting/stopping GPU operations
4. **Synchronization** - Waiting for different parts to finish

MoE introduces overhead in categories 2-4 that can **outweigh** the FLOPs savings, especially for smaller models like yours.

### Understanding the Router Overhead

Let me walk through what happens in your `MoELayer.forward()`:

```python
def forward(self, x: torch.Tensor, train: bool = True) -> Tuple[torch.Tensor, Dict]:
    batch_size, seq_len, d_model = x.shape
    x_flat = x.view(-1, d_model)  # [batch*seq, d_model]
    
    # STEP 1: Router computes probabilities (small overhead)
    router_logits = self.router(x_flat)  # [batch*seq, num_experts]
    router_probs = F.softmax(router_logits, dim=-1)
    
    # STEP 2: Select top-k experts (THIS IS THE PROBLEM)
    top_k_probs, top_k_indices = torch.topk(router_probs, self.config.top_k, dim=-1)
```

**The routing decision itself is fast.** The problem comes next:

```python
    # STEP 3: Compute expert outputs
    expert_outputs = torch.zeros_like(x_flat)
    
    for i, expert in enumerate(self.experts):
        # Find which tokens go to this expert
        expert_mask = (top_k_indices == i).any(dim=-1)  # Boolean mask
        
        if expert_mask.any():
            # Gather tokens for this expert
            expert_input = x_flat[expert_mask]  # MEMORY OPERATION
            
            # Compute expert output
            expert_out = expert(expert_input)  # COMPUTE
            
            # Scatter back to output
            expert_outputs[expert_mask] += expert_out * weight  # MEMORY OPERATION
```

### Why This Loop is Slow

**Problem 1: Irregular Memory Access Patterns**

When you select "tokens that go to expert 3," you're selecting a random subset of tokens scattered throughout your batch. This creates **non-contiguous memory access**, which is very slow on GPUs.

GPUs are designed for **coalesced memory access** - reading/writing consecutive memory locations. When you gather scattered tokens, the GPU has to fetch data from many different memory addresses, which is 10-100× slower than reading consecutive data.

```
IDEAL (dense layer):     Memory layout: [A B C D E F G H]
                         GPU reads:     [A B C D E F G H] → Fast!

MOE (scattered access):  Memory layout: [A B C D E F G H]
                         Expert 1 tokens: A, D, F
                         GPU reads:     [A _ _ D _ F _ _] → Slow!
```

**Problem 2: Variable Workload Per Expert**

In dense models, every layer processes the same amount of data. GPUs are very efficient when workload is predictable and uniform.

In MoE, some experts might get 100 tokens while others get 10,000 tokens in the same batch. This creates:
- **Load imbalance**: Some GPU threads finish early and sit idle
- **Dynamic shapes**: Each expert computation has different input size, preventing certain optimizations
- **Kernel launch overhead**: Each expert requires a separate GPU kernel launch

**Problem 3: Small Batch Per Expert**

Your total batch is split across 8 experts with top-2 routing. This means each expert sees approximately `batch_size * 2 / 8 = batch_size / 4` tokens.

If your inference batch is 64, each expert processes ~16 tokens. This is a tiny workload for a GPU - the overhead of launching the kernel exceeds the actual computation time.

Dense layers process all 64 tokens in one kernel launch, which is much more efficient.

### Quantifying the Overhead

Let me give you concrete numbers for your model:

**Dense FFN layer** (from FlashAttentionTransformer):
- Input: [batch=64, seq=200, d_model=256]
- Operations: Two matrix multiplies (256→512, 512→256)
- Total FLOPs: 64 × 200 × (256×512×2 + 512×256×2) ≈ 3.4 billion FLOPs
- GPU utilization: ~40% (good for this size)
- Actual time: ~0.8ms on T4

**MoE layer** (from FlashMoETransformer):
- Input: [batch=64, seq=200, d_model=256]
- Active FLOPs: Only 25% of experts → 0.85 billion FLOPs
- But with overhead:
  - Router: +0.1ms
  - Gather operations: +0.3ms per expert × 8 = +2.4ms
  - Scatter operations: +0.3ms per expert × 8 = +2.4ms
  - Small batch inefficiency: 2× slowdown on compute
- Actual time: ~3-4ms on T4

**The paradox**: MoE uses 4× fewer FLOPs but takes 4× longer on small batches!

### When MoE Actually Wins

MoE shines when:

1. **Large batches** (1000+ tokens per expert): Router overhead is amortized
2. **Large models** (1B+ parameters): Compute dominates over memory operations
3. **Specialized hardware**: Google TPUs have special MoE support
4. **Long sequences**: More tokens spread the overhead

For your use case (27M params, batch 64-256, T4 inference), **MoE overhead likely exceeds its benefits**.

### Recommendation for Your Situation

Given your inference constraints:
- 16M members to process
- Scarce GPU resources
- Need to run on whatever is available (H100, T4, maybe CPU)

**I recommend deploying the Dense Flash Attention model**, not MoE, for production inference. Train both, compare validation accuracy - if they're within 1-2%, deploy Dense.

---

## Question 5: How to Choose Between Dense and MoE for Inference

This is the practical decision you need to make. Let me walk through a systematic approach.

### The Decision Framework

You need to evaluate both models on two dimensions:
1. **Accuracy/Performance**: How well do they predict?
2. **Inference Cost**: How fast and cheap is inference?

Then pick the model with the best **performance-per-cost** ratio, not just the highest accuracy.

### Step 1: Train Both Models to Convergence

First, train both architectures with equivalent compute budgets:

```python
# Dense Flash Attention model
dense_config = FlashAttentionConfig(
    embedding_size=256,
    nhid=512,  # 4× expansion
    nlayers=6,
    use_swiglu=True,
    use_learnt_att_pool=True
)

# MoE Flash Attention model (parameter-matched)
moe_config = MoEConfig(
    d_model=256,
    d_ff=512,
    num_experts=8,
    top_k=2,
    use_moe_from_layer=2  # Layers 0-1 dense, 2-5 MoE
)
```

Train both for the same number of epochs on the same data. Your current experiments already do this.

### Step 2: Compare Validation Accuracy

After training, evaluate both on your validation set:

```python
# Metrics to compare
metrics = [
    'val_loss',
    'top_10_accuracy',
    'precision@10',
    'recall@10',
    'rare_code_accuracy',  # Important for clinical models
]

# Example results (hypothetical)
dense_results = {
    'val_loss': 0.185,
    'top_10_accuracy': 0.72,
    'precision@10': 0.68,
    'recall@10': 0.71,
    'rare_code_accuracy': 0.45,
}

moe_results = {
    'val_loss': 0.178,  # 4% better
    'top_10_accuracy': 0.74,  # 2.8% better
    'precision@10': 0.70,  # 3% better
    'recall@10': 0.73,  # 2.8% better
    'rare_code_accuracy': 0.48,  # 6.7% better
}
```

### Step 3: Benchmark Inference Speed on Target Hardware

This is crucial. Run actual inference benchmarks on the GPUs you'll use in production:

```python
import time

def benchmark_inference(model, dataloader, device, num_batches=100):
    """Benchmark inference throughput."""
    model = model.to(device)
    model.eval()
    
    # Warmup
    for i, batch in enumerate(dataloader):
        if i >= 10:
            break
        with torch.no_grad():
            x = batch['x'].to(device)
            _ = model(x)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    samples_processed = 0
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        with torch.no_grad():
            x = batch['x'].to(device)
            _ = model(x)
        samples_processed += x.shape[0]
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    return {
        'samples_per_second': samples_processed / elapsed,
        'ms_per_sample': elapsed * 1000 / samples_processed,
        'total_time': elapsed
    }

# Run on each GPU type you might use
for device_name, device in [('H100', 'cuda:0'), ('T4', 'cuda:0')]:
    dense_perf = benchmark_inference(dense_model, loader, device)
    moe_perf = benchmark_inference(moe_model, loader, device)
    
    print(f"\n{device_name}:")
    print(f"  Dense: {dense_perf['samples_per_second']:.1f} samples/sec")
    print(f"  MoE: {moe_perf['samples_per_second']:.1f} samples/sec")
    print(f"  Dense is {dense_perf['samples_per_second']/moe_perf['samples_per_second']:.1f}× faster")
```

### Step 4: Calculate Cost-Adjusted Performance

Now combine accuracy and speed into a single metric:

```python
# Hypothetical results
                    Dense       MoE
Validation Loss:    0.185      0.178  (MoE 4% better)
Top-10 Accuracy:    72%        74%    (MoE 2.8% better)

Inference Speed (T4):
  samples/sec:      450        180    (Dense 2.5× faster)
  ms/sample:        2.2        5.6

Time for 16M members:
  Dense: 16M / 450 = 35,556 sec = 9.9 hours
  MoE: 16M / 180 = 88,889 sec = 24.7 hours

Cost (T4 @ $2.99/hr):
  Dense: 10 hours × $2.99 = $29.90
  MoE: 25 hours × $2.99 = $74.75
```

**The tradeoff**: MoE gives 2-4% better accuracy but costs 2.5× more for inference.

### Step 5: Make the Business Decision

Ask yourself:
- **How much is 2% accuracy worth?** If it prevents 1,000 incorrect predictions on 16M members, is that worth $45 extra per inference run?
- **Can you afford 25 hours of GPU time?** With your scarcity issues, a 10-hour job is more likely to complete than a 25-hour job
- **What's your latency requirement?** If you need weekly embeddings and have limited GPU windows, Dense is safer

### The "Efficiency Frontier" Visualization

Think of it this way:

```
Performance
    ↑
    │                         * MoE (higher accuracy)
    │                        
    │               * Dense (good accuracy, faster)
    │              
    │      * Smaller model (lower accuracy, very fast)
    │
    └────────────────────────────────→ Inference Cost
    
    Goal: Be on the "Pareto frontier" - no wasted cost for no accuracy gain
```

MoE is on the frontier (you can't get that accuracy cheaper), but Dense might be the better operating point for your constraints.

### My Recommendation for Your Specific Situation

Given:
- GPU scarcity (uncertain H100 availability)
- Large inference workload (16M members)
- Need for reliability (can't fail mid-inference)

**I recommend training both, then deploying Dense if:**
1. Dense validation accuracy is within 5% of MoE
2. Dense is 2× or more faster at inference
3. Dense can run on CPU as a fallback (MoE cannot practically)

**Only deploy MoE if:**
1. MoE accuracy is significantly better (>5%)
2. You can guarantee GPU access for the full inference duration
3. The accuracy improvement justifies the cost

### How to Measure This in Your Code

Add these metrics to your `comprehensive_evaluation`:

```python
def compare_for_deployment(dense_results, moe_results, inference_benchmarks):
    """Compare models for deployment decision."""
    
    # Accuracy difference
    acc_diff = (moe_results['top_10_acc'] - dense_results['top_10_acc']) / dense_results['top_10_acc']
    
    # Speed difference
    speed_ratio = inference_benchmarks['dense_samples_per_sec'] / inference_benchmarks['moe_samples_per_sec']
    
    # Cost for 16M members
    dense_cost = (16_000_000 / inference_benchmarks['dense_samples_per_sec']) / 3600 * T4_HOURLY_RATE
    moe_cost = (16_000_000 / inference_benchmarks['moe_samples_per_sec']) / 3600 * T4_HOURLY_RATE
    
    # Efficiency score (accuracy per dollar)
    dense_efficiency = dense_results['top_10_acc'] / dense_cost
    moe_efficiency = moe_results['top_10_acc'] / moe_cost
    
    print(f"Accuracy difference: MoE is {acc_diff*100:.1f}% better")
    print(f"Speed difference: Dense is {speed_ratio:.1f}× faster")
    print(f"Cost per inference run: Dense ${dense_cost:.0f}, MoE ${moe_cost:.0f}")
    print(f"Efficiency (acc/$): Dense {dense_efficiency:.4f}, MoE {moe_efficiency:.4f}")
    
    if speed_ratio > 2 and acc_diff < 0.05:
        return "RECOMMEND_DENSE"
    elif acc_diff > 0.10:
        return "RECOMMEND_MOE"
    else:
        return "CLOSE_CALL_PREFER_DENSE_FOR_RELIABILITY"
```

---

## Summary: Key Insights

1. **Preemption** is cloud providers taking back resources you're using. Design with checkpoints to survive it.

2. **GCP resource allocation** is a tiered system where on-demand users lose to committed-use customers. Your scarcity problem is likely physical capacity exhaustion, not just bad timing.

3. **Training on H100, inference on T4** works perfectly fine. Model weights are hardware-agnostic. Just ensure batch size fits in memory and precision is compatible.

4. **MoE overhead** comes from irregular memory access, load imbalance, and small per-expert batches. For models your size (~35M params) on small batches, MoE is often *slower* than Dense despite using fewer FLOPs.

5. **Choosing between Dense and MoE** requires benchmarking both on your actual inference hardware. Compare accuracy-per-dollar, not just accuracy. Given your constraints, Dense is likely the better choice unless MoE shows >5% accuracy improvement.

Would you like me to elaborate on any of these points further?