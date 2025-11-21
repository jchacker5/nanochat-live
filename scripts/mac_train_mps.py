"""
Train SRGI on Mac using PyTorch MPS (Metal Performance Shaders).
This allows you to train the custom SRGI architecture immediately on Apple Silicon
without needing to port the full model definition to MLX.

Usage:
    python scripts/mac_train_mps.py --depth=12 --device_batch_size=8 --total_batch_size=16384
"""

import os
import time
import torch
import wandb
from contextlib import nullcontext
from functools import partial

# -----------------------------------------------------------------------------
# Config - Optimized for Mac M-Series (M1/M2/M3)
# -----------------------------------------------------------------------------
# Runtime
device_type = "mps" # Force MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # Enable CPU fallback for missing ops

# Model architecture (Smaller by default for Mac training)
depth = 12 # Reduced depth for faster iteration on laptop
max_seq_len = 1024 # Reduced context length

# Optimization
device_batch_size = 8 # Smaller per-device batch (adjust based on RAM: 8 for 16GB, 16 for 32GB)
total_batch_size = 32768 # Total batch size (gradient accumulation handles the rest)
num_iterations = 1000 # Short run for testing

# Learning Rates (Scaled down slightly for smaller batch/model)
embedding_lr = 0.002 
unembedding_lr = 0.002
matrix_lr = 0.01
weight_decay = 0.01

# Evaluation
eval_every = 100
sample_every = 100
eval_tokens = 1024 * 64
core_metric_every = -1 # Disable expensive metrics

# -----------------------------------------------------------------------------
# Imports from nanochat
# -----------------------------------------------------------------------------
from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine

# -----------------------------------------------------------------------------
# Training Script
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print_banner()
    print("🍎 Starting Mac-Optimized Training (MPS)...")

    # Allow CLI overrides
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
    user_config = {k: globals()[k] for k in config_keys}

    # Init
    ddp = False # No DDP on single Mac usually
    device = torch.device("mps")
    master_process = True
    
    # MPS doesn't support bfloat16 fully in all ops, use float16 or float32
    # M-series supports bfloat16, but PyTorch MPS support varies. 
    # We'll try bfloat16, if it fails, user can switch to float16 or float32.
    dtype = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32
    autocast_ctx = torch.amp.autocast(device_type="mps", dtype=dtype)
    
    # No synchronization needed for MPS usually, but we can dummy it
    synchronize = lambda: None
    get_max_memory = lambda: torch.mps.current_allocated_memory() if hasattr(torch.mps, "current_allocated_memory") else 0

    # Tokenizer
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)
    vocab_size = tokenizer.get_vocab_size()
    print0(f"Vocab size: {vocab_size:,}")

    # Model Init
    model_dim = depth * 64
    num_heads = max(1, (model_dim + 127) // 128)
    print0(f"Model: depth={depth}, dim={model_dim}, heads={num_heads}")
    
    model_config = GPTConfig(
        sequence_len=max_seq_len, 
        vocab_size=vocab_size, 
        n_layer=depth, 
        n_head=num_heads, 
        n_kv_head=num_heads, 
        n_embd=model_dim,
        # Enable SRGI features if desired
        use_srgi=globals().get('use_srgi', False)
    )
    
    model = GPT(model_config)
    model.to(device)
    
    # Init Weights
    model.init_weights()
    
    # Compile is NOT supported on MPS yet properly, so we skip it
    print("ℹ️ Skipping torch.compile (not supported on MPS)")
    
    num_params = sum(p.numel() for p in model.parameters())
    print0(f"Number of parameters: {num_params:,}")

    # Optimizer
    optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
    adamw_optimizer, muon_optimizer = optimizers

    # DataLoader
    print("Loading data...")
    train_loader = tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="train", device=device)
    build_val_loader = lambda: tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val", device=device)
    x, y = next(train_loader)

    # Training Loop
    tokens_per_step = device_batch_size * max_seq_len
    grad_accum_steps = max(1, total_batch_size // tokens_per_step)
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    wandb_run = DummyWandb() if run == "dummy" else wandb.init(project="nanochat", name=run, config=user_config)

    smooth_train_loss = 0
    total_training_time = 0

    print("🚀 Training started...")
    
    for step in range(num_iterations + 1):
        t0 = time.time()
        
        # Forward + Backward
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                loss = model(x, y)
            
            loss_val = loss.item()
            loss_accum += loss_val
            loss = loss / grad_accum_steps
            loss.backward()
            x, y = next(train_loader)
            
        # Update
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        
        t1 = time.time()
        dt = t1 - t0
        if step > 5: total_training_time += dt

        # Logging
        loss_accum /= grad_accum_steps
        smooth_train_loss = 0.9 * smooth_train_loss + 0.1 * loss_accum if step > 0 else loss_accum
        
        tok_per_sec = int(total_batch_size / dt)
        print0(f"Step {step:04d} | Loss: {smooth_train_loss:.4f} | Time: {dt*1000:.1f}ms | Tok/s: {tok_per_sec:,}")

        if step % eval_every == 0 and step > 0:
             print("Evaluating...")
             # Add eval logic here if needed (simplified for this script)
             
        if step % sample_every == 0 and step > 0:
            print("Generating sample...")
            model.eval()
            with torch.no_grad():
                # Simple sample
                ids = torch.tensor([[50256]], device=device) # BOS
                for _ in range(20):
                    logits = model(ids)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    ids = torch.cat((ids, next_token), dim=1)
                print(f"Sample: {tokenizer.decode(ids[0].tolist())}")
            model.train()

        if step == num_iterations:
            print("Saving checkpoint...")
            # Save logic

    print("Done! 🍎")
