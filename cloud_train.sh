#!/bin/bash
# Cloud Training Script for SRGI
# Use this on Google Cloud Notebooks, AWS SageMaker, or any cloud GPU instance

set -e

echo "=========================================="
echo "SRGI Cloud Training Setup"
echo "=========================================="

# Check GPU
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  No GPU detected!")
    exit(1)
EOF

# Install dependencies
echo "Installing dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install datasets tokenizers tiktoken wandb numpy matplotlib pytest
pip install jax jaxlib equinox scipy
pip install git+https://github.com/extropic-ai/thrml.git || echo "THRML optional"

# Build Rust tokenizer
echo "Building Rust tokenizer..."
if ! command -v rustc &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    export PATH="$HOME/.cargo/bin:$PATH"
fi

pip install maturin
cd rustbpe
maturin build --release
pip install target/wheels/*.whl
cd ..

# Download data
echo "Downloading training data..."
python -m nanochat.dataset -n 8  # Start with 8 shards

# Train tokenizer
echo "Training tokenizer..."
python -m scripts.tok_train --max_chars=2000000000 --vocab_size=65536

# Run theory validation
echo "Running theory validation tests..."
python scripts/test_srgi_theory.py
python scripts/ebm_experiments.py

# Start training
echo "Starting training..."
echo "This will train a depth=20 model for full validation"
echo "Adjust parameters in the command below as needed"

# Small validation run
python -m scripts.base_train \
    --depth=8 \
    --max_seq_len=2048 \
    --device_batch_size=16 \
    --total_batch_size=65536 \
    --num_iterations=1000 \
    --run=srgi-cloud-validation

echo "✅ Training complete!"
echo "Checkpoints saved in ./checkpoints/"

