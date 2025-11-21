# THRML Installation Guide

## Requirements

**THRML requires Python 3.10 or later** and the following dependencies:
- JAX
- Equinox
- NumPy
- SciPy

## Installation Status

**Current Status**: THRML installation attempted but requires Python 3.10+

The system Python is 3.9, which doesn't support `TypeAlias` from typing (added in Python 3.10).

## Installation Options

### Option 1: Use Python 3.10+ (Recommended)

If you have Python 3.10+ available:

```bash
# Using Python 3.10+
python3.10 -m pip install git+https://github.com/extropic-ai/thrml.git

# Or clone and install
git clone https://github.com/extropic-ai/thrml.git
cd thrml
python3.10 -m pip install -e .
```

### Option 2: Use pyenv or conda to install Python 3.10+

```bash
# Using pyenv
pyenv install 3.10.0
pyenv local 3.10.0
pip install git+https://github.com/extropic-ai/thrml.git

# Using conda
conda create -n thrml python=3.10
conda activate thrml
pip install git+https://github.com/extropic-ai/thrml.git
```

### Option 3: Use Without THRML (Current Setup)

The EBM implementation works **without THRML** using PyTorch-based sampling:

```python
from nanochat.ebm_hopfield import EBMHopfieldMemory

# This will use PyTorch fallback (works with Python 3.9)
memory = EBMHopfieldMemory(
    n_embd=768,
    memory_size=1024,
    use_thrml=False,  # Use PyTorch fallback
    sampling_method='block_gibbs'
)
```

## Verification

Once THRML is installed, verify it works:

```python
import thrml
print("THRML version:", thrml.__version__)
print("Available modules:", dir(thrml))
```

## Current Implementation

The `EBMHopfieldMemory` class automatically detects if THRML is available:

- **If THRML available**: Uses THRML for block Gibbs sampling (when `use_thrml=True`)
- **If THRML not available**: Falls back to PyTorch-based sampling (works with Python 3.9)

This allows development and training to proceed even without THRML installed.

## Next Steps

1. **For immediate use**: Continue with PyTorch fallback (already implemented)
2. **For THRML integration**: Upgrade to Python 3.10+ and install THRML
3. **For production**: Use THRML for hardware-accelerated sampling on TSUs

## References

- THRML GitHub: https://github.com/extropic-ai/thrml
- THRML Documentation: https://docs.thrml.ai
- Extropic Website: https://extropic.ai

