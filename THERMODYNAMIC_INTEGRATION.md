# Thermodynamic Computing Integration for SRGI

**Integration Guide: Enhancing SRGI with Thermodynamic Computing Principles**

This document outlines how to integrate thermodynamic computing principles and the **thermox** simulator into the SRGI architecture, improving energy efficiency, sampling capabilities, and computational stability.

---

## 1. Overview: Why Thermodynamic Computing for SRGI?

### 1.1 The Connection

**SRGI's Phase-3 Hopfield Memory** implements energy minimization:
- Energy function: `E = -log Σ exp(β x^T ξᵢ)`
- Iterative convergence to attractor states
- Stochastic sampling for pattern retrieval

**Thermodynamic Computing** leverages:
- Natural thermal fluctuations for computation
- Stochastic processes for efficient sampling
- Energy-based dynamics (exactly what Hopfield networks use!)

### 1.2 Key Benefits

1. **Energy Efficiency**: Thermodynamic systems harness natural stochastic processes, reducing energy consumption
2. **Enhanced Sampling**: Better Gaussian sampling for probabilistic operations in Hopfield memory
3. **Natural Dynamics**: The energy minimization in Hopfield networks aligns perfectly with thermodynamic principles
4. **Scalability**: Thermodynamic computing scales efficiently for large-scale computations

---

## 2. Thermodynamic Computing Simulators

### 2.1 Normal Computing's thermox Simulator

**thermox** is the first thermodynamic computing simulator developed by Normal Computing. It emulates their Stochastic Processing Unit (SPU) hardware, allowing you to:

- Simulate thermodynamic computing processes
- Test algorithms leveraging thermodynamic principles
- Perform Gaussian sampling and probabilistic computations
- Model energy-based dynamics

**Installation & Setup:**

```bash
# Install thermox (check Normal Computing's GitHub/website for latest)
pip install thermox

# Or if available via conda
conda install -c normal-computing thermox
```

**Resources:**
- Blog: https://www.normalcomputing.com/blog-posts/thermox-the-first-thermodynamic-computing-simulator
- GitHub: Search for "normal-computing/thermox" or similar

### 2.2 Extropic's THRML Simulator

**THRML** (Thermodynamic Hypergraphical Modeling Library) is an open-source Python library developed by Extropic. It's designed to:

- Construct thermodynamic hypergraphical models
- Simulate execution on Thermodynamic Sampling Units (TSUs)
- Accelerate simulations using JAX
- Develop energy-based models for Extropic hardware

**Key Features:**
- Built on JAX for GPU acceleration
- Supports probabilistic graphical models
- Energy-based model development
- Open-source and available on GitHub

**Installation & Setup:**

```bash
# Install THRML from GitHub
pip install git+https://github.com/extropic-ai/thrml.git

# Or clone and install
git clone https://github.com/extropic-ai/thrml.git
cd thrml
pip install -e .
```

**Resources:**
- GitHub: https://github.com/extropic-ai/thrml
- Website: https://extropic.ai/software
- Documentation: Check GitHub repository for docs

### 2.3 Comparison: thermox vs THRML

| Feature | thermox (Normal Computing) | THRML (Extropic) |
|---------|---------------------------|------------------|
| **Hardware Target** | Stochastic Processing Unit (SPU) | Thermodynamic Sampling Units (TSUs) |
| **Acceleration** | Not specified | JAX (GPU acceleration) |
| **Focus** | General thermodynamic computing | Hypergraphical models, energy-based models |
| **License** | Check their website | Open-source (check GitHub) |
| **Best For** | General thermodynamic simulations | Energy-based models, probabilistic graphical models |

**Recommendation**: 
- Use **thermox** for general thermodynamic computing simulations
- Use **THRML** for energy-based models and probabilistic graphical models (especially if you're already using JAX)

---

## 3. Integration Strategy: SRGI + Thermodynamic Computing

### 3.1 Phase-3 Hopfield Memory Enhancement

The `ModernHopfieldMemory` class in `nanochat/hopfield_memory.py` can be enhanced with thermodynamic principles:

#### Current Implementation:
```python
# Current: Deterministic energy minimization
sim = (state @ patterns.T) * beta
attn = F.softmax(sim, dim=-1)
retrieved = attn @ patterns
```

#### Thermodynamic Enhancement:
```python
# Enhanced: Thermodynamic sampling with thermal fluctuations
def forward_thermodynamic(self, x, temperature=1.0, return_energy=False):
    """
    Forward pass with thermodynamic sampling.
    
    Args:
        x: Input states (B, T, n_embd)
        temperature: Thermal temperature (controls stochasticity)
        return_energy: If True, return energy values
    """
    B, T, C = x.shape
    
    # Project to query space
    q = self.query(x)
    k = self.key(self.patterns)
    
    # Thermodynamic sampling: add thermal noise
    # Higher temperature = more exploration, lower = more exploitation
    state = q
    
    for step in range(self.n_steps):
        # Compute energy (similarities)
        sim = (state @ k.T) * self.beta
        
        # Add thermal fluctuations (Gaussian noise scaled by temperature)
        thermal_noise = torch.randn_like(sim) * temperature
        sim_thermal = sim + thermal_noise
        
        # Softmax with temperature scaling
        attn = F.softmax(sim_thermal / temperature, dim=-1)
        
        # Retrieve weighted combination
        retrieved = attn @ self.patterns
        
        # Update state (with momentum for smoother convergence)
        state = 0.5 * state + 0.5 * retrieved
    
    # Final retrieval
    sim_final = (state @ k.T) * self.beta
    attn_final = F.softmax(sim_final / temperature, dim=-1)
    output = attn_final @ self.patterns
    
    if return_energy:
        energy_val = self.energy(output, self.patterns)
        return output, energy_val
    
    return output
```

### 3.2 Using thermox for Gaussian Sampling

If thermox provides Gaussian sampling capabilities, use it for:

1. **Pattern Initialization**: Initialize memory patterns using thermodynamic sampling
2. **Noise Injection**: Add controlled thermal noise during retrieval
3. **Energy Landscape Exploration**: Sample from energy distributions

```python
# Example: Using thermox for Gaussian sampling (pseudo-code)
import thermox

class ThermodynamicHopfieldMemory(ModernHopfieldMemory):
    """Enhanced Hopfield memory with thermodynamic computing."""
    
    def __init__(self, n_embd, memory_size=1024, beta=1.0, n_steps=3, 
                 use_thermox=True, temperature=1.0):
        super().__init__(n_embd, memory_size, beta, n_steps)
        self.use_thermox = use_thermox
        self.temperature = temperature
        
        if use_thermox:
            # Initialize thermox simulator
            self.thermox_sim = thermox.Simulator()
    
    def sample_thermal_noise(self, shape, temperature=None):
        """Sample thermal noise using thermox."""
        if temperature is None:
            temperature = self.temperature
        
        if self.use_thermox:
            # Use thermox for Gaussian sampling
            return self.thermox_sim.sample_gaussian(
                shape=shape, 
                mean=0.0, 
                std=temperature
            )
        else:
            # Fallback to PyTorch
            return torch.randn(shape, device=self.patterns.device) * temperature
    
    def forward(self, x, return_energy=False):
        """Forward with thermodynamic sampling."""
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(self.patterns)
        
        state = q
        for step in range(self.n_steps):
            sim = (state @ k.T) * self.beta
            
            # Add thermodynamic noise
            thermal_noise = self.sample_thermal_noise(sim.shape)
            sim_thermal = sim + thermal_noise
            
            attn = F.softmax(sim_thermal / self.temperature, dim=-1)
            retrieved = attn @ self.patterns
            state = 0.5 * state + 0.5 * retrieved
        
        sim_final = (state @ k.T) * self.beta
        attn_final = F.softmax(sim_final / self.temperature, dim=-1)
        output = attn_final @ self.patterns
        
        if return_energy:
            energy_val = self.energy(output, self.patterns)
            return output, energy_val
        
        return output
```

---

## 4. Paper Integration: How This Fits Into SRGI

### 4.1 Theoretical Foundation

**Add to Section 2.4 (Information-Geometric Foundation):**

> **Thermodynamic Interpretation**: The energy minimization in Phase-3 Hopfield memory can be interpreted through the lens of statistical thermodynamics. The energy function `E = -log Σ exp(β x^T ξᵢ)` corresponds to the Helmholtz free energy in a canonical ensemble, where `β` acts as the inverse temperature. The iterative convergence to attractor states follows a thermodynamic relaxation process, where the system evolves toward equilibrium states (attractors) through stochastic dynamics.

**Mathematical Formulation:**

The Hopfield energy can be written as:
$$E(x) = -\frac{1}{\beta} \log \sum_{i=1}^{M} \exp(\beta x^T \xi_i)$$

where:
- `β` = inverse temperature (controls sharpness of attractors)
- `ξ_i` = stored memory patterns
- `x` = query state

In the thermodynamic limit, this corresponds to:
$$F = -kT \log Z$$

where `F` is the Helmholtz free energy, `k` is Boltzmann's constant, `T` is temperature, and `Z` is the partition function.

### 4.2 New Section: "Thermodynamic Computing Integration"

**Add after Section 3 (Architecture):**

#### 3.5 Thermodynamic Computing Integration

We enhance Phase-3 Hopfield memory with thermodynamic computing principles, leveraging the **thermox** simulator developed by Normal Computing [REF]. This integration provides:

1. **Stochastic Sampling**: Using thermox's Gaussian sampling capabilities for pattern initialization and noise injection
2. **Thermal Fluctuations**: Controlled thermal noise enables exploration of the energy landscape
3. **Energy Efficiency**: Thermodynamic processes naturally minimize energy consumption

**Implementation**: We modify the `ModernHopfieldMemory` forward pass to include:
- Temperature-scaled softmax: `softmax(sim / T)` where `T` is thermal temperature
- Thermal noise injection: `sim_thermal = sim + ε` where `ε ~ N(0, T)`
- Stochastic convergence: Multiple samples converge to attractors with controlled variance

**Benefits**:
- **Better Exploration**: Thermal fluctuations prevent premature convergence to local minima
- **Robustness**: Stochastic dynamics improve generalization
- **Energy Efficiency**: Natural thermodynamic processes reduce computational overhead

### 4.3 Experimental Section Addition

**Add to Section 5 (Experiments):**

#### 5.X Thermodynamic Computing Ablation

We compare three variants:
1. **Baseline**: Standard Hopfield memory (deterministic)
2. **Thermodynamic (PyTorch)**: Added thermal noise using PyTorch's Gaussian sampling
3. **Thermodynamic (thermox)**: Using thermox simulator for Gaussian sampling

**Metrics**:
- Memory retrieval accuracy
- Energy consumption (if measurable)
- Convergence speed
- Robustness to noise

**Expected Results**:
- Thermodynamic variants show improved exploration and robustness
- thermox may provide better energy efficiency (hardware-dependent)
- Temperature parameter provides control over exploration vs. exploitation trade-off

---

## 5. Implementation Steps

### Step 1: Install Simulator(s)

**Option A: Install thermox (Normal Computing)**
```bash
# Check Normal Computing's website for latest installation
pip install thermox
# Or follow their installation guide
```

**Option B: Install THRML (Extropic)**
```bash
# Install THRML from GitHub
pip install git+https://github.com/extropic-ai/thrml.git

# Or clone and install
git clone https://github.com/extropic-ai/thrml.git
cd thrml
pip install -e .
```

**Option C: Install Both**
```bash
# Install both simulators for comparison
pip install thermox
pip install git+https://github.com/extropic-ai/thrml.git
```

### Step 2: Create Enhanced Module

Create `nanochat/thermodynamic_hopfield.py`:

```python
"""
Thermodynamic-enhanced Hopfield Memory for SRGI.

Integrates thermodynamic computing principles using thermox simulator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nanochat.hopfield_memory import ModernHopfieldMemory

try:
    import thermox
    HAS_THERMOX = True
except ImportError:
    HAS_THERMOX = False

try:
    import thrml
    HAS_THRML = True
except ImportError:
    HAS_THRML = False

if not HAS_THERMOX and not HAS_THRML:
    print("Warning: Neither thermox nor THRML available. Using PyTorch fallback.")


class ThermodynamicHopfieldMemory(ModernHopfieldMemory):
    """
    Enhanced Hopfield memory with thermodynamic computing.
    
    Adds thermal fluctuations and stochastic sampling for better
    exploration and energy efficiency.
    """
    
    def __init__(self, n_embd, memory_size=1024, beta=1.0, n_steps=3,
                 temperature=1.0, simulator='auto'):
        """
        Args:
            simulator: 'thermox', 'thrml', 'pytorch', or 'auto'
                'auto' will use the first available simulator
        """
        super().__init__(n_embd, memory_size, beta, n_steps)
        self.temperature = temperature
        
        # Determine which simulator to use
        if simulator == 'auto':
            if HAS_THERMOX:
                simulator = 'thermox'
            elif HAS_THRML:
                simulator = 'thrml'
            else:
                simulator = 'pytorch'
        
        self.simulator_type = simulator
        
        if simulator == 'thermox' and HAS_THERMOX:
            # Initialize thermox simulator
            self.thermox_sim = thermox.Simulator()
        elif simulator == 'thrml' and HAS_THRML:
            # Initialize THRML (may need specific initialization)
            self.thrml_sim = thrml  # Adjust based on actual API
        else:
            self.simulator_type = 'pytorch'
    
    def sample_thermal_noise(self, shape, device=None):
        """Sample thermal noise using available simulator or PyTorch."""
        if device is None:
            device = self.patterns.device
        
        if self.simulator_type == 'thermox' and HAS_THERMOX:
            # Use thermox for Gaussian sampling
            noise = self.thermox_sim.sample_gaussian(
                shape=shape,
                mean=0.0,
                std=self.temperature
            )
            return torch.tensor(noise, device=device, dtype=self.patterns.dtype)
        
        elif self.simulator_type == 'thrml' and HAS_THRML:
            # Use THRML for sampling (adjust API based on actual implementation)
            # THRML uses JAX, so we may need to convert
            import numpy as np
            import jax.numpy as jnp
            # Note: Adjust this based on actual THRML API
            # THRML may have different sampling functions
            noise = thrml.sample_gaussian(shape, mean=0.0, std=self.temperature)
            # Convert JAX array to PyTorch tensor
            noise_np = np.array(noise)
            return torch.tensor(noise_np, device=device, dtype=self.patterns.dtype)
        
        else:
            # Fallback to PyTorch
            return torch.randn(shape, device=device, dtype=self.patterns.dtype) * self.temperature
    
    def forward(self, x, return_energy=False):
        """
        Forward pass with thermodynamic sampling.
        
        Args:
            x: Input states (B, T, n_embd)
            return_energy: If True, return energy values
        
        Returns:
            output: Retrieved memory states
            energy_val (optional): Energy values
        """
        B, T, C = x.shape
        
        # Project to query space
        q = self.query(x)
        k = self.key(self.patterns)
        
        # Iterative updates with thermal fluctuations
        state = q
        for step in range(self.n_steps):
            # Compute similarities
            sim = (state @ k.T) * self.beta
            
            # Add thermal noise for exploration
            thermal_noise = self.sample_thermal_noise(sim.shape, device=x.device)
            sim_thermal = sim + thermal_noise
            
            # Temperature-scaled softmax
            attn = F.softmax(sim_thermal / self.temperature, dim=-1)
            
            # Retrieve weighted combination
            retrieved = attn @ self.patterns
            
            # Update state
            state = 0.5 * state + 0.5 * retrieved
        
        # Final retrieval
        sim_final = (state @ k.T) * self.beta
        thermal_noise_final = self.sample_thermal_noise(sim_final.shape, device=x.device)
        sim_final_thermal = sim_final + thermal_noise_final
        attn_final = F.softmax(sim_final_thermal / self.temperature, dim=-1)
        output = attn_final @ self.patterns
        
        if return_energy:
            energy_val = self.energy(output, self.patterns)
            return output, energy_val
        
        return output
```

### Step 3: Update Tests

Create `tests/test_thermodynamic_hopfield.py`:

```python
"""Tests for Thermodynamic Hopfield Memory."""

import torch
import pytest
from nanochat.thermodynamic_hopfield import ThermodynamicHopfieldMemory


def test_thermodynamic_hopfield_basic():
    """Test basic forward pass."""
    n_embd = 64
    memory_size = 128
    batch_size = 2
    seq_len = 10
    
    memory = ThermodynamicHopfieldMemory(
        n_embd=n_embd,
        memory_size=memory_size,
        temperature=1.0,
        use_thermox=False  # Use PyTorch fallback for testing
    )
    
    x = torch.randn(batch_size, seq_len, n_embd)
    output = memory(x)
    
    assert output.shape == (batch_size, seq_len, n_embd)


def test_temperature_effect():
    """Test that temperature affects output."""
    n_embd = 32
    memory_size = 64
    x = torch.randn(1, 5, n_embd)
    
    memory_low = ThermodynamicHopfieldMemory(
        n_embd=n_embd,
        memory_size=memory_size,
        temperature=0.5,
        use_thermox=False
    )
    
    memory_high = ThermodynamicHopfieldMemory(
        n_embd=n_embd,
        memory_size=memory_size,
        temperature=2.0,
        use_thermox=False
    )
    
    output_low = memory_low(x)
    output_high = memory_high(x)
    
    # Outputs should be different due to temperature
    assert not torch.allclose(output_low, output_high, atol=1e-5)
```

### Step 4: Update Configuration

Add to `SRGI_ROADMAP.md` or config:

```python
# Configuration option
config.use_thermodynamic_hopfield = True
config.thermox_temperature = 1.0
config.use_thermox_simulator = True  # Use thermox if available
```

---

## 6. Research Questions & Experiments

### 6.1 Key Questions

1. **Does thermodynamic sampling improve memory retrieval?**
   - Compare retrieval accuracy with/without thermal noise
   - Test across different temperature values

2. **Energy efficiency gains?**
   - Measure computational overhead
   - Compare thermox vs. PyTorch implementations

3. **Robustness improvements?**
   - Test with corrupted/noisy inputs
   - Measure convergence stability

4. **Temperature parameter tuning?**
   - Optimal temperature for different tasks
   - Adaptive temperature scheduling

### 6.2 Experimental Design

```python
# Experiment: Compare baseline vs. thermodynamic
experiments = {
    'baseline': ModernHopfieldMemory(n_embd=768, memory_size=1024),
    'thermodynamic_pytorch': ThermodynamicHopfieldMemory(
        n_embd=768, memory_size=1024, temperature=1.0, use_thermox=False
    ),
    'thermodynamic_thermox': ThermodynamicHopfieldMemory(
        n_embd=768, memory_size=1024, temperature=1.0, use_thermox=True
    )
}

# Metrics
metrics = [
    'retrieval_accuracy',
    'convergence_speed',
    'energy_consumption',  # If measurable
    'robustness_to_noise'
]
```

---

## 7. References & Resources

### Papers

1. **Normal Computing**: Thermodynamic computing papers (check their website)
2. **Extropic**: Thermodynamic computing papers (check their website)
3. **"Training Thermodynamic Computers by Gradient Descent"** - Stephen Whitelam (2025)
4. **"Error Mitigation for Thermodynamic Computing"** - Maxwell Aifer et al. (2024)
5. **Modern Hopfield Networks** - Ramsauer et al. (2020)

### Resources

**Normal Computing:**
- **thermox Simulator**: https://www.normalcomputing.com/blog-posts/thermox-the-first-thermodynamic-computing-simulator
- **Normal Computing Blog**: https://www.normalcomputing.com/blog
- **Normal Computing GitHub**: Search for "normal-computing" repositories

**Extropic:**
- **THRML GitHub**: https://github.com/extropic-ai/thrml
- **Extropic Software**: https://extropic.ai/software
- **Extropic Blog**: https://extropic.ai/writing
- **XTR-0 Platform**: https://extropic.ai/writing/inside-x0-and-xtr-0

### Key Concepts

- **Thermodynamic Computing**: Using thermal fluctuations for computation
- **Stochastic Processing Unit (SPU)**: Hardware implementation
- **Gaussian Sampling**: Efficient probabilistic sampling
- **Energy Minimization**: Natural convergence to low-energy states

---

## 8. Next Steps

1. **Install simulators** (thermox and/or THRML) and verify they work
2. **Compare simulators** - test both and see which works better for SRGI
3. **Implement `ThermodynamicHopfieldMemory`** class with support for both
4. **Run ablation studies** comparing:
   - Baseline (no thermodynamic)
   - thermox implementation
   - THRML implementation
   - PyTorch fallback
5. **Write up results** in paper Section 5.X
6. **Update visualizations** showing thermodynamic effects
7. **Benchmark** energy efficiency (if possible)

---

## 9. Integration Checklist

- [ ] Install thermox simulator (Normal Computing)
- [ ] Install THRML simulator (Extropic)
- [ ] Compare both simulators and choose best fit
- [ ] Create `ThermodynamicHopfieldMemory` class with multi-simulator support
- [ ] Write tests for thermodynamic variant
- [ ] Run ablation experiments (baseline vs. thermox vs. THRML)
- [ ] Add theoretical section to paper (Section 3.5)
- [ ] Add experimental results (Section 5.X)
- [ ] Update visualizations showing thermodynamic effects
- [ ] Update README and documentation
- [ ] Benchmark performance improvements
- [ ] Document which simulator works best for SRGI

---

**Last Updated**: November 2025  
**Status**: Draft - Ready for Implementation

