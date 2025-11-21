# Energy-Based Model (EBM) Integration: Extropic Research + SRGI

**Comprehensive Guide: Integrating Extropic's EBM Research into SRGI Architecture**

This document outlines how to leverage Extropic's Energy-Based Model (EBM) research and THRML library to enhance SRGI's Phase-3 Hopfield Memory with advanced thermodynamic sampling and energy-based dynamics.

---

## 1. Overview: EBMs and SRGI

### 1.1 The Perfect Match

**SRGI's Phase-3 Hopfield Memory** is fundamentally an **Energy-Based Model**:
- Energy function: `E(x) = -log Σ exp(β x^T ξᵢ)`
- States evolve toward energy minima (attractors)
- Natural fit for EBM frameworks

**Extropic's EBM Research** provides:
- Efficient block Gibbs sampling
- Thermodynamic Sampling Units (TSUs) for hardware acceleration
- THRML library for probabilistic graphical models
- Energy-efficient sampling from complex distributions

### 1.2 Key Connections

1. **Hopfield Networks = EBMs**: Modern Hopfield networks are a type of energy-based model
2. **Energy Minimization**: Both use energy functions to find stable states
3. **Stochastic Sampling**: Both leverage probabilistic sampling for inference
4. **Thermodynamic Computing**: Both benefit from thermodynamic principles

---

## 2. Extropic's EBM Research & THRML

### 2.1 What is THRML?

**THRML** (Thermodynamic Hypergraphical Modeling Library) is Extropic's open-source JAX library for:

- Building probabilistic graphical models (PGMs)
- Efficient block Gibbs sampling on sparse, heterogeneous graphs
- Simulating Thermodynamic Sampling Units (TSUs)
- Developing discrete energy-based models
- Prototyping thermodynamic algorithms

**Key Features:**
- JAX-based for GPU acceleration
- Block Gibbs sampling (more efficient than standard Gibbs)
- Sparse graph support
- Hardware simulation capabilities

### 2.2 Extropic's EBM Approach

**Core Principle**: EBMs are implemented as **parameterized stochastic analog circuits** that physically settle into low-energy states representing likely outcomes.

**Key Advantages:**
- **Energy Efficiency**: TSUs use significantly less energy than GPUs
- **Natural Sampling**: Hardware naturally samples from energy distributions
- **Fast Inference**: Direct hardware implementation accelerates sampling
- **Scalability**: Efficient for large-scale probabilistic models

### 2.3 Installation

```bash
# Install THRML
pip install thrml

# Or from GitHub
git clone https://github.com/extropic-ai/thrml.git
cd thrml
pip install -e .

# Requirements
# - Python 3.10+
# - JAX (for GPU acceleration)
# - NumPy
```

**Documentation:**
- GitHub: https://github.com/extropic-ai/thrml
- Core Concepts: https://www.github.gg/wiki/extropic-ai/thrml/core-concepts
- Extropic Blog: https://extropic.ai/writing

---

## 3. SRGI Hopfield Memory as an EBM

### 3.1 Energy Function Formulation

The Hopfield energy function can be written as an EBM:

```python
# Standard Hopfield Energy (already an EBM!)
E(x) = -log Σᵢ exp(β x^T ξᵢ)

# This is equivalent to:
# E(x) = -log Z(x) where Z(x) = Σᵢ exp(β x^T ξᵢ)
# 
# Probability distribution:
# P(x) = (1/Z) exp(-E(x)) = (1/Z) exp(log Σᵢ exp(β x^T ξᵢ))
#      = (1/Z) Σᵢ exp(β x^T ξᵢ)
```

### 3.2 EBM Interpretation

**Energy Function**: `E(x) = -log Σᵢ exp(β x^T ξᵢ)`
- Lower energy = higher probability
- Attractors = energy minima
- Sampling = exploring energy landscape

**Partition Function**: `Z = Σᵢ exp(β x^T ξᵢ)`
- Normalizes the probability distribution
- Computationally expensive (exponential in number of patterns)

**Sampling Strategy**: 
- Current: Deterministic iterative updates
- Enhanced: Block Gibbs sampling via THRML

---

## 4. Integration Strategy: THRML + SRGI Hopfield Memory

### 4.1 Block Gibbs Sampling for Hopfield Networks

**Standard Gibbs Sampling** (one variable at a time):
- Slow convergence
- High autocorrelation
- Inefficient for high-dimensional spaces

**Block Gibbs Sampling** (multiple variables simultaneously):
- Faster convergence
- Lower autocorrelation
- Better for structured models like Hopfield networks

**For Hopfield Memory:**
- **Block 1**: Query state `q`
- **Block 2**: Attention weights `α`
- **Block 3**: Retrieved pattern `r`

### 4.2 Enhanced Hopfield Memory with THRML

```python
"""
Enhanced Hopfield Memory using Extropic's THRML for EBM sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import thrml
    import jax
    import jax.numpy as jnp
    HAS_THRML = True
except ImportError:
    HAS_THRML = False
    print("Warning: THRML not available. Using PyTorch fallback.")

from nanochat.hopfield_memory import ModernHopfieldMemory


class EBMHopfieldMemory(ModernHopfieldMemory):
    """
    Energy-Based Model Hopfield Memory using THRML for efficient sampling.
    
    This class enhances the standard Hopfield memory with:
    - Block Gibbs sampling via THRML
    - Energy-based inference
    - Thermodynamic sampling capabilities
    """
    
    def __init__(self, n_embd, memory_size=1024, beta=1.0, n_steps=3,
                 use_thrml=True, sampling_method='block_gibbs'):
        """
        Args:
            n_embd: Embedding dimension
            memory_size: Number of memory patterns
            beta: Inverse temperature
            n_steps: Number of sampling steps
            use_thrml: Use THRML for sampling (if available)
            sampling_method: 'block_gibbs', 'gibbs', or 'deterministic'
        """
        super().__init__(n_embd, memory_size, beta, n_steps)
        self.use_thrml = use_thrml and HAS_THRML
        self.sampling_method = sampling_method
        
        if self.use_thrml:
            # Initialize THRML components
            self._init_thrml_model()
    
    def _init_thrml_model(self):
        """Initialize THRML model for block Gibbs sampling."""
        # Define energy function for THRML
        # This will be used for block Gibbs sampling
        
        # Energy function: E(x, patterns) = -log Σ exp(β x^T ξᵢ)
        def energy_fn(query_state, patterns, beta):
            """
            Energy function for THRML.
            
            Args:
                query_state: Query vector (n_embd,)
                patterns: Memory patterns (memory_size, n_embd)
                beta: Inverse temperature
            
            Returns:
                Energy value (scalar)
            """
            # Compute similarities
            similarities = jnp.dot(query_state, patterns.T) * beta
            
            # Energy = -log sum exp(similarities)
            energy = -jnp.log(jnp.sum(jnp.exp(similarities)) + 1e-10)
            
            return energy
        
        self.energy_fn = energy_fn
        
        # Define sampling blocks
        # Block 1: Query state
        # Block 2: Attention weights (derived from query)
        # We'll use THRML's block Gibbs sampling
    
    def sample_with_thrml(self, query_state, patterns, n_samples=1):
        """
        Sample using THRML's block Gibbs sampling.
        
        Args:
            query_state: Initial query (n_embd,)
            patterns: Memory patterns (memory_size, n_embd)
            n_samples: Number of samples to generate
        
        Returns:
            Sampled query states
        """
        if not self.use_thrml:
            raise RuntimeError("THRML not available")
        
        # Convert to JAX arrays
        query_jax = jnp.array(query_state.detach().cpu().numpy())
        patterns_jax = jnp.array(patterns.detach().cpu().numpy())
        
        # Define conditional distributions for block Gibbs
        def sample_query_given_attention(query, patterns, beta, attention_weights):
            """Sample query state given attention weights."""
            # Retrieve pattern based on attention
            retrieved = jnp.dot(attention_weights, patterns)
            
            # Add some noise for exploration
            noise = jnp.random.normal(0, 1.0 / beta, query.shape)
            new_query = 0.5 * query + 0.5 * retrieved + noise
            
            return new_query
        
        def sample_attention_given_query(query, patterns, beta):
            """Sample attention weights given query state."""
            # Compute similarities
            similarities = jnp.dot(query, patterns.T) * beta
            
            # Softmax to get attention
            attention = jax.nn.softmax(similarities)
            
            return attention
        
        # Block Gibbs sampling loop
        current_query = query_jax
        samples = []
        
        for _ in range(n_samples):
            # Block 1: Sample attention given query
            attention = sample_attention_given_query(current_query, patterns_jax, self.beta)
            
            # Block 2: Sample query given attention
            current_query = sample_query_given_attention(
                current_query, patterns_jax, self.beta, attention
            )
            
            samples.append(current_query)
        
        # Convert back to PyTorch
        samples_torch = torch.tensor(
            np.array(samples),
            device=query_state.device,
            dtype=query_state.dtype
        )
        
        return samples_torch
    
    def forward(self, x, return_energy=False, use_ebm_sampling=True):
        """
        Forward pass with optional EBM sampling via THRML.
        
        Args:
            x: Input states (B, T, n_embd)
            return_energy: If True, return energy values
            use_ebm_sampling: Use THRML block Gibbs sampling if available
        
        Returns:
            output: Retrieved memory states
            energy_val (optional): Energy values
        """
        B, T, C = x.shape
        
        # Project to query space
        q = self.query(x)  # (B, T, n_embd)
        k = self.key(self.patterns)  # (memory_size, n_embd)
        
        if use_ebm_sampling and self.use_thrml and self.sampling_method == 'block_gibbs':
            # Use THRML block Gibbs sampling
            outputs = []
            energies = []
            
            for b in range(B):
                batch_outputs = []
                batch_energies = []
                
                for t in range(T):
                    query = q[b, t]  # (n_embd,)
                    
                    # Sample using THRML
                    samples = self.sample_with_thrml(query, self.patterns, n_samples=self.n_steps)
                    final_sample = samples[-1]  # Use last sample
                    
                    # Compute final attention and retrieve
                    sim_final = (final_sample @ k.T) * self.beta
                    attn_final = F.softmax(sim_final, dim=-1)
                    output = attn_final @ self.patterns
                    
                    batch_outputs.append(output)
                    
                    if return_energy:
                        energy_val = self.energy(output.unsqueeze(0), self.patterns)
                        batch_energies.append(energy_val.squeeze(0))
                
                outputs.append(torch.stack(batch_outputs))
                if return_energy:
                    energies.append(torch.stack(batch_energies))
            
            output = torch.stack(outputs)
            
            if return_energy:
                energy_val = torch.stack(energies)
                return output, energy_val
            
            return output
        
        else:
            # Fallback to standard iterative updates
            return super().forward(x, return_energy=return_energy)
```

---

## 5. Advanced EBM Features

### 5.1 Energy-Based Training

EBMs can be trained using contrastive divergence or persistent contrastive divergence:

```python
class EBMHopfieldMemoryTrainer:
    """Trainer for EBM Hopfield Memory using contrastive divergence."""
    
    def __init__(self, model, learning_rate=1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    def contrastive_divergence_step(self, positive_samples, n_negative_steps=1):
        """
        Contrastive divergence training step.
        
        Args:
            positive_samples: Positive examples (B, T, n_embd)
            n_negative_steps: Number of negative sampling steps
        """
        # Positive phase: compute energy on data
        pos_energy = self.model.energy(positive_samples, self.model.patterns)
        
        # Negative phase: sample from model
        # Start from random initialization
        negative_samples = torch.randn_like(positive_samples)
        
        # Run Gibbs sampling for n_negative_steps
        for _ in range(n_negative_steps):
            # Sample attention
            sim = (negative_samples @ self.model.key(self.model.patterns).T) * self.model.beta
            attn = F.softmax(sim, dim=-1)
            
            # Sample query given attention
            retrieved = attn @ self.model.patterns
            noise = torch.randn_like(negative_samples) / self.model.beta
            negative_samples = 0.5 * negative_samples + 0.5 * retrieved + noise
        
        # Compute negative energy
        neg_energy = self.model.energy(negative_samples, self.model.patterns)
        
        # Contrastive divergence loss
        loss = (pos_energy - neg_energy).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

### 5.2 Persistent Contrastive Divergence

More stable training using persistent negative samples:

```python
class PersistentEBMTrainer(EBMHopfieldMemoryTrainer):
    """Trainer using persistent contrastive divergence."""
    
    def __init__(self, model, learning_rate=1e-3):
        super().__init__(model, learning_rate)
        self.persistent_negative_samples = None
    
    def persistent_contrastive_divergence_step(self, positive_samples):
        """Persistent contrastive divergence with maintained negative samples."""
        # Initialize persistent samples if needed
        if self.persistent_negative_samples is None:
            self.persistent_negative_samples = torch.randn_like(positive_samples)
        
        # Positive phase
        pos_energy = self.model.energy(positive_samples, self.model.patterns)
        
        # Negative phase: update persistent samples
        # Run a few Gibbs steps from current persistent samples
        for _ in range(5):  # Fewer steps needed with persistence
            sim = (self.persistent_negative_samples @ 
                   self.model.key(self.model.patterns).T) * self.model.beta
            attn = F.softmax(sim, dim=-1)
            retrieved = attn @ self.model.patterns
            noise = torch.randn_like(self.persistent_negative_samples) / self.model.beta
            self.persistent_negative_samples = (
                0.5 * self.persistent_negative_samples + 
                0.5 * retrieved + noise
            )
        
        neg_energy = self.model.energy(self.persistent_negative_samples, self.model.patterns)
        
        # Loss
        loss = (pos_energy - neg_energy).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

---

## 6. Paper Integration: How This Fits Into SRGI

### 6.1 Theoretical Section Addition

**Add to Section 3.5 (or create new Section 3.6):**

#### 3.6 Energy-Based Model Formulation

We reformulate Phase-3 Hopfield Memory as an **Energy-Based Model (EBM)**, enabling integration with Extropic's thermodynamic computing research [REF]. The energy function:

$$E(x) = -\frac{1}{\beta} \log \sum_{i=1}^{M} \exp(\beta x^T \xi_i)$$

defines a probability distribution over query states:

$$P(x) = \frac{1}{Z} \exp(-E(x)) = \frac{1}{Z} \sum_{i=1}^{M} \exp(\beta x^T \xi_i)$$

where $Z$ is the partition function and $\xi_i$ are stored memory patterns.

**Block Gibbs Sampling**: We enhance the standard iterative updates with **block Gibbs sampling** via Extropic's THRML library, sampling query states and attention weights in alternating blocks. This approach:

1. **Faster Convergence**: Block sampling reduces autocorrelation compared to standard Gibbs
2. **Better Exploration**: Stochastic sampling explores energy landscape more effectively
3. **Hardware Acceleration**: THRML enables simulation of Thermodynamic Sampling Units (TSUs)

**Energy-Based Training**: We train the EBM using contrastive divergence, minimizing the difference between energies on positive examples (data) and negative examples (samples from the model).

### 6.2 Experimental Section Addition

**Add to Section 5 (Experiments):**

#### 5.X Energy-Based Model Ablation

We compare four variants of Phase-3 Hopfield Memory:

1. **Baseline**: Standard deterministic iterative updates
2. **EBM (PyTorch)**: Energy-based formulation with PyTorch sampling
3. **EBM (THRML Block Gibbs)**: Using THRML for block Gibbs sampling
4. **EBM (Contrastive Divergence Training)**: Trained with CD instead of standard backprop

**Metrics**:
- Memory retrieval accuracy
- Sampling efficiency (samples per convergence)
- Energy landscape exploration
- Training stability

**Expected Results**:
- THRML block Gibbs provides faster convergence
- EBM training improves pattern storage
- Better exploration of energy landscape

---

## 7. Implementation Steps

### Step 1: Install THRML

```bash
pip install thrml
# Or from GitHub
git clone https://github.com/extropic-ai/thrml.git
cd thrml
pip install -e .
```

### Step 2: Create EBM Module

Create `nanochat/ebm_hopfield.py` with the `EBMHopfieldMemory` class (see Section 4.2).

### Step 3: Create Trainer

Create `nanochat/ebm_trainer.py` with contrastive divergence training (see Section 5.1).

### Step 4: Update Tests

Create `tests/test_ebm_hopfield.py`:

```python
"""Tests for EBM Hopfield Memory."""

import torch
import pytest
from nanochat.ebm_hopfield import EBMHopfieldMemory


def test_ebm_hopfield_basic():
    """Test basic forward pass."""
    n_embd = 64
    memory_size = 128
    batch_size = 2
    seq_len = 10
    
    memory = EBMHopfieldMemory(
        n_embd=n_embd,
        memory_size=memory_size,
        use_thrml=False  # Use PyTorch fallback for testing
    )
    
    x = torch.randn(batch_size, seq_len, n_embd)
    output = memory(x)
    
    assert output.shape == (batch_size, seq_len, n_embd)


def test_ebm_sampling_methods():
    """Test different sampling methods."""
    n_embd = 32
    memory_size = 64
    x = torch.randn(1, 5, n_embd)
    
    memory_det = EBMHopfieldMemory(
        n_embd=n_embd,
        memory_size=memory_size,
        sampling_method='deterministic',
        use_thrml=False
    )
    
    memory_gibbs = EBMHopfieldMemory(
        n_embd=n_embd,
        memory_size=memory_size,
        sampling_method='gibbs',
        use_thrml=False
    )
    
    output_det = memory_det(x)
    output_gibbs = memory_gibbs(x)
    
    assert output_det.shape == output_gibbs.shape
```

### Step 5: Integration with SRGI Block

Update `SRGIBlock` to use EBM variant:

```python
# In SRGIBlock.__init__
if config.use_ebm_hopfield:
    from nanochat.ebm_hopfield import EBMHopfieldMemory
    self.hopfield = EBMHopfieldMemory(
        config.n_embd,
        memory_size=1024,
        use_thrml=config.use_thrml
    )
else:
    from nanochat.hopfield_memory import ModernHopfieldMemory
    self.hopfield = ModernHopfieldMemory(
        config.n_embd,
        memory_size=1024
    )
```

---

## 8. Research Questions & Experiments

### 8.1 Key Questions

1. **Does block Gibbs sampling improve convergence?**
   - Compare convergence speed: deterministic vs. block Gibbs
   - Measure autocorrelation in samples

2. **Energy-based training benefits?**
   - Compare standard backprop vs. contrastive divergence
   - Measure pattern storage capacity

3. **THRML hardware simulation advantages?**
   - Compare PyTorch vs. THRML sampling
   - Measure energy efficiency (if measurable)

4. **Exploration vs. exploitation trade-off?**
   - Test different temperature schedules
   - Measure diversity of retrieved patterns

### 8.2 Experimental Design

```python
# Experiment: Compare EBM variants
experiments = {
    'baseline': ModernHopfieldMemory(n_embd=768, memory_size=1024),
    'ebm_pytorch': EBMHopfieldMemory(
        n_embd=768, memory_size=1024, use_thrml=False
    ),
    'ebm_thrml': EBMHopfieldMemory(
        n_embd=768, memory_size=1024, use_thrml=True
    ),
    'ebm_trained_cd': EBMHopfieldMemory(
        n_embd=768, memory_size=1024, use_thrml=True
    )  # Trained with contrastive divergence
}

# Metrics
metrics = [
    'retrieval_accuracy',
    'convergence_speed',
    'sampling_efficiency',
    'energy_landscape_exploration',
    'pattern_diversity'
]
```

---

## 9. References & Resources

### Extropic Resources

- **THRML GitHub**: https://github.com/extropic-ai/thrml
- **Extropic Website**: https://extropic.ai
- **Extropic Blog**: https://extropic.ai/writing
- **Core Concepts**: https://www.github.gg/wiki/extropic-ai/thrml/core-concepts

### Key Papers

1. **Extropic EBM Research**: Check Extropic's publications on energy-based models
2. **Block Gibbs Sampling**: Papers on efficient Gibbs sampling for PGMs
3. **Contrastive Divergence**: Hinton (2002) - Training products of experts
4. **Modern Hopfield Networks**: Ramsauer et al. (2020)

### Related Work

- **Energy-Based Models**: LeCun et al. (2006) - A tutorial on energy-based learning
- **Thermodynamic Computing**: Extropic's research on TSUs
- **Probabilistic Graphical Models**: Koller & Friedman (2009)

---

## 10. Next Steps

1. **Install THRML** and explore the library
2. **Implement `EBMHopfieldMemory`** class
3. **Implement contrastive divergence trainer**
4. **Run ablation studies** comparing variants
5. **Write up results** in paper Section 5.X
6. **Update visualizations** showing EBM sampling
7. **Benchmark** sampling efficiency and convergence

---

## 11. Integration Checklist

- [ ] Install THRML library
- [ ] Study THRML documentation and examples
- [ ] Create `EBMHopfieldMemory` class
- [ ] Implement block Gibbs sampling
- [ ] Create contrastive divergence trainer
- [ ] Write tests for EBM variant
- [ ] Run ablation experiments
- [ ] Add theoretical section to paper (Section 3.6)
- [ ] Add experimental results (Section 5.X)
- [ ] Update visualizations
- [ ] Update README and documentation
- [ ] Benchmark performance improvements

---

## 12. Code Examples: Quick Start

### Basic Usage

```python
from nanochat.ebm_hopfield import EBMHopfieldMemory

# Create EBM Hopfield memory
memory = EBMHopfieldMemory(
    n_embd=768,
    memory_size=1024,
    beta=1.0,
    use_thrml=True,  # Use THRML if available
    sampling_method='block_gibbs'
)

# Forward pass
x = torch.randn(2, 10, 768)  # batch=2, seq_len=10
output = memory(x, use_ebm_sampling=True)

# With energy
output, energy = memory(x, return_energy=True, use_ebm_sampling=True)
```

### Training with Contrastive Divergence

```python
from nanochat.ebm_trainer import PersistentEBMTrainer

# Create trainer
trainer = PersistentEBMTrainer(memory, learning_rate=1e-3)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = trainer.persistent_contrastive_divergence_step(batch)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

---

**Last Updated**: November 2025  
**Status**: Draft - Ready for Implementation  
**Related Documents**: `THERMODYNAMIC_INTEGRATION.md`, `SRGI_ROADMAP.md`

