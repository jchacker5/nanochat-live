# SRGI Architecture Roadmap

**Spin-Resonant Geometric Intelligence — Implementation Plan**

This document outlines the complete SRGI architecture and implementation roadmap for NanoChat-Live. The architecture is based on the neuroscience findings that the cortex operates as a coherent wave system with gamma-stabilized rotating waves.

---

## Architecture Overview

```
Input → Spinor Embeddings → [Resonant SSM + Phase-Aware Attention + Geometric Bottlenecks]* → Hopfield Attractor → Output
           (Phase-2)              (Phase-1)         (Phase-2)              (Phase-2)           (Phase-3)
```

### Three-Layer Design Philosophy

1. **Resonance Layer** (Phase-1) — Temporal stability via oscillators
2. **Phase Layer** (Phase-2) — Coherence via synchronization
3. **Geometry Layer** (Phase-2/3) — Structure via Riemannian manifolds + attractors

---

## Phase-1: Resonant Foundation ✅ COMPLETE

### StableResonantSSM (`nanochat/ssm.py`)

**Brain Mapping**: PV interneurons → 40 Hz gamma stability

**Mathematical Foundation**:
- Continuous dynamics: `dh/dt = A h + B u` where `A` has complex eigenvalues
- Bilinear discretization: `A_d = (I - dt/2*A)^{-1} (I + dt/2*A)`
- Damping constraint: `Re(λ) ≤ -damp_min` (prevents pure imaginary)
- State evolution: `h[t+1] = A_d h[t] + B_d u[t]`, output `y = Re(C h)`

**Key Properties**:
- ✅ Unitary-like dynamics (stable norms)
- ✅ Phase-preserving (complex state)
- ✅ Gradient-friendly (spectral constraints)
- ✅ Interpretable (phase/magnitude extraction)

**Implementation Status**: ✅ Done
- File: `nanochat/ssm.py`
- Classes: `StableResonantSSM`, `ResonantBlock`
- Demo: `scripts/ssm_demo.py`
- Verification: All checks passed

**Phase-2 Implementation Status**: ✅ Complete
- File: `nanochat/phase_attention.py`
- File: `nanochat/spinor_embeddings.py`
- File: `nanochat/geometric_bottleneck.py`
- Classes: `PhaseAwareAttention`, `SpinorEmbedding`, `GeometricBottleneck`
- Tests: `tests/test_phase_attention.py`, `tests/test_spinor_embeddings.py`, `tests/test_geometric_bottleneck.py`

**Integration**:
```python
from nanochat.ssm import ResonantBlock
block = ResonantBlock(n_embd=768, state_dim=384)
x = block(hidden_states)  # Parallel to attention
```

---

## Phase-2: Phase-Aware Dynamics ✅ COMPLETE

### 2.1 Phase-Aware Attention (PAA)

**Brain Mapping**: Gamma phase-locking for coherence

**Mathematical Foundation**:
- Standard attention: `Attn(Q, K, V) = softmax(QK^T / √d) V`
- Phase-aware: `PAA(Q, K, V) = softmax((QK^T / √d) · (1 + β cos(Δφ))) V`
- Phase difference: `Δφ = angle(Q) - angle(K)` (from RoPE or explicit phase)
- Coherence gate: `β ∈ [0, 1]` learned per head

**Key Properties**:
- Tokens "in phase" get higher attention weight
- Out-of-phase tokens are downweighted
- Preserves standard attention when `β=0`
- Enables temporal binding via phase synchronization

**Implementation Plan**:

```python
# File: nanochat/phase_attention.py

class PhaseAwareAttention(nn.Module):
    """Attention with phase coherence gating."""

    def __init__(self, n_embd, n_head, beta_init=0.5):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # Standard Q, K, V projections
        self.c_q = nn.Linear(n_embd, n_embd, bias=False)
        self.c_k = nn.Linear(n_embd, n_embd, bias=False)
        self.c_v = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        # Phase coherence weight (learned per head)
        self.beta = nn.Parameter(torch.ones(n_head) * beta_init)

    def forward(self, x, cos_sin):
        B, T, C = x.shape

        # Project and split heads
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE (which gives us phase information)
        cos, sin = cos_sin
        q_rot = apply_rotary_emb(q, cos, sin)
        k_rot = apply_rotary_emb(k, cos, sin)

        # Extract phase from complex representation of RoPE
        # Phase difference: Δφ_ij between token i and j
        # Approximate via position difference (RoPE encodes this)
        position_diff = torch.arange(T, device=x.device).unsqueeze(0) - \
                       torch.arange(T, device=x.device).unsqueeze(1)

        # Coherence modulation: 1 + β cos(θ * position_diff)
        # θ is implicit in RoPE frequency
        theta = 10000 ** (-torch.arange(0, self.head_dim, 2, device=x.device) / self.head_dim)
        phase_diff = position_diff.unsqueeze(-1) * theta.unsqueeze(0).unsqueeze(0)
        coherence = 1 + self.beta.view(1, -1, 1, 1) * torch.cos(phase_diff).mean(-1, keepdim=True)

        # Standard attention scores
        attn_scores = (q_rot @ k_rot.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Modulate by phase coherence
        attn_scores = attn_scores * coherence

        # Apply softmax and attend to values
        attn_weights = F.softmax(attn_scores, dim=-1)
        y = attn_weights @ v

        # Re-assemble and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y
```

**Expected Benefits**:
- Improved long-range binding
- Temporal coherence in sequences
- Reduced attention dilution
- Phase-based selective routing

**Testing**:
- Copy task (phase alignment critical)
- Associative recall (binding test)
- Long-context coherence metrics

---

### 2.2 Spinor Embeddings

**Brain Mapping**: Spinor-like orientation invariance in cortex

**Mathematical Foundation**:
- Standard embeddings: `E ∈ R^{vocab × d}`
- Quaternion embeddings: `E ∈ H^{vocab × d/4}` where `H` is quaternion space
- Or complex: `E ∈ C^{vocab × d/2}`
- Unitary operations preserve norm: `||R(θ)e|| = ||e||`

**Implementation Plan**:

```python
# File: nanochat/spinor_embeddings.py

class SpinorEmbedding(nn.Module):
    """Complex-valued embeddings with unitary operations."""

    def __init__(self, vocab_size, n_embd):
        super().__init__()
        assert n_embd % 2 == 0, "n_embd must be even for complex embeddings"
        self.n_embd = n_embd

        # Complex embedding: real and imaginary parts
        self.embed_real = nn.Embedding(vocab_size, n_embd // 2)
        self.embed_imag = nn.Embedding(vocab_size, n_embd // 2)

    def forward(self, idx):
        # Get complex embedding
        real = self.embed_real(idx)
        imag = self.embed_imag(idx)

        # Normalize to unit magnitude (optional, for true spinors)
        # magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
        # real, imag = real / magnitude, imag / magnitude

        # Interleave real and imaginary for compatibility with real-valued layers
        # [r1, i1, r2, i2, ...] allows treating as real while preserving structure
        embed = torch.stack([real, imag], dim=-1).flatten(-2, -1)

        return embed  # Shape: (B, T, n_embd)

    def rotate(self, x, theta):
        """Apply phase rotation: e^{iθ} * z = (cos θ + i sin θ)(a + ib)"""
        # Split into real and imaginary
        real, imag = x[..., 0::2], x[..., 1::2]

        # Rotate: real' = real cos θ - imag sin θ, imag' = real sin θ + imag cos θ
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        real_rot = real * cos_theta - imag * sin_theta
        imag_rot = real * sin_theta + imag * cos_theta

        # Interleave back
        return torch.stack([real_rot, imag_rot], dim=-1).flatten(-2, -1)
```

---

### 2.3 Geometric Bottlenecks (Hyperbolic + Toroidal)

**Brain Mapping**: Rotating cortical waves on curved manifolds

**Mathematical Foundation**:

**Hyperbolic space** (Poincaré ball) for hierarchies:
- Metric: `ds² = 4/(1 - ||x||²)² ||dx||²`
- Exponential map: `exp_p(v) = p ⊕ tanh(||v||/2) v/||v||`
- Logarithmic map: `log_p(q) = 2 arctanh(||q ⊖ p||) (q ⊖ p)/||q ⊖ p||`
- Geodesics are hyperbolic arcs
- **Property**: Encodes tree-like structures naturally (parent-child distances)

**Toroidal space** (S¹ × S¹ × ... ) for periodic phenomena:
- Metric: `ds² = dθ₁² + dθ₂² + ...` (angles on circles)
- Operations: Angular addition/subtraction
- **Property**: Encodes cyclic patterns (time, rotation, periodic structures)

**Implementation Plan**:

```python
# File: nanochat/geometric_bottleneck.py

import torch
import torch.nn as nn
import geoopt  # Riemannian optimization library

class HyperbolicBottleneck(nn.Module):
    """Poincaré ball bottleneck for hierarchical structure."""

    def __init__(self, n_embd, hyperbolic_dim, curvature=-1.0):
        super().__init__()
        self.n_embd = n_embd
        self.hyperbolic_dim = hyperbolic_dim

        # Manifold (Poincaré ball)
        self.manifold = geoopt.PoincareBall(c=-curvature)

        # Project to hyperbolic space
        self.to_hyp = nn.Linear(n_embd, hyperbolic_dim)

        # Riemannian layer (geodesic operations)
        self.hyp_transform = geoopt.layers.RadialNd(
            hyperbolic_dim, hyperbolic_dim, self.manifold
        )

        # Project back to Euclidean
        self.from_hyp = nn.Linear(hyperbolic_dim, n_embd)

    def forward(self, x):
        # x: (B, T, n_embd)
        B, T, C = x.shape

        # Project to hyperbolic space
        x_hyp = self.to_hyp(x)
        x_hyp = self.manifold.expmap0(x_hyp)  # Map to Poincaré ball

        # Transform in hyperbolic space (geodesic operations)
        x_hyp = self.hyp_transform(x_hyp)

        # Project back to Euclidean
        x_euclid = self.manifold.logmap0(x_hyp)
        x_out = self.from_hyp(x_euclid)

        return x_out


class ToroidalBottleneck(nn.Module):
    """Toroidal space for periodic/cyclic structure."""

    def __init__(self, n_embd, n_circles=4):
        super().__init__()
        self.n_embd = n_embd
        self.n_circles = n_circles

        # Project to angular representation (each circle has 2 dims: sin, cos)
        self.to_torus = nn.Linear(n_embd, n_circles * 2)

        # Transform on torus (via angle mixing)
        self.torus_transform = nn.Linear(n_circles * 2, n_circles * 2)

        # Project back
        self.from_torus = nn.Linear(n_circles * 2, n_embd)

    def forward(self, x):
        # Project to angular space
        angles = self.to_torus(x)  # (B, T, n_circles * 2)

        # Normalize to unit circles: [sin θ, cos θ] for each circle
        angles = angles.view(*angles.shape[:-1], self.n_circles, 2)
        angles = F.normalize(angles, p=2, dim=-1)  # Unit circle constraint
        angles = angles.flatten(-2, -1)

        # Transform (mixing angles while preserving circle structure)
        angles_transformed = self.torus_transform(angles)
        angles_transformed = angles_transformed.view(*angles.shape[:-1], self.n_circles, 2)
        angles_transformed = F.normalize(angles_transformed, p=2, dim=-1)
        angles_transformed = angles_transformed.flatten(-2, -1)

        # Project back to Euclidean
        x_out = self.from_torus(angles_transformed)

        return x_out


class GeometricBottleneck(nn.Module):
    """Combined hyperbolic + toroidal bottleneck."""

    def __init__(self, n_embd, hyperbolic_dim=64, n_circles=4):
        super().__init__()
        self.hyperbolic = HyperbolicBottleneck(n_embd, hyperbolic_dim)
        self.toroidal = ToroidalBottleneck(n_embd, n_circles)

        # Mixing weight (learned)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x_hyp = self.hyperbolic(x)
        x_tor = self.toroidal(x)

        # Weighted combination
        alpha = torch.sigmoid(self.alpha)  # Ensure [0, 1]
        return alpha * x_hyp + (1 - alpha) * x_tor
```

**Dependencies**:
- `geoopt` for Riemannian operations: `pip install geoopt`

**Testing**:
- Hierarchical reasoning (tree structures)
- Periodic pattern recognition (time series)
- Multi-scale representation learning

---

## Phase-3: Attractor Memory 📋 PLANNED

### Modern Hopfield Networks

**Brain Mapping**: Wave attractors = perceptual clarity

**Mathematical Foundation**:
- Classical Hopfield: `E = -½ x^T W x`, updates via `sign(Wx)`
- Modern Hopfield (2016): `E = -log Σ exp(x^T ξᵢ / β)`, continuous attractor
- Dense associative memory with exponential capacity
- Energy minimization → convergence to stored patterns

**Implementation Plan**:

```python
# File: nanochat/hopfield_memory.py

class ModernHopfieldMemory(nn.Module):
    """Dense associative memory via modern Hopfield networks."""

    def __init__(self, n_embd, memory_size=1024, beta=1.0, n_steps=3):
        super().__init__()
        self.n_embd = n_embd
        self.memory_size = memory_size
        self.beta = beta  # Inverse temperature
        self.n_steps = n_steps  # Inner loop iterations

        # Memory patterns (learned or initialized)
        self.patterns = nn.Parameter(torch.randn(memory_size, n_embd) * 0.02)

        # Query/Key projections
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)

    def energy(self, x, patterns):
        """Compute Hopfield energy: -log Σ exp(β x^T ξᵢ)"""
        similarities = (x @ patterns.T) * self.beta  # (B, T, memory_size)
        return -torch.logsumexp(similarities, dim=-1)  # (B, T)

    def forward(self, x, return_energy=False):
        """
        Forward with iterative energy minimization.
        x: (B, T, n_embd)
        """
        B, T, C = x.shape

        # Project to query space
        q = self.query(x)
        k = self.key(self.patterns)  # Project memory patterns

        # Iterative updates (energy minimization)
        state = q
        for step in range(self.n_steps):
            # Compute similarities to all patterns
            sim = (state @ k.T) * self.beta  # (B, T, memory_size)
            attn = F.softmax(sim, dim=-1)

            # Retrieve weighted combination of patterns
            retrieved = attn @ self.patterns  # (B, T, n_embd)

            # Update state (move toward attractor)
            state = 0.5 * state + 0.5 * retrieved

        # Final retrieval
        sim_final = (state @ k.T) * self.beta
        attn_final = F.softmax(sim_final, dim=-1)
        output = attn_final @ self.patterns

        if return_energy:
            energy_val = self.energy(output, self.patterns)
            return output, energy_val

        return output
```

**Key Properties**:
- Fixed-point attractors (stable memory states)
- Exponential storage capacity
- Denoising (corrupted inputs converge to clean patterns)
- Associative recall (partial cues retrieve full patterns)

**Integration**:
```python
# Add to transformer block
hopfield_head = ModernHopfieldMemory(n_embd=768)
memory_state = hopfield_head(hidden_states)
output = hidden_states + 0.3 * memory_state  # Residual blend
```

**Testing**:
- Associative recall tasks
- Long-context memory retrieval
- Hallucination reduction (attractor stability)

---

## Full SRGI Block Architecture

```python
class SRGIBlock(nn.Module):
    """
    Full SRGI transformer block integrating all components.

    Architecture:
        Input → Layer Norm
              → Phase-Aware Attention (Phase-2)
              → Resonant SSM (Phase-1)
              → Geometric Bottleneck (Phase-2)
              → MLP
              → Hopfield Memory (Phase-3, optional)
              → Output
    """

    def __init__(self, config):
        super().__init__()

        # Phase-Aware Attention
        self.paa = PhaseAwareAttention(
            config.n_embd,
            config.n_head,
            beta_init=0.5
        )

        # Resonant SSM
        self.ssm = ResonantBlock(
            config.n_embd,
            state_dim=config.n_embd // 2,
            residual_weight=0.1
        )

        # Geometric bottleneck
        self.geom = GeometricBottleneck(
            config.n_embd,
            hyperbolic_dim=64,
            n_circles=4
        )

        # Standard MLP
        self.mlp = MLP(config)

        # Optional: Hopfield attractor memory
        if config.use_hopfield:
            self.hopfield = ModernHopfieldMemory(
                config.n_embd,
                memory_size=1024,
                n_steps=3
            )
        else:
            self.hopfield = None

    def forward(self, x, cos_sin, kv_cache=None):
        # Phase-aware attention
        x = x + self.paa(norm(x), cos_sin)

        # Resonant SSM (parallel pathway)
        x = self.ssm(x)

        # Geometric structure
        x = x + 0.2 * self.geom(norm(x))

        # Standard MLP
        x = x + self.mlp(norm(x))

        # Attractor memory (optional)
        if self.hopfield is not None:
            memory = self.hopfield(x)
            x = x + 0.3 * memory

        return x
```

---

## Regularization & Training Losses

### Phase Consistency Loss

Encourage phase coherence across tokens:

```python
def phase_consistency_loss(hidden_states, window=10):
    """
    Penalize large phase jumps within local windows.
    hidden_states: (B, T, n_embd) — assumed to have phase structure
    """
    # Extract phase from complex representation
    real, imag = hidden_states[..., 0::2], hidden_states[..., 1::2]
    phase = torch.atan2(imag, real)  # (B, T, n_embd/2)

    # Phase difference between adjacent tokens
    phase_diff = phase[:, 1:, :] - phase[:, :-1, :]

    # Wrap to [-π, π]
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))

    # Penalize large jumps
    loss = (phase_diff ** 2).mean()

    return loss
```

### Spectral Regularization

Encourage eigenvalues near unit circle:

```python
def spectral_regularization(A_real, A_imag, target_radius=1.0):
    """
    Encourage eigenvalues to stay near target radius in complex plane.
    A_real, A_imag: Real and imaginary parts of diagonal eigenvalues
    """
    radius = torch.sqrt(A_real ** 2 + A_imag ** 2)
    loss = ((radius - target_radius) ** 2).mean()
    return loss
```

### Total Training Loss

```python
loss_total = (
    loss_lm  # Standard language modeling loss
    + 0.01 * phase_consistency_loss(hidden_states)
    + 0.001 * spectral_regularization(model.A_real, model.A_imag)
)
```

---

## Implementation Timeline

| Phase | Component | ETA | Dependencies |
|-------|-----------|-----|--------------|
| ✅ Phase-1 | StableResonantSSM | Done | PyTorch |
| ✅ Phase-2 | PhaseAwareAttention | Done | Phase-1 |
| ✅ Phase-2 | SpinorEmbeddings | Done | None |
| ✅ Phase-2 | Geometric Bottlenecks | Done | geoopt (optional) |
| 📋 Phase-3 | Hopfield Memory | Week 3-4 | Phase-2 |
| 📋 Phase-3 | Full Integration | Week 4-5 | All above |
| 📋 Phase-3 | Benchmark Suite | Week 5-6 | Full model |

---

## Testing & Validation

### Unit Tests

- [x] `test_phase_aware_attention.py` — Phase coherence gating ✅
- [x] `test_spinor_embeddings.py` — Unitary operations ✅
- [x] `test_geometric_bottleneck.py` — Manifold constraints ✅
- [ ] `test_hopfield_memory.py` — Attractor convergence

### Integration Tests

- [ ] `test_srgi_block.py` — Full block forward/backward
- [ ] `test_gradient_flow.py` — Deep network stability
- [ ] `test_phase_consistency.py` — Phase preservation through layers

### Benchmark Tasks

- [ ] **NIAH (Needle In A Haystack)** — 64k/128k context exact recall
- [ ] **Long-range coreference** — Entity tracking across documents
- [ ] **Associative recall** — Binding and retrieval
- [ ] **Copy task** — Phase alignment verification
- [ ] **Perplexity** — Language modeling quality
- [ ] **Hallucination rate** — Attractor stability test

---

## References & Related Work

### Neuroscience Papers (November 2025)
- **PV Interneurons & Gamma**: Neuron, November 2025
- **Rotating Cortical Waves**: Miller Lab @ MIT, November 2025

### State-Space Models
- **S4** (Gu et al., 2021): Structured State Spaces for Sequence Modeling
- **Mamba** (Gu & Dao, 2023): Fast SSM with selective state updates

### Geometric Deep Learning
- **Hyperbolic Neural Networks** (Ganea et al., 2018)
- **Toroidal Embeddings** (Daza et al., 2021)
- **Riemannian Optimization** (geoopt library)

### Associative Memory
- **Modern Hopfield Networks** (Ramsauer et al., 2020)
- **Dense Associative Memory** (Krotov & Hopfield, 2016)

### Phase & Oscillations
- **Neural Oscillations** (Buzsáki & Draguhn, 2004)
- **Phase Precession** (O'Keefe & Recce, 1993)

---

## Contributing

We welcome contributions to any phase:

1. **Phase-1 optimizations**: Parallel scan for SSM, numerical stability improvements
2. **Phase-2 implementations**: PAA, spinor embeddings, geometric bottlenecks
3. **Phase-3 memory**: Hopfield networks, attractor visualization
4. **Benchmarks**: Long-context tasks, phase coherence metrics
5. **Documentation**: Tutorials, ablations, visualizations

See `CONTRIBUTING.md` for guidelines.

---

**Let's stop scaling and start resonating.**
