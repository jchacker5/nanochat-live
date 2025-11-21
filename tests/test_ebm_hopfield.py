"""
Tests for EBM Hopfield Memory
"""

import torch
import pytest
from nanochat.ebm_hopfield import EBMHopfieldMemory


def test_ebm_hopfield_basic():
    """Test basic forward pass of EBM Hopfield Memory."""
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
    
    memory_block = EBMHopfieldMemory(
        n_embd=n_embd,
        memory_size=memory_size,
        sampling_method='block_gibbs',
        use_thrml=False
    )
    
    output_det = memory_det(x, use_ebm_sampling=False)
    output_gibbs = memory_gibbs(x, use_ebm_sampling=True)
    output_block = memory_block(x, use_ebm_sampling=True)
    
    assert output_det.shape == output_gibbs.shape == output_block.shape


def test_ebm_temperature_effect():
    """Test that temperature affects sampling."""
    n_embd = 32
    memory_size = 64
    x = torch.randn(1, 5, n_embd)
    
    memory_low = EBMHopfieldMemory(
        n_embd=n_embd,
        memory_size=memory_size,
        temperature=0.5,
        use_thrml=False
    )
    
    memory_high = EBMHopfieldMemory(
        n_embd=n_embd,
        memory_size=memory_size,
        temperature=2.0,
        use_thrml=False
    )
    
    output_low = memory_low(x, use_ebm_sampling=True)
    output_high = memory_high(x, use_ebm_sampling=True)
    
    # Outputs should be different due to temperature
    assert not torch.allclose(output_low, output_high, atol=1e-5)


def test_ebm_energy():
    """Test energy computation."""
    n_embd = 64
    memory_size = 128
    batch_size = 2
    seq_len = 10
    
    memory = EBMHopfieldMemory(
        n_embd=n_embd,
        memory_size=memory_size,
        use_thrml=False
    )
    x = torch.randn(batch_size, seq_len, n_embd)
    
    # Compute energy
    energy = memory.energy(x)
    
    # Check energy shape
    assert energy.shape == (batch_size, seq_len)
    
    # Energy should be finite
    assert torch.isfinite(energy).all()


def test_ebm_return_energy():
    """Test forward pass with return_energy=True."""
    n_embd = 64
    memory_size = 128
    batch_size = 1
    seq_len = 5
    
    memory = EBMHopfieldMemory(
        n_embd=n_embd,
        memory_size=memory_size,
        use_thrml=False
    )
    x = torch.randn(batch_size, seq_len, n_embd)
    
    # Forward pass with energy
    output, energy = memory(x, return_energy=True, use_ebm_sampling=True)
    
    # Check shapes
    assert output.shape == (batch_size, seq_len, n_embd)
    assert energy.shape == (batch_size, seq_len)
    
    # Energy should be finite
    assert torch.isfinite(energy).all()


def test_ebm_sample_negative():
    """Test negative sampling for contrastive divergence."""
    n_embd = 32
    memory_size = 64
    batch_size = 2
    seq_len = 5
    
    memory = EBMHopfieldMemory(
        n_embd=n_embd,
        memory_size=memory_size,
        use_thrml=False
    )
    
    positive_samples = torch.randn(batch_size, seq_len, n_embd)
    negative_samples = memory.sample_negative(positive_samples, n_negative_steps=3)
    
    # Check shape
    assert negative_samples.shape == positive_samples.shape
    
    # Negative samples should be different from positive
    assert not torch.allclose(positive_samples, negative_samples, atol=1e-5)


def test_ebm_gradient_flow():
    """Test that gradients flow through the module."""
    n_embd = 32
    memory_size = 64
    batch_size = 1
    seq_len = 5
    
    memory = EBMHopfieldMemory(
        n_embd=n_embd,
        memory_size=memory_size,
        use_thrml=False
    )
    x = torch.randn(batch_size, seq_len, n_embd, requires_grad=True)
    
    # Forward pass
    output = memory(x, use_ebm_sampling=True)
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    # Check gradients
    assert x.grad is not None
    assert memory.patterns.grad is not None
    assert memory.query.weight.grad is not None
    assert memory.key.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

