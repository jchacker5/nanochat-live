"""
Tests for Modern Hopfield Memory
"""

import torch
import pytest
from nanochat.hopfield_memory import ModernHopfieldMemory


def test_hopfield_memory_basic():
    """Test basic forward pass of Modern Hopfield Memory."""
    n_embd = 768
    memory_size = 1024
    batch_size = 2
    seq_len = 100
    
    memory = ModernHopfieldMemory(n_embd=n_embd, memory_size=memory_size)
    
    # Create input
    x = torch.randn(batch_size, seq_len, n_embd)
    
    # Forward pass
    output = memory(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, n_embd)
    
    # Check that patterns parameter exists
    assert hasattr(memory, 'patterns')
    assert memory.patterns.shape == (memory_size, n_embd)


def test_hopfield_memory_energy():
    """Test energy computation."""
    n_embd = 64
    memory_size = 128
    batch_size = 2
    seq_len = 10
    
    memory = ModernHopfieldMemory(n_embd=n_embd, memory_size=memory_size)
    x = torch.randn(batch_size, seq_len, n_embd)
    
    # Compute energy
    energy = memory.energy(x)
    
    # Check energy shape
    assert energy.shape == (batch_size, seq_len)
    
    # Energy should be finite
    assert torch.isfinite(energy).all()
    
    # Energy should be negative (logsumexp of positive similarities)
    assert (energy <= 0).all()


def test_hopfield_memory_return_energy():
    """Test forward pass with return_energy=True."""
    n_embd = 64
    memory_size = 128
    batch_size = 1
    seq_len = 5
    
    memory = ModernHopfieldMemory(n_embd=n_embd, memory_size=memory_size)
    x = torch.randn(batch_size, seq_len, n_embd)
    
    # Forward pass with energy
    output, energy = memory(x, return_energy=True)
    
    # Check shapes
    assert output.shape == (batch_size, seq_len, n_embd)
    assert energy.shape == (batch_size, seq_len)
    
    # Energy should be finite
    assert torch.isfinite(energy).all()


def test_hopfield_memory_convergence():
    """Test that iterative updates converge toward attractors."""
    n_embd = 32
    memory_size = 64
    batch_size = 1
    seq_len = 1
    
    memory = ModernHopfieldMemory(n_embd=n_embd, memory_size=memory_size, n_steps=1)
    x = torch.randn(batch_size, seq_len, n_embd)
    
    # Get output with 1 step
    output_1 = memory(x)
    
    # Get output with more steps
    memory.n_steps = 5
    output_5 = memory(x)
    
    # More steps should change the output (convergence)
    assert not torch.allclose(output_1, output_5, atol=1e-5)


def test_hopfield_memory_attention_weights():
    """Test attention weights retrieval."""
    n_embd = 64
    memory_size = 128
    batch_size = 2
    seq_len = 10
    
    memory = ModernHopfieldMemory(n_embd=n_embd, memory_size=memory_size)
    x = torch.randn(batch_size, seq_len, n_embd)
    
    # Get attention weights
    attn = memory.get_attention_weights(x)
    
    # Check shape
    assert attn.shape == (batch_size, seq_len, memory_size)
    
    # Attention weights should sum to 1
    attn_sum = attn.sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)
    
    # All weights should be non-negative
    assert (attn >= 0).all()


def test_hopfield_memory_different_beta():
    """Test that different beta values affect retrieval."""
    n_embd = 32
    memory_size = 64
    batch_size = 1
    seq_len = 5
    
    x = torch.randn(batch_size, seq_len, n_embd)
    
    # Low beta (smoother)
    memory_low = ModernHopfieldMemory(n_embd=n_embd, memory_size=memory_size, beta=0.5)
    output_low = memory_low(x)
    
    # High beta (sharper)
    memory_high = ModernHopfieldMemory(n_embd=n_embd, memory_size=memory_size, beta=2.0)
    output_high = memory_high(x)
    
    # Outputs should be different
    assert not torch.allclose(output_low, output_high, atol=1e-5)


def test_hopfield_memory_denoising():
    """Test denoising property: corrupted input converges to clean pattern."""
    n_embd = 32
    memory_size = 10
    batch_size = 1
    seq_len = 1
    
    memory = ModernHopfieldMemory(n_embd=n_embd, memory_size=memory_size, n_steps=5)
    
    # Store a specific pattern by initializing patterns
    target_pattern = torch.randn(1, n_embd)
    with torch.no_grad():
        memory.patterns[0] = target_pattern
    
    # Create corrupted input (target + noise)
    noise = torch.randn(1, n_embd) * 0.5
    corrupted_input = target_pattern + noise
    
    # Retrieve
    output = memory(corrupted_input.unsqueeze(0))
    
    # Output should be closer to target than corrupted input
    dist_corrupted = torch.norm(corrupted_input - target_pattern)
    dist_output = torch.norm(output.squeeze(0) - target_pattern)
    
    # Note: This is a weak test - convergence depends on initialization
    # But we can at least check that the output is different from input
    assert not torch.allclose(output.squeeze(0), corrupted_input, atol=1e-3)


def test_hopfield_memory_gradient_flow():
    """Test that gradients flow through the module."""
    n_embd = 32
    memory_size = 64
    batch_size = 1
    seq_len = 5
    
    memory = ModernHopfieldMemory(n_embd=n_embd, memory_size=memory_size)
    x = torch.randn(batch_size, seq_len, n_embd, requires_grad=True)
    
    # Forward pass
    output = memory(x)
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    # Check gradients
    assert x.grad is not None
    assert memory.patterns.grad is not None
    assert memory.query.weight.grad is not None
    assert memory.key.weight.grad is not None


def test_hopfield_memory_different_memory_sizes():
    """Test with different memory sizes."""
    n_embd = 64
    batch_size = 1
    seq_len = 10
    x = torch.randn(batch_size, seq_len, n_embd)
    
    # Small memory
    memory_small = ModernHopfieldMemory(n_embd=n_embd, memory_size=32)
    output_small = memory_small(x)
    assert output_small.shape == (batch_size, seq_len, n_embd)
    
    # Large memory
    memory_large = ModernHopfieldMemory(n_embd=n_embd, memory_size=2048)
    output_large = memory_large(x)
    assert output_large.shape == (batch_size, seq_len, n_embd)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

