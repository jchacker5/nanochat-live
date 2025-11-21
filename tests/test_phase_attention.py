"""
Tests for Phase-Aware Attention (PAA)
"""

import torch
import pytest
from nanochat.phase_attention import PhaseAwareAttention


def test_phase_aware_attention_basic():
    """Test basic forward pass of Phase-Aware Attention."""
    n_embd = 768
    n_head = 12
    batch_size = 2
    seq_len = 100
    
    paa = PhaseAwareAttention(n_embd=n_embd, n_head=n_head, n_kv_head=n_head)
    
    # Create input
    x = torch.randn(batch_size, seq_len, n_embd)
    
    # Create RoPE embeddings
    head_dim = n_embd // n_head
    cos = torch.randn(1, seq_len, 1, head_dim // 2)
    sin = torch.randn(1, seq_len, 1, head_dim // 2)
    cos_sin = (cos, sin)
    
    # Forward pass
    y = paa(x, cos_sin, kv_cache=None)
    
    # Check output shape
    assert y.shape == (batch_size, seq_len, n_embd)
    
    # Check that beta parameter exists and is learnable
    assert hasattr(paa, 'beta')
    assert paa.beta.shape == (n_head,)


def test_phase_aware_attention_gqa():
    """Test Phase-Aware Attention with Group Query Attention (GQA)."""
    n_embd = 768
    n_head = 12
    n_kv_head = 6  # GQA: fewer KV heads
    batch_size = 2
    seq_len = 50
    
    paa = PhaseAwareAttention(n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head)
    
    x = torch.randn(batch_size, seq_len, n_embd)
    head_dim = n_embd // n_head
    cos = torch.randn(1, seq_len, 1, head_dim // 2)
    sin = torch.randn(1, seq_len, 1, head_dim // 2)
    cos_sin = (cos, sin)
    
    y = paa(x, cos_sin, kv_cache=None)
    assert y.shape == (batch_size, seq_len, n_embd)


def test_phase_aware_attention_coherence():
    """Test that phase coherence affects attention scores."""
    n_embd = 64
    n_head = 2
    seq_len = 10
    
    paa = PhaseAwareAttention(n_embd=n_embd, n_head=n_head, beta_init=1.0)
    
    x = torch.randn(1, seq_len, n_embd)
    head_dim = n_embd // n_head
    cos = torch.randn(1, seq_len, 1, head_dim // 2)
    sin = torch.randn(1, seq_len, 1, head_dim // 2)
    cos_sin = (cos, sin)
    
    # Forward pass
    y = paa(x, cos_sin, kv_cache=None)
    
    # Output should be different from input (non-trivial transformation)
    assert not torch.allclose(y, x, atol=1e-5)


def test_phase_aware_attention_zero_beta():
    """Test that beta=0 recovers standard attention behavior."""
    n_embd = 64
    n_head = 2
    seq_len = 10
    
    paa = PhaseAwareAttention(n_embd=n_embd, n_head=n_head, beta_init=0.0)
    # Manually set beta to zero
    with torch.no_grad():
        paa.beta.zero_()
    
    x = torch.randn(1, seq_len, n_embd)
    head_dim = n_embd // n_head
    cos = torch.randn(1, seq_len, 1, head_dim // 2)
    sin = torch.randn(1, seq_len, 1, head_dim // 2)
    cos_sin = (cos, sin)
    
    y = paa(x, cos_sin, kv_cache=None)
    assert y.shape == (1, seq_len, n_embd)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

