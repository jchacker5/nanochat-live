"""
Tests for Spinor Embeddings
"""

import torch
import pytest
from nanochat.spinor_embeddings import SpinorEmbedding


def test_spinor_embedding_basic():
    """Test basic forward pass of Spinor Embedding."""
    vocab_size = 1000
    n_embd = 768
    batch_size = 2
    seq_len = 50
    
    embed = SpinorEmbedding(vocab_size=vocab_size, n_embd=n_embd)
    
    # Create token indices
    idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    x = embed(idx)
    
    # Check output shape
    assert x.shape == (batch_size, seq_len, n_embd)
    
    # Check that embeddings are interleaved (real, imag, real, imag, ...)
    # This is verified by checking that we have both embedding components
    assert embed.embed_real.weight.shape == (vocab_size, n_embd // 2)
    assert embed.embed_imag.weight.shape == (vocab_size, n_embd // 2)


def test_spinor_embedding_normalize():
    """Test Spinor Embedding with normalization."""
    vocab_size = 100
    n_embd = 64
    
    embed = SpinorEmbedding(vocab_size=vocab_size, n_embd=n_embd, normalize=True)
    
    idx = torch.randint(0, vocab_size, (1, 10))
    x = embed(idx)
    
    # Check that magnitudes are approximately unit (if normalized)
    # Extract real and imaginary parts
    real = x[..., 0::2]
    imag = x[..., 1::2]
    magnitudes = torch.sqrt(real**2 + imag**2)
    
    # Should be close to 1.0 (within tolerance)
    assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5)


def test_spinor_embedding_rotate():
    """Test phase rotation operation."""
    vocab_size = 100
    n_embd = 64
    
    embed = SpinorEmbedding(vocab_size=vocab_size, n_embd=n_embd)
    
    idx = torch.randint(0, vocab_size, (2, 10))
    x = embed(idx)
    
    # Apply rotation
    theta = torch.tensor(0.5)  # Rotate by 0.5 radians
    x_rotated = embed.rotate(x, theta)
    
    # Check shape preserved
    assert x_rotated.shape == x.shape
    
    # Rotation should change the values (unless rotation is trivial)
    assert not torch.allclose(x_rotated, x, atol=1e-5)


def test_spinor_embedding_phase_magnitude():
    """Test phase and magnitude extraction."""
    vocab_size = 100
    n_embd = 64
    
    embed = SpinorEmbedding(vocab_size=vocab_size, n_embd=n_embd)
    
    idx = torch.randint(0, vocab_size, (1, 10))
    x = embed(idx)
    
    # Extract phase and magnitude
    phase = embed.get_phase(x)
    magnitude = embed.get_magnitude(x)
    
    # Check shapes
    assert phase.shape == (*x.shape[:-1], n_embd // 2)
    assert magnitude.shape == (*x.shape[:-1], n_embd // 2)
    
    # Phase should be in [-π, π]
    assert torch.all(phase >= -3.14159) and torch.all(phase <= 3.14159)
    
    # Magnitude should be positive
    assert torch.all(magnitude >= 0)


def test_spinor_embedding_unitary_property():
    """Test that rotation preserves magnitude (unitary property)."""
    vocab_size = 100
    n_embd = 64
    
    embed = SpinorEmbedding(vocab_size=vocab_size, n_embd=n_embd)
    
    idx = torch.randint(0, vocab_size, (1, 10))
    x = embed(idx)
    
    # Get original magnitude
    mag_original = embed.get_magnitude(x)
    
    # Rotate by various angles
    for theta_val in [0.1, 0.5, 1.0, 2.0]:
        theta = torch.tensor(theta_val)
        x_rotated = embed.rotate(x, theta)
        mag_rotated = embed.get_magnitude(x_rotated)
        
        # Magnitude should be preserved (unitary operation)
        assert torch.allclose(mag_original, mag_rotated, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

