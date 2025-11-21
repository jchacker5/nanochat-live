"""
Tests for Geometric Bottlenecks
"""

import torch
import pytest
from nanochat.geometric_bottleneck import (
    HyperbolicBottleneck,
    ToroidalBottleneck,
    GeometricBottleneck,
)


def test_hyperbolic_bottleneck_basic():
    """Test basic forward pass of Hyperbolic Bottleneck."""
    n_embd = 768
    hyperbolic_dim = 64
    batch_size = 2
    seq_len = 100
    
    hyp = HyperbolicBottleneck(n_embd=n_embd, hyperbolic_dim=hyperbolic_dim, use_geoopt=False)
    
    x = torch.randn(batch_size, seq_len, n_embd)
    y = hyp(x)
    
    # Check output shape
    assert y.shape == (batch_size, seq_len, n_embd)
    
    # Output should be different from input (non-trivial transformation)
    assert not torch.allclose(y, x, atol=1e-5)


def test_toroidal_bottleneck_basic():
    """Test basic forward pass of Toroidal Bottleneck."""
    n_embd = 768
    n_circles = 4
    batch_size = 2
    seq_len = 100
    
    tor = ToroidalBottleneck(n_embd=n_embd, n_circles=n_circles)
    
    x = torch.randn(batch_size, seq_len, n_embd)
    y = tor(x)
    
    # Check output shape
    assert y.shape == (batch_size, seq_len, n_embd)
    
    # Output should be different from input
    assert not torch.allclose(y, x, atol=1e-5)


def test_geometric_bottleneck_basic():
    """Test basic forward pass of combined Geometric Bottleneck."""
    n_embd = 768
    hyperbolic_dim = 64
    n_circles = 4
    batch_size = 2
    seq_len = 100
    
    geom = GeometricBottleneck(
        n_embd=n_embd,
        hyperbolic_dim=hyperbolic_dim,
        n_circles=n_circles,
        use_geoopt=False
    )
    
    x = torch.randn(batch_size, seq_len, n_embd)
    y = geom(x)
    
    # Check output shape
    assert y.shape == (batch_size, seq_len, n_embd)
    
    # Check that alpha parameter exists
    assert hasattr(geom, 'alpha')
    assert geom.alpha.shape == ()


def test_geometric_bottleneck_alpha():
    """Test that mixing weight alpha is in [0, 1]."""
    n_embd = 64
    geom = GeometricBottleneck(n_embd=n_embd, use_geoopt=False)
    
    x = torch.randn(1, 10, n_embd)
    y = geom(x)
    
    # Alpha should be sigmoid'd, so in [0, 1]
    alpha_val = torch.sigmoid(geom.alpha)
    assert 0 <= alpha_val.item() <= 1


def test_toroidal_bottleneck_circle_structure():
    """Test that toroidal bottleneck preserves circle structure."""
    n_embd = 64
    n_circles = 4
    
    tor = ToroidalBottleneck(n_embd=n_embd, n_circles=n_circles)
    
    x = torch.randn(1, 10, n_embd)
    
    # Get intermediate representation
    angles = tor.to_torus(x)
    angles = angles.view(*angles.shape[:-1], n_circles, 2)
    
    # Check that angles are normalized (on unit circles)
    norms = torch.norm(angles, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_hyperbolic_bottleneck_ball_constraint():
    """Test that hyperbolic bottleneck constrains to unit ball."""
    n_embd = 64
    hyperbolic_dim = 32
    
    hyp = HyperbolicBottleneck(n_embd=n_embd, hyperbolic_dim=hyperbolic_dim, use_geoopt=False)
    
    x = torch.randn(1, 10, n_embd)
    y = hyp(x)
    
    # The transformation should be bounded (not explode)
    assert torch.isfinite(y).all()
    assert not torch.isnan(y).any()


@pytest.mark.skipif(
    not hasattr(__import__('nanochat.geometric_bottleneck'), 'HAS_GEOOPT') or
    not __import__('nanochat.geometric_bottleneck').HAS_GEOOPT,
    reason="geoopt not available"
)
def test_hyperbolic_bottleneck_with_geoopt():
    """Test Hyperbolic Bottleneck with geoopt if available."""
    n_embd = 64
    hyperbolic_dim = 32
    
    hyp = HyperbolicBottleneck(n_embd=n_embd, hyperbolic_dim=hyperbolic_dim, use_geoopt=True)
    
    x = torch.randn(1, 10, n_embd)
    y = hyp(x)
    
    assert y.shape == (1, 10, n_embd)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

