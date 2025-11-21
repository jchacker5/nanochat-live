"""
Tests for Čech-de Rham theorem implementations.

Tests all the topological deep learning modules added to SRGI.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_phase_attention_commutativity():
    """Test that PhaseAwareAttention can compute commutativity loss."""
    from nanochat.phase_attention import PhaseAwareAttention
    
    n_embd = 64
    n_head = 4
    batch_size = 2
    seq_len = 10
    
    paa = PhaseAwareAttention(n_embd=n_embd, n_head=n_head)
    
    x = torch.randn(batch_size, seq_len, n_embd)
    cos = torch.randn(1, seq_len, 1, n_embd // n_head // 2)
    sin = torch.randn(1, seq_len, 1, n_embd // n_head // 2)
    cos_sin = (cos, sin)
    
    # Test forward pass
    output = paa(x, cos_sin)
    assert output.shape == (batch_size, seq_len, n_embd)
    
    # Test with commutativity loss
    output, comm_loss = paa(x, cos_sin, return_commutativity_loss=True)
    assert output.shape == (batch_size, seq_len, n_embd)
    assert comm_loss.shape == ()  # Scalar
    assert comm_loss.item() >= 0  # Loss should be non-negative


def test_double_complex_network():
    """Test DoubleComplexNetwork forward pass."""
    from nanochat.double_complex_network import DoubleComplexNetwork
    
    n_embd = 64
    n_head = 4
    batch_size = 2
    seq_len = 10
    
    dcn = DoubleComplexNetwork(n_embd=n_embd, n_head=n_head)
    
    x = torch.randn(batch_size, seq_len, n_embd)
    cos = torch.randn(1, seq_len, 1, n_embd // n_head // 2)
    sin = torch.randn(1, seq_len, 1, n_embd // n_head // 2)
    cos_sin = (cos, sin)
    
    # Test forward pass
    output = dcn(x, cos_sin)
    assert output.shape == (batch_size, seq_len, n_embd)
    
    # Test with commutativity loss
    output, comm_loss = dcn(x, cos_sin, return_commutativity=True)
    assert output.shape == (batch_size, seq_len, n_embd)
    assert comm_loss.shape == ()  # Scalar
    assert comm_loss.item() >= 0


def test_simplicial_attention():
    """Test SimplicialAttention forward pass."""
    from nanochat.simplicial_attention import SimplicialAttention
    
    n_embd = 64
    n_head = 4
    batch_size = 2
    n_faces = 20
    
    sim_attn = SimplicialAttention(n_embd=n_embd, n_head=n_head, k=1)
    
    x = torch.randn(batch_size, n_faces, n_embd)
    
    # Test forward pass without boundary matrix
    output = sim_attn(x)
    assert output.shape == (batch_size, n_faces, n_embd)
    
    # Test with boundary matrix
    boundary_matrix = torch.eye(n_faces)  # Identity (no boundaries)
    output = sim_attn(x, boundary_matrix=boundary_matrix)
    assert output.shape == (batch_size, n_faces, n_embd)


def test_persistence_homology_tracker():
    """Test PersistenceHomologyTracker."""
    try:
        from nanochat.persistence_homology import PersistenceHomologyTracker
        
        batch_size = 2
        seq_len = 10
        n_embd = 64
        
        tracker = PersistenceHomologyTracker(max_dim=1)
        
        x_before = torch.randn(batch_size, seq_len, n_embd)
        x_after = torch.randn(batch_size, seq_len, n_embd)
        
        # Test Betti number computation
        betti_before = tracker.compute_betti_number(x_before)
        assert betti_before.shape == (batch_size,)
        assert betti_before.dtype == torch.long
        
        # Test persistence loss
        loss = tracker.persistence_loss(x_before, x_after)
        assert loss.shape == ()  # Scalar
        assert loss.item() >= 0
    except ImportError:
        pytest.skip("scipy not available, skipping persistence homology test")


def test_geometric_bottleneck_betti_tracking():
    """Test GeometricBottleneck with Betti number tracking."""
    from nanochat.geometric_bottleneck import GeometricBottleneck
    
    n_embd = 64
    batch_size = 2
    seq_len = 10
    
    geom = GeometricBottleneck(n_embd=n_embd, track_betti=True)
    
    x = torch.randn(batch_size, seq_len, n_embd)
    
    # Test forward pass
    output = geom(x)
    assert output.shape == (batch_size, seq_len, n_embd)
    
    # Test with Betti number tracking (only if tracker is available)
    if geom.betti_tracker is not None:
        output, betti_before, betti_after = geom(x, return_betti=True)
        assert output.shape == (batch_size, seq_len, n_embd)
        assert betti_before.shape == (batch_size,)
        assert betti_after.shape == (batch_size,)
    else:
        # If tracker not available, just verify forward pass works
        output = geom(x, return_betti=False)
        assert output.shape == (batch_size, seq_len, n_embd)


def test_integration_all_modules():
    """Integration test: use all modules together."""
    from nanochat.double_complex_network import DoubleComplexNetwork
    
    n_embd = 64
    n_head = 4
    batch_size = 2
    seq_len = 10
    
    # Create modules
    dcn = DoubleComplexNetwork(n_embd=n_embd, n_head=n_head)
    
    x = torch.randn(batch_size, seq_len, n_embd)
    cos = torch.randn(1, seq_len, 1, n_embd // n_head // 2)
    sin = torch.randn(1, seq_len, 1, n_embd // n_head // 2)
    cos_sin = (cos, sin)
    
    # Forward pass with commutativity
    output, comm_loss = dcn(x, cos_sin, return_commutativity=True)
    
    # Verify outputs
    assert output.shape == (batch_size, seq_len, n_embd)
    assert comm_loss.item() >= 0
    
    # Try to add persistence loss if available
    try:
        from nanochat.persistence_homology import PersistenceHomologyTracker
        tracker = PersistenceHomologyTracker(max_dim=1)
        pers_loss = tracker.persistence_loss(x, output)
        assert pers_loss.item() >= 0
        
        # Total topology loss
        total_topology_loss = comm_loss + 0.1 * pers_loss
        assert total_topology_loss.item() >= 0
    except ImportError:
        # If scipy not available, just test commutativity loss
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

