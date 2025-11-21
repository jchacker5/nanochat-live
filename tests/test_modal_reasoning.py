"""
Tests for modal logic reasoning modules.
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_kripke_frame():
    """Test KripkeFrame forward pass."""
    from nanochat.modal_reasoning import KripkeFrame
    
    n_worlds = 4
    n_embd = 64
    batch_size = 2
    seq_len = 10
    
    kf = KripkeFrame(n_worlds=n_worlds, n_embd=n_embd, relation_type='S5')
    
    x = torch.randn(batch_size, seq_len, n_embd)
    output = kf(x)
    
    assert output.shape == (batch_size, seq_len, n_embd)
    assert kf.accessibility.shape == (n_worlds, n_worlds)


def test_modal_attention():
    """Test ModalAttention forward pass."""
    from nanochat.modal_reasoning import ModalAttention, KripkeFrame
    
    n_embd = 64
    n_head = 4
    batch_size = 2
    seq_len = 10
    
    modal_attn = ModalAttention(n_embd=n_embd, n_head=n_head, n_worlds=4)
    kf = KripkeFrame(n_worlds=4, n_embd=n_embd)
    
    x = torch.randn(batch_size, seq_len, n_embd)
    
    # Test without Kripke frame
    output = modal_attn(x)
    assert output.shape == (batch_size, seq_len, n_embd)
    
    # Test with Kripke frame
    output = modal_attn(x, kripke_frame=kf)
    assert output.shape == (batch_size, seq_len, n_embd)


def test_modal_cot_reasoning():
    """Test ModalCoTReasoning forward pass."""
    from nanochat.modal_reasoning import ModalCoTReasoning
    
    n_embd = 64
    batch_size = 2
    seq_len = 10
    
    modal_cot = ModalCoTReasoning(n_embd=n_embd, n_worlds=4, max_steps=3)
    
    x = torch.randn(batch_size, seq_len, n_embd)
    
    # Test forward pass
    output = modal_cot(x)
    assert output.shape == (batch_size, seq_len, n_embd)
    
    # Test with verification scores
    output, verification = modal_cot(x, return_verification=True)
    assert output.shape == (batch_size, seq_len, n_embd)
    assert verification.shape == (batch_size, seq_len, 3)  # max_steps=3


def test_modal_geometric_bottleneck():
    """Test ModalGeometricBottleneck forward pass."""
    from nanochat.modal_reasoning import ModalGeometricBottleneck
    
    n_embd = 64
    batch_size = 2
    seq_len = 10
    
    modal_geom = ModalGeometricBottleneck(n_embd=n_embd, n_worlds=4)
    
    x = torch.randn(batch_size, seq_len, n_embd)
    
    # Test without fidelity scores
    output = modal_geom(x)
    assert output.shape == (batch_size, seq_len, n_embd)
    
    # Test with fidelity scores (OCR compression simulation)
    fidelity_scores = torch.rand(batch_size, seq_len)
    output = modal_geom(x, fidelity_scores=fidelity_scores)
    assert output.shape == (batch_size, seq_len, n_embd)


def test_modal_integration():
    """Integration test: modal reasoning with existing SRGI components."""
    from nanochat.modal_reasoning import ModalCoTReasoning, KripkeFrame
    from nanochat.phase_attention import PhaseAwareAttention
    
    n_embd = 64
    n_head = 4
    batch_size = 2
    seq_len = 10
    
    # Create modules
    modal_cot = ModalCoTReasoning(n_embd=n_embd, n_worlds=4)
    kf = KripkeFrame(n_worlds=4, n_embd=n_embd)
    
    # Phase-aware attention (existing SRGI component)
    cos = torch.randn(1, seq_len, 1, n_embd // n_head // 2)
    sin = torch.randn(1, seq_len, 1, n_embd // n_head // 2)
    cos_sin = (cos, sin)
    
    paa = PhaseAwareAttention(n_embd=n_embd, n_head=n_head)
    
    x = torch.randn(batch_size, seq_len, n_embd)
    
    # Modal reasoning
    x_modal = modal_cot(x)
    
    # Phase-aware attention on modal-reasoned output
    x_phase = paa(x_modal, cos_sin)
    
    assert x_phase.shape == (batch_size, seq_len, n_embd)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

