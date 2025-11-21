#!/usr/bin/env python3
"""
Integration test for SRGI architecture with entanglement.
Tests that the full SRGI model can be instantiated and run forward passes.
"""

import torch
import sys
import os
sys.path.append('/Users/jchacker5/Documents/nanochat-live')

from nanochat.gpt import GPT, GPTConfig

def test_srgi_model_instantiation():
    """Test that SRGI model can be created with entanglement enabled."""
    print("Testing SRGI model instantiation...")

    # Create SRGI config
    config = GPTConfig(
        vocab_size=1000,
        n_embd=128,
        n_layer=4,
        n_head=4,
        n_kv_head=4,  # Must be <= n_head
        sequence_len=512,
        use_srgi=True,  # Enable SRGI
        use_entangle=True,  # Enable entanglement
        entangle_bond_dim=8,
        lambda_entangle=0.1
    )

    # Create model
    model = GPT(config)
    model.eval()

    print("✓ SRGI model instantiated successfully")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    return model

def test_srgi_forward_pass():
    """Test forward pass through SRGI model."""
    print("Testing SRGI forward pass...")

    model = test_srgi_model_instantiation()

    # Create dummy input
    B, T = 2, 16
    idx = torch.randint(0, 1000, (B, T))

    # Forward pass
    with torch.no_grad():
        loss = model(idx, targets=idx)

    print(f"✓ Forward pass successful, loss: {loss:.4f}")

    return model

def test_srgi_generation():
    """Test token generation with SRGI model."""
    print("Testing SRGI token generation...")

    model = test_srgi_model_instantiation()

    # Create dummy input
    B, T = 1, 8
    idx = torch.randint(0, 1000, (B, T)).tolist()[0]  # Convert to list

    # Generate
    with torch.no_grad():
        generated = list(model.generate(idx, max_tokens=10, temperature=0.8))

    print(f"✓ Generation successful, generated {len(generated)} tokens")

def test_srgi_with_different_configs():
    """Test SRGI with different configurations."""
    print("Testing SRGI with different configurations...")

    configs = [
        {"use_srgi": True, "use_entangle": False, "name": "SRGI without entanglement"},
        {"use_srgi": True, "use_entangle": True, "entangle_bond_dim": 4, "name": "SRGI with small entanglement"},
        {"use_srgi": True, "use_entangle": True, "entangle_bond_dim": 16, "name": "SRGI with large entanglement"},
    ]

    for cfg in configs:
        print(f"  Testing {cfg['name']}...")

        config = GPTConfig(
            vocab_size=1000,
            n_embd=64,  # Smaller for faster testing
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            sequence_len=256,
            **{k: v for k, v in cfg.items() if k != 'name'}
        )

        model = GPT(config)

        # Quick forward test
        B, T = 1, 8
        idx = torch.randint(0, 1000, (B, T))
        with torch.no_grad():
            loss = model(idx, targets=idx)

        print(".4f")

if __name__ == "__main__":
    print("Running SRGI Integration Tests...\n")

    try:
        test_srgi_forward_pass()
        test_srgi_generation()
        test_srgi_with_different_configs()

        print("\n🎉 All SRGI integration tests passed!")
        print("\nSRGI with quantum-inspired entanglement is ready for production! 🚀")

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
