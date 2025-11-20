#!/usr/bin/env python3
"""
Demo script for Stable Resonant SSM (SRGI Phase-1)

This script demonstrates the usage of the StableResonantSSM layer and provides
a simple training loop to verify it works end-to-end. It includes:

1. Basic instantiation and forward pass
2. Phase information extraction
3. A minimal training loop with synthetic data
4. Integration example with transformer blocks

Usage:
    python scripts/ssm_demo.py

For integration into full model training, see scripts/base_train.py and
add ResonantBlock to the transformer architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from nanochat.ssm import StableResonantSSM, ResonantBlock


def demo_basic_usage():
    """Demonstrate basic SSM instantiation and forward pass."""
    print("=" * 70)
    print("DEMO 1: Basic StableResonantSSM Usage")
    print("=" * 70)

    # Create SSM layer
    state_dim = 64
    input_dim = 768  # Typical transformer embedding dimension
    batch_size = 4
    seq_len = 100

    ssm = StableResonantSSM(state_dim=state_dim, input_dim=input_dim)

    # Generate dummy input
    x = torch.randn(batch_size, seq_len, input_dim)

    print(f"Input shape: {x.shape}")
    print(f"SSM parameters: {sum(p.numel() for p in ssm.parameters()):,}")

    # Forward pass
    y = ssm(x)
    print(f"Output shape: {y.shape}")
    print(f"Output mean: {y.mean().item():.4f}, std: {y.std().item():.4f}")

    # Check that gradients flow
    loss = y.mean()
    loss.backward()
    print(f"Gradients computed successfully ✓")

    print()


def demo_phase_extraction():
    """Demonstrate phase information extraction from SSM."""
    print("=" * 70)
    print("DEMO 2: Phase Information Extraction")
    print("=" * 70)

    # Create SSM layer
    ssm = StableResonantSSM(state_dim=32, input_dim=64)

    # Generate input with some temporal structure (sine wave)
    t = torch.linspace(0, 4 * torch.pi, 100).unsqueeze(0).unsqueeze(-1)
    x = torch.sin(t).repeat(1, 1, 64)  # (1, 100, 64)

    # Extract phase information
    magnitudes, phases = ssm.get_phase_info(x)

    print(f"Magnitude shape: {magnitudes.shape}")
    print(f"Phase shape: {phases.shape}")
    print(f"Phase range: [{phases.min().item():.2f}, {phases.max().item():.2f}]")
    print(f"Mean magnitude: {magnitudes.mean().item():.4f}")

    # Check phase evolution over time
    phase_changes = (phases[:, 1:, :] - phases[:, :-1, :]).abs().mean()
    print(f"Mean phase change per timestep: {phase_changes.item():.4f} radians")

    print()


def demo_training_loop():
    """Demonstrate training the SSM on a simple sequence modeling task."""
    print("=" * 70)
    print("DEMO 3: Training Loop on Synthetic Data")
    print("=" * 70)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    state_dim = 64
    input_dim = 128
    model = StableResonantSSM(state_dim=state_dim, input_dim=input_dim).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Generate synthetic data: predict next step of a noisy sine wave
    def generate_batch(batch_size=8, seq_len=50):
        t = torch.linspace(0, 4 * torch.pi, seq_len).unsqueeze(0).repeat(batch_size, 1)
        # Multiple frequencies
        freqs = torch.rand(batch_size, 1) * 3 + 1  # Random frequencies 1-4
        signal = torch.sin(t * freqs).unsqueeze(-1).repeat(1, 1, input_dim)
        noise = torch.randn_like(signal) * 0.1
        x = signal + noise
        # Target: clean signal
        y = signal
        return x.to(device), y.to(device)

    # Training loop
    print("Training for 100 iterations...")
    losses = []

    for epoch in range(100):
        model.train()
        model._clamp_eig()  # Apply eigenvalue constraints

        x, y_target = generate_batch()

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y_target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"Iteration {epoch+1:3d}: Loss = {loss.item():.6f}")

    # Check convergence
    initial_loss = sum(losses[:10]) / 10
    final_loss = sum(losses[-10:]) / 10
    improvement = (initial_loss - final_loss) / initial_loss * 100

    print(f"\nInitial loss (avg first 10): {initial_loss:.6f}")
    print(f"Final loss (avg last 10):    {final_loss:.6f}")
    print(f"Improvement: {improvement:.1f}%")

    if final_loss < initial_loss:
        print("✓ Loss decreased - training successful!")
    else:
        print("⚠ Warning: Loss did not decrease")

    print()


def demo_transformer_integration():
    """Demonstrate how to integrate SSM into a transformer block."""
    print("=" * 70)
    print("DEMO 4: Transformer Integration with ResonantBlock")
    print("=" * 70)

    # Typical transformer embedding dimension
    n_embd = 768
    batch_size = 2
    seq_len = 50

    # Create a resonant block (can replace or augment standard transformer blocks)
    block = ResonantBlock(n_embd=n_embd, state_dim=n_embd // 2, residual_weight=0.1)

    # Generate dummy transformer hidden states
    x = torch.randn(batch_size, seq_len, n_embd)

    print(f"Input shape: {x.shape}")

    # Forward pass
    y = block(x)

    print(f"Output shape: {y.shape}")
    print(f"Residual preserved: {torch.allclose(y.mean(), x.mean(), atol=0.5)}")

    # The block can be stacked or interleaved with standard transformer blocks
    # Example architecture:
    print("\nExample integration in transformer:")
    print("  1. Standard Attention Block")
    print("  2. ResonantBlock (adds phase dynamics)")
    print("  3. Standard MLP Block")
    print("  4. ResonantBlock (maintains resonance)")
    print("  ... repeat ...")

    print()


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Stable Resonant SSM (SRGI Phase-1) - Demo Script".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Run demos
    demo_basic_usage()
    demo_phase_extraction()
    demo_training_loop()
    demo_transformer_integration()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("All demos completed successfully! ✓")
    print()
    print("Next steps:")
    print("  1. Integrate ResonantBlock into nanochat/gpt.py Block class")
    print("  2. Add SSM layers to model config (e.g., every 2-3 standard blocks)")
    print("  3. Train on long-context tasks to evaluate memory improvements")
    print("  4. Visualize phase dynamics during inference")
    print("  5. Implement Phase-Aware Attention (SRGI Phase-2)")
    print()
    print("For full SRGI architecture, see the paper:")
    print("  Defendre, J. (2025). Spin-Resonant Geometric Intelligence.")
    print()


if __name__ == "__main__":
    main()
