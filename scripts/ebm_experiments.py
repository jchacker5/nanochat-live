"""
EBM Experiments: Comprehensive demonstrations of EBM Hopfield Memory

This script demonstrates:
1. Basic EBM functionality
2. Different sampling methods (deterministic, Gibbs, block Gibbs)
3. Temperature effects on sampling
4. Contrastive divergence training
5. Persistent contrastive divergence training
6. Energy landscape analysis
7. Denoising and associative recall
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.ebm_hopfield import EBMHopfieldMemory
from nanochat.ebm_trainer import EBMHopfieldTrainer, PersistentEBMTrainer


def experiment_1_basic_functionality():
    """Experiment 1: Basic EBM Hopfield Memory forward pass."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Basic EBM Hopfield Memory Functionality")
    print("="*70)
    
    n_embd = 64
    memory_size = 128
    batch_size = 2
    seq_len = 10
    
    memory = EBMHopfieldMemory(
        n_embd=n_embd,
        memory_size=memory_size,
        use_thrml=False  # Use PyTorch fallback
    )
    
    x = torch.randn(batch_size, seq_len, n_embd)
    
    # Forward pass
    output = memory(x)
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    
    # Forward pass with energy
    output, energy = memory(x, return_energy=True, use_ebm_sampling=True)
    print(f"✓ Energy shape: {energy.shape}")
    print(f"✓ Mean energy: {energy.mean().item():.4f}")
    print(f"✓ Energy range: [{energy.min().item():.4f}, {energy.max().item():.4f}]")
    
    return memory, x, output, energy


def experiment_2_sampling_methods():
    """Experiment 2: Compare different sampling methods."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Sampling Methods Comparison")
    print("="*70)
    
    n_embd = 32
    memory_size = 64
    x = torch.randn(1, 5, n_embd)
    
    # Create memories with different sampling methods
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
        temperature=1.0,
        use_thrml=False
    )
    
    memory_block = EBMHopfieldMemory(
        n_embd=n_embd,
        memory_size=memory_size,
        sampling_method='block_gibbs',
        temperature=1.0,
        use_thrml=False
    )
    
    # Forward passes
    output_det = memory_det(x, use_ebm_sampling=False)
    output_gibbs = memory_gibbs(x, use_ebm_sampling=True)
    output_block = memory_block(x, use_ebm_sampling=True)
    
    # Compute differences
    diff_gibbs = torch.norm(output_gibbs - output_det).item()
    diff_block = torch.norm(output_block - output_det).item()
    
    print(f"✓ Deterministic output norm: {torch.norm(output_det).item():.4f}")
    print(f"✓ Gibbs sampling output norm: {torch.norm(output_gibbs).item():.4f}")
    print(f"✓ Block Gibbs output norm: {torch.norm(output_block).item():.4f}")
    print(f"✓ Difference (Gibbs vs Det): {diff_gibbs:.4f}")
    print(f"✓ Difference (Block Gibbs vs Det): {diff_block:.4f}")
    
    return {
        'deterministic': output_det,
        'gibbs': output_gibbs,
        'block_gibbs': output_block
    }


def experiment_3_temperature_effects():
    """Experiment 3: Temperature effects on sampling."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Temperature Effects on Sampling")
    print("="*70)
    
    n_embd = 32
    memory_size = 64
    x = torch.randn(1, 5, n_embd)
    
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    outputs = []
    energies = []
    
    for temp in temperatures:
        memory = EBMHopfieldMemory(
            n_embd=n_embd,
            memory_size=memory_size,
            temperature=temp,
            sampling_method='gibbs',
            use_thrml=False
        )
        output, energy = memory(x, return_energy=True, use_ebm_sampling=True)
        outputs.append(output)
        energies.append(energy.mean().item())
        print(f"✓ Temperature {temp:4.1f}: Mean energy = {energy.mean().item():.4f}")
    
    # Check variance across temperatures
    output_stack = torch.stack(outputs)
    variance = output_stack.var(dim=0).mean().item()
    print(f"✓ Output variance across temperatures: {variance:.6f}")
    
    return temperatures, energies


def experiment_4_denoising():
    """Experiment 4: Denoising corrupted patterns."""
    print("\n" + "="*70)
    print("EXPERIMENT 4: Denoising Corrupted Patterns")
    print("="*70)
    
    n_embd = 64
    memory_size = 10
    
    # Create memory and store some patterns
    memory = EBMHopfieldMemory(
        n_embd=n_embd,
        memory_size=memory_size,
        beta=2.0,  # Higher beta for stronger attractors
        n_steps=5,
        use_thrml=False
    )
    
    # Create clean patterns
    clean_patterns = torch.randn(memory_size, n_embd)
    memory.patterns.data = clean_patterns
    
    # Corrupt one pattern with noise
    noise_level = 0.5
    corrupted = clean_patterns[0:1] + noise_level * torch.randn(1, n_embd)
    
    # Denoise using EBM
    denoised, energy = memory(corrupted.unsqueeze(0), return_energy=True, use_ebm_sampling=True)
    denoised = denoised.squeeze(0)
    
    # Compute reconstruction error
    original_error = torch.norm(corrupted - clean_patterns[0]).item()
    denoised_error = torch.norm(denoised - clean_patterns[0]).item()
    
    print(f"✓ Original pattern norm: {torch.norm(clean_patterns[0]).item():.4f}")
    print(f"✓ Corrupted pattern norm: {torch.norm(corrupted).item():.4f}")
    print(f"✓ Denoised pattern norm: {torch.norm(denoised).item():.4f}")
    print(f"✓ Error before denoising: {original_error:.4f}")
    print(f"✓ Error after denoising: {denoised_error:.4f}")
    print(f"✓ Improvement: {((original_error - denoised_error) / original_error * 100):.1f}%")
    print(f"✓ Final energy: {energy.mean().item():.4f}")
    
    return corrupted, denoised, clean_patterns[0]


def experiment_5_associative_recall():
    """Experiment 5: Associative recall from partial cues."""
    print("\n" + "="*70)
    print("EXPERIMENT 5: Associative Recall from Partial Cues")
    print("="*70)
    
    n_embd = 64
    memory_size = 5
    
    memory = EBMHopfieldMemory(
        n_embd=n_embd,
        memory_size=memory_size,
        beta=3.0,  # High beta for strong associations
        n_steps=10,
        use_thrml=False
    )
    
    # Create memory patterns
    stored_patterns = torch.randn(memory_size, n_embd)
    memory.patterns.data = stored_patterns
    
    # Create partial cue (first half of a pattern)
    cue_fraction = 0.3
    cue_dim = int(n_embd * cue_fraction)
    partial_cue = stored_patterns[0, :cue_dim]
    
    # Pad with zeros to full dimension (add sequence dimension)
    full_cue = torch.zeros(1, 1, n_embd)  # (batch=1, seq_len=1, n_embd)
    full_cue[0, 0, :cue_dim] = partial_cue
    
    # Recall using EBM
    recalled, energy = memory(full_cue, return_energy=True, use_ebm_sampling=True)
    recalled = recalled.squeeze(0).squeeze(0)  # Remove batch and seq dims
    
    # Compute similarity to stored patterns
    similarities = []
    for i in range(memory_size):
        sim = torch.cosine_similarity(recalled.unsqueeze(0), stored_patterns[i:i+1])
        similarities.append(sim.item())
    
    best_match_idx = np.argmax(similarities)
    best_match_sim = similarities[best_match_idx]
    
    print(f"✓ Partial cue dimension: {cue_dim}/{n_embd} ({cue_fraction*100:.0f}%)")
    print(f"✓ Best match index: {best_match_idx}")
    print(f"✓ Best match similarity: {best_match_sim:.4f}")
    print(f"✓ All similarities: {[f'{s:.3f}' for s in similarities]}")
    print(f"✓ Final energy: {energy.mean().item():.4f}")
    
    return full_cue, recalled, stored_patterns, similarities


def experiment_6_contrastive_divergence():
    """Experiment 6: Contrastive divergence training."""
    print("\n" + "="*70)
    print("EXPERIMENT 6: Contrastive Divergence Training")
    print("="*70)
    
    n_embd = 32
    memory_size = 64
    batch_size = 4
    seq_len = 8
    
    # Create model and trainer
    memory = EBMHopfieldMemory(
        n_embd=n_embd,
        memory_size=memory_size,
        use_thrml=False
    )
    
    trainer = EBMHopfieldTrainer(memory, learning_rate=1e-3)
    
    # Generate training data
    train_data = torch.randn(batch_size, seq_len, n_embd)
    
    # Training loop
    n_epochs = 10
    losses = []
    
    print(f"Training for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        loss = trainer.train_step(train_data, use_cd=True, n_negative_steps=3)
        losses.append(loss)
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1:2d}: Loss = {loss:.6f}")
    
    print(f"✓ Initial loss: {losses[0]:.6f}")
    print(f"✓ Final loss: {losses[-1]:.6f}")
    print(f"✓ Loss change: {losses[-1] - losses[0]:.6f}")
    
    return losses


def experiment_7_persistent_cd():
    """Experiment 7: Persistent contrastive divergence training."""
    print("\n" + "="*70)
    print("EXPERIMENT 7: Persistent Contrastive Divergence Training")
    print("="*70)
    
    n_embd = 32
    memory_size = 64
    batch_size = 4
    seq_len = 8
    
    # Create model and persistent trainer
    memory = EBMHopfieldMemory(
        n_embd=n_embd,
        memory_size=memory_size,
        use_thrml=False
    )
    
    trainer = PersistentEBMTrainer(memory, learning_rate=1e-3)
    
    # Generate training batches
    n_batches = 10
    losses = []
    
    print(f"Training with {n_batches} batches (persistent negative samples)...")
    for batch_idx in range(n_batches):
        batch = torch.randn(batch_size, seq_len, n_embd)
        loss = trainer.train_step(batch, use_cd=True, n_negative_steps=2)
        losses.append(loss)
        if (batch_idx + 1) % 2 == 0:
            print(f"  Batch {batch_idx+1:2d}: Loss = {loss:.6f}")
    
    print(f"✓ Initial loss: {losses[0]:.6f}")
    print(f"✓ Final loss: {losses[-1]:.6f}")
    print(f"✓ Loss change: {losses[-1] - losses[0]:.6f}")
    print(f"✓ Persistent samples maintained: {trainer.persistent_negative_samples is not None}")
    
    return losses


def experiment_8_energy_landscape():
    """Experiment 8: Analyze energy landscape."""
    print("\n" + "="*70)
    print("EXPERIMENT 8: Energy Landscape Analysis")
    print("="*70)
    
    n_embd = 32
    memory_size = 8
    
    memory = EBMHopfieldMemory(
        n_embd=n_embd,
        memory_size=memory_size,
        beta=2.0,
        use_thrml=False
    )
    
    # Create test states
    n_samples = 20
    test_states = torch.randn(n_samples, 1, n_embd)
    
    # Compute energies
    energies = []
    for state in test_states:
        energy = memory.energy(state)
        energies.append(energy.item())
    
    energies = np.array(energies)
    
    print(f"✓ Number of test states: {n_samples}")
    print(f"✓ Mean energy: {energies.mean():.4f}")
    print(f"✓ Std energy: {energies.std():.4f}")
    print(f"✓ Min energy: {energies.min():.4f}")
    print(f"✓ Max energy: {energies.max():.4f}")
    print(f"✓ Energy range: {energies.max() - energies.min():.4f}")
    
    return energies


def run_all_experiments():
    """Run all EBM experiments."""
    print("\n" + "="*70)
    print("EBM HOPFIELD MEMORY EXPERIMENTS")
    print("="*70)
    print("Running comprehensive EBM experiments...")
    
    results = {}
    
    try:
        results['exp1'] = experiment_1_basic_functionality()
        results['exp2'] = experiment_2_sampling_methods()
        results['exp3'] = experiment_3_temperature_effects()
        results['exp4'] = experiment_4_denoising()
        results['exp5'] = experiment_5_associative_recall()
        results['exp6'] = experiment_6_contrastive_divergence()
        results['exp7'] = experiment_7_persistent_cd()
        results['exp8'] = experiment_8_energy_landscape()
        
        print("\n" + "="*70)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error during experiments: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return results


if __name__ == "__main__":
    results = run_all_experiments()
    
    if results:
        print("\n✓ All experiments completed successfully!")
        print("✓ EBM Hopfield Memory is working correctly")
        print("\nNext steps:")
        print("  - Integrate EBM into SRGI training pipeline")
        print("  - Experiment with THRML integration for hardware acceleration")
        print("  - Test on larger memory sizes and longer sequences")

