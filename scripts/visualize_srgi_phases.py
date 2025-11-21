#!/usr/bin/env python3
"""
SRGI Phase Visualizations

Generate comprehensive visualizations for each phase of the SRGI architecture:
- Phase-1: Resonant SSM (eigenvalues, phase diagrams, state evolution)
- Phase-2: Phase-Aware Attention, Spinor Embeddings, Geometric Bottlenecks
- Phase-3: Hopfield Attractors (conceptual)

Run with: python scripts/visualize_srgi_phases.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

# Try to import SRGI components (optional - visualizations work without them)
try:
    import torch
    try:
        from nanochat.ssm import StableResonantSSM, ResonantBlock
        HAS_SSM = True
    except ImportError:
        HAS_SSM = False
        
    try:
        from nanochat.phase_attention import PhaseAwareAttention
        HAS_PAA = True
    except ImportError:
        HAS_PAA = False
        
    try:
        from nanochat.spinor_embeddings import SpinorEmbedding
        HAS_SPINOR = True
    except ImportError:
        HAS_SPINOR = False
        
    try:
        from nanochat.geometric_bottleneck import GeometricBottleneck
        HAS_GEOM = True
    except ImportError:
        HAS_GEOM = False
except ImportError:
    # torch not available, use simulated data
    HAS_SSM = False
    HAS_PAA = False
    HAS_SPINOR = False
    HAS_GEOM = False


def plot_phase1_eigenvalues():
    """Phase-1: Plot eigenvalue distribution in complex plane."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Simulate eigenvalues for resonant SSM
    # They should be near the imaginary axis with small negative real parts
    n_eigenvalues = 64
    np.random.seed(42)
    
    # Generate eigenvalues: Re(λ) ∈ [-0.1, -0.01], Im(λ) ∈ [-2π, 2π]
    real_parts = -np.random.uniform(0.01, 0.1, n_eigenvalues)
    imag_parts = np.random.uniform(-2*np.pi, 2*np.pi, n_eigenvalues)
    
    # Plot eigenvalues
    scatter = ax.scatter(real_parts, imag_parts, c=np.abs(imag_parts), 
                        cmap='hsv', s=100, alpha=0.7, edgecolors='black', linewidths=1)
    
    # Draw unit circle
    circle = Circle((0, 0), 1, fill=False, linestyle='--', color='gray', linewidth=2, label='Unit Circle')
    ax.add_patch(circle)
    
    # Draw imaginary axis
    ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Imaginary Axis')
    
    # Draw damping region
    ax.axvspan(-0.1, 0, alpha=0.1, color='green', label='Damping Region')
    
    ax.set_xlabel('Real Part (Re(λ))', fontsize=12)
    ax.set_ylabel('Imaginary Part (Im(λ))', fontsize=12)
    ax.set_title('Phase-1: Resonant SSM Eigenvalue Distribution\n(PV Interneurons → 40 Hz Gamma Stability)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.set_xlim(-0.15, 0.05)
    ax.set_ylim(-7, 7)
    
    plt.colorbar(scatter, ax=ax, label='|Im(λ)|')
    plt.tight_layout()
    return fig


def plot_phase1_state_evolution():
    """Phase-1: Plot state evolution over time showing resonance."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Simulate state evolution
    t = np.linspace(0, 10, 1000)
    
    # Multiple oscillators with different frequencies
    frequencies = [0.5, 1.0, 2.0, 4.0]  # Hz
    damping = 0.05
    
    for i, freq in enumerate(frequencies):
        # Damped oscillator: e^(-damping*t) * sin(2π*freq*t)
        state = np.exp(-damping * t) * np.sin(2 * np.pi * freq * t)
        axes[0].plot(t, state, label=f'{freq} Hz', linewidth=2, alpha=0.8)
    
    axes[0].set_xlabel('Time (s)', fontsize=11)
    axes[0].set_ylabel('State Amplitude', fontsize=11)
    axes[0].set_title('Phase-1: Resonant State Evolution\n(Lightly Damped Oscillators)', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Phase portrait (state vs derivative)
    t_short = np.linspace(0, 5, 500)
    freq = 1.0
    state = np.exp(-damping * t_short) * np.sin(2 * np.pi * freq * t_short)
    state_deriv = np.exp(-damping * t_short) * (2 * np.pi * freq * np.cos(2 * np.pi * freq * t_short) - 
                                                 damping * np.sin(2 * np.pi * freq * t_short))
    
    axes[1].plot(state, state_deriv, linewidth=2, alpha=0.7)
    axes[1].scatter([state[0]], [state_deriv[0]], color='green', s=100, zorder=5, label='Start')
    axes[1].scatter([state[-1]], [state_deriv[-1]], color='red', s=100, zorder=5, label='End')
    axes[1].set_xlabel('State', fontsize=11)
    axes[1].set_ylabel('State Derivative', fontsize=11)
    axes[1].set_title('Phase Portrait (Spiral to Origin)', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_phase2_attention_patterns():
    """Phase-2: Visualize phase-aware attention patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    seq_len = 50
    n_head = 4
    
    # Simulate attention scores with and without phase coherence
    position_diff = np.arange(seq_len)[:, None] - np.arange(seq_len)[None, :]
    
    # Standard attention (no phase)
    attn_standard = np.exp(-0.1 * np.abs(position_diff))
    attn_standard = attn_standard / attn_standard.sum(axis=1, keepdims=True)
    
    # Phase-aware attention (with coherence modulation)
    beta = 0.5
    theta_base = 10000.0
    head_dim = 64
    theta = theta_base ** (-np.arange(0, head_dim, 2) / head_dim)
    phase_diff = position_diff[:, :, None] * theta[None, None, :]
    coherence = 1 + beta * np.cos(phase_diff).mean(axis=-1)
    attn_phase = attn_standard * coherence
    attn_phase = attn_phase / attn_phase.sum(axis=1, keepdims=True)
    
    # Plot standard attention
    im1 = axes[0, 0].imshow(attn_standard, cmap='viridis', aspect='auto', origin='lower')
    axes[0, 0].set_title('Standard Attention\n(No Phase Coherence)', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Key Position', fontsize=10)
    axes[0, 0].set_ylabel('Query Position', fontsize=10)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot phase-aware attention
    im2 = axes[0, 1].imshow(attn_phase, cmap='viridis', aspect='auto', origin='lower')
    axes[0, 1].set_title('Phase-Aware Attention\n(β=0.5, Coherence Gating)', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Key Position', fontsize=10)
    axes[0, 1].set_ylabel('Query Position', fontsize=10)
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot difference
    attn_diff = attn_phase - attn_standard
    im3 = axes[1, 0].imshow(attn_diff, cmap='RdBu_r', aspect='auto', origin='lower', 
                           vmin=-attn_diff.max(), vmax=attn_diff.max())
    axes[1, 0].set_title('Difference (Phase-Aware - Standard)', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Key Position', fontsize=10)
    axes[1, 0].set_ylabel('Query Position', fontsize=10)
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Plot coherence modulation factor
    im4 = axes[1, 1].imshow(coherence, cmap='plasma', aspect='auto', origin='lower')
    axes[1, 1].set_title('Phase Coherence Modulation\n(1 + β cos(Δφ))', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Key Position', fontsize=10)
    axes[1, 1].set_ylabel('Query Position', fontsize=10)
    plt.colorbar(im4, ax=axes[1, 1])
    
    fig.suptitle('Phase-2: Phase-Aware Attention Patterns\n(Gamma Phase-Locking for Coherence)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig


def plot_phase2_spinor_embeddings():
    """Phase-2: Visualize spinor embeddings in complex plane."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Simulate spinor embeddings
    vocab_size = 100
    n_embd = 64
    n_complex = n_embd // 2
    
    np.random.seed(42)
    # Generate random complex embeddings
    real_parts = np.random.randn(vocab_size, n_complex) * 0.5
    imag_parts = np.random.randn(vocab_size, n_complex) * 0.5
    
    # Plot 1: Complex plane for first dimension
    for i in range(min(20, vocab_size)):
        axes[0, 0].scatter(real_parts[i, 0], imag_parts[i, 0], alpha=0.6, s=50)
    axes[0, 0].set_xlabel('Real Part', fontsize=11)
    axes[0, 0].set_ylabel('Imaginary Part', fontsize=11)
    axes[0, 0].set_title('Spinor Embeddings in Complex Plane\n(First Dimension)', fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_aspect('equal')
    
    # Plot 2: Magnitude distribution
    magnitudes = np.sqrt(real_parts**2 + imag_parts**2)
    axes[0, 1].hist(magnitudes.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_xlabel('Magnitude |z|', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Magnitude Distribution', fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Phase distribution
    phases = np.arctan2(imag_parts, real_parts)
    axes[1, 0].hist(phases.flatten(), bins=50, alpha=0.7, color='coral', edgecolor='black')
    axes[1, 0].set_xlabel('Phase (radians)', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Phase Distribution', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Rotation demonstration
    theta = np.pi / 4  # 45 degrees
    real_rot = real_parts[:10, 0] * np.cos(theta) - imag_parts[:10, 0] * np.sin(theta)
    imag_rot = real_parts[:10, 0] * np.sin(theta) + imag_parts[:10, 0] * np.cos(theta)
    
    axes[1, 1].scatter(real_parts[:10, 0], imag_parts[:10, 0], alpha=0.6, s=100, 
                      label='Original', color='blue')
    axes[1, 1].scatter(real_rot, imag_rot, alpha=0.6, s=100, 
                      label=f'Rotated (θ=π/4)', color='red', marker='x')
    axes[1, 1].set_xlabel('Real Part', fontsize=11)
    axes[1, 1].set_ylabel('Imaginary Part', fontsize=11)
    axes[1, 1].set_title('Unitary Rotation (Preserves Norm)', fontsize=11, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect('equal')
    
    fig.suptitle('Phase-2: Spinor Embeddings\n(Complex-Valued with Unitary Operations)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig


def plot_phase2_geometric_manifolds():
    """Phase-2: Visualize hyperbolic and toroidal manifolds."""
    fig = plt.figure(figsize=(16, 6))
    
    # Hyperbolic space (Poincaré disk)
    ax1 = fig.add_subplot(131)
    
    # Draw Poincaré disk (unit circle)
    circle = Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax1.add_patch(circle)
    
    # Draw geodesics (hyperbolic arcs)
    n_geodesics = 8
    for i in range(n_geodesics):
        angle = 2 * np.pi * i / n_geodesics
        # Geodesic through origin
        t = np.linspace(-0.9, 0.9, 100)
        x = t * np.cos(angle)
        y = t * np.sin(angle)
        ax1.plot(x, y, 'b-', alpha=0.5, linewidth=1.5)
    
    # Draw some points
    points = np.array([[0.3, 0.2], [0.5, -0.3], [-0.4, 0.4]])
    ax1.scatter(points[:, 0], points[:, 1], c='red', s=100, zorder=5)
    
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_aspect('equal')
    ax1.set_title('Hyperbolic Space (Poincaré Disk)\n(Hierarchical Structures)', 
                  fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Toroidal space (2D torus)
    ax2 = fig.add_subplot(132, projection='3d')
    
    # Parametric equations for torus
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, 2 * np.pi, 50)
    u, v = np.meshgrid(u, v)
    
    R = 1.0  # Major radius
    r = 0.5  # Minor radius
    x_torus = (R + r * np.cos(v)) * np.cos(u)
    y_torus = (R + r * np.cos(v)) * np.sin(u)
    z_torus = r * np.sin(v)
    
    ax2.plot_surface(x_torus, y_torus, z_torus, alpha=0.7, color='coral', edgecolor='none')
    ax2.set_title('Toroidal Space (2D Torus)\n(Periodic/Cyclic Patterns)', 
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Combined visualization: projection
    ax3 = fig.add_subplot(133)
    
    # Show how embeddings are projected
    n_points = 100
    euclidean_points = np.random.randn(n_points, 2) * 0.5
    
    # Project to hyperbolic (simplified)
    hyp_norms = np.linalg.norm(euclidean_points, axis=1)
    hyp_points = euclidean_points / (1 + hyp_norms[:, None])
    
    # Project to toroidal (angles)
    angles = np.arctan2(euclidean_points[:, 1], euclidean_points[:, 0])
    torus_points = np.column_stack([np.cos(angles), np.sin(angles)]) * 0.3
    
    ax3.scatter(euclidean_points[:, 0], euclidean_points[:, 1], 
               alpha=0.5, s=30, label='Euclidean', color='blue')
    ax3.scatter(hyp_points[:, 0], hyp_points[:, 1], 
               alpha=0.5, s=30, label='Hyperbolic', color='green')
    ax3.scatter(torus_points[:, 0], torus_points[:, 1], 
               alpha=0.5, s=30, label='Toroidal', color='red')
    ax3.set_title('Geometric Bottleneck Projections', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    fig.suptitle('Phase-2: Geometric Bottlenecks\n(Rotating Cortical Waves on Curved Manifolds)', 
                 fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    return fig


def plot_phase3_hopfield_attractors():
    """Phase-3: Visualize Hopfield attractor basins (conceptual)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Simulate energy landscape
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Multiple attractors (energy minima)
    n_attractors = 3
    attractors = np.array([[1.5, 1.0], [-1.0, 1.5], [0.0, -1.5]])
    
    # Energy function: sum of Gaussian wells at attractor positions
    energy = np.zeros_like(X)
    for att in attractors:
        energy += -np.exp(-((X - att[0])**2 + (Y - att[1])**2) / 0.5)
    
    # Plot energy landscape
    im1 = axes[0].contourf(X, Y, energy, levels=20, cmap='viridis')
    axes[0].contour(X, Y, energy, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    axes[0].scatter(attractors[:, 0], attractors[:, 1], c='red', s=200, 
                   marker='*', edgecolors='white', linewidths=2, zorder=5, label='Attractors')
    
    # Draw trajectories converging to attractors
    np.random.seed(42)
    for _ in range(5):
        start = np.random.uniform(-2, 2, 2)
        trajectory = [start]
        current = start.copy()
        for _ in range(20):
            # Gradient descent toward nearest attractor
            dists = np.linalg.norm(attractors - current, axis=1)
            nearest_idx = np.argmin(dists)
            direction = attractors[nearest_idx] - current
            current += 0.1 * direction / (np.linalg.norm(direction) + 1e-8)
            trajectory.append(current.copy())
        trajectory = np.array(trajectory)
        axes[0].plot(trajectory[:, 0], trajectory[:, 1], 'w-', alpha=0.6, linewidth=2)
        axes[0].scatter([start[0]], [start[1]], c='yellow', s=50, zorder=5)
    
    axes[0].set_xlabel('State Dimension 1', fontsize=11)
    axes[0].set_ylabel('State Dimension 2', fontsize=11)
    axes[0].set_title('Hopfield Energy Landscape\n(Attractor Basins)', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0], label='Energy')
    
    # Plot convergence dynamics
    t = np.linspace(0, 10, 100)
    axes[1].plot(t, 1 - np.exp(-t), 'b-', linewidth=2, label='Convergence Rate')
    axes[1].axhline(1, color='r', linestyle='--', alpha=0.5, label='Attractor')
    axes[1].set_xlabel('Iteration', fontsize=11)
    axes[1].set_ylabel('State Similarity to Attractor', fontsize=11)
    axes[1].set_title('Attractor Convergence Dynamics', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle('Phase-3: Modern Hopfield Attractor Memory\n(Wave Attractors = Perceptual Clarity)', 
                 fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    return fig


def main():
    """Generate all visualizations."""
    print("Generating SRGI Phase Visualizations...")
    print("=" * 60)
    
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Phase-1 visualizations
    print("\n[Phase-1] Generating Resonant SSM visualizations...")
    fig1 = plot_phase1_eigenvalues()
    fig1.savefig(f"{output_dir}/phase1_eigenvalues.png", dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/phase1_eigenvalues.png")
    
    fig2 = plot_phase1_state_evolution()
    fig2.savefig(f"{output_dir}/phase1_state_evolution.png", dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/phase1_state_evolution.png")
    
    # Phase-2 visualizations
    print("\n[Phase-2] Generating Phase-Aware Dynamics visualizations...")
    fig3 = plot_phase2_attention_patterns()
    fig3.savefig(f"{output_dir}/phase2_attention_patterns.png", dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/phase2_attention_patterns.png")
    
    fig4 = plot_phase2_spinor_embeddings()
    fig4.savefig(f"{output_dir}/phase2_spinor_embeddings.png", dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/phase2_spinor_embeddings.png")
    
    fig5 = plot_phase2_geometric_manifolds()
    fig5.savefig(f"{output_dir}/phase2_geometric_manifolds.png", dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/phase2_geometric_manifolds.png")
    
    # Phase-3 visualizations
    print("\n[Phase-3] Generating Hopfield Attractor visualizations...")
    fig6 = plot_phase3_hopfield_attractors()
    fig6.savefig(f"{output_dir}/phase3_hopfield_attractors.png", dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/phase3_hopfield_attractors.png")
    
    print("\n" + "=" * 60)
    print("✅ All visualizations generated successfully!")
    print(f"📁 Output directory: {output_dir}/")
    print("\nGenerated files:")
    print("  - phase1_eigenvalues.png")
    print("  - phase1_state_evolution.png")
    print("  - phase2_attention_patterns.png")
    print("  - phase2_spinor_embeddings.png")
    print("  - phase2_geometric_manifolds.png")
    print("  - phase3_hopfield_attractors.png")


if __name__ == "__main__":
    main()

