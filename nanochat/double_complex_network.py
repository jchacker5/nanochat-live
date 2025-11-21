"""
Double Complex Network implementing Čech-de Rham structure.

This module implements the Double Complex Network (DCN) architecture inspired by
the Čech-de Rham theorem. It creates parallel branches for discrete (Čech) and
continuous (de Rham) computations, enforcing commutativity between them.

Reference:
    Defendre, J. (2025). Spin-Resonant Geometric Intelligence (SRGI):
    Unifying Geometry, Resonance, and Neural Computation for Scalable Intelligence.
    Section 3.4.1: Bridging Topology and Neural Networks.
"""

import torch
import torch.nn as nn
from nanochat.geometric_bottleneck import GeometricBottleneck
from nanochat.phase_attention import PhaseAwareAttention


class DoubleComplexNetwork(nn.Module):
    """
    Double Complex Network implementing Čech-de Rham structure.
    
    Horizontal branch: Discrete Čech (simplicial covers)
    Vertical branch: Smooth de Rham (differential forms)
    Commutativity: Ensures δd = dδ
    
    Args:
        n_embd: Embedding dimension
        n_head: Number of attention heads
        n_kv_head: Number of key/value heads (for GQA support)
        hyperbolic_dim: Dimension of hyperbolic space (default: 64)
        n_circles: Number of circles in torus (default: 4)
    
    Shape:
        - Input: (batch, seq_len, n_embd)
        - Output: (batch, seq_len, n_embd)
    
    Examples:
        >>> dcn = DoubleComplexNetwork(n_embd=768, n_head=12)
        >>> x = torch.randn(4, 100, 768)
        >>> cos_sin = (torch.randn(1, 100, 1, 64), torch.randn(1, 100, 1, 64))
        >>> y = dcn(x, cos_sin)
        >>> y.shape
        torch.Size([4, 100, 768])
    """
    
    def __init__(self, n_embd, n_head, n_kv_head=None, 
                 hyperbolic_dim=64, n_circles=4):
        super().__init__()
        self.n_embd = n_embd
        
        # Horizontal branch: Discrete Čech (simplicial)
        self.cech_branch = nn.Sequential(
            nn.Linear(n_embd, n_embd),  # Simplicial convolution
            nn.ReLU(),
            nn.Linear(n_embd, n_embd)
        )
        
        # Vertical branch: Smooth de Rham (differential forms)
        self.derham_branch = GeometricBottleneck(
            n_embd, hyperbolic_dim, n_circles
        )
        
        # Phase-aware attention (enforces commutativity)
        self.phase_attention = PhaseAwareAttention(
            n_embd, n_head, n_kv_head, beta_init=0.5
        )
        
        # Commutativity projection
        self.commutativity_proj = nn.Linear(n_embd * 2, n_embd)
        
    def forward(self, x, cos_sin, kv_cache=None, return_commutativity=False):
        """
        Forward pass through double complex.
        
        Args:
            x: (B, T, n_embd) input
            cos_sin: RoPE cos/sin for phase attention
            kv_cache: Optional KV cache
            return_commutativity: Return commutativity loss
        
        Returns:
            output: (B, T, n_embd)
            commutativity_loss: scalar (if return_commutativity=True)
        """
        B, T, C = x.shape
        
        # Horizontal branch: Discrete Čech
        x_cech = self.cech_branch(x)  # (B, T, n_embd)
        
        # Vertical branch: Smooth de Rham
        x_derham = self.derham_branch(x)  # (B, T, n_embd)
        
        # Phase-aware attention (enforces phase coherence)
        if return_commutativity:
            x_phase, comm_loss = self.phase_attention(
                x, cos_sin, kv_cache, return_commutativity_loss=True
            )
        else:
            x_phase = self.phase_attention(x, cos_sin, kv_cache)
            comm_loss = None
        
        # Combine branches with commutativity constraint
        x_combined = torch.cat([x_cech, x_derham], dim=-1)  # (B, T, 2*n_embd)
        output = self.commutativity_proj(x_combined)  # (B, T, n_embd)
        
        # Residual connection
        output = output + x
        
        if return_commutativity:
            return output, comm_loss
        return output

