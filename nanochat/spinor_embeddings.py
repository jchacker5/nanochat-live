"""
Spinor Embeddings for SRGI Architecture

This module implements complex-valued embeddings (spinor embeddings) that enable
unitary operations and preserve geometric structure. Spinor embeddings map tokens
to complex vectors, allowing phase rotations and unitary transformations that
preserve norms.

Key features:
- Complex embeddings: real and imaginary parts stored separately
- Unitary operations preserve norm: ||R(θ)e|| = ||e||
- Compatible with standard real-valued layers via interleaving
- Optional normalization for true spinor behavior

Reference:
    Defendre, J. (2025). Spin-Resonant Geometric Intelligence (SRGI):
    Unifying Geometry, Resonance, and Neural Computation for Scalable Intelligence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpinorEmbedding(nn.Module):
    """
    Complex-valued embeddings with unitary operations.
    
    Maps tokens to complex vectors by storing real and imaginary parts separately.
    The embeddings are interleaved [r1, i1, r2, i2, ...] for compatibility with
    standard real-valued layers while preserving complex structure.
    
    Args:
        vocab_size: Size of the vocabulary
        n_embd: Embedding dimension (must be even for complex embeddings)
        normalize: Whether to normalize to unit magnitude (default: False)
    
    Shape:
        - Input: (batch, seq_len) of token indices
        - Output: (batch, seq_len, n_embd)
    
    Examples:
        >>> embed = SpinorEmbedding(vocab_size=50304, n_embd=768)
        >>> idx = torch.randint(0, 50304, (4, 100))
        >>> x = embed(idx)
        >>> x.shape
        torch.Size([4, 100, 768])
    """
    
    def __init__(self, vocab_size, n_embd, normalize=False):
        super().__init__()
        assert n_embd % 2 == 0, "n_embd must be even for complex embeddings"
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.normalize = normalize
        
        # Complex embedding: real and imaginary parts
        self.embed_real = nn.Embedding(vocab_size, n_embd // 2)
        self.embed_imag = nn.Embedding(vocab_size, n_embd // 2)
    
    def forward(self, idx):
        """
        Forward pass: get complex embeddings for token indices.
        
        Args:
            idx: Token indices of shape (B, T)
        
        Returns:
            Complex embeddings of shape (B, T, n_embd) with interleaved real/imaginary parts
        """
        # Get complex embedding components
        real = self.embed_real(idx)
        imag = self.embed_imag(idx)
        
        # Optional: normalize to unit magnitude (for true spinors)
        if self.normalize:
            magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
            real = real / magnitude
            imag = imag / magnitude
        
        # Interleave real and imaginary for compatibility with real-valued layers
        # [r1, i1, r2, i2, ...] allows treating as real while preserving structure
        embed = torch.stack([real, imag], dim=-1).flatten(-2, -1)
        
        return embed  # Shape: (B, T, n_embd)
    
    def rotate(self, x, theta):
        """
        Apply phase rotation: e^{iθ} * z = (cos θ + i sin θ)(a + ib)
        
        This implements a unitary transformation that preserves the norm of the
        complex embeddings.
        
        Args:
            x: Input tensor of shape (..., n_embd) with interleaved real/imaginary parts
            theta: Rotation angle(s) of shape (...,) or broadcastable
        
        Returns:
            Rotated embeddings of same shape as x
        """
        # Split into real and imaginary parts
        real = x[..., 0::2]  # Even indices: real parts
        imag = x[..., 1::2]  # Odd indices: imaginary parts
        
        # Ensure theta has compatible shape
        if theta.dim() < x.dim():
            # Add dimensions to match x's shape (except last dim)
            for _ in range(x.dim() - theta.dim() - 1):
                theta = theta.unsqueeze(-1)
        
        # Rotate: real' = real cos θ - imag sin θ, imag' = real sin θ + imag cos θ
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        real_rot = real * cos_theta - imag * sin_theta
        imag_rot = real * sin_theta + imag * cos_theta
        
        # Interleave back
        result = torch.zeros_like(x)
        result[..., 0::2] = real_rot
        result[..., 1::2] = imag_rot
        
        return result
    
    def get_phase(self, x):
        """
        Extract phase from complex embeddings.
        
        Args:
            x: Input tensor of shape (..., n_embd) with interleaved real/imaginary parts
        
        Returns:
            Phase angles of shape (..., n_embd // 2) in range [-π, π]
        """
        real = x[..., 0::2]
        imag = x[..., 1::2]
        phase = torch.atan2(imag, real)
        return phase
    
    def get_magnitude(self, x):
        """
        Extract magnitude from complex embeddings.
        
        Args:
            x: Input tensor of shape (..., n_embd) with interleaved real/imaginary parts
        
        Returns:
            Magnitudes of shape (..., n_embd // 2)
        """
        real = x[..., 0::2]
        imag = x[..., 1::2]
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
        return magnitude

