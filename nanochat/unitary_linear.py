"""
Unitary Linear Layer for SRGI Architecture

This module implements unitary linear transformations using Givens rotation
parametrization. Unitary layers preserve norms, enabling resonant propagation
without information loss.

Key features:
- Deterministic pair generation for reproducibility
- Givens rotation parametrization for unitary constraints
- Efficient computation with torch.no_grad() for rotation matrices
- Complex input support (real and imaginary parts)

Reference:
    Defendre, J. (2025). Spin-Resonant Geometric Intelligence (SRGI):
    Unifying Geometry, Resonance, and Neural Computation for Scalable Intelligence.
"""

import torch
import torch.nn as nn


class UnitaryLinear(nn.Module):
    """
    Unitary linear layer using Givens rotation parametrization.
    
    Maintains ||Uz||_2 = ||z||_2 for resonant propagation. Uses deterministic
    pair generation for reproducibility and efficient computation with torch.no_grad()
    for rotation matrices.
    
    Args:
        n_embd: Embedding dimension
        n_rotations: Number of Givens rotations (default: n_embd // 2)
    
    Shape:
        - Input: (..., n_embd, 2) complex tensor [real, imag]
        - Output: (..., n_embd, 2) unitarily transformed tensor
    
    Examples:
        >>> layer = UnitaryLinear(n_embd=768)
        >>> x = torch.randn(2, 10, 768, 2)  # batch=2, seq=10, dim=768, [real, imag]
        >>> y = layer(x)
        >>> y.shape
        torch.Size([2, 10, 768, 2])
    """
    
    def __init__(self, n_embd: int, n_rotations: int = None):
        super().__init__()
        self.n_embd = n_embd
        # Use n_embd//2 Givens rotations by default
        self.n_rotations = n_rotations or (n_embd // 2)
        
        # Parametrize as angles θ for Givens rotations
        self.angles = nn.Parameter(torch.randn(self.n_rotations) * 0.1)
        # Pairs of indices to rotate (deterministic for reproducibility)
        self.register_buffer('pairs', self._generate_pairs())
    
    def _generate_pairs(self):
        """
        Generate deterministic pairs of dimensions to rotate for reproducibility.
        
        Uses sequential pairs: (0,1), (2,3), ... for deterministic behavior
        instead of random pairs. This ensures reproducible results across runs.
        """
        # Use sequential pairs: (0,1), (2,3), ... for deterministic behavior
        pairs = list(zip(range(0, self.n_embd, 2), range(1, self.n_embd, 2)))
        # Truncate to n_rotations if needed
        pairs = pairs[:self.n_rotations]
        return torch.tensor(pairs)
    
    def forward(self, x):
        """
        Apply composition of Givens rotations (unitary transformation).
        
        Args:
            x: (..., n_embd, 2) complex tensor [real, imag]
        
        Returns:
            Unitarily transformed x with same shape
        """
        # Process real and imaginary separately
        x_real, x_imag = x[..., 0], x[..., 1]
        
        # Use torch.no_grad() for efficiency when computing rotations
        # Rotation matrices don't need gradients during forward pass
        with torch.no_grad():
            cos_sin = [(torch.cos(angle), torch.sin(angle)) for angle in self.angles]
        
        for (cos_a, sin_a), (i, j) in zip(cos_sin, self.pairs):
            # Givens rotation on real part
            x_real_i = x_real[..., i] * cos_a - x_real[..., j] * sin_a
            x_real_j = x_real[..., i] * sin_a + x_real[..., j] * cos_a
            x_real = x_real.clone()
            x_real[..., i] = x_real_i
            x_real[..., j] = x_real_j
            
            # Givens rotation on imaginary part
            x_imag_i = x_imag[..., i] * cos_a - x_imag[..., j] * sin_a
            x_imag_j = x_imag[..., i] * sin_a + x_imag[..., j] * cos_a
            x_imag = x_imag.clone()
            x_imag[..., i] = x_imag_i
            x_imag[..., j] = x_imag_j
        
        return torch.stack([x_real, x_imag], dim=-1)

