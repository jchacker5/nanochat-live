"""
Geometric Bottlenecks for SRGI Architecture

This module implements geometric bottlenecks using hyperbolic and toroidal spaces
to encode hierarchical and periodic structures. These manifolds provide built-in
geometric structure that helps the model learn tree-like hierarchies and cyclic
patterns without explicit supervision.

Key features:
- Hyperbolic space (Poincaré ball) for hierarchical structures
- Toroidal space (S¹ × S¹ × ...) for periodic/cyclic patterns
- Combined bottleneck with learned mixing weight
- Optional geoopt dependency for advanced Riemannian operations

Reference:
    Defendre, J. (2025). Spin-Resonant Geometric Intelligence (SRGI):
    Unifying Geometry, Resonance, and Neural Computation for Scalable Intelligence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import geoopt, but make it optional
try:
    import geoopt
    HAS_GEOOPT = True
except ImportError:
    HAS_GEOOPT = False


class HyperbolicBottleneck(nn.Module):
    """
    Poincaré ball bottleneck for hierarchical structure.
    
    Projects embeddings to hyperbolic space (Poincaré ball), applies transformations
    in that space, then projects back to Euclidean. This naturally encodes tree-like
    hierarchical structures where parent-child distances follow hyperbolic geometry.
    
    Args:
        n_embd: Embedding dimension
        hyperbolic_dim: Dimension of hyperbolic space (default: 64)
        curvature: Curvature of hyperbolic space (default: -1.0)
        use_geoopt: Whether to use geoopt library if available (default: True)
    
    Shape:
        - Input: (batch, seq_len, n_embd)
        - Output: (batch, seq_len, n_embd)
    
    Examples:
        >>> hyp = HyperbolicBottleneck(n_embd=768, hyperbolic_dim=64)
        >>> x = torch.randn(4, 100, 768)
        >>> y = hyp(x)
        >>> y.shape
        torch.Size([4, 100, 768])
    """
    
    def __init__(self, n_embd, hyperbolic_dim=64, curvature=-1.0, use_geoopt=True):
        super().__init__()
        self.n_embd = n_embd
        self.hyperbolic_dim = hyperbolic_dim
        self.curvature = curvature
        self.use_geoopt = use_geoopt and HAS_GEOOPT
        
        # Project to hyperbolic space
        self.to_hyp = nn.Linear(n_embd, hyperbolic_dim)
        
        if self.use_geoopt:
            # Use geoopt for proper Riemannian operations
            self.manifold = geoopt.PoincareBall(c=-curvature)
            self.hyp_transform = geoopt.layers.RadialNd(
                hyperbolic_dim, hyperbolic_dim, self.manifold
            )
        else:
            # Fallback: simple linear transformation with tanh constraint
            self.hyp_transform = nn.Sequential(
                nn.Linear(hyperbolic_dim, hyperbolic_dim),
                nn.Tanh()  # Constrain to unit ball
            )
        
        # Project back to Euclidean
        self.from_hyp = nn.Linear(hyperbolic_dim, n_embd)
    
    def _expmap0(self, x):
        """Exponential map at origin (maps to Poincaré ball)."""
        if self.use_geoopt:
            return self.manifold.expmap0(x)
        else:
            # Simple approximation: tanh constrains to unit ball
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-8
            return torch.tanh(x_norm) * (x / x_norm)
    
    def _logmap0(self, x):
        """Logarithmic map at origin (maps from Poincaré ball)."""
        if self.use_geoopt:
            return self.manifold.logmap0(x)
        else:
            # Simple approximation: inverse of expmap0
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-8
            return torch.atanh(torch.clamp(x_norm, max=0.999)) * (x / x_norm)
    
    def forward(self, x):
        """
        Forward pass through hyperbolic bottleneck.
        
        Args:
            x: Input tensor of shape (B, T, n_embd)
        
        Returns:
            Output tensor of shape (B, T, n_embd)
        """
        B, T, C = x.shape
        
        # Project to hyperbolic space
        x_hyp = self.to_hyp(x)  # (B, T, hyperbolic_dim)
        
        # Map to Poincaré ball
        x_hyp = self._expmap0(x_hyp)
        
        # Transform in hyperbolic space (geodesic operations)
        x_hyp = self.hyp_transform(x_hyp)
        
        # Ensure we stay in the ball (for numerical stability)
        if not self.use_geoopt:
            x_norm = torch.norm(x_hyp, dim=-1, keepdim=True)
            x_hyp = x_hyp * torch.clamp(x_norm, max=0.999) / (x_norm + 1e-8)
        
        # Project back to Euclidean
        x_euclid = self._logmap0(x_hyp)
        x_out = self.from_hyp(x_euclid)
        
        return x_out


class ToroidalBottleneck(nn.Module):
    """
    Toroidal space bottleneck for periodic/cyclic structure.
    
    Projects embeddings to angular representation on multiple circles (torus),
    applies transformations while preserving circle structure, then projects back.
    This naturally encodes cyclic patterns like time, rotation, and periodic structures.
    
    Args:
        n_embd: Embedding dimension
        n_circles: Number of circles in the torus (default: 4)
    
    Shape:
        - Input: (batch, seq_len, n_embd)
        - Output: (batch, seq_len, n_embd)
    
    Examples:
        >>> tor = ToroidalBottleneck(n_embd=768, n_circles=4)
        >>> x = torch.randn(4, 100, 768)
        >>> y = tor(x)
        >>> y.shape
        torch.Size([4, 100, 768])
    """
    
    def __init__(self, n_embd, n_circles=4):
        super().__init__()
        self.n_embd = n_embd
        self.n_circles = n_circles
        
        # Project to angular representation (each circle has 2 dims: sin, cos)
        self.to_torus = nn.Linear(n_embd, n_circles * 2)
        
        # Transform on torus (via angle mixing)
        self.torus_transform = nn.Linear(n_circles * 2, n_circles * 2)
        
        # Project back
        self.from_torus = nn.Linear(n_circles * 2, n_embd)
    
    def forward(self, x):
        """
        Forward pass through toroidal bottleneck.
        
        Args:
            x: Input tensor of shape (B, T, n_embd)
        
        Returns:
            Output tensor of shape (B, T, n_embd)
        """
        # Project to angular space
        angles = self.to_torus(x)  # (B, T, n_circles * 2)
        
        # Normalize to unit circles: [sin θ, cos θ] for each circle
        angles = angles.view(*angles.shape[:-1], self.n_circles, 2)
        angles = F.normalize(angles, p=2, dim=-1)  # Unit circle constraint
        angles = angles.flatten(-2, -1)
        
        # Transform (mixing angles while preserving circle structure)
        angles_transformed = self.torus_transform(angles)
        angles_transformed = angles_transformed.view(*angles.shape[:-1], self.n_circles, 2)
        angles_transformed = F.normalize(angles_transformed, p=2, dim=-1)
        angles_transformed = angles_transformed.flatten(-2, -1)
        
        # Project back to Euclidean
        x_out = self.from_torus(angles_transformed)
        
        return x_out


class GeometricBottleneck(nn.Module):
    """
    Combined hyperbolic + toroidal bottleneck.
    
    Combines both geometric spaces with a learned mixing weight to leverage
    both hierarchical (hyperbolic) and periodic (toroidal) structures.
    
    Args:
        n_embd: Embedding dimension
        hyperbolic_dim: Dimension of hyperbolic space (default: 64)
        n_circles: Number of circles in torus (default: 4)
        use_geoopt: Whether to use geoopt for hyperbolic operations (default: True)
    
    Shape:
        - Input: (batch, seq_len, n_embd)
        - Output: (batch, seq_len, n_embd)
    
    Examples:
        >>> geom = GeometricBottleneck(n_embd=768, hyperbolic_dim=64, n_circles=4)
        >>> x = torch.randn(4, 100, 768)
        >>> y = geom(x)
        >>> y.shape
        torch.Size([4, 100, 768])
    """
    
    def __init__(self, n_embd, hyperbolic_dim=64, n_circles=4, use_geoopt=True):
        super().__init__()
        self.hyperbolic = HyperbolicBottleneck(n_embd, hyperbolic_dim, use_geoopt=use_geoopt)
        self.toroidal = ToroidalBottleneck(n_embd, n_circles)
        
        # Mixing weight (learned)
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        """
        Forward pass through combined geometric bottleneck.
        
        Args:
            x: Input tensor of shape (B, T, n_embd)
        
        Returns:
            Output tensor of shape (B, T, n_embd)
        """
        x_hyp = self.hyperbolic(x)
        x_tor = self.toroidal(x)
        
        # Weighted combination
        alpha = torch.sigmoid(self.alpha)  # Ensure [0, 1]
        return alpha * x_hyp + (1 - alpha) * x_tor

