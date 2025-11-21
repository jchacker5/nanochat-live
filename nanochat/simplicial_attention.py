"""
Simplicial Attention for graph-structured data.

Extends attention mechanism to work on simplicial complexes (graphs with
higher-order structures like edges, triangles), respecting cohomological structure.

Reference:
    Defendre, J. (2025). Spin-Resonant Geometric Intelligence (SRGI):
    Unifying Geometry, Resonance, and Neural Computation for Scalable Intelligence.
    Section 3.4.1: Topological Deep Learning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplicialAttention(nn.Module):
    """
    Attention mechanism for simplicial complexes.
    
    Extends standard attention to k-faces (vertices, edges, triangles, etc.)
    respecting cohomological structure.
    
    Args:
        n_embd: Embedding dimension
        n_head: Number of attention heads
        k: Simplex dimension (0=vertices, 1=edges, 2=triangles, ...)
    
    Shape:
        - Input: (batch, n_faces, n_embd)
        - Output: (batch, n_faces, n_embd)
    
    Examples:
        >>> sim_attn = SimplicialAttention(n_embd=768, n_head=12, k=1)
        >>> x = torch.randn(4, 50, 768)  # 50 edges
        >>> y = sim_attn(x)
        >>> y.shape
        torch.Size([4, 50, 768])
    """
    
    def __init__(self, n_embd, n_head, k=1):
        """
        Args:
            n_embd: Embedding dimension
            n_head: Number of attention heads
            k: Simplex dimension (0=vertices, 1=edges, 2=triangles, ...)
        """
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.k = k
        self.head_dim = n_embd // n_head
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        
        # Q, K, V projections for k-faces
        self.c_q = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        
    def forward(self, x, boundary_matrix=None):
        """
        Forward pass on simplicial complex.
        
        Args:
            x: (B, n_faces, n_embd) features on k-faces
            boundary_matrix: (n_faces, n_faces) boundary operator (optional)
        
        Returns:
            output: (B, n_faces, n_embd)
        """
        B, n_faces, C = x.shape
        
        # Standard attention
        q = self.c_q(x).view(B, n_faces, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, n_faces, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, n_faces, self.n_head, self.head_dim)
        
        # Transpose for attention: (B, n_faces, H, D) -> (B, H, n_faces, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention scores
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # If boundary matrix provided, respect cohomological structure
        if boundary_matrix is not None:
            # Mask attention to respect boundary relationships
            boundary_mask = boundary_matrix.unsqueeze(0).unsqueeze(0)  # (1, 1, n_faces, n_faces)
            attn_scores = attn_scores.masked_fill(boundary_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = (attn_weights @ v).transpose(1, 2).contiguous()
        output = output.view(B, n_faces, C)
        output = self.c_proj(output)
        
        return output

