"""
Modern Hopfield Memory for SRGI Architecture

This module implements Modern Hopfield Networks as dense associative memory,
providing stable attractor states for improved memory retrieval and perceptual clarity.

Key features:
- Exponential storage capacity (vs. linear in classical Hopfield)
- Iterative energy minimization for convergence to attractors
- Denoising: corrupted inputs converge to clean patterns
- Associative recall: partial cues retrieve full patterns
- Fixed-point attractors for stable memory states

Mathematical Foundation:
- Energy function: E = -log Σ exp(β x^T ξᵢ)
- Iterative updates move state toward energy minima
- Modern Hopfield (2016) provides continuous attractor dynamics

Brain Mapping:
- Wave attractors = perceptual clarity
- Stable attractor states correspond to clear perceptual representations

Reference:
    Ramsauer, H., et al. (2020). Hopfield Networks is All You Need.
    Krotov, D., & Hopfield, J. J. (2016). Dense Associative Memory for Pattern Recognition.
    Defendre, J. (2025). Spin-Resonant Geometric Intelligence (SRGI).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModernHopfieldMemory(nn.Module):
    """
    Dense associative memory via modern Hopfield networks.
    
    This module implements a modern Hopfield network that acts as an associative
    memory system. Given a query, it iteratively converges to the nearest stored
    pattern (attractor) through energy minimization.
    
    Args:
        n_embd: Embedding dimension
        memory_size: Number of memory patterns to store (default: 1024)
        beta: Inverse temperature parameter (default: 1.0)
            Higher beta = sharper attractors, more selective retrieval
        n_steps: Number of iterative update steps (default: 3)
            More steps = better convergence but slower
    
    Shape:
        - Input: (batch, seq_len, n_embd)
        - Output: (batch, seq_len, n_embd)
    
    Example:
        >>> memory = ModernHopfieldMemory(n_embd=768, memory_size=1024)
        >>> x = torch.randn(2, 10, 768)  # batch=2, seq_len=10
        >>> output = memory(x)
        >>> output.shape
        torch.Size([2, 10, 768])
    """
    
    def __init__(self, n_embd: int, memory_size: int = 1024, beta: float = 1.0, n_steps: int = 3):
        super().__init__()
        self.n_embd = n_embd
        self.memory_size = memory_size
        self.beta = beta  # Inverse temperature
        self.n_steps = n_steps  # Inner loop iterations
        
        # Memory patterns (learned or initialized)
        # Initialize with small random values
        self.patterns = nn.Parameter(torch.randn(memory_size, n_embd) * 0.02)
        
        # Query/Key projections for flexible retrieval
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
    
    def energy(self, x: torch.Tensor, patterns: torch.Tensor = None) -> torch.Tensor:
        """
        Compute Hopfield energy: E = -log Σ exp(β x^T ξᵢ)
        
        Lower energy = closer to an attractor (stored pattern)
        
        Args:
            x: Query states of shape (batch, seq_len, n_embd)
            patterns: Memory patterns (default: self.patterns)
        
        Returns:
            Energy values of shape (batch, seq_len)
        """
        if patterns is None:
            patterns = self.patterns
        
        # Compute similarities: x^T ξᵢ for all patterns
        similarities = (x @ patterns.T) * self.beta  # (B, T, memory_size)
        
        # Energy = -log Σ exp(similarities)
        energy_val = -torch.logsumexp(similarities, dim=-1)  # (B, T)
        
        return energy_val
    
    def forward(self, x: torch.Tensor, return_energy: bool = False):
        """
        Forward pass with iterative energy minimization.
        
        The algorithm iteratively updates the state to move toward the nearest
        attractor (stored pattern) through energy minimization.
        
        Args:
            x: Input states of shape (batch, seq_len, n_embd)
            return_energy: If True, also return energy values
        
        Returns:
            output: Retrieved memory states of shape (batch, seq_len, n_embd)
            energy_val (optional): Energy values of shape (batch, seq_len)
        """
        B, T, C = x.shape
        
        # Project to query space
        q = self.query(x)  # (B, T, n_embd)
        k = self.key(self.patterns)  # (memory_size, n_embd)
        
        # Iterative updates (energy minimization)
        state = q
        for step in range(self.n_steps):
            # Compute similarities to all patterns
            sim = (state @ k.T) * self.beta  # (B, T, memory_size)
            
            # Soft attention over patterns
            attn = F.softmax(sim, dim=-1)  # (B, T, memory_size)
            
            # Retrieve weighted combination of patterns
            retrieved = attn @ self.patterns  # (B, T, n_embd)
            
            # Update state (move toward attractor)
            # Mix current state with retrieved pattern
            state = 0.5 * state + 0.5 * retrieved
        
        # Final retrieval
        sim_final = (state @ k.T) * self.beta
        attn_final = F.softmax(sim_final, dim=-1)
        output = attn_final @ self.patterns
        
        if return_energy:
            energy_val = self.energy(output, self.patterns)
            return output, energy_val
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for visualization/debugging.
        
        Args:
            x: Input states of shape (batch, seq_len, n_embd)
        
        Returns:
            Attention weights of shape (batch, seq_len, memory_size)
        """
        q = self.query(x)
        k = self.key(self.patterns)
        sim = (q @ k.T) * self.beta
        attn = F.softmax(sim, dim=-1)
        return attn

