"""
Phase-Aware Attention (PAA) for SRGI Architecture

This module implements Phase-Aware Attention, which modulates attention scores
based on phase coherence between tokens. This enables temporal binding via
phase synchronization, aligning with the neuroscience finding that gamma
phase-locking provides coherence in cortical computation.

Key features:
- Phase coherence gating: tokens "in phase" get higher attention weight
- Learned coherence parameter β per head
- Compatible with RoPE (Rotary Position Embeddings)
- Preserves standard attention when β=0

Reference:
    Defendre, J. (2025). Spin-Resonant Geometric Intelligence (SRGI):
    Unifying Geometry, Resonance, and Neural Computation for Scalable Intelligence.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import apply_rotary_emb, norm


class PhaseAwareAttention(nn.Module):
    """
    Attention with phase coherence gating.
    
    Modulates attention scores based on phase differences between tokens,
    enabling tokens "in phase" to communicate preferentially. This implements
    gamma phase-locking for coherence as described in the SRGI architecture.
    
    Args:
        n_embd: Embedding dimension
        n_head: Number of attention heads
        n_kv_head: Number of key/value heads (for GQA support)
        beta_init: Initial value for phase coherence weight (default: 0.5)
    
    Shape:
        - Input: (batch, seq_len, n_embd)
        - Output: (batch, seq_len, n_embd)
    
    Examples:
        >>> paa = PhaseAwareAttention(n_embd=768, n_head=12, n_kv_head=12)
        >>> x = torch.randn(4, 100, 768)
        >>> cos_sin = (torch.randn(1, 100, 1, 64), torch.randn(1, 100, 1, 64))
        >>> y = paa(x, cos_sin, kv_cache=None)
        >>> y.shape
        torch.Size([4, 100, 768])
    """
    
    def __init__(self, n_embd, n_head, n_kv_head=None, beta_init=0.5, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        
        # Standard Q, K, V projections
        self.c_q = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        # Phase coherence weight (learned per head)
        # β ∈ [0, 1] controls how much phase coherence affects attention
        self.beta = nn.Parameter(torch.ones(n_head) * beta_init)
    
    def forward(self, x, cos_sin, kv_cache=None, return_commutativity_loss=False):
        """
        Forward pass with phase-aware attention.
        
        Args:
            x: Input tensor of shape (B, T, n_embd)
            cos_sin: Tuple of (cos, sin) tensors for RoPE, each shape (1, T, 1, head_dim//2)
            kv_cache: Optional KV cache for inference
            return_commutativity_loss: If True, return commutativity loss (δd - dδ)
        
        Returns:
            Output tensor of shape (B, T, n_embd)
            If return_commutativity_loss=True, also returns scalar commutativity loss
        """
        B, T, C = x.shape
        
        # Project and split heads
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        
        # Apply RoPE (which gives us phase information)
        cos, sin = cos_sin
        q_rot = apply_rotary_emb(q, cos, sin)
        k_rot = apply_rotary_emb(k, cos, sin)
        
        # Apply QK norm (matching the standard attention in gpt.py)
        q_rot, k_rot = norm(q_rot), norm(k_rot)
        
        # Transpose for attention: (B, T, H, D) -> (B, H, T, D)
        q_rot = q_rot.transpose(1, 2)
        k_rot = k_rot.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply KV cache if provided
        if kv_cache is not None:
            k_rot, v = kv_cache.insert_kv(self.layer_idx, k_rot, v)
        
        Tq = q_rot.size(2)  # number of queries in this forward pass
        Tk = k_rot.size(2)  # number of keys/values in total
        
        # Compute phase coherence modulation
        # Phase difference: Δφ_ij between token i and j
        # Approximate via position difference (RoPE encodes this)
        position_diff = torch.arange(Tq, device=x.device, dtype=torch.float32).unsqueeze(0) - \
                       torch.arange(Tk, device=x.device, dtype=torch.float32).unsqueeze(1)
        
        # Coherence modulation: 1 + β cos(θ * position_diff)
        # θ is implicit in RoPE frequency (base=10000)
        theta_base = 10000.0
        theta = theta_base ** (-torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32) / self.head_dim)
        
        # Compute phase difference for each head dimension pair
        # Shape: (1, 1, Tq, Tk, head_dim//2)
        phase_diff = position_diff.unsqueeze(-1) * theta.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # Average across frequency dimensions and apply coherence
        # Shape: (1, n_head, Tq, Tk, 1)
        coherence = 1 + self.beta.view(1, -1, 1, 1, 1) * torch.cos(phase_diff).mean(-1, keepdim=True)
        coherence = coherence.squeeze(-1)  # (1, n_head, Tq, Tk)
        
        # Standard attention scores
        attn_scores = (q_rot @ k_rot.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Modulate by phase coherence
        attn_scores = attn_scores * coherence
        
        # Handle causal masking and GQA
        enable_gqa = self.n_head != self.n_kv_head
        
        if kv_cache is None or Tq == Tk:
            # Training mode or full sequence: causal attention
            attn_weights = F.softmax(attn_scores, dim=-1)
            # Apply causal mask
            causal_mask = torch.tril(torch.ones((Tq, Tk), device=x.device, dtype=torch.bool))
            attn_weights = attn_weights * causal_mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
        elif Tq == 1:
            # Single query inference: attend to all cached keys
            attn_weights = F.softmax(attn_scores, dim=-1)
        else:
            # Chunk inference: handle prefix + causal chunk
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=x.device)
            prefix_len = Tk - Tq
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=x.device))
            
            # Apply mask and softmax
            attn_scores = attn_scores.masked_fill(~attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Attend to values
        y = attn_weights @ v
        
        # Handle GQA: duplicate values if needed
        if enable_gqa:
            # Repeat values to match query heads
            repeat_factor = self.n_head // self.n_kv_head
            y = y.repeat_interleave(repeat_factor, dim=1)
        
        # Re-assemble and project
        y = y.transpose(1, 2).contiguous().view(B, Tq, C)
        y = self.c_proj(y)
        
        if return_commutativity_loss:
            # Compute commutativity loss: ||δd - dδ||
            # δ = discrete coboundary (token adjacency)
            # d = continuous differential (phase gradient)
            
            if Tq > 2:  # Need at least 3 tokens for coboundary
                # Discrete coboundary: δx[i] = x[i+1] - x[i] (token differences)
                discrete_coboundary = x[:, 1:, :] - x[:, :-1, :]  # (B, T-1, n_embd)
                
                # Continuous differential: dφ = phase gradient from q_rot
                q_rot_flat = q_rot.transpose(1, 2).contiguous().view(B, Tq, self.n_head * self.head_dim)
                phase_gradient = q_rot_flat[:, 1:, :] - q_rot_flat[:, :-1, :]  # (B, T-1, n_head*head_dim)
                
                # Apply operations in both orders
                # δd: discrete coboundary of continuous differential
                if discrete_coboundary.shape[1] > 1:
                    delta_d = discrete_coboundary[:, 1:, :] - discrete_coboundary[:, :-1, :]
                    
                    # dδ: continuous differential of discrete coboundary
                    d_delta = phase_gradient[:, 1:, :] - phase_gradient[:, :-1, :]
                    
                    # Project to same dimension for comparison
                    if delta_d.shape[-1] != d_delta.shape[-1]:
                        proj = nn.Linear(d_delta.shape[-1], delta_d.shape[-1], device=x.device, dtype=x.dtype)
                        d_delta_proj = proj(d_delta)
                    else:
                        d_delta_proj = d_delta
                    
                    # Commutativity loss: ||δd - dδ||
                    if delta_d.shape == d_delta_proj.shape:
                        commutativity_loss = torch.mean((delta_d - d_delta_proj).pow(2))
                    else:
                        # Fallback: use norm difference
                        commutativity_loss = torch.mean((torch.norm(delta_d, dim=-1) - torch.norm(d_delta_proj, dim=-1)).pow(2))
                else:
                    commutativity_loss = torch.tensor(0.0, device=x.device)
            else:
                commutativity_loss = torch.tensor(0.0, device=x.device)
            
            return y, commutativity_loss
        
        return y

