"""
Modal Logic Reasoning for SRGI Architecture

This module implements modal logic concepts (Kripke semantics, possible worlds)
to enhance reasoning, self-verification, and chain-of-thought capabilities.

Inspired by DeepSeek-R1's approach to using modal structures for:
- Possible worlds exploration (◊p: "it is possible that p")
- Necessity verification (□p: "necessarily p")
- Epistemic accessibility relations (K_a p: "agent knows p")
- Self-verification via S4/S5 semantics

Reference:
    Defendre, J. (2025). Spin-Resonant Geometric Intelligence (SRGI):
    Unifying Geometry, Resonance, and Neural Computation for Scalable Intelligence.
    Section: Modal Logic Integration for Enhanced Reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class KripkeFrame(nn.Module):
    """
    Kripke frame representing possible worlds and accessibility relations.
    
    In modal logic, a Kripke frame (W, R) consists of:
    - W: Set of possible worlds (states)
    - R: Accessibility relation between worlds
    
    For SRGI, we use:
    - Worlds = different reasoning paths or phase states
    - Accessibility = phase coherence or geometric similarity
    
    Args:
        n_worlds: Number of possible worlds to maintain
        n_embd: Embedding dimension for each world
        relation_type: Type of accessibility relation ('S4', 'S5', 'custom')
    
    Shape:
        - Input: (batch, seq_len, n_embd)
        - Output: (batch, seq_len, n_embd)
    """
    
    def __init__(self, n_worlds=4, n_embd=768, relation_type='S5'):
        super().__init__()
        self.n_worlds = n_worlds
        self.n_embd = n_embd
        self.relation_type = relation_type
        
        # World embeddings (learnable)
        self.world_embeddings = nn.Parameter(torch.randn(n_worlds, n_embd))
        
        # Accessibility relation matrix (learnable)
        # R[i, j] = 1 means world i can access world j
        if relation_type == 'S5':
            # S5: Equivalence relation (reflexive, symmetric, transitive)
            # All worlds accessible to all (full connectivity)
            self.register_buffer('accessibility', torch.ones(n_worlds, n_worlds))
        elif relation_type == 'S4':
            # S4: Reflexive and transitive (but not necessarily symmetric)
            # Sequential reasoning paths
            accessibility = torch.tril(torch.ones(n_worlds, n_worlds))
            self.register_buffer('accessibility', accessibility)
        else:
            # Custom: Learnable accessibility
            self.accessibility = nn.Parameter(torch.rand(n_worlds, n_worlds))
        
        # World mixing weights
        self.world_mixer = nn.Linear(n_embd * n_worlds, n_embd)
    
    def forward(self, x):
        """
        Forward pass through Kripke frame.
        
        Args:
            x: (B, T, n_embd) input embeddings
        
        Returns:
            output: (B, T, n_embd) modal-reasoned embeddings
        """
        B, T, C = x.shape
        
        # Project input to each world
        # Shape: (B, T, n_worlds, n_embd)
        world_states = x.unsqueeze(2) + self.world_embeddings.unsqueeze(0).unsqueeze(0)
        
        # Apply accessibility relations
        # For each world, aggregate accessible worlds
        # Shape: (B, T, n_worlds, n_embd)
        accessible_states = torch.zeros_like(world_states)
        
        for i in range(self.n_worlds):
            # Find accessible worlds
            accessible_mask = self.accessibility[i].unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            # Aggregate accessible states
            accessible_states[:, :, i, :] = (world_states * accessible_mask).sum(dim=2)
        
        # Combine worlds
        world_combined = accessible_states.view(B, T, self.n_worlds * C)
        output = self.world_mixer(world_combined)
        
        return output


class ModalAttention(nn.Module):
    """
    Modal-aware attention using Kripke semantics.
    
    Implements modal operators:
    - □ (necessity): Attention over all accessible worlds
    - ◊ (possibility): Attention over at least one accessible world
    
    Args:
        n_embd: Embedding dimension
        n_head: Number of attention heads
        n_worlds: Number of possible worlds
        use_necessity: Use necessity operator (□) vs possibility (◊)
    
    Shape:
        - Input: (batch, seq_len, n_embd)
        - Output: (batch, seq_len, n_embd)
    """
    
    def __init__(self, n_embd, n_head, n_worlds=4, use_necessity=True):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_worlds = n_worlds
        self.head_dim = n_embd // n_head
        self.use_necessity = use_necessity
        
        # Standard Q, K, V projections
        self.c_q = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        # Modal operator weights
        self.modal_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x, kripke_frame=None):
        """
        Forward pass with modal operators.
        
        Args:
            x: (B, T, n_embd) input
            kripke_frame: Optional KripkeFrame for world structure
        
        Returns:
            output: (B, T, n_embd)
        """
        B, T, C = x.shape
        
        # Standard attention
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention scores
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply modal operators
        if kripke_frame is not None:
            # Use accessibility relations from Kripke frame
            accessibility = kripke_frame.accessibility  # (n_worlds, n_worlds)
            
            # Expand accessibility to attention dimensions
            # For simplicity, map tokens to worlds (round-robin)
            n_worlds = accessibility.shape[0]
            world_map_i = torch.arange(T, device=x.device) % n_worlds  # (T,)
            world_map_j = torch.arange(T, device=x.device) % n_worlds  # (T,)
            
            # Build modal mask: (T, T) where mask[i,j] = accessibility[world_i, world_j]
            modal_mask = accessibility[world_map_i][:, world_map_j]  # (T, T)
            modal_mask = modal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
            
            if self.use_necessity:
                # □: Necessity - require all accessible worlds
                # Mask out inaccessible worlds
                attn_scores = attn_scores.masked_fill(modal_mask == 0, float('-inf'))
            else:
                # ◊: Possibility - at least one accessible world
                # Boost accessible worlds
                attn_scores = attn_scores + self.modal_weight * modal_mask.float()
        
        # Softmax and attend
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = (attn_weights @ v).transpose(1, 2).contiguous()
        output = output.view(B, T, C)
        output = self.c_proj(output)
        
        return output


class ModalCoTReasoning(nn.Module):
    """
    Modal Chain-of-Thought reasoning module.
    
    Implements DeepSeek-R1 style reasoning with:
    - Possible worlds exploration (◊p)
    - Necessity verification (□p)
    - Self-verification via epistemic accessibility
    
    Args:
        n_embd: Embedding dimension
        n_worlds: Number of reasoning paths to explore
        max_steps: Maximum CoT steps
    
    Shape:
        - Input: (batch, seq_len, n_embd)
        - Output: (batch, seq_len, n_embd)
    """
    
    def __init__(self, n_embd=768, n_worlds=4, max_steps=5):
        super().__init__()
        self.n_embd = n_embd
        self.n_worlds = n_worlds
        self.max_steps = max_steps
        
        # Kripke frame for world structure
        self.kripke_frame = KripkeFrame(n_worlds=n_worlds, n_embd=n_embd, relation_type='S5')
        
        # Modal attention layers
        self.necessity_attn = ModalAttention(n_embd, n_head=8, n_worlds=n_worlds, use_necessity=True)
        self.possibility_attn = ModalAttention(n_embd, n_head=8, n_worlds=n_worlds, use_necessity=False)
        
        # Verification layer (epistemic: K_a p)
        self.verification = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_embd),
            nn.Sigmoid()  # Verification confidence
        )
        
        # Step controller
        self.step_controller = nn.Linear(n_embd, 1)
    
    def forward(self, x, return_verification=False):
        """
        Forward pass with modal CoT reasoning.
        
        Args:
            x: (B, T, n_embd) input
            return_verification: Return verification scores
        
        Returns:
            output: (B, T, n_embd) reasoned output
            verification_scores: (B, T) if return_verification=True
        """
        B, T, C = x.shape
        
        # Initial state through Kripke frame
        current_state = self.kripke_frame(x)
        
        # CoT steps
        verification_scores = []
        for step in range(self.max_steps):
            # Possibility exploration (◊): Explore possible paths
            possible_state = self.possibility_attn(current_state, kripke_frame=self.kripke_frame)
            
            # Necessity verification (□): Verify across all accessible worlds
            necessary_state = self.necessity_attn(possible_state, kripke_frame=self.kripke_frame)
            
            # Epistemic verification: K_a p (agent knows p)
            verification = self.verification(necessary_state)  # (B, T, n_embd)
            verification_score = verification.mean(dim=-1)  # (B, T)
            verification_scores.append(verification_score)
            
            # Update state
            current_state = current_state + verification * necessary_state
            
            # Early stopping if verification is high
            if verification_score.mean() > 0.9:
                break
        
        output = current_state
        
        if return_verification:
            # Stack verification scores: (B, T, max_steps)
            verification_stack = torch.stack(verification_scores, dim=-1)
            return output, verification_stack
        
        return output


class ModalGeometricBottleneck(nn.Module):
    """
    Geometric bottleneck enhanced with modal reasoning.
    
    Combines geometric structure (hyperbolic/toroidal) with modal logic
    for reasoning over compressed/uncertain states.
    
    This addresses DeepSeek-OCR style compression where:
    - Compressed states = possible worlds with varying fidelity
    - Modal operators explore uncertainty (blurred text = ◊, clear = □)
    """
    
    def __init__(self, n_embd=768, hyperbolic_dim=64, n_circles=4, n_worlds=4):
        super().__init__()
        from nanochat.geometric_bottleneck import GeometricBottleneck
        
        self.geometric = GeometricBottleneck(n_embd, hyperbolic_dim, n_circles)
        self.modal_cot = ModalCoTReasoning(n_embd, n_worlds=n_worlds)
        
        # Fidelity-aware mixing
        self.fidelity_mixer = nn.Linear(n_embd * 2, n_embd)
    
    def forward(self, x, fidelity_scores=None):
        """
        Forward pass with modal-geometric reasoning.
        
        Args:
            x: (B, T, n_embd) input embeddings
            fidelity_scores: (B, T) optional fidelity scores (for OCR compression)
        
        Returns:
            output: (B, T, n_embd)
        """
        # Geometric transformation
        x_geom = self.geometric(x)
        
        # Modal reasoning (treats low-fidelity as possible worlds)
        if fidelity_scores is not None:
            # Mask low-fidelity regions for possibility exploration
            low_fid_mask = (fidelity_scores < 0.5).unsqueeze(-1)  # (B, T, 1)
            x_masked = x * (1 - low_fid_mask.float()) + x_geom * low_fid_mask.float()
        else:
            x_masked = x
        
        x_modal = self.modal_cot(x_masked)
        
        # Combine geometric and modal
        x_combined = torch.cat([x_geom, x_modal], dim=-1)
        output = self.fidelity_mixer(x_combined)
        
        return output

