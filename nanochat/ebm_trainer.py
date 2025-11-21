"""
Energy-Based Model Trainer for SRGI Hopfield Memory

This module provides training utilities for EBM Hopfield Memory using
contrastive divergence and persistent contrastive divergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class EBMHopfieldTrainer:
    """
    Trainer for EBM Hopfield Memory using contrastive divergence.
    
    Contrastive divergence minimizes the difference between energies on
    positive examples (data) and negative examples (samples from model).
    """
    
    def __init__(self, model, learning_rate=1e-3, optimizer_type='adam'):
        """
        Args:
            model: EBMHopfieldMemory instance
            learning_rate: Learning rate for optimizer
            optimizer_type: 'adam' or 'sgd'
        """
        self.model = model
        
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    def contrastive_divergence_step(self, positive_samples, n_negative_steps=1):
        """
        Contrastive divergence training step.
        
        Args:
            positive_samples: Positive examples (B, T, n_embd)
            n_negative_steps: Number of negative sampling steps
        
        Returns:
            loss: Contrastive divergence loss
        """
        # Positive phase: compute energy on data
        pos_energy = self.model.energy(positive_samples, self.model.patterns)
        
        # Negative phase: sample from model
        negative_samples = self.model.sample_negative(
            positive_samples, n_negative_steps=n_negative_steps
        )
        
        # Compute negative energy
        neg_energy = self.model.energy(negative_samples, self.model.patterns)
        
        # Contrastive divergence loss
        # Minimize positive energy, maximize negative energy
        loss = (pos_energy - neg_energy).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_step(self, batch, use_cd=True, n_negative_steps=1):
        """
        Single training step.
        
        Args:
            batch: Input batch (B, T, n_embd)
            use_cd: Use contrastive divergence if True
            n_negative_steps: Number of negative sampling steps
        
        Returns:
            loss: Training loss
        """
        if use_cd:
            return self.contrastive_divergence_step(batch, n_negative_steps)
        else:
            # Standard backpropagation (if model supports it)
            # This would require a different loss function
            raise NotImplementedError("Standard backprop not implemented for EBM")


class PersistentEBMTrainer(EBMHopfieldTrainer):
    """
    Trainer using persistent contrastive divergence.
    
    Maintains persistent negative samples across batches for more stable training.
    """
    
    def __init__(self, model, learning_rate=1e-3, optimizer_type='adam'):
        super().__init__(model, learning_rate, optimizer_type)
        self.persistent_negative_samples = None
    
    def persistent_contrastive_divergence_step(self, positive_samples, n_negative_steps=5):
        """
        Persistent contrastive divergence training step.
        
        Args:
            positive_samples: Positive examples (B, T, n_embd)
            n_negative_steps: Number of Gibbs steps (fewer needed with persistence)
        
        Returns:
            loss: Contrastive divergence loss
        """
        # Initialize persistent samples if needed
        if self.persistent_negative_samples is None:
            self.persistent_negative_samples = torch.randn_like(positive_samples)
        
        # Positive phase
        pos_energy = self.model.energy(positive_samples, self.model.patterns)
        
        # Negative phase: update persistent samples
        # Run fewer Gibbs steps since we maintain state across batches
        k = self.model.key(self.model.patterns)
        
        for _ in range(n_negative_steps):
            sim = (self.persistent_negative_samples @ k.T) * self.model.beta
            thermal_noise = self.model.sample_thermal_noise(
                sim.shape, device=positive_samples.device
            )
            sim_thermal = sim + thermal_noise
            attn = F.softmax(sim_thermal / self.model.temperature, dim=-1)
            retrieved = attn @ self.model.patterns
            noise = self.model.sample_thermal_noise(
                self.persistent_negative_samples.shape, device=positive_samples.device
            )
            self.persistent_negative_samples = (
                0.5 * self.persistent_negative_samples + 
                0.5 * retrieved + noise
            )
        
        neg_energy = self.model.energy(self.persistent_negative_samples, self.model.patterns)
        
        # Loss
        loss = (pos_energy - neg_energy).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_step(self, batch, use_cd=True, n_negative_steps=5):
        """
        Single training step with persistent contrastive divergence.
        
        Args:
            batch: Input batch (B, T, n_embd)
            use_cd: Use contrastive divergence if True
            n_negative_steps: Number of negative sampling steps
        
        Returns:
            loss: Training loss
        """
        if use_cd:
            return self.persistent_contrastive_divergence_step(batch, n_negative_steps)
        else:
            raise NotImplementedError("Standard backprop not implemented for EBM")

