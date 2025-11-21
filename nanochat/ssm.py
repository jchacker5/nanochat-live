"""
Stable Resonant State-Space Model (R-SSM) for SRGI Architecture

This module implements the Resonant State-Space Layer from the Spin-Resonant
Geometric Intelligence (SRGI) paper. The R-SSM uses lightly damped oscillators
with eigenvalues near the imaginary axis to preserve information and enable
phase-based routing of information through the network.

Key features:
- Bilinear discretization for numerical stability
- Complex diagonal state matrix with damping constraints
- Phase-aware dynamics for selective information routing
- Unitary-like updates to prevent gradient explosion/vanishing

Reference:
    Defendre, J. (2025). Spin-Resonant Geometric Intelligence (SRGI):
    Unifying Geometry, Resonance, and Neural Computation for Scalable Intelligence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StableResonantSSM(nn.Module):
    """
    Lightly damped oscillator layer with eigenvalues on a soft circle.

    This layer implements a state-space model with complex eigenvalues constrained
    to have small negative real parts (damping) and significant imaginary parts
    (resonance). The bilinear discretization ensures numerical stability during
    training and inference.

    Args:
        state_dim: Dimension of the complex state space
        input_dim: Dimension of input/output features
        damp_min: Minimum damping magnitude (prevents pure imaginary eigenvalues)
        dt: Discretization timestep for the state-space model

    Shape:
        - Input: (batch, seq_len, input_dim)
        - Output: (batch, seq_len, input_dim)

    Examples:
        >>> ssm = StableResonantSSM(state_dim=64, input_dim=768)
        >>> x = torch.randn(4, 100, 768)  # batch=4, seq=100, features=768
        >>> y = ssm(x)
        >>> y.shape
        torch.Size([4, 100, 768])
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        damp_min: float = 5e-4,
        dt: float = 0.01
    ):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.dt = torch.tensor(dt)  # Make tensor for gradient ops
        self.damp_min = damp_min  # Clamps real part away from pure imaginary

        # Learn A: diagonal complex matrix with |Re(λ)| >= damp_min
        # Real part provides damping, imaginary part provides resonance
        self.A_real = nn.Parameter(torch.randn(state_dim) * 0.1 - damp_min)
        self.A_imag = nn.Parameter(torch.randn(state_dim))

        # B: input projection matrix (state_dim x input_dim)
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * 0.02)

        # C: output mixing matrix (input_dim x state_dim)
        self.C = nn.Parameter(torch.randn(input_dim, state_dim) * 0.02)

    def _clamp_eig(self):
        """
        Enforce stability constraint on eigenvalues.

        Clamps real parts to be negative and at least damp_min in magnitude.
        This ensures the system is stable (damped) but still oscillatory.
        Called during training before each forward pass.
        """
        with torch.no_grad():
            self.A_real.clamp_(max=-self.damp_min)

    def forward(self, u):
        """
        Forward pass through the resonant state-space layer with complex extension.

        Implements the bilinear discretization:
            A_d = (I - dt/2 * A)^{-1} (I + dt/2 * A)
            B_d = (I - dt/2 * A)^{-1} (sqrt(dt) * B)

        Then evolves the state:
            h[t+1] = A_d @ h[t] + B_d @ u[t]
            y[t] = C @ h[t] (complex output)

        Supports both real and complex inputs:
        - Real input: (batch, seq_len, input_dim) -> real output
        - Complex input: (batch, seq_len, input_dim, 2) [real, imag] -> complex output

        Args:
            u: Input sequence of shape (batch, seq_len, input_dim) or (batch, seq_len, input_dim, 2)

        Returns:
            Output sequence of shape (batch, seq_len, input_dim) or (batch, seq_len, input_dim, 2)
        """
        # Check if input is complex (has last dimension of size 2)
        is_complex = u.dim() == 4 and u.shape[-1] == 2
        
        if is_complex:
            # Complex input: (B, T, n_embd, 2) [real, imag]
            return self._forward_complex(u)
        else:
            # Real input: (B, T, n_embd)
            return self._forward_real(u)
    
    def _forward_real(self, u):
        """Forward pass for real-valued inputs (backward compatibility)."""
        batch, seq_len, _ = u.shape
        device = u.device
        self.dt = self.dt.to(device)

        # Initialize complex state to zeros
        h = torch.zeros(batch, self.state_dim, dtype=torch.complex64, device=device)

        outputs = []
        dt = self.dt

        # Build complex diagonal state matrix A
        A = torch.diag_embed(torch.complex(self.A_real, self.A_imag))

        # Bilinear discretization for numerical stability
        I = torch.eye(self.state_dim, device=device, dtype=torch.complex64)
        A_disc = torch.linalg.solve(I - (dt / 2) * A, I + (dt / 2) * A)
        B_disc = torch.linalg.solve(
            I - (dt / 2) * A,
            torch.sqrt(dt) * self.B.to(torch.complex64)
        )

        # Process sequence step-by-step
        for t in range(seq_len):
            h = (A_disc @ h.unsqueeze(-1)).squeeze(-1)
            h = h + (B_disc @ u[:, t, :].unsqueeze(-1).to(torch.complex64)).squeeze(-1)
            y = (self.C.to(torch.complex64) @ h.unsqueeze(-1)).real.squeeze(-1)
            outputs.append(y)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, input_dim)
    
    def _forward_complex(self, x):
        """
        Forward pass with complex extension as described in paper.
        
        Args:
            x: (B, T, n_embd, 2) complex input [real, imag]
        
        Returns:
            (B, T, n_embd, 2) with resonant evolution
        """
        B, T, n_embd, _ = x.shape
        device = x.device
        self.dt = self.dt.to(device)
        
        # Extract real and imaginary parts
        x_real = x[..., 0]  # (B, T, n_embd)
        x_imag = x[..., 1]  # (B, T, n_embd)
        
        # Project to state space (complex extension)
        u_real = self.B(x_real)  # (B, T, n_state)
        u_imag = self.B(x_imag)  # (B, T, n_state)
        u_complex = u_real + 1j * u_imag  # Complex state
        
        # Initialize complex state to zeros
        h = torch.zeros(B, self.state_dim, dtype=torch.complex64, device=device)
        
        outputs_real = []
        outputs_imag = []
        dt = self.dt
        
        # Build complex diagonal state matrix A
        A = torch.diag_embed(torch.complex(self.A_real, self.A_imag))
        
        # Bilinear discretization
        I = torch.eye(self.state_dim, device=device, dtype=torch.complex64)
        A_disc = torch.linalg.solve(I - (dt / 2) * A, I + (dt / 2) * A)
        B_disc = torch.linalg.solve(
            I - (dt / 2) * A,
            torch.sqrt(dt) * self.B.to(torch.complex64)
        )
        
        # Process sequence step-by-step
        for t in range(T):
            # State update: h[t+1] = A_d @ h[t] + B_d @ u[t]
            h = (A_disc @ h.unsqueeze(-1)).squeeze(-1)
            u_t_complex = u_complex[:, t, :].to(torch.complex64)
            h = h + (B_disc @ u_t_complex.unsqueeze(-1)).squeeze(-1)
            
            # Project back to embedding space (separate real/imag)
            y_complex = self.C.to(torch.complex64) @ h.unsqueeze(-1)
            y_real = y_complex.real.squeeze(-1)
            y_imag = y_complex.imag.squeeze(-1)
            
            outputs_real.append(y_real)
            outputs_imag.append(y_imag)
        
        out_real = torch.stack(outputs_real, dim=1)  # (B, T, n_embd)
        out_imag = torch.stack(outputs_imag, dim=1)  # (B, T, n_embd)
        
        return torch.stack([out_real, out_imag], dim=-1)  # (B, T, n_embd, 2)

    def get_phase_info(self, u):
        """
        Extract phase information from the hidden states.

        Useful for visualization and debugging phase-based routing.

        Args:
            u: Input sequence of shape (batch, seq_len, input_dim)

        Returns:
            Tuple of (magnitudes, phases) each of shape (batch, seq_len, state_dim)
        """
        batch, seq_len, _ = u.shape
        device = u.device
        self.dt = self.dt.to(device)

        h = torch.zeros(batch, self.state_dim, dtype=torch.complex64, device=device)

        magnitudes = []
        phases = []
        dt = self.dt

        A = torch.diag_embed(torch.complex(self.A_real, self.A_imag))
        I = torch.eye(self.state_dim, device=device, dtype=torch.complex64)
        A_disc = torch.linalg.solve(I - (dt / 2) * A, I + (dt / 2) * A)
        B_disc = torch.linalg.solve(
            I - (dt / 2) * A,
            torch.sqrt(dt) * self.B.to(torch.complex64)
        )

        for t in range(seq_len):
            h = (A_disc @ h.unsqueeze(-1)).squeeze(-1)
            h = h + (B_disc @ u[:, t, :].unsqueeze(-1).to(torch.complex64)).squeeze(-1)

            # Extract magnitude and phase
            mag = torch.abs(h)
            phase = torch.angle(h)

            magnitudes.append(mag)
            phases.append(phase)

        magnitudes = torch.stack(magnitudes, dim=1)
        phases = torch.stack(phases, dim=1)

        return magnitudes, phases


class ResonantBlock(nn.Module):
    """
    Transformer block augmented with resonant state-space dynamics.

    This can be used as a drop-in replacement for standard transformer blocks,
    adding resonant dynamics as a parallel pathway that preserves phase information
    and enables longer-range dependencies.

    Args:
        n_embd: Embedding dimension
        state_dim: Dimension of the SSM state (typically same as n_embd or smaller)
        residual_weight: Weight for combining SSM output with residual (default: 0.1)

    Shape:
        - Input: (batch, seq_len, n_embd)
        - Output: (batch, seq_len, n_embd)
    """

    def __init__(self, n_embd: int, state_dim: int = None, residual_weight: float = 0.1):
        super().__init__()
        if state_dim is None:
            state_dim = n_embd // 2  # Use half the embedding dim by default

        self.ssm = StableResonantSSM(state_dim=state_dim, input_dim=n_embd)
        self.residual_weight = residual_weight

    def forward(self, x):
        """
        Forward pass with resonant augmentation.

        Args:
            x: Input of shape (batch, seq_len, n_embd)

        Returns:
            Output of shape (batch, seq_len, n_embd)
        """
        # Apply SSM and blend with residual connection
        ssm_out = self.ssm(x)
        return x + self.residual_weight * ssm_out
