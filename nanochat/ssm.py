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
- **FFT-based Convolution** for O(L log L) efficient training

Reference:
    Defendre, J. (2025). Spin-Resonant Geometric Intelligence (SRGI):
    Unifying Geometry, Resonance, and Neural Computation for Scalable Intelligence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class StableResonantSSM(nn.Module):
    """
    Lightly damped oscillator layer with eigenvalues on a soft circle.
    
    Uses FFT convolution for efficient training (O(L log L)).
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
        self.dt = torch.tensor(dt)
        self.damp_min = damp_min

        # Learn A: diagonal complex matrix with |Re(λ)| >= damp_min
        self.A_real = nn.Parameter(torch.randn(state_dim) * 0.1 - damp_min)
        self.A_imag = nn.Parameter(torch.randn(state_dim))

        # B: input projection matrix (state_dim x input_dim)
        # Factorized for efficiency: (D, N)
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * 0.02)

        # C: output mixing matrix (input_dim x state_dim)
        self.C = nn.Parameter(torch.randn(input_dim, state_dim) * 0.02)
        
        # D: skip connection
        self.D = nn.Parameter(torch.randn(input_dim))

    def forward(self, u):
        """
        Forward pass using FFT convolution.
        
        h_t = A h_{t-1} + B u_t
        y_t = C h_t + D u_t
        
        Since A is diagonal and constant, this is a convolution y = u * K
        """
        # Check complex input
        if u.dim() == 4 and u.shape[-1] == 2:
             # Treat complex input as 2x channels for simplicity in this version
             # or handle separately. For now, flattening complex channels or taking norm.
             # A rigorous complex SSM would convolve complex inputs.
             # Let's process real/imag separately or combined.
             u_real = u[..., 0]
             u_imag = u[..., 1]
             # Process as combined features for the SSM
             # Or better, run SSM on complex input natively
             return self._forward_complex_fft(u)
        
        return self._forward_real_fft(u)

    def _get_discretized_params(self, device):
        dt = self.dt.to(device)
        
        # Enforce damping constraint
        A_real = -torch.abs(self.A_real) - self.damp_min
        A = torch.complex(A_real, self.A_imag)
        
        # Bilinear discretization
        # A_bar = (I - dt/2 A)^-1 (I + dt/2 A)
        # B_bar = (I - dt/2 A)^-1 (dt B)
        # Since A is diagonal, inverse is element-wise division
        
        denom = 1 - (dt / 2) * A
        A_bar = (1 + (dt / 2) * A) / denom
        B_bar = (dt * self.B.to(torch.complex64)) / denom.unsqueeze(-1) # (state, input)
        
        return A_bar, B_bar

    def _forward_real_fft(self, u):
        B_batch, L, D = u.shape
        device = u.device
        
        # Get discretized parameters
        A_bar, B_bar = self._get_discretized_params(device) # A_bar: (state,), B_bar: (state, input)
        
        # Compute Kernel K
        # K_t = C @ (A_bar^t) @ B_bar
        # We can compute (A_bar^t) efficiently for all t
        
        # 1. Compute powers of A_bar: (A_bar^0, ..., A_bar^{L-1})
        # A_bar is diagonal (state_dim,)
        # vandermonde style
        k = torch.arange(L, device=device)
        # A_bar_pow: (state_dim, L)
        A_bar_pow = A_bar.unsqueeze(1) ** k.unsqueeze(0) 
        
        # 2. Compute Convolution Kernel K
        # We want y = u * K. 
        # The SSM equation implies: h = (u @ B_bar.T) * A_bar_pow
        # u: (B, L, input)
        # B_bar: (state, input)
        
        # Project input u to state space: v = u @ B_bar.T
        # v: (B, L, state) (complex)
        v = torch.matmul(u.to(torch.complex64), B_bar.T)
        
        # Convolve v with A_bar_pow
        # v: (B, L, state), A_bar_pow: (state, L)
        # We can use FFT.
        
        # FFT length needs to be 2*L to avoid circular aliasing
        n_fft = 2 * L
        
        # FFT of input v
        # v is complex, so we use fft, not rfft
        v_f = torch.fft.fft(v, n=n_fft, dim=1) # (B, n_fft, state)
        
        # FFT of kernel (A_bar_pow)
        # Transpose A_bar_pow to (L, state) for FFT
        k_f = torch.fft.fft(A_bar_pow.T, n=n_fft, dim=0) # (n_fft, state)
        
        # Element-wise multiplication in frequency domain
        y_f = v_f * k_f.unsqueeze(0)
        
        # Inverse FFT
        y = torch.fft.ifft(y_f, n=n_fft, dim=1) # (B, n_fft, state)
        
        # Crop to original length
        y = y[:, :L, :]
        
        # Project to output: y_out = y @ C.T + u @ D
        # C: (input, state) -> C.T: (state, input)
        y_out = torch.matmul(y.to(torch.complex64), self.C.T.to(torch.complex64)).real
        
        # Skip connection
        y_out = y_out + u * self.D
        
        return y_out

    def _forward_complex_fft(self, u):
        # u is (B, L, D, 2)
        u_c = torch.complex(u[..., 0], u[..., 1]) # (B, L, D)
        
        # Similar logic as real, but keep complex throughout
        B_batch, L, D = u_c.shape
        device = u_c.device
        
        A_bar, B_bar = self._get_discretized_params(device)
        
        # v = u @ B_bar.T
        v = torch.matmul(u_c, B_bar.T) # (B, L, state)
        
        # Convolution
        n_fft = 2 * L
        k = torch.arange(L, device=device)
        A_bar_pow = A_bar.unsqueeze(1) ** k.unsqueeze(0) # (state, L)
        
        # Use full fft for complex signals
        v_f = torch.fft.fft(v, n=n_fft, dim=1)
        k_f = torch.fft.fft(A_bar_pow.T, n=n_fft, dim=0)
        
        y_f = v_f * k_f.unsqueeze(0)
        y = torch.fft.ifft(y_f, n=n_fft, dim=1)
        y = y[:, :L, :]
        
        # Output projection
        y_out = torch.matmul(y, self.C.T.to(torch.complex64))
        
        # Skip
        y_out = y_out + u_c * self.D
        
        return torch.stack([y_out.real, y_out.imag], dim=-1)

    def get_phase_info(self, u):
        # Helper for visualization - uses slower recurrence for exact state tracking if needed
        # Or reconstructs from convolution
        return None, None

class ResonantBlock(nn.Module):
    def __init__(self, n_embd: int, state_dim: int = None, residual_weight: float = 0.1):
        super().__init__()
        if state_dim is None:
            state_dim = n_embd
        self.ssm = StableResonantSSM(state_dim=state_dim, input_dim=n_embd)
        self.residual_weight = residual_weight

    def forward(self, x):
        ssm_out = self.ssm(x)
        return x + self.residual_weight * ssm_out
