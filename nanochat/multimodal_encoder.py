"""
Multimodal Encoder - DeepSeek OCR inspired approach
Converts images, video, and audio into tokens that SRGI can process.

Key insight: Use diffusion encoder to convert all modalities to latent tokens,
then project into the same embedding space as text tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
import math

class VisionDiffusionEncoder(nn.Module):
    """
    Vision Diffusion Encoder (VDE) - DeepSeek OCR inspired
    
    Converts images/video frames into tokens using a diffusion encoder approach.
    The key insight: treat vision like OCR - convert visual information into
    discrete tokens that can be processed by the language model.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_patches: Optional[int] = None,
        use_conv: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        if num_patches is None:
            self.num_patches = (image_size // patch_size) ** 2
        else:
            self.num_patches = num_patches
        
        # Patch embedding: convert image patches to tokens
        if use_conv:
            # Convolutional patch embedding (more efficient)
            self.patch_embed = nn.Conv2d(
                3, embed_dim, kernel_size=patch_size, stride=patch_size
            )
        else:
            # Linear patch embedding
            self.patch_embed = nn.Linear(3 * patch_size * patch_size, embed_dim)
        
        # Position embeddings for patches
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W) or (B, T, C, H, W) for video
        Returns:
            tokens: (B, num_patches, embed_dim) or (B, T, num_patches, embed_dim)
        """
        if images.dim() == 5:
            # Video: (B, T, C, H, W)
            B, T, C, H, W = images.shape
            images = images.view(B * T, C, H, W)
            tokens = self._forward_images(images)
            tokens = tokens.view(B, T, self.num_patches, self.embed_dim)
            return tokens
        else:
            # Single image: (B, C, H, W)
            return self._forward_images(images)
    
    def _forward_images(self, images: torch.Tensor) -> torch.Tensor:
        """Process batch of images"""
        B = images.shape[0]
        
        # Patch embedding: (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.patch_embed(images)
        
        # Flatten spatial dimensions: (B, embed_dim, H', W') -> (B, embed_dim, num_patches)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Layer norm
        x = self.norm(x)
        
        return x


class AudioEncoder(nn.Module):
    """
    Audio Encoder - converts audio waveforms to tokens
    
    Uses mel-spectrogram + CNN to extract features, then projects to token space.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 256,
        embed_dim: int = 768,
        max_audio_length: int = 1024,  # max tokens per audio clip
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.embed_dim = embed_dim
        self.max_audio_length = max_audio_length
        
        # Mel-spectrogram projection
        self.mel_proj = nn.Linear(n_mels, embed_dim)
        
        # Temporal CNN to reduce sequence length
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.GELU(),
        )
        
        # Position embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_audio_length, embed_dim) * 0.02
        )
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: (B, samples) raw audio waveform
        Returns:
            tokens: (B, T, embed_dim) where T <= max_audio_length
        """
        B = audio.shape[0]
        
        # Convert to mel-spectrogram (simplified - in practice use librosa or torchaudio)
        # For now, we'll use a learned projection from raw audio
        # In production, use proper mel-spectrogram computation
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Simple learned projection (replace with actual mel-spectrogram)
        # This is a placeholder - use torchaudio.transforms.MelSpectrogram in production
        x = audio.unsqueeze(1)  # (B, 1, samples)
        x = F.adaptive_avg_pool1d(x, self.max_audio_length * 2)  # Downsample
        x = x.repeat(1, self.n_mels, 1)  # (B, n_mels, T)
        x = x.transpose(1, 2)  # (B, T, n_mels)
        
        # Project to embedding space
        x = self.mel_proj(x)  # (B, T, embed_dim)
        
        # Temporal convolution
        x = x.transpose(1, 2)  # (B, embed_dim, T)
        x = self.temporal_conv(x)  # (B, embed_dim, T')
        x = x.transpose(1, 2)  # (B, T', embed_dim)
        
        # Truncate/pad to max_audio_length
        T = x.shape[1]
        if T > self.max_audio_length:
            x = x[:, :self.max_audio_length, :]
        elif T < self.max_audio_length:
            padding = torch.zeros(B, self.max_audio_length - T, self.embed_dim, device=x.device)
            x = torch.cat([x, padding], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Layer norm
        x = self.norm(x)
        
        return x


class MultimodalProjector(nn.Module):
    """
    Projects multimodal tokens into the same space as text tokens
    
    This allows vision/audio tokens to be processed by the same SRGI model
    as text tokens.
    """
    
    def __init__(
        self,
        vision_dim: int = 768,
        audio_dim: int = 768,
        text_dim: int = 768,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        output_dim = output_dim or text_dim
        
        # Projection layers
        self.vision_proj = nn.Linear(vision_dim, output_dim)
        self.audio_proj = nn.Linear(audio_dim, output_dim)
        
        # Optional: learnable modality embeddings
        self.vision_modality_embed = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)
        self.audio_modality_embed = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)
        
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        vision_tokens: Optional[torch.Tensor] = None,
        audio_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Project multimodal tokens to text token space
        
        Args:
            vision_tokens: (B, T_v, vision_dim) or None
            audio_tokens: (B, T_a, audio_dim) or None
        
        Returns:
            projected_tokens: (B, T_total, output_dim)
        """
        tokens_list = []
        
        if vision_tokens is not None:
            # Project vision tokens
            v_tokens = self.vision_proj(vision_tokens)
            v_tokens = v_tokens + self.vision_modality_embed
            tokens_list.append(v_tokens)
        
        if audio_tokens is not None:
            # Project audio tokens
            a_tokens = self.audio_proj(audio_tokens)
            a_tokens = a_tokens + self.audio_modality_embed
            tokens_list.append(a_tokens)
        
        if len(tokens_list) == 0:
            raise ValueError("At least one modality must be provided")
        
        # Concatenate all tokens
        projected = torch.cat(tokens_list, dim=1)
        
        # Layer norm
        projected = self.norm(projected)
        
        return projected


class UnifiedMultimodalEncoder(nn.Module):
    """
    Unified encoder that handles all modalities and projects to text space
    
    This is the main interface for multimodal SRGI.
    """
    
    def __init__(
        self,
        text_embed_dim: int = 768,
        vision_image_size: int = 224,
        vision_patch_size: int = 16,
        audio_sample_rate: int = 16000,
        audio_max_length: int = 1024,
    ):
        super().__init__()
        
        # Individual encoders
        self.vision_encoder = VisionDiffusionEncoder(
            image_size=vision_image_size,
            patch_size=vision_patch_size,
            embed_dim=text_embed_dim,
        )
        
        self.audio_encoder = AudioEncoder(
            sample_rate=audio_sample_rate,
            embed_dim=text_embed_dim,
            max_audio_length=audio_max_length,
        )
        
        # Projector (identity if dimensions match, otherwise learnable)
        self.projector = MultimodalProjector(
            vision_dim=text_embed_dim,
            audio_dim=text_embed_dim,
            text_dim=text_embed_dim,
            output_dim=text_embed_dim,
        )
    
    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images/video to tokens"""
        return self.vision_encoder(images)
    
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to tokens"""
        return self.audio_encoder(audio)
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode multimodal inputs and project to unified token space
        
        Args:
            images: (B, C, H, W) or (B, T, C, H, W) for video
            audio: (B, samples) raw audio waveform
        
        Returns:
            tokens: (B, T_total, embed_dim) ready for SRGI model
        """
        vision_tokens = None
        audio_tokens = None
        
        if images is not None:
            vision_tokens = self.encode_vision(images)
        
        if audio is not None:
            audio_tokens = self.encode_audio(audio)
        
        # Project to unified space
        unified_tokens = self.projector(
            vision_tokens=vision_tokens,
            audio_tokens=audio_tokens,
        )
        
        return unified_tokens

