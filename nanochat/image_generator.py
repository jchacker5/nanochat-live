"""
Image/Video Generation for SRGI Mini-AGI

Implements multimodal output heads for image and video generation.
Uses diffusion models to generate images/videos from text prompts or latent representations.

This provides the "Image/Video head" from the architecture diagram.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
import math

# Optional imports for advanced generation
try:
    from diffusers import StableDiffusionPipeline, DiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    import PIL
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class SimpleImageGenerator(nn.Module):
    """
    Simple image generator using learned latents + decoder.

    This is a placeholder for more advanced diffusion models.
    In production, replace with Stable Diffusion or similar.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        latent_dim: int = 256,
        image_size: int = 256,
        channels: int = 3
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.channels = channels

        # Project from text embedding to image latent
        self.text_to_latent = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim * 16 * 16),  # 16x16 spatial latent
        )

        # Simple decoder (replace with VAE/Diffusion in production)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 2, 1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),         # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),          # 64x64 -> 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(64, channels, 4, 2, 1),      # 128x128 -> 256x256
            nn.Sigmoid(),  # Output in [0,1]
        )

    def forward(self, text_embedding: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate image from text embedding.

        Args:
            text_embedding: (B, embed_dim) text representation
            noise: Optional noise for stochastic generation

        Returns:
            image: (B, C, H, W) in [0,1]
        """
        B = text_embedding.shape[0]

        # Generate latent from text
        latent_flat = self.text_to_latent(text_embedding)  # (B, latent_dim * 256)
        latent = latent_flat.view(B, self.latent_dim, 16, 16)  # (B, latent_dim, 16, 16)

        # Add optional noise for variation
        if noise is not None:
            latent = latent + noise

        # Decode to image
        image = self.decoder(latent)  # (B, C, H, W)

        return image


class StableDiffusionGenerator:
    """
    Stable Diffusion image generator.

    Uses HuggingFace diffusers for high-quality image generation.
    """

    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", device: str = "auto"):
        self.model_id = model_id
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None

        if DIFFUSERS_AVAILABLE:
            try:
                print(f"🎨 Loading Stable Diffusion: {model_id}")
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                self.pipe = self.pipe.to(self.device)

                # Speed optimizations
                if self.device == "cuda":
                    self.pipe.enable_attention_slicing()
                    # self.pipe.enable_xformers_memory_efficient_attention()  # Optional

                print("✅ Stable Diffusion ready")
            except Exception as e:
                print(f"❌ Stable Diffusion failed to load: {e}")
                print("   Falling back to simple generator")
        else:
            print("⚠️  Diffusers not available. Install with: pip install diffusers transformers")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_images: int = 1,
        height: int = 512,
        width: int = 512,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20
    ) -> List[Image.Image]:
        """
        Generate images from text prompt.

        Args:
            prompt: Text description
            negative_prompt: What to avoid
            num_images: Number of images to generate
            height/width: Image dimensions
            guidance_scale: Classifier-free guidance strength
            num_inference_steps: Number of denoising steps

        Returns:
            List of PIL Images
        """
        if not self.pipe:
            return []

        try:
            with torch.autocast(self.device):
                images = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_images,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                ).images

            return images

        except Exception as e:
            print(f"❌ Image generation failed: {e}")
            return []


class VideoGenerator:
    """
    Video generation from image sequences.

    Creates simple videos by interpolating between generated images.
    In production, replace with proper video generation models.
    """

    def __init__(self, fps: int = 30):
        self.fps = fps

    def create_video_from_images(
        self,
        images: List[Image.Image],
        output_path: str,
        duration_per_image: float = 2.0
    ) -> bool:
        """
        Create video from sequence of images.

        Args:
            images: List of PIL Images
            output_path: Output video file path
            duration_per_image: Seconds per image

        Returns:
            Success status
        """
        if not PIL_AVAILABLE:
            print("❌ PIL not available for video generation")
            return False

        try:
            # Simple frame duplication for now
            # In production, use proper video encoding libraries
            frames = []
            frames_per_image = int(self.fps * duration_per_image)

            for img in images:
                for _ in range(frames_per_image):
                    frames.append(img)

            # Save as GIF for now (replace with proper video encoding)
            if output_path.endswith('.gif'):
                frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=int(1000 / self.fps),
                    loop=0
                )
                return True

            print("⚠️  Video generation limited to GIF format")
            return False

        except Exception as e:
            print(f"❌ Video generation failed: {e}")
            return False


class MultimodalOutputHeads(nn.Module):
    """
    Parallel multimodal output heads.

    Implements the "Multimodal output heads (parallel)" from the architecture diagram.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Text head (standard language model head)
        self.text_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Image generation head
        if DIFFUSERS_AVAILABLE:
            self.image_generator = StableDiffusionGenerator()
        else:
            self.image_generator = SimpleImageGenerator(embed_dim=config.n_embd)

        # Video generation (optional)
        self.video_generator = VideoGenerator()

    def forward(
        self,
        hidden_state: torch.Tensor,
        generate_images: bool = False,
        generate_video: bool = False,
        image_prompt: Optional[str] = None,
        video_prompt: Optional[str] = None
    ) -> dict:
        """
        Generate multimodal outputs in parallel.

        Args:
            hidden_state: Final hidden state from SRGI (B, T, n_embd)
            generate_images: Whether to generate images
            generate_video: Whether to generate video
            image_prompt: Text prompt for image generation
            video_prompt: Text prompt for video generation

        Returns:
            Dict with text_logits, images, video_path
        """
        # Text generation head
        text_logits = self.text_head(hidden_state)  # (B, T, vocab_size)

        result = {
            'text_logits': text_logits,
            'images': None,
            'video_path': None
        }

        # Image generation (if requested)
        if generate_images and image_prompt:
            if hasattr(self.image_generator, 'generate'):
                # Stable Diffusion
                images = self.image_generator.generate(
                    prompt=image_prompt,
                    num_images=1,
                    height=512,
                    width=512
                )
                result['images'] = images if images else None
            else:
                # Simple generator (placeholder)
                # Use last token embedding as text representation
                text_embedding = hidden_state[:, -1, :]  # (B, n_embd)
                noise = torch.randn_like(text_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 16, 16))
                image_tensor = self.image_generator(text_embedding, noise)
                # Convert to PIL (placeholder)
                result['images'] = [self._tensor_to_pil(image_tensor[0])]

        # Video generation (if requested and images available)
        if generate_video and result['images'] and len(result['images']) > 1:
            video_path = f"/tmp/srgi_video_{int(torch.randint(0, 1000, (1,)).item())}.gif"
            success = self.video_generator.create_video_from_images(
                result['images'],
                video_path
            )
            if success:
                result['video_path'] = video_path

        return result

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image (placeholder implementation)."""
        if not PIL_AVAILABLE:
            return None

        # Convert from (C, H, W) to PIL
        tensor = tensor.detach().cpu()
        tensor = torch.clamp(tensor * 255, 0, 255).byte()
        array = tensor.permute(1, 2, 0).numpy()
        return Image.fromarray(array)


# Convenience functions
def generate_image_from_text(
    prompt: str,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    **kwargs
) -> Optional[List[Image.Image]]:
    """Generate images from text prompt."""
    if DIFFUSERS_AVAILABLE:
        generator = StableDiffusionGenerator(model_id)
        return generator.generate(prompt, **kwargs)
    else:
        print("❌ Image generation requires diffusers. Install with: pip install diffusers transformers")
        return None

def generate_video_from_prompts(
    prompts: List[str],
    output_path: str = "output.mp4",
    **kwargs
) -> bool:
    """Generate video from sequence of text prompts."""
    # Generate images for each prompt
    images = []
    for prompt in prompts:
        imgs = generate_image_from_text(prompt, **kwargs)
        if imgs:
            images.extend(imgs)

    if len(images) < 2:
        print("❌ Need at least 2 images for video generation")
        return False

    # Create video
    video_gen = VideoGenerator()
    return video_gen.create_video_from_images(images, output_path)
