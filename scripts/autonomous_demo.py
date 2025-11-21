"""
Demo script for autonomous curiosity-driven operation (Phase-5).

Run as:
python scripts/autonomous_demo.py

This demonstrates the complete mini-AGI loop with intrinsic curiosity,
self-driven exploration, and memory consolidation.
"""

import torch
import time
from nanochat.gpt import GPT, GPTConfig
from nanochat.autonomous import enable_autonomous_mode

def main():
    # Configure model with SRGI + autonomous mode
    config = GPTConfig()
    config.use_srgi = True  # Enable full SRGI architecture
    config.use_entangle = True  # Enable entanglement bottleneck
    config.use_ebm_hopfield = True  # Enable attractor memory
    config.autonomous_mode = True  # Enable autonomous curiosity loop

    # Autonomous settings
    config.curiosity_threshold = 0.1  # Lower = more curious
    config.nap_duration = 5  # Shorter nap for demo (normally 60 seconds)
    config.consolidation_interval = 30  # Shorter consolidation for demo (normally 3600 seconds)

    print("🤖 Initializing SRGI mini-AGI with autonomous curiosity...")
    print("This will run the complete Phase-5 autonomous loop.")
    print("Press Ctrl+C to stop the autonomous agent.")
    print()

    # Initialize model (load from checkpoint if available)
    model = GPT(config)

    # Check for checkpoint
    checkpoint_path = "checkpoints/srgi_checkpoint.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print(f"⚠️  No checkpoint found at {checkpoint_path}")
        print("Running with random weights - results will be nonsense but demonstrates the loop")
        print()

    model.eval()

    # Optional: Initialize webcam/audio streams (placeholder for now)
    webcam_stream = None  # Would be cv2.VideoCapture(0) in real implementation
    audio_stream = None   # Would be pyaudio stream in real implementation

    print("🚀 Starting autonomous operation...")
    print("The agent will now:")
    print("  - Monitor the environment for interesting patterns")
    print("  - Generate its own goals and questions")
    print("  - Execute actions to maximize information gain")
    print("  - Consolidate memories into stable attractors")
    print()

    # Enable autonomous mode - this runs forever until interrupted
    enable_autonomous_mode(model, config, webcam_stream, audio_stream)

if __name__ == "__main__":
    main()
