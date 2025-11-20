# NanoChat-Live: Multimodal Streaming Agent Fork of NanoChat

![nanochat logo](dev/nanochat.png)

> The best multimodal ChatGPT that $100 can buy – now with live streaming for vision/audio inputs, latent chain-of-thought reasoning, and multi-head outputs (text/audio/diffusion/actions).

This repo is a fork of [karpathy/nanochat](https://github.com/karpathy/nanochat), extending the full-stack implementation of a minimal, hackable LLM like ChatGPT with live, persistent agent capabilities. Inspired by Andrej Karpathy's October 20, 2025 tweet praising DeepSeek-OCR for pixel-based inputs (calling tokenizers "ugly" and expressing interest in an "image-input-only version of nanochat"), this fork shifts toward pixel intelligence: streaming webcam/audio inputs, unified latent processing (with implicit CoT in transformer hiddens), and hybrid decoders for dynamic outputs. NanoChat-Live runs on consumer hardware, maintaining the original's low-cost training (~$100 for base, +$100 for multimodal extensions) and reproducibility.

NanoChat-Live turns one-shot chat into a continuous loop: The model "sees" via webcam, "hears" via mic, reasons in latents, and responds via text, synthesized audio, visual refinements, or tool actions. It's designed for real-time agents, while keeping the codebase clean and dependency-lite.

## 🌀 NEW: Spin-Resonant Geometric Intelligence (SRGI)

**We're enhancing NanoChat with physics-inspired neural architecture for better memory and reasoning!**

This fork now implements components from **Spin-Resonant Geometric Intelligence (SRGI)**, a novel architecture that augments transformers with:

- **🎯 Resonant State-Space Layers (R-SSM)**: Lightly damped oscillators with complex eigenvalues that preserve information through phase dynamics. Unlike standard transformers that struggle with long-range dependencies, R-SSM maintains stable resonances that encode temporal patterns without gradient vanishing.

- **🌊 Phase-Aware Dynamics**: Information routing based on phase synchronization, inspired by neural oscillations in the brain. Tokens that are "in phase" communicate preferentially, enabling selective attention and more coherent reasoning chains.

- **📐 Geometric State Spaces**: Future work will add hyperbolic embeddings for hierarchies and toroidal components for periodic phenomena—encoding structure directly in the latent geometry.

- **🧲 Attractor Memory**: Planned integration of modern Hopfield networks for stable, re-enterable memory states that reduce hallucination and improve long-context consistency.

### Why SRGI?

Standard transformers excel at pattern matching but struggle with:
- **Persistent memory** beyond context windows
- **Binding problems** in multi-entity reasoning
- **Phase/timing** information in sequences
- **Systematic generalization** to novel compositions

SRGI addresses these by treating neural computation as a physical system with resonance, geometry, and symmetry—offering **structure over scale**. The Resonant SSM layer is our Phase-1 implementation, providing:

✅ **Stable long-range dependencies** via unitary-like dynamics
✅ **Phase-based information routing** for selective communication
✅ **Reduced gradient issues** through spectral constraints
✅ **Interpretable dynamics** via phase/magnitude visualization

**Current Status**: We've implemented `StableResonantSSM` in `nanochat/ssm.py` as a drop-in module. It can be added to any transformer block with minimal overhead (~1.3× FLOPs). Training and evaluation scripts demonstrate memory improvements on long-context tasks.

**Paper Reference**: Defendre, J. (2025). *Spin-Resonant Geometric Intelligence: Unifying Geometry, Resonance, and Neural Computation for Scalable Intelligence.* Draft v0.2.

See `nanochat/ssm.py` for implementation details and `scripts/ssm_demo.py` for usage examples.

---

## Talk to it

For the original text-only nanochat, see upstream. For NanoChat-Live, run the extended web UI with live mode (after training or loading a checkpoint):

```bash
python -m scripts.chat_web --live
```

This launches a server with webcam/audio streaming. Visit the URL (e.g., http://localhost:8000/) to interact in real-time – the model perceives you, updates its belief state, and selects output heads dynamically.

Currently, no hosted demo (fork-specific), but you can train and run locally. The d32 model (1.9B params, ~$800 train) now supports multimodal with minor VRAM tweaks.

## Quick Start

Follow the original quick start for base training, then enable live mode. On an 8XH100 node (~$24/hr), the speedrun takes ~4 hours:

```bash
bash speedrun.sh
```

For live multimodal:

1. Activate env: `source .venv/bin/activate`
2. Run CLI with live: `python scripts/chat_cli.py --checkpoint checkpoints/d20.pt --live`
   - Webcam/mic activate; model processes streams and responds (e.g., "I see you waving – hello!").
3. Or web: `python -m scripts.chat_web --live`
   - Browser UI with real-time video/audio feed.

To train with multimodal data (e.g., video-audio pairs):

```bash
python scripts/live_train.py  # Adapts base_train for latents
```

Expect kindergartener-level responses, now with visual/audio awareness. Ask it to describe what it sees or respond to sounds.

## Bigger Models

Same as upstream, but multimodal adds ~20% compute. For d26 (GPT-2 level):

- Increase data shards to include video/audio (e.g., synthetic from `dev/gen_synthetic_data.py`).
- Reduce `--device_batch_size` to 16 for VRAM.
- Train: ~14 hours on 8XH100.

Runs on A100s (slower), single GPUs (8x longer), or CPU/MPS for tiny models (see `dev/runcpu.sh`).

## Running on CPU / MPS

Same as upstream, with multimodal caveats: Vision/audio capture works, but latents/diffusion are slow without GPU. Use `dev/runcpu.sh` for small-scale testing.

## Customization

See upstream guide for identity infusion. For multimodal:

- Add synthetic video/audio data in `dev/gen_synthetic_data.py` (e.g., scripted scenes with transcripts).
- Mix into midtraining/SFT: Teach vision tasks like object description.
- Extend abilities: See upstream "counting r in strawberry" guide; adapt for visual counting (e.g., "how many fingers?").

## Questions

Package the repo as upstream suggests, or use DeepWiki on your fork. Ask LLMs about new `live/` features.

## Tests

Original tests + new: Run `python -m pytest tests/test_rustbpe.py -v -s`. Add multimodal tests soon.

## File Structure

Builds on upstream, with new `live/` for streaming extensions:

```
.
├── LICENSE
├── README.md  # This file (fork-specific)
├── dev
│   ├── gen_synthetic_data.py       # Extended for video/audio synthetics
│   ├── generate_logo.html
│   ├── nanochat.png
│   ├── repackage_data_reference.py # Supports multimodal shards
│   └── runcpu.sh                   # Now with live mode flags
├── live  # New: Multimodal streaming
│   ├── __init__.py
│   ├── capture.py                 # Webcam/mic rolling canvas
│   ├── live_agent.py              # Persistent loop + heads
│   └── vde.py                     # Vision Diffusion Encoder (DeepSeek-OCR inspired)
├── nanochat
│   ├── __init__.py
│   ├── adamw.py
│   ├── checkpoint_manager.py
│   ├── common.py
│   ├── configurator.py
│   ├── core_eval.py               # Extended for multimodal evals
│   ├── dataloader.py              # Supports latent loading
│   ├── dataset.py                 # Multimodal data download
│   ├── engine.py                  # Online KV cache for streaming
│   ├── execution.py
│   ├── gpt.py                     # Latent input projection + hidden exposure
│   ├── logo.svg
│   ├── loss_eval.py
│   ├── muon.py
│   ├── report.py
│   ├── ssm.py                     # NEW: Stable Resonant SSM (SRGI Phase-1)
│   ├── tokenizer.py               # Bypass for pixel-only
│   └── ui.html                    # Updated for live UI
├── pyproject.toml  # Added opencv-python, pyaudio, torchaudio, gtts
├── run1000.sh
├── rustbpe
│   ├── Cargo.lock
│   ├── Cargo.toml
│   ├── README.md
│   └── src
│       └── lib.rs
├── scripts
│   ├── base_eval.py
│   ├── base_loss.py
│   ├── base_train.py
│   ├── chat_cli.py                # --live flag
│   ├── chat_eval.py               # Visual/audio tasks
│   ├── chat_rl.py
│   ├── chat_sft.py
│   ├── chat_web.py                # --live flag
│   ├── live_train.py              # New: Multimodal training
│   ├── mid_train.py
│   ├── tok_eval.py
│   └── tok_train.py
├── speedrun.sh
├── tasks
│   ├── arc.py
│   ├── common.py
│   ├── customjson.py
│   ├── gsm8k.py
│   ├── humaneval.py
│   ├── mmlu.py
│   ├── smoltalk.py
│   └── spellingbee.py              # Potential visual extensions
├── tests
│   └── test_rustbpe.py
└── uv.lock
```

## Contributing

Build on upstream goals: Improve micro multimodal agents for <$1000. Keep minimal – no giant configs. PRs welcome for live features (e.g., better VDE, RL for latency).

LLM policy: Same as upstream.

## Acknowledgements

- Upstream: Andrej Karpathy and contributors.
- Inspiration: Karpathy's DeepSeek-OCR tweet.
- Thanks: HuggingFace for datasets; Lambda for compute.

## Cite

For the fork:

```bibtex
@misc{nanochat-live,
  author = {jchacker5}, 
  title = {NanoChat-Live: Multimodal Streaming Fork of NanoChat},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/jchacker5/nanochat-live}
}
```

Original: See upstream.

## License

MIT
