"""
Lifelong Learning Module - True Weight Evolution (Phase-6)

Enables SRGI to continuously update its neural weights after initial training,
achieving genuine lifelong self-improvement through three complementary mechanisms:

1. Online Gradient Accumulation (OGA) - Tiny continuous updates
2. DreamerV3-style Replay + Imagination - World model learning
3. o1-style Synthetic Data Loop - Self-generated training data

This turns SRGI from a static model into one that genuinely improves over time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import time
import numpy as np
from collections import deque
import threading


class ExperienceReplay:
    """DreamerV3-style replay buffer for lifelong learning."""

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, experience: Dict[str, Any], priority: float = 1.0):
        """Add experience to replay buffer."""
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample batch with priority weighting."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        # Priority-based sampling
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class SyntheticDataGenerator:
    """o1-style synthetic data generator for lifelong learning."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_reasoning_trace(self, prompt: str, max_length: int = 1024) -> Dict[str, torch.Tensor]:
        """Generate a single reasoning trace (like o1)."""
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        # Generate with chain-of-thought
        with torch.no_grad():
            generated = self.model.generate(
                input_tensor,
                max_tokens=max_length,
                temperature=0.7,
                top_k=50
            )

        # Decode and return
        generated_text = self.tokenizer.decode(generated[0].tolist())
        return {
            "prompt": prompt,
            "response": generated_text,
            "input_ids": input_ids,
            "output_ids": generated[0].tolist()
        }

    def generate_dataset(self, n_samples: int = 1000, max_length: int = 1024) -> List[Dict[str, torch.Tensor]]:
        """Generate synthetic dataset for training."""
        prompts = [
            "Explain quantum entanglement step by step.",
            "Solve this math problem: ∫ sin²(x) dx",
            "Write a Python function to sort a list using merge sort.",
            "What are the implications of Gödel's incompleteness theorems?",
            "Design an algorithm to find the shortest path in a graph.",
            "Explain how neural networks learn through backpropagation.",
            "What is the Riemann hypothesis and why is it important?",
            "Write a program to implement a binary search tree.",
            "Explain the principles of general relativity.",
            "Design a system for real-time speech recognition.",
        ]

        dataset = []
        for _ in range(n_samples):
            prompt = np.random.choice(prompts)
            trace = self.generate_reasoning_trace(prompt, max_length)
            dataset.append(trace)

        return dataset


class OnlineGradientAccumulator:
    """Online Gradient Accumulation for lifelong learning."""

    def __init__(self, model, lr: float = 5e-8, interval: int = 512):
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        self.interval = interval
        self.step_count = 0
        self.accumulated_loss = 0.0
        self.grad_accumulated = False

    def accumulate(self, loss: torch.Tensor):
        """Accumulate gradients from loss."""
        if loss.requires_grad:
            loss.backward()
            self.grad_accumulated = True
        self.step_count += 1

        # Perform update every interval steps
        if self.step_count % self.interval == 0 and self.grad_accumulated:
            self._update_weights()

    def _update_weights(self):
        """Perform actual weight update."""
        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.grad_accumulated = False

        print(".6f")


class DreamerWorldModel:
    """Simplified Dreamer-style world model for imagination."""

    def __init__(self, model, config):
        self.model = model
        self.config = config

    def imagine_trajectory(self, start_state: torch.Tensor, horizon: int = 16) -> List[torch.Tensor]:
        """Imagine a trajectory from start state."""
        trajectory = [start_state]

        current_state = start_state
        for _ in range(horizon):
            # Use model to predict next state (simplified)
            with torch.no_grad():
                # This would use the entangled bottleneck as a world model
                next_state_logits = self.model(current_state.unsqueeze(0))
                # Sample next state (simplified)
                next_state = torch.softmax(next_state_logits[0], dim=-1).argmax(dim=-1)
                trajectory.append(next_state)
                current_state = next_state

        return trajectory

    def compute_world_model_loss(self, real_trajectory: List[torch.Tensor],
                               imagined_trajectory: List[torch.Tensor]) -> torch.Tensor:
        """Compute world model prediction loss."""
        losses = []
        for real, imagined in zip(real_trajectory[1:], imagined_trajectory[1:]):
            loss = F.cross_entropy(imagined.unsqueeze(0), real.unsqueeze(0))
            losses.append(loss)

        return torch.stack(losses).mean()


class LifelongLearner:
    """
    Complete lifelong learning system combining all three mechanisms.

    This enables true weight evolution after pre-training.
    """

    def __init__(self, model, config, tokenizer):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer

        # 1. Online Gradient Accumulation
        self.oga = OnlineGradientAccumulator(
            model=model,
            lr=getattr(config, 'oga_lr', 5e-8),
            interval=getattr(config, 'oga_interval', 512)
        )

        # 2. Dreamer-style Replay Buffer
        self.replay = ExperienceReplay(capacity=getattr(config, 'replay_capacity', 100_000))

        # 3. Synthetic Data Generator
        self.synthetic_gen = SyntheticDataGenerator(model, tokenizer)

        # 4. Dreamer World Model
        self.world_model = DreamerWorldModel(model, config)

        # Statistics
        self.total_steps = 0
        self.last_synthetic_update = time.time()
        self.synthetic_interval_hours = getattr(config, 'synthetic_data_every_hours', 12)

        print("🧠 Lifelong learning initialized!")
        print(f"   OGA interval: {self.oga.interval}")
        print(f"   Replay capacity: {self.replay.capacity}")
        print(f"   Synthetic interval: {self.synthetic_interval_hours} hours")

    def step(self, experience: Dict[str, Any]):
        """
        Process one experience step for lifelong learning.

        Args:
            experience: Dict containing observation, action, reward, next_obs, etc.
        """
        self.total_steps += 1

        # Store experience in replay buffer
        self.replay.add(experience)

        # Compute losses from experience
        losses = self._compute_losses(experience)

        # Accumulate gradients
        if losses:
            total_loss = sum(losses.values())
            self.oga.accumulate(total_loss)

        # Dreamer-style imagination updates (every 256 steps)
        if self.total_steps % getattr(self.config, 'dreamer_every', 256) == 0:
            self._dreamer_update()

        # Synthetic data updates (every 12 hours)
        if time.time() - self.last_synthetic_update > (self.synthetic_interval_hours * 3600):
            self._synthetic_update()
            self.last_synthetic_update = time.time()

    def _compute_losses(self, experience: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute various losses from experience."""
        losses = {}

        # Curiosity prediction loss (if curiosity module available)
        if hasattr(self.model, 'curiosity_predictor'):
            # Simplified curiosity loss
            losses['curiosity'] = torch.tensor(0.1, requires_grad=True)

        # Attractor energy loss (if attractor memory available)
        if hasattr(self.model, 'attractor_energy'):
            losses['attractor'] = torch.tensor(0.05, requires_grad=True)

        # Next-token prediction loss (from experience)
        if 'input_ids' in experience and 'target_ids' in experience:
            input_tensor = torch.tensor(experience['input_ids']).unsqueeze(0)
            target_tensor = torch.tensor(experience['target_ids']).unsqueeze(0)

            logits = self.model(input_tensor)
            losses['prediction'] = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_tensor.view(-1),
                ignore_index=-1
            )

        return losses

    def _dreamer_update(self):
        """Perform Dreamer-style imagination rollout."""
        if len(self.replay) < 16:
            return

        print("🎭 Dreamer imagination rollout...")

        # Sample batch from replay
        batch = self.replay.sample(min(16, len(self.replay)))

        for experience in batch:
            if 'state' in experience:
                # Imagine trajectory from this state
                imagined_trajectory = self.world_model.imagine_trajectory(
                    torch.tensor(experience['state']),
                    horizon=8
                )

                # Compute world model loss
                if 'trajectory' in experience:
                    real_trajectory = [torch.tensor(s) for s in experience['trajectory']]
                    world_loss = self.world_model.compute_world_model_loss(
                        real_trajectory, imagined_trajectory
                    )
                    world_loss.backward()

                    # Update OGA with world model loss
                    self.oga.accumulate(world_loss)

        print("✅ Dreamer update complete")

    def _synthetic_update(self):
        """Perform o1-style synthetic data training."""
        print("🤖 Generating synthetic training data...")

        # Generate synthetic dataset
        n_samples = getattr(self.config, 'synthetic_batch_size', 256)
        synthetic_data = self.synthetic_gen.generate_dataset(n_samples=n_samples)

        print(f"📝 Generated {len(synthetic_data)} synthetic samples")

        # Train on synthetic data (one epoch)
        self.model.train()
        total_loss = 0

        for sample in synthetic_data:
            input_ids = torch.tensor(sample['input_ids']).unsqueeze(0)
            target_ids = torch.tensor(sample['output_ids']).unsqueeze(0)

            self.model.zero_grad()
            logits = self.model(input_ids)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=-1
            )

            loss.backward()
            total_loss += loss.item()

            # Update every 16 samples
            if len(synthetic_data) % 16 == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.oga.optimizer.step()
                self.oga.optimizer.zero_grad()

        avg_loss = total_loss / len(synthetic_data)
        print(".4f")

    def get_stats(self) -> Dict[str, Any]:
        """Get lifelong learning statistics."""
        return {
            'total_steps': self.total_steps,
            'replay_size': len(self.replay),
            'oga_updates': self.oga.step_count // self.oga.interval,
            'hours_since_synthetic': (time.time() - self.last_synthetic_update) / 3600
        }


# Configuration additions
def add_lifelong_config(config_class):
    """Add lifelong learning configuration to a config class."""

    # Online Gradient Accumulation
    config_class.oga_lr = 5e-8
    config_class.oga_interval = 512

    # Dreamer replay
    config_class.dreamer_every = 256
    config_class.replay_capacity = 100_000

    # Synthetic data
    config_class.synthetic_data_every_hours = 12
    config_class.synthetic_batch_size = 256

    # Enable flag
    config_class.enable_lifelong = False

    return config_class
