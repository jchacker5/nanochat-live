# nanochat/autonomous.py

# Autonomous Curiosity & Consolidation Loop (Phase-5)

# Active Inference / Free Energy Principle implementation
# References:
#   - Friston et al. (2017–2025) Active Inference reviews
#   - LeCun (2022–2025) JEPA & self-supervised world models
#   - Schmidhuber (2010–2025) Formal Theory of Fun & Artificial Curiosity
#   - DeepMind Adaptive Agent (2025) intrinsic motivation cycles

import torch
import torch.nn as nn
import time
from typing import Optional, Tuple, Dict, Any


class CuriosityEngine(nn.Module):
    """
    Active Inference / Free Energy Principle loop

    Implements intrinsic curiosity drive that predicts information gain
    and autonomously explores high-entropy states.

    The key insight: predict how much the entangled entropy will change
    after an action, then choose actions that maximize expected surprise.
    """

    def __init__(self, config):
        super().__init__()

        # Intrinsic curiosity: predict how much the entangled entropy will change
        self.entropy_predictor = nn.Sequential(
            nn.Linear(config.n_embd * 2, 256), nn.ReLU(),
            nn.Linear(256, 1)  # predicted ΔS_vN after action
        )

        # Goal generator: sample high-predicted-entropy states as subgoals
        # Reuse the entangled bottleneck from Phase-4
        from nanochat.entangle import EntangledBottleneck
        self.goal_sampler = EntangledBottleneck(config.n_embd, bond_dim=32)

        # Action space: define what the agent can do
        self.n_actions = 32  # Number of possible actions to consider

        # Curiosity threshold for "nap" behavior (from config)
        self.surprise_threshold = getattr(config, 'curiosity_threshold', 0.1)

    def forward(self, current_state_complex: torch.Tensor,
                webcam_frame: Optional[torch.Tensor] = None,
                audio_frame: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute intrinsic curiosity and select next action.

        Args:
            current_state_complex: (B, T, n_embd, 2) current complex state
            webcam_frame: optional webcam input for multimodal perception
            audio_frame: optional audio input for multimodal perception

        Returns:
            surprise: scalar tensor measuring current state novelty
            best_action_idx: index of action that maximizes expected information gain
        """
        # 1. Predict surprise (negative log-likelihood + entropy change)
        state_mean = current_state_complex.mean(dim=1)  # (B, n_embd, 2)
        state_flat = state_mean.view(state_mean.shape[0], -1)  # (B, n_embd*2)
        surprise = self.entropy_predictor(state_flat)  # (B, 1)

        # 2. Imagine N possible actions (or camera movements, questions, tool calls)
        # Sample from the entangled bottleneck to generate possible future states
        imagined_states = self.goal_sampler(
            current_state_complex.unsqueeze(0).repeat(self.n_actions, 1, 1, 1)
        )[0]  # Take just the states, ignore entropy for now

        # 3. Choose action that maximizes expected information gain
        predicted_entropy = []
        for i in range(self.n_actions):
            s = imagined_states[i:i+1]  # (1, T, n_embd, 2)
            # Reuse the entangle method to compute entropy of imagined state
            _, entropy = self.goal_sampler(s)
            predicted_entropy.append(entropy)

        predicted_entropy = torch.stack(predicted_entropy)  # (n_actions,)
        best_action_idx = torch.argmax(predicted_entropy)

        return surprise.squeeze(), best_action_idx

    def update_model_weights(self, learning_rate: float = 1e-5):
        """
        THEORETICAL: Enable continuous weight updates through online learning.

        CURRENT STATUS: NOT IMPLEMENTED
        This would require significant architectural changes including:
        1. Online gradient computation from prediction errors
        2. Continuous optimizer (Adam, SGD, etc.)
        3. Catastrophic forgetting prevention (regularization)
        4. Meta-learning components for rapid adaptation
        5. Memory replay mechanisms

        Current SRGI only consolidates attractor memories.
        """
        print("⚠️  Continuous weight updates not implemented in current SRGI")
        print("   Only attractor memory consolidation occurs")
        print("   True online learning would require major architectural extension")

        # Theoretical implementation concept:
        # if self.online_optimizer:
        #     loss = self.compute_adaptation_loss(current_experience)
        #     loss.backward()
        #     self.online_optimizer.step()
        #     self.online_optimizer.zero_grad()

    def execute_action(self, action_idx: torch.Tensor) -> torch.Tensor:
        """
        Execute the chosen action in the real world.
        This is a placeholder - in practice this would interface with:
        - Robot motors
        - Camera servos
        - Tool execution APIs
        - Question generation systems

        Args:
            action_idx: which action to execute

        Returns:
            new_state: resulting state after action execution
        """
        # Placeholder implementation
        # In a real system, this would:
        # - Move camera to new position
        # - Ask a question via text generation
        # - Execute a tool call
        # - Move robot to new location

        print(f"Executing action {action_idx.item()}: [placeholder - implement robot/webcam/tool interface]")

        # Return a dummy new state for now
        # In practice, this would capture new sensor data
        return torch.randn(1, 10, 768, 2)  # Dummy complex state


class AutonomousAgent:
    """
    Complete autonomous agent that runs the curiosity loop 24/7.

    Integrates with the main SRGI model to provide intrinsic motivation
    and memory consolidation.
    """

    def __init__(self, model, config, tokenizer=None):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.curiosity = CuriosityEngine(config)

        # Autonomous operation settings (from config)
        self.nap_duration = getattr(config, 'nap_duration', 60)  # seconds to sleep when bored
        self.consolidation_interval = getattr(config, 'consolidation_interval', 3600)  # seconds between memory consolidation

        # Lifelong learning (Phase-6)
        self.lifelong_learner = None
        if getattr(config, 'enable_lifelong', False):
            from nanochat.lifelong import LifelongLearner
            self.lifelong_learner = LifelongLearner(model, config, tokenizer)
            print("🧠 Lifelong learning enabled - weights will evolve continuously!")
        self.last_consolidation = time.time()

    def run_autonomous_loop(self, webcam_stream=None, audio_stream=None):
        """
        Main autonomous loop - runs forever until interrupted.

        Args:
            webcam_stream: optional webcam input stream
            audio_stream: optional audio input stream
        """
        print("🤖 Starting autonomous curiosity loop...")

        while True:
            try:
                # Get current state (would come from sensors/environment)
                current_state = self._get_current_state(webcam_stream, audio_stream)

                # Compute curiosity and choose action
                surprise, action_idx = self.curiosity(current_state, webcam_stream, audio_stream)

                print(".3f")

                if surprise < self.curiosity.surprise_threshold:
                    print(f"😴 World is boring (surprise = {surprise:.3f}), taking a nap...")
                    time.sleep(self.nap_duration)
                else:
                    print(f"🎯 World is interesting! Executing action {action_idx.item()}...")

                    # Execute action and get new state
                    new_state = self.curiosity.execute_action(action_idx)

                    # Create experience record for lifelong learning
                    experience = {
                        'state': current_state,
                        'action': action_idx.item(),
                        'surprise': surprise.item(),
                        'next_state': new_state,
                        'timestamp': time.time()
                    }

                    # Lifelong learning step (if enabled)
                    if self.lifelong_learner:
                        self.lifelong_learner.step(experience)

                    # Store in episodic memory and potentially consolidate
                    self._store_episodic_memory(new_state)

                    # Periodic consolidation into slow attractors
                    if time.time() - self.last_consolidation > self.consolidation_interval:
                        self._consolidate_memory()
                        self.last_consolidation = time.time()

            except KeyboardInterrupt:
                print("🛑 Autonomous loop interrupted by user")
                break
            except Exception as e:
                print(f"⚠️  Error in autonomous loop: {e}")
                time.sleep(5)  # Brief pause before retrying

    def _get_current_state(self, webcam_stream, audio_stream):
        """
        Get current state from sensors/environment.
        Placeholder - in practice would interface with actual sensors.
        """
        # Placeholder: generate random complex state
        # In practice: process webcam/audio through multimodal encoder
        return torch.randn(1, self.config.sequence_len, self.config.n_embd, 2)

    def _store_episodic_memory(self, new_state):
        """
        Store new experience in episodic memory.
        """
        # Access the model's attractor memory
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # Find the SRGI block with attractor memory
            for block in self.model.transformer.h:
                if hasattr(block, 'memory') and block.memory is not None:
                    # Store the new state in attractor memory
                    block.memory.store_episodic(new_state)
                    print("💾 Stored experience in attractor memory")
                    break

    def _consolidate_memory(self):
        """
        Run memory consolidation - compress episodic memories into stable attractors.
        """
        print("🧠 Running memory consolidation...")

        # Access attractor memory for consolidation
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            for block in self.model.transformer.h:
                if hasattr(block, 'memory') and block.memory is not None:
                    # Trigger consolidation (implementation depends on memory type)
                    if hasattr(block.memory, 'consolidate'):
                        block.memory.consolidate()
                    print("✨ Memory consolidation complete")


# Integration function for easy use
def enable_autonomous_mode(model, config, webcam_stream=None, audio_stream=None):
    """
    Enable autonomous curiosity-driven operation.

    Args:
        model: SRGI model instance
        config: model configuration
        webcam_stream: optional webcam input
        audio_stream: optional audio input
    """
    agent = AutonomousAgent(model, config)
    agent.run_autonomous_loop(webcam_stream, audio_stream)
