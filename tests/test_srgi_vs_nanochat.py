"""
Test comparing SRGI (Spinor-Resonant Geometric Intelligence) to vanilla Transformer.

This test simulates the core differences between:
1. Original NanoChat: Standard decoder-only Transformer (GPT-style)
2. SRGI: Physics/neuroscience-inspired augmentations including:
   - Resonant SSM layers (lightly damped oscillators for long-term memory)
   - Phase-aware attention (coherence gating for selective binding)
   - Geometric bottlenecks (hyperbolic+toroidal for hierarchy/periodicity)

Benchmark: Needle-in-a-Haystack (NIAH)
- Train on random sequences with a "needle" token hidden early
- Measure recall accuracy at the end (predicting the needle after "haystack" noise)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Toy Vanilla Transformer (mimicking original NanoChat core)
class VanillaTransformer(nn.Module):
    def __init__(self, vocab_size=100, d_model=64, nhead=4, num_layers=2, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x) + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        return self.fc_out(x)

# Toy SRGI: Add simple resonant SSM and phase modulation
class SimpleSSM(nn.Module):
    def __init__(self, state_dim=32, d_model=64):
        super().__init__()
        self.A = nn.Parameter(torch.randn(state_dim) * 0.1 - 0.01)  # Lightly damped
        self.B = nn.Parameter(torch.randn(state_dim, d_model))
        self.C = nn.Parameter(torch.randn(d_model, state_dim))

    def forward(self, h_prev, u):
        A_disc = torch.exp(self.A)  # Simple discretization
        h = A_disc.unsqueeze(0) * h_prev + (self.B @ u.t()).t()  # Batch handling
        return h, (self.C @ h.t()).t()

class SRGIToy(nn.Module):
    def __init__(self, vocab_size=100, d_model=64, nhead=4, num_layers=2, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.ssm = SimpleSSM()
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.phase = nn.Parameter(torch.randn(max_len))  # Toy phase

    def forward(self, x):
        emb = self.embed(x) + self.pos_embed[:, :x.size(1), :]
        # Simple phase-aware modulation
        phase_mod = 1 + 0.1 * torch.cos(self.phase[:x.size(1)]).unsqueeze(0).unsqueeze(-1)
        emb = emb * phase_mod
        x_trans = self.transformer(emb)
        # Add SSM residual for memory preservation
        h = torch.zeros(x.size(0), 32, device=x.device)
        for t in range(x.size(1)):
            h, out = self.ssm(h, x_trans[:, t, :])
            x_trans[:, t, :] += out
        return self.fc_out(x_trans)

# Synthetic NIAH benchmark: Hide needle early, predict it after haystack
def generate_data(seq_len=256, vocab=100):
    hay = np.random.randint(0, vocab, seq_len - 2)
    needle_pos = np.random.randint(0, seq_len - 2)
    needle = np.random.randint(0, vocab)
    seq = np.concatenate([hay[:needle_pos], [needle], hay[needle_pos:], [needle]])
    seq = torch.tensor(seq).unsqueeze(0)  # Batch dim 1
    return seq[:, :-1], seq[:, 1:]

# Train and evaluate recall
def train_and_test(model, epochs=50, lr=0.01, seq_len=256):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    recalls = []
    for ep in range(epochs):
        x, y = generate_data(seq_len)
        out = model(x)
        loss = loss_fn(out.view(-1, 100), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 10 == 0:
            with torch.no_grad():
                pred = out.argmax(dim=-1)
                recall = (pred[0, -1] == y[0, -1]).item()  # Did it recall the needle?
                recalls.append(recall)
    return np.mean(recalls)

def run_comparison():
    """
    Run comparison between vanilla Transformer and SRGI on NIAH benchmark.

    Expected results:
    - Vanilla: ~40-60% recall (standard attention fades quadratically)
    - SRGI: ~60-80% recall (resonant SSM preserves info over long horizons)
    """
    print("=" * 80)
    print("SRGI vs Original NanoChat Comparison Test")
    print("Benchmark: Needle-in-a-Haystack (NIAH)")
    print("=" * 80)
    print()

    # Run comparison with different sequence lengths
    for seq_len in [128, 256, 512]:
        print(f"\nTesting with sequence length: {seq_len}")
        print("-" * 80)

        vanilla = VanillaTransformer()
        srgi = SRGIToy()

        print("Training Vanilla Transformer...")
        vanilla_recall = train_and_test(vanilla, seq_len=seq_len, epochs=50)

        print("Training SRGI model...")
        srgi_recall = train_and_test(srgi, seq_len=seq_len, epochs=50)

        print(f"\nResults for seq_len={seq_len}:")
        print(f"  Vanilla average recall: {vanilla_recall:.2%}")
        print(f"  SRGI average recall:    {srgi_recall:.2%}")
        print(f"  SRGI improvement:       {(srgi_recall - vanilla_recall):.2%}")

        if srgi_recall > vanilla_recall:
            print(f"  ✓ SRGI outperformed vanilla by {((srgi_recall / vanilla_recall - 1) * 100):.1f}%")
        else:
            print(f"  ✗ SRGI did not outperform vanilla")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    run_comparison()
