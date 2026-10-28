import torch
import torch.nn as nn
from pathlib import Path
import sys

# Make the project root importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.transformer import TinyTransformer

torch.manual_seed(42)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Training device:", device)

# Tiny demonstration dataset: character IDs 0-127.
text = ("def add(a, b): return a + b\n" * 100)
data = torch.tensor(
    [ord(ch) for ch in text],
    dtype=torch.long,
    device=device,
)

model = TinyTransformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

sequence_length = 32
steps = 300

model.train()

for step in range(steps):
    starts = torch.randint(
        0,
        len(data) - sequence_length - 1,
        (16,),
        device=device,
    )

    x = torch.stack([data[s:s + sequence_length] for s in starts])
    y = torch.stack([
        data[s + 1:s + sequence_length + 1] for s in starts
    ])

    logits = model(x)
    loss = loss_fn(logits.reshape(-1, 128), y.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step + 1) % 50 == 0:
        print(f"Step {step + 1}/{steps}, loss: {loss.item():.4f}")

# Save the trained weights.
checkpoint = ROOT / "checkpoints" / "tiny_model.pt"
checkpoint.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), checkpoint)

print("Training finished!")
print("Checkpoint saved to:", checkpoint)