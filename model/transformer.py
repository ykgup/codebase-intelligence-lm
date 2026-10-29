import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        max_length=128,
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.vocab_size = vocab_size
        self.max_length = max_length

        self.token_embedding = nn.Embedding(
            vocab_size, embedding_dim
        )
        self.position_embedding = nn.Embedding(
            max_length, embedding_dim
        )

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )

        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_ids):
        batch_size, sequence_length = token_ids.shape

        if sequence_length > self.max_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"maximum {self.max_length}"
            )

        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        x = self.token_embedding(token_ids)
        x = x + self.position_embedding(positions)

        # True means a position cannot attend to that future position.
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=token_ids.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        x = self.transformer(x, mask=causal_mask)
        return self.output(x)