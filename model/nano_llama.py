"""
CodebaseLlama: A modern ~6.2M parameter Llama-style Transformer architecture
with Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU, and FlashAttention
designed for fast local repository-context adaptation on 4GB VRAM GPUs.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CodebaseLlamaConfig:
    vocab_size: int = 4096
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    max_seq_len: int = 256
    dropout: float = 0.05
    multiple_of: int = 64


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm + self.eps).type_as(x)
        return self.weight * x_normed


def precompute_rope_cache(dim: int, max_seq_len: int, base: float = 10000.0, device=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_heads, T, head_dim)
    T = x.size(2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim // 2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack((out1, out2), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: CodebaseLlamaConfig):
        super().__init__()
        assert config.dim % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.dropout = config.dropout

        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.proj = nn.Linear(config.dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLUFFN(nn.Module):
    def __init__(self, config: CodebaseLlamaConfig):
        super().__init__()
        hidden_dim = int(2 * (4 * config.dim) / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class LlamaBlock(nn.Module):
    def __init__(self, config: CodebaseLlamaConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.dim)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class CodebaseLlama(nn.Module):
    def __init__(self, config: CodebaseLlamaConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([LlamaBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying between input embeddings and output projection
        self.output.weight = self.tok_embeddings.weight

        cos, sin = precompute_rope_cache(
            config.dim // config.n_heads, config.max_seq_len
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        if T > self.config.max_seq_len:
            idx = idx[:, -self.config.max_seq_len :]

        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)
        x = self.norm(x)
        return self.output(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CodebaseLlamaConfig()
    model = CodebaseLlama(cfg).to(device)
    dummy_x = torch.randint(0, cfg.vocab_size, (4, cfg.max_seq_len), device=device)
    out = model(dummy_x)
    print(f"CodebaseLlama initialized on {device}")
    print(f"Total Parameters : {model.count_parameters():,}")
    print(f"Input Shape      : {tuple(dummy_x.shape)}")
    print(f"Output Shape     : {tuple(out.shape)}")