"""A GPT-style decoder-only transformer, written from scratch against raw
torch.nn primitives -- no transformers/AutoModel involved. The point of this
file existing at all is to see every piece (attention, rotary embeddings,
norm, MLP) rather than import a black box.

Architecture is LLaMA-style rather than original-GPT-2-style, since that's
what every current open model actually uses: RMSNorm instead of LayerNorm,
rotary position embeddings instead of learned/absolute position embeddings,
SwiGLU instead of a GELU MLP, no bias terms. Attention uses
F.scaled_dot_product_attention, which dispatches to a fused flash-attention
kernel on Ada -- meaningfully faster and more memory-efficient than a
hand-written softmax(QK^T)V, without giving up readability.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def _rotary_cos_sin(seq_len: int, head_dim: int, theta: float, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
    freqs = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (batch, n_head, seq_len, head_dim); cos/sin: (seq_len, head_dim)
    return x * cos + _rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.rope_theta = cfg.rope_theta

        self.qkv_proj = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = _rotary_cos_sin(T, self.head_dim, self.rope_theta, x.device, q.dtype)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # standard SwiGLU sizing: hidden dim = 8/3 * n_embd, rounded to a
        # multiple of 64 for clean tensor-core tiling
        hidden = int(8 * cfg.n_embd / 3)
        hidden = ((hidden + 63) // 64) * 64
        self.gate_proj = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.up_proj = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.down_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.final_norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying: halves embedding-table params, standard practice

        self.apply(self._init_weights)
        # scaled init on residual projections, per GPT-2, so residual stream
        # variance doesn't grow with depth
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters()) - self.token_emb.weight.numel()  # embedding counted once via tying

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.token_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        # inference: only compute logits for the last position, saves memory/time
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    def configure_optimizer(self, weight_decay: float, lr: float, betas: tuple[float, float]) -> torch.optim.Optimizer:
        # weight decay only on matrix params (Linear weights), not on norms/embeddings -- standard practice
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused_available and torch.cuda.is_available())

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8, top_k: int | None = 50) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
