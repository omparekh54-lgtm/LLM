"""
gpt_model.py -- Step 4 of the pipeline (not run directly for training;
imported by 05_pretrain.py and 07_instruction_tune.py). Run this file
directly (`python gpt_model.py`) to sanity-check the parameter count.

A standard decoder-only transformer (GPT-2 style), written from scratch in
plain PyTorch. No external model libraries -- this is the actual "built
from scratch" part of the project.

Default config lands at ~143M parameters:
  - 10 layers, 16 heads, d_model=1024, ffn=4096, context=512, vocab=16384
  - weight tying between input embedding and output projection (standard
    trick, saves ~17M params and slightly improves quality)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 16384
    block_size: int = 512      # max context length
    n_layer: int = 10
    n_head: int = 16
    n_embd: int = 1024
    ffn_mult: int = 4          # feedforward hidden size = n_embd * ffn_mult
    dropout: float = 0.1
    bias: bool = False         # GPT-2 uses bias in linears/layernorms; False is a bit faster and works fine
    grad_checkpointing: bool = False  # trade ~25% more compute for much lower activation memory


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # use PyTorch's fused scaled_dot_product_attention (flash-attention
        # kernel on supported GPUs) instead of a hand-rolled softmax --
        # meaningfully faster and lower memory on the RTX 4050.
        self.use_flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = config.n_embd * config.ffn_mult
        self.fc_in = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.fc_out = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.dropout(self.fc_out(self.act(self.fc_in(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying: share the embedding matrix with the output projection
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 paper trick, helps
        # deeper models train stably)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc_out.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        n = sum(p.numel() for p in self.parameters())
        n -= self.pos_emb.weight.numel()  # conventional: exclude positional embeddings from the headline count
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} exceeds block_size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            if self.config.grad_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            # reshape (not view) -- targets can be a non-contiguous slice
            # (e.g. y[:, 1:] in the instruction-tuning script), which .view()
            # can't handle directly
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, top_p=None, repetition_penalty=1.0):
        """
        top_p (nucleus sampling): instead of a fixed top_k cutoff, keeps the
        smallest set of tokens whose cumulative probability reaches top_p.
        This adapts to how "confident" the model is at each step -- narrow
        when confident, wider when uncertain -- which top_k alone can't do.
        Combine with top_k (top_k applied first) or use top_p alone by
        passing top_k=None.

        repetition_penalty > 1.0 discourages repeating tokens already
        present in the sequence so far -- a cheap fix for the common small-
        model failure mode of looping ("return return return...") since
        this model has no other mechanism to notice it's repeating itself.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    seen = torch.unique(idx[b])
                    seen_logits = logits[b, seen]
                    # divide positive logits, multiply negative ones -- both push the
                    # token's probability down, matching the standard repetition-penalty definition
                    logits[b, seen] = torch.where(seen_logits > 0, seen_logits / repetition_penalty,
                                                   seen_logits * repetition_penalty)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # keep the smallest prefix whose cumulative prob >= top_p; always keep at least 1 token
                sorted_mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_mask] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


if __name__ == "__main__":
    # quick sanity check: instantiate and print param count
    cfg = GPTConfig()
    model = GPT(cfg)
    n = model.num_params()
    print(f"Model config: {cfg}")
    print(f"Parameter count: {n:,} ({n / 1e6:.1f}M)")