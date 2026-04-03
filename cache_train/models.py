# ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model
# Copyright (c) 2024-2026 Northeastern University.
# Developed in NEU SMILE LAB by Haichao Zhang (https://zhanghaichao.xyz)
# and Yun Raymond Fu (https://www1.ece.neu.edu/~yunfu/).
# SPDX-style identifier: LicenseRef-ThinkJEPA-Attribution
# Original source: https://github.com/Hai-chao-Zhang/ThinkJEPA
# See the root LICENSE, NOTICE, CITATION.cff, and CITATION.bib for attribution and citation requirements.

import torch
import torch.nn as nn
import torch.nn.functional as F


def _run_attention_in_module_dtype(attn_module, query, key, value):
    target_dtype = getattr(attn_module.in_proj_weight, "dtype", query.dtype)
    output_dtype = query.dtype
    if query.dtype != target_dtype:
        query = query.to(dtype=target_dtype)
    if key.dtype != target_dtype:
        key = key.to(dtype=target_dtype)
    if value.dtype != target_dtype:
        value = value.to(dtype=target_dtype)
    attn_out, attn_weights = attn_module(query, key, value)
    if attn_out.dtype != output_dtype:
        attn_out = attn_out.to(dtype=output_dtype)
    return attn_out, attn_weights


class TrajectoryReadoutMLP(nn.Module):
    def __init__(
        self,
        d=1024,
        n_tokens=128,
        out_dims=52 * 3,
        mlp_hidden=1024,
        p=0.1,
        use_attn_pool=True,
        nhead=8,
        downsample=False,
    ):
        super().__init__()
        self.d = d
        self.n_tokens = n_tokens
        self.out_dims = out_dims
        self.use_attn_pool = use_attn_pool

        if use_attn_pool:
            self.token_query = nn.Parameter(torch.randn(1, 1, d))
            self.token_attn = nn.MultiheadAttention(
                d, nhead, batch_first=True, dropout=p
            )
        else:
            self.token_query = None
            self.token_attn = None

        self.ln_in = nn.LayerNorm(d)
        self.mlp_in = nn.Sequential(
            nn.Linear(d, mlp_hidden),
            nn.GELU(),
            nn.Dropout(p),
        )

        self.temporal_mlp = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(p),
        )

        self.downsample = downsample
        if downsample:
            # Average-pool over time with kernel=2, stride=2 => T -> floor(T/2)
            self.time_pool = nn.AvgPool1d(kernel_size=2, stride=2)

        self.ln_out = nn.LayerNorm(mlp_hidden)
        self.fc_out = nn.Linear(mlp_hidden, out_dims)

        self._initialize_readout_parameters()

    def _initialize_readout_parameters(self):
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def _pool_temporal_tokens(self, x):
        if not self.use_attn_pool:
            return x.mean(dim=2)

        B, T, N, d = x.shape
        x = x.view(B * T, N, d)
        q = self.token_query.expand(B * T, -1, -1)
        y, _ = _run_attention_in_module_dtype(self.token_attn, q, x, x)
        y = y.squeeze(1)
        y = y.view(B, T, d)
        return y

    def forward(self, x):
        if x.ndim == 3:
            B, T, d = x.shape
            x = x.view(B, T, -1, d)
        else:
            B, T, _, _ = x.shape

        x = self._pool_temporal_tokens(x)
        x = self.ln_in(x)
        x = self.mlp_in(x)
        H = x.size(-1)

        x_flat = x.reshape(B * T, H)
        y = self.temporal_mlp(x_flat).view(B, T, H)
        x = x + y

        if self.downsample:
            # Downsample over time: convert to [B, H, T], pool, then return to [B, T', H]
            x = x.transpose(1, 2)  # [B, H, T]
            x = self.time_pool(x)  # [B, H, T']
            x = x.transpose(1, 2).contiguous()  # [B, T', H]

        x = self.ln_out(x)
        out = self.fc_out(x)
        return out
