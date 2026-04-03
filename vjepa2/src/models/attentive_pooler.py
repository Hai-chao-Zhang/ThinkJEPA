# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math

import torch
import torch.nn as nn

from src.models.utils.modules import Block, CrossAttention, CrossAttentionBlock
from src.utils.tensors import trunc_normal_

from .pose_transformer import TransformerDecoder


class TrajTFhead(nn.Module):
    """
    输入不变：接收 encoder 的单个 clip 表征 x（形如 [B, L, D] 或 [B, N_tokens, D]，也兼容 [B, C, H, W]）
    输出不变：该 clip 的状态序列 [B, T, 6]，其中 T=frames_per_clip

    与原始版本的区别：
    - 将基于注意力的 Pooler 改为 TransformerDecoder 风格的 cross-attention：
      使用 T 个 query token（零 token）作为查询，context 为 x。
    """

    def __init__(
        self,
        embed_dim: int = 768,
        frames_per_clip: int = 16,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        depth: int = 1,
        qkv_bias: bool = True,
        init_std: float = 0.02,
        use_activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.frames_per_clip = frames_per_clip
        self.embed_dim = embed_dim

        # —— 与 MANOTransformerDecoderHead 保持思路一致：
        # 使用 token 查询（零 token/learnable 均可）。这里用零 token，更贴合示例的默认做法。
        # 关键参数沿用当前 Head 的入参，而非 cfg。
        self.transformer = TransformerDecoder(
            heads=num_heads,
            mlp_dim=embed_dim,
            num_tokens=frames_per_clip,  # T 个查询
            token_dim=1,  # 与示例一致：零 token（1 维占位）
            dim=embed_dim,  # 输出隐空间维度与下游 head 对齐
            depth=depth,
        )

        # 回归到 6 维（xyz + euler 或你下游定义的 6 维）
        self.head = nn.Linear(embed_dim, 3)
        nn.init.normal_(self.head.weight, std=init_std)
        nn.init.zeros_(self.head.bias)

    def _ensure_seq(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入统一为 [B, L, D] 形式：
        - 若是 [B, L, D] 直接返回
        - 若是 [B, C, H, W]，展平为空间 token：[B, H*W, C]
        - 若是 [B, D]，视作单 token：[B, 1, D]
        其他情况则抛错，方便尽早定位不一致。
        """
        if x.dim() == 3:
            # [B, L, D]
            return x
        elif x.dim() == 4:
            # 兼容 ViT 风格特征 [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = x.shape
            x = einops.rearrange(x, "b c h w -> b (h w) c")
            return x
        elif x.dim() == 2:
            # [B, D] -> [B, 1, D]
            return x.unsqueeze(1)
        else:
            raise ValueError(
                f"Unsupported input shape {tuple(x.shape)}; expected [B,L,D] or [B,C,H,W] or [B,D]."
            )

    def forward(self, initp, x: torch.Tensor) -> torch.Tensor:
        """
        x: encoder 对 "一个 clip" 的输出；形如 [B, L, D]（或可被 _ensure_seq 规整）
        return: [B, T, 6]，T=self.frames_per_clip
        """
        x = self._ensure_seq(x)  # [B, L, D]
        B = x.shape[0]
        T = self.frames_per_clip

        # 与示例保持一致：用零 token 作为查询（形状 [B, T, 1]）
        # initp = torch.zeros(B, T, 1, device=x.device, dtype=x.dtype)

        # Cross-attention：queries = token，context = x
        # 期望输出 [B, T, D]
        q = self.transformer(initp, context=x)

        # 回归到每帧 3 维
        y = self.head(q)  # [B, T, 3]
        return y


class AttentivePooler(nn.Module):
    """Attentive Pooler"""

    def __init__(
        self,
        num_queries=1,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=3,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.use_activation_checkpointing = use_activation_checkpointing
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))

        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
            )
        else:
            self.cross_attention_block = CrossAttention(
                dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias
            )

        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=False,
                        norm_layer=norm_layer,
                    )
                    for i in range(depth - 1)
                ]
            )

        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        layer_id = 0
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

        if self.complete_block:
            rescale(self.cross_attention_block.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.blocks is not None:
            for blk in self.blocks:
                if self.use_activation_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
        q = self.query_tokens.repeat(len(x), 1, 1)
        q = self.cross_attention_block(q, x)
        return q


class AttentiveClassifier(nn.Module):
    """Attentive Classifier"""

    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        num_classes=1000,
        complete_block=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        self.linear = nn.Linear(embed_dim, num_classes, bias=True)

    def forward(self, x):
        x = self.pooler(x).squeeze(1)
        x = self.linear(x)
        return x


class TrajRegressorHead(nn.Module):
    """
    输入不变：接收 encoder 的单个 clip 表征 x（形如 [B, N_tokens, D] 或 [B, L, D]）
    输出：该 clip 的状态序列 [B, T, 6]，其中 T=frames_per_clip
    """

    def __init__(
        self,
        embed_dim=768,
        frames_per_clip=16,
        num_heads=8,
        mlp_ratio=4.0,
        depth=1,
        qkv_bias=True,
        init_std=0.02,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.frames_per_clip = frames_per_clip

        # 用注意力查询 T 个时间点
        self.pool = AttentivePooler(
            num_queries=frames_per_clip * 2,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            qkv_bias=qkv_bias,
            complete_block=True,
            init_std=init_std,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        # 每个时刻输出 6 维（xyz + euler）
        self.head = nn.Linear(embed_dim, 3)

    def forward(self, initp, x):
        """
        x: encoder 对"一个 clip"的输出，通常形如 [B, L, D]（tokens）或可被 pooler 消化的序列特征
        return: [B, T, 6]，T=self.frames_per_clip
        """
        # 先用注意力把序列对齐到 T 个时间查询
        q = self.pool(x)  # [B, T, D]
        y = self.head(q)  # [B, T, 6]
        return y


class VJEPA2SOTAHandTrackingHead(nn.Module):
    """
    SOTA Hand Tracking Head - drop-in replacement for TrajRegressorHead
    Supports joint-level and rigid 6DoF tracking with temporal modeling.

    Input: [B, L, D] vJEPA2 tokens
    Output: [B, T, out_dim] where out_dim depends on mode
    """

    def __init__(
        self,
        embed_dim=768,
        frames_per_clip=16,
        num_joints=1,  # 1 for rigid, 21+ for hand
        out_mode="rigid6d",  # "rigid6d", "joints3d", "joints2d"
        pool_depth=3,
        num_heads=8,
        temporal_type="transformer",  # "gru", "transformer", "tcn"
        temporal_layers=2,
        temporal_dim=384,
        mlp_ratio=4.0,
        qkv_bias=True,
        init_std=0.02,
        use_activation_checkpointing=False,
        dropout=0.1,
    ):
        super().__init__()
        self.frames_per_clip = frames_per_clip
        self.num_joints = num_joints
        self.out_mode = out_mode
        self.use_activation_checkpointing = use_activation_checkpointing

        T = frames_per_clip
        J = num_joints

        # Learnable queries for T*J points
        self.queries = nn.Parameter(torch.zeros(1, T * J, embed_dim))

        # Sinusoidal positional encoding
        time_pos = self._make_sinusoidal_pos(T, embed_dim)
        self.register_buffer("time_pos", time_pos)
        joint_pos = self._make_sinusoidal_pos(J, embed_dim)
        self.register_buffer("joint_pos", joint_pos)

        # Cross-attention pooling (multi-block)
        self.pool_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(pool_depth)
            ]
        )

        # Project to temporal dim
        self.pool_proj = (
            nn.Linear(embed_dim, temporal_dim)
            if embed_dim != temporal_dim
            else nn.Identity()
        )

        # Temporal modeling
        if temporal_type == "gru":
            self.temporal = nn.GRU(
                input_size=temporal_dim,
                hidden_size=temporal_dim // 2,
                num_layers=temporal_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if temporal_layers > 1 else 0.0,
            )
            self.temporal_proj = nn.Linear(temporal_dim, temporal_dim)
        elif temporal_type == "transformer":
            self.temporal = nn.ModuleList(
                [
                    Block(
                        dim=temporal_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=False,
                        norm_layer=nn.LayerNorm,
                    )
                    for _ in range(temporal_layers)
                ]
            )
            self.temporal_proj = nn.Identity()
        elif temporal_type == "tcn":
            tcn_layers = []
            for i in range(temporal_layers):
                dilation = 2**i
                padding = dilation
                tcn_layers.append(
                    nn.Sequential(
                        nn.Conv1d(
                            temporal_dim,
                            temporal_dim,
                            3,
                            padding=padding,
                            dilation=dilation,
                        ),
                        nn.LayerNorm(temporal_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout),
                    )
                )
            self.temporal = nn.ModuleList(tcn_layers)
            self.temporal_proj = nn.Identity()
        else:
            raise ValueError(f"Unknown temporal_type: {temporal_type}")

        self.temporal_type = temporal_type

        # Prediction heads
        if out_mode == "joints3d":
            self.coord_head = nn.Linear(temporal_dim, 3)
            self.conf_head = nn.Linear(temporal_dim, 1)
        elif out_mode == "joints2d":
            self.coord_head = nn.Linear(temporal_dim, 2)
            self.conf_head = nn.Linear(temporal_dim, 1)
        elif out_mode == "rigid6d":
            self.coord_head = nn.Linear(temporal_dim, 6)  # xyz + euler
            self.conf_head = nn.Linear(temporal_dim, 1)
        else:
            raise ValueError(f"Unknown out_mode: {out_mode}")

        # Initialize
        trunc_normal_(self.queries, std=init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _make_sinusoidal_pos(self, length: int, dim: int) -> torch.Tensor:
        """Create sinusoidal positional encodings."""
        pos = torch.arange(length).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # [1, L, D]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        """Rescale residual connections."""

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, block in enumerate(self.pool_blocks):
            if hasattr(block, "mlp") and hasattr(block.mlp, "fc2"):
                rescale(block.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, x):
        """
        x: [B, L, D] vJEPA2 tokens
        return: [B, T, out_dim]
        """
        # Ensure FP32 for GRU compatibility
        x = x.float()

        B, L, D = x.shape
        T = self.frames_per_clip
        J = self.num_joints

        # Construct queries with positional encoding
        q = self.queries.reshape(1, T, J, D)
        q = q + self.time_pos.unsqueeze(2)
        q = q + self.joint_pos.unsqueeze(1)
        q = q.reshape(1, T * J, D).repeat(B, 1, 1)  # [B, T*J, D]

        # Cross-attention pooling
        for block in self.pool_blocks:
            if self.use_activation_checkpointing:
                q = torch.utils.checkpoint.checkpoint(block, q, x, use_reentrant=False)
            else:
                q = block(q, x)

        # Project and reshape
        q = self.pool_proj(q)  # [B, T*J, temporal_dim]
        q = q.reshape(B, T, J, -1)  # [B, T, J, temporal_dim]

        # Temporal modeling
        if self.temporal_type == "gru":
            # Process per joint: [B*J, T, D] - ensure FP32
            q_flat = q.permute(0, 2, 1, 3).reshape(B * J, T, -1).float()
            q_flat, _ = self.temporal(q_flat)
            q_flat = self.temporal_proj(q_flat)
            q = q_flat.reshape(B, J, T, -1).permute(0, 2, 1, 3)  # [B, T, J, D]
        elif self.temporal_type == "transformer":
            q_flat = q.permute(0, 2, 1, 3).reshape(B * J, T, -1)
            for blk in self.temporal:
                q_flat = blk(q_flat)
            q = q_flat.reshape(B, J, T, -1).permute(0, 2, 1, 3)
        elif self.temporal_type == "tcn":
            q_flat = q.permute(0, 2, 3, 1).reshape(B * J, -1, T)  # [B*J, D, T]
            for layer in self.temporal:
                residual = q_flat
                q_flat = layer(q_flat) + residual
            q = q_flat.reshape(B, J, -1, T).permute(0, 3, 1, 2)  # [B, T, J, D]

        # Prediction
        coords = self.coord_head(q)  # [B, T, J, C]

        # For rigid mode, squeeze joint dim
        if self.out_mode == "rigid6d" and J == 1:
            coords = coords.squeeze(2)  # [B, T, 6]

        return coords
