# ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model
# Copyright (c) 2024-2026 Northeastern University.
# Developed in NEU SMILE LAB by Haichao Zhang (https://zhanghaichao.xyz)
# and Yun Raymond Fu (https://www1.ece.neu.edu/~yunfu/).
# SPDX-style identifier: LicenseRef-ThinkJEPA-Attribution
# Original source: https://github.com/Hai-chao-Zhang/ThinkJEPA
# See the root LICENSE, NOTICE, CITATION.cff, and CITATION.bib for attribution and citation requirements.

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.masks.utils import apply_masks
from src.models.utils.modules import Block
from src.models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from src.utils.tensors import repeat_interleave_batch, trunc_normal_


VALID_GUIDANCE_MODES = {"film", "crossattn", "adaln"}


@dataclass
class LayerwiseGuidanceBank:
    mode: str
    layers: list

    def repeat_for_views(self, repeat_factor):
        if repeat_factor == 1:
            return self
        if self.mode == "crossattn":
            repeated_layers = [
                layer_tokens.repeat(repeat_factor, 1, 1) for layer_tokens in self.layers
            ]
        else:
            repeated_layers = [
                (scale_tokens.repeat(repeat_factor, 1), shift_tokens.repeat(repeat_factor, 1))
                for scale_tokens, shift_tokens in self.layers
            ]
        return LayerwiseGuidanceBank(mode=self.mode, layers=repeated_layers)


class CortexGuidedVideoPredictor(nn.Module):
    """Roll out masked latent tokens with optional cortex-style guidance."""

    _LEGACY_KEY_PREFIXES = {
        "predictor_embed.": "context_adapter.",
        "mask_tokens.": "query_templates.",
        "predictor_pos_embed.": "fixed_position_table.",
        "predictor_blocks.": "rollout_layers.",
        "predictor_norm.": "rollout_norm.",
        "predictor_proj.": "output_adapter.",
        "vlm_old_proj.": "guidance_old_adapter.",
        "vlm_new_proj.": "guidance_new_adapter.",
        "vlm_adapters.": "guidance_fusion_mlps.",
        "vlm_adaln_norms.": "guidance_prenorms.",
        "vlm_cross_attn_norms.": "guidance_query_norms.",
        "vlm_cross_attn.": "guidance_memory_readers.",
        "vlm_gate.": "guidance_layer_scale.",
    }

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=False,
        use_mask_tokens=False,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
        return_all_tokens=False,
        chop_last_n_tokens=0,
        use_rope=False,
        use_vlm_merge=True,
        vlm_cond_mode="film",
        vlm_old_dim=3584,
        vlm_new_dim=3584,
        vlm_hidden=512,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.return_all_tokens = return_all_tokens
        self.tail_crop_count = chop_last_n_tokens

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.uses_video_layout = num_frames > 1
        self.grid_height = self.img_height // patch_size
        self.grid_width = self.img_width // patch_size
        self.grid_depth = num_frames // tubelet_size
        self.use_activation_checkpointing = use_activation_checkpointing
        self.position_uniform_power = uniform_power
        self.use_rope = use_rope

        self.context_adapter = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        self.query_templates = None
        self.num_query_templates = 0
        if use_mask_tokens:
            self.num_query_templates = num_mask_tokens
            self.query_templates = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
                    for _ in range(num_mask_tokens)
                ]
            )

        if self.uses_video_layout:
            patch_count = (
                (num_frames // tubelet_size)
                * (self.img_height // patch_size)
                * (self.img_width // patch_size)
            )
        else:
            patch_count = (self.img_height // patch_size) * (self.img_width // patch_size)
        self.num_patches = patch_count

        self.fixed_position_table = None
        if not use_rope:
            self.fixed_position_table = nn.Parameter(
                torch.zeros(1, patch_count, predictor_embed_dim), requires_grad=False
            )

        drop_path_schedule = [
            value.item() for value in torch.linspace(0, drop_path_rate, depth)
        ]
        self.rollout_layers = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    grid_depth=self.grid_depth,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_schedule[layer_idx],
                    norm_layer=norm_layer,
                )
                for layer_idx in range(depth)
            ]
        )
        self.rollout_norm = norm_layer(predictor_embed_dim)
        self.output_adapter = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        if self.fixed_position_table is not None:
            self._seed_position_table(self.fixed_position_table.data)
        self.init_std = init_std
        if self.query_templates is not None and not zero_init_mask_tokens:
            for template in self.query_templates:
                trunc_normal_(template, std=init_std)
        self.apply(self._init_module_parameters)
        self._stabilize_residual_branches()

        self.use_guidance_merge = use_vlm_merge
        self.guidance_mode = self._normalize_guidance_mode(vlm_cond_mode)
        if self.use_guidance_merge:
            self.guidance_old_adapter = nn.Linear(
                vlm_old_dim, predictor_embed_dim, bias=False
            )
            self.guidance_new_adapter = nn.Linear(
                vlm_new_dim, predictor_embed_dim, bias=False
            )
            if self.guidance_mode in {"film", "adaln"}:
                fusion_width = 4 * predictor_embed_dim
                self.guidance_fusion_mlps = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.LayerNorm(fusion_width),
                            nn.Linear(fusion_width, vlm_hidden, bias=True),
                            nn.GELU(),
                            nn.Linear(vlm_hidden, 2 * predictor_embed_dim, bias=True),
                        )
                        for _ in range(depth)
                    ]
                )
                if self.guidance_mode == "adaln":
                    self.guidance_prenorms = nn.ModuleList(
                        [nn.LayerNorm(predictor_embed_dim) for _ in range(depth)]
                    )
            else:
                self.guidance_query_norms = nn.ModuleList(
                    [nn.LayerNorm(predictor_embed_dim) for _ in range(depth)]
                )
                self.guidance_memory_readers = nn.ModuleList(
                    [
                        nn.MultiheadAttention(
                            embed_dim=predictor_embed_dim,
                            num_heads=num_heads,
                            dropout=attn_drop_rate,
                            batch_first=True,
                        )
                        for _ in range(depth)
                    ]
                )
            self.guidance_layer_scale = nn.Parameter(torch.zeros(depth, 1, 1))

    @staticmethod
    def _normalize_guidance_mode(vlm_cond_mode):
        normalized_mode = str(vlm_cond_mode).lower().strip()
        if normalized_mode not in VALID_GUIDANCE_MODES:
            return "film"
        return normalized_mode

    @classmethod
    def _upgrade_legacy_state_dict(cls, state_dict):
        if not isinstance(state_dict, dict):
            return state_dict
        upgraded = {}
        changed = False
        for key, value in state_dict.items():
            new_key = key
            for old_prefix, new_prefix in cls._LEGACY_KEY_PREFIXES.items():
                if key.startswith(old_prefix):
                    new_key = f"{new_prefix}{key[len(old_prefix):]}"
                    changed = True
                    break
            upgraded[new_key] = value
        return upgraded if changed else state_dict

    def load_state_dict(self, state_dict, strict=True):
        upgraded = self._upgrade_legacy_state_dict(state_dict)
        return super().load_state_dict(upgraded, strict=strict)

    def _seed_position_table(self, position_table):
        embed_dim = position_table.size(-1)
        spatial_grid = self.img_height // self.patch_size
        if self.uses_video_layout:
            depth_grid = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                embed_dim,
                spatial_grid,
                depth_grid,
                cls_token=False,
                uniform_power=self.position_uniform_power,
            )
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, spatial_grid, cls_token=False)
        position_table.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_module_parameters(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _stabilize_residual_branches(self):
        for layer_index, block in enumerate(self.rollout_layers, start=1):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * layer_index))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * layer_index))

    @staticmethod
    def _normalize_mask_views(mask_groups):
        return mask_groups if isinstance(mask_groups, list) else [mask_groups]

    @staticmethod
    def _maybe_move_tensor(value, device):
        if value is not None and isinstance(value, torch.Tensor):
            return value.to(device)
        return value

    @staticmethod
    def _normalize_guidance_mask(mask_tensor):
        if mask_tensor is not None and mask_tensor.dim() == 3 and mask_tensor.size(-1) == 1:
            return mask_tensor.squeeze(-1)
        return mask_tensor

    @staticmethod
    def _reduce_token_set(token_set, token_mask=None, dim=1, eps=1e-6):
        if token_set is None:
            return None
        if token_mask is None:
            return token_set.mean(dim=dim)
        while token_mask.dim() < token_set.dim():
            token_mask = token_mask.unsqueeze(-1)
        weight = token_mask.to(dtype=token_set.dtype)
        numer = (token_set * weight).sum(dim=dim)
        denom = weight.sum(dim=dim).clamp_min(eps)
        return numer / denom

    def _project_guidance_layer(self, guidance_stream, layer_index, projector):
        if guidance_stream is None or layer_index >= guidance_stream.size(0):
            return None
        return projector(guidance_stream[layer_index])

    @staticmethod
    def _select_valid_guidance_tokens(token_set, token_mask):
        if token_set is None:
            return None
        if token_mask is None:
            return token_set
        valid_mask = token_mask.to(dtype=torch.bool)
        return token_set[valid_mask]

    def _build_cross_attention_memory(self, old_tokens, new_tokens, old_mask, new_mask, device):
        memory_parts = []
        valid_old = self._select_valid_guidance_tokens(old_tokens, old_mask)
        valid_new = self._select_valid_guidance_tokens(new_tokens, new_mask)
        if valid_old is not None and valid_old.numel() > 0:
            memory_parts.append(valid_old)
        if valid_new is not None and valid_new.numel() > 0:
            memory_parts.append(valid_new)
        if len(memory_parts) == 0:
            hidden_width = self.context_adapter.out_features
            return torch.zeros(1, hidden_width, device=device)
        return torch.cat(memory_parts, dim=0)

    def _build_residual_style_guidance(self, old_tokens, new_tokens, old_mask, new_mask, layer_index, batch_size, device):
        old_summary = self._reduce_token_set(old_tokens, old_mask, dim=0)
        new_summary = self._reduce_token_set(new_tokens, new_mask, dim=0)
        if old_summary is None and new_summary is None:
            hidden_width = self.context_adapter.out_features
            old_summary = torch.zeros(hidden_width, device=device)
            new_summary = torch.zeros(hidden_width, device=device)
        elif old_summary is None:
            old_summary = torch.zeros_like(new_summary)
        elif new_summary is None:
            new_summary = torch.zeros_like(old_summary)

        fusion_signature = torch.cat(
            [
                old_summary,
                new_summary,
                (old_summary - new_summary).abs(),
                old_summary * new_summary,
            ],
            dim=-1,
        )
        scale_shift = self.guidance_fusion_mlps[layer_index](fusion_signature)
        scale_tokens, shift_tokens = scale_shift.chunk(2, dim=-1)
        residual_gate = torch.tanh(self.guidance_layer_scale[layer_index]).reshape(1)
        scale_tokens = (scale_tokens * residual_gate).view(-1)
        shift_tokens = (shift_tokens * residual_gate).view(-1)
        return (
            scale_tokens.unsqueeze(0).expand(batch_size, -1),
            shift_tokens.unsqueeze(0).expand(batch_size, -1),
        )

    def _build_layerwise_guidance(self, guidance_payload, batch_size, device):
        if guidance_payload is None or not self.use_guidance_merge:
            return None

        old_stream = self._maybe_move_tensor(guidance_payload.get("vlm_old"), device)
        new_stream = self._maybe_move_tensor(guidance_payload.get("vlm_new"), device)
        if old_stream is None and new_stream is None:
            return None

        old_mask = self._normalize_guidance_mask(
            self._maybe_move_tensor(guidance_payload.get("vlm_old_mask"), device)
        )
        new_mask = self._normalize_guidance_mask(
            self._maybe_move_tensor(guidance_payload.get("vlm_new_mask"), device)
        )

        if old_stream is not None and old_stream.dim() != 3:
            raise ValueError(f"vlm_old must be [L,S,D], got {tuple(old_stream.shape)}")
        if new_stream is not None and new_stream.dim() != 3:
            raise ValueError(f"vlm_new must be [L,S,D], got {tuple(new_stream.shape)}")

        if self.guidance_mode == "crossattn":
            layer_count = len(self.guidance_memory_readers)
        else:
            layer_count = len(self.guidance_fusion_mlps)

        def _select_mask_layer(mask_bank, layer_index):
            if mask_bank is None:
                return None
            if mask_bank.dim() == 1:
                return mask_bank.to(dtype=torch.bool)
            if layer_index < mask_bank.size(0):
                return mask_bank[layer_index].to(dtype=torch.bool)
            return mask_bank[-1].to(dtype=torch.bool)

        layer_payloads = []
        for layer_index in range(layer_count):
            old_tokens = self._project_guidance_layer(
                old_stream, layer_index, self.guidance_old_adapter
            )
            new_tokens = self._project_guidance_layer(
                new_stream, layer_index, self.guidance_new_adapter
            )
            old_mask_layer = _select_mask_layer(old_mask, layer_index)
            new_mask_layer = _select_mask_layer(new_mask, layer_index)
            if self.guidance_mode == "crossattn":
                memory_tokens = self._build_cross_attention_memory(
                    old_tokens, new_tokens, old_mask_layer, new_mask_layer, device
                )
                layer_payloads.append(
                    memory_tokens.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
                )
            else:
                layer_payloads.append(
                    self._build_residual_style_guidance(
                        old_tokens,
                        new_tokens,
                        old_mask_layer,
                        new_mask_layer,
                        layer_index,
                        batch_size,
                        device,
                    )
                )

        return LayerwiseGuidanceBank(mode=self.guidance_mode, layers=layer_payloads)

    def _embed_visible_context(self, context_tokens, visible_groups, batch_size, has_cls):
        rollout_context = self.context_adapter(context_tokens)
        cls_token = None
        if has_cls:
            cls_token = rollout_context[:, :1, :]
            rollout_context = rollout_context[:, 1:, :]

        visible_token_count = rollout_context.shape[1]
        if self.fixed_position_table is not None:
            visible_positions = self.fixed_position_table.repeat(batch_size, 1, 1)
            rollout_context = rollout_context + apply_masks(visible_positions, visible_groups)
        return rollout_context, cls_token, visible_token_count

    def _build_query_stream(self, target_groups, batch_size, device, view_count, query_template_index):
        rollout_width = self.context_adapter.out_features
        if self.query_templates is None:
            template_tokens = torch.zeros(1, 1, rollout_width, device=device)
        else:
            template_index = query_template_index % self.num_query_templates
            template_tokens = self.query_templates[template_index]
        query_stream = template_tokens.repeat(batch_size, self.num_patches, 1)
        query_stream = apply_masks(query_stream, target_groups)
        if self.fixed_position_table is not None:
            query_positions = self.fixed_position_table.repeat(batch_size, 1, 1)
            query_positions = apply_masks(query_positions, target_groups)
            query_positions = repeat_interleave_batch(
                query_positions, batch_size, repeat=view_count
            )
            query_stream = query_stream + query_positions
        return query_stream

    def _stack_rollout_sequence(self, context_stream, query_stream, visible_groups, target_groups, cls_token):
        view_count = len(visible_groups)
        repeated_context = context_stream.repeat(view_count, 1, 1)
        rollout_stream = torch.cat([repeated_context, query_stream], dim=1)

        visible_lookup = torch.cat(visible_groups, dim=0)
        target_lookup = torch.cat(target_groups, dim=0)
        joint_lookup = torch.cat([visible_lookup, target_lookup], dim=1)
        sort_order = torch.argsort(joint_lookup, dim=1)
        sorted_lookup = torch.stack(
            [joint_lookup[row_index, row_order] for row_index, row_order in enumerate(sort_order)],
            dim=0,
        )
        rollout_stream = torch.stack(
            [rollout_stream[row_index, row_order, :] for row_index, row_order in enumerate(sort_order)],
            dim=0,
        )

        if self.tail_crop_count > 0:
            rollout_stream = rollout_stream[:, :-self.tail_crop_count]
            sorted_lookup = sorted_lookup[:, :-self.tail_crop_count]

        if cls_token is not None:
            rollout_stream = torch.cat([cls_token.repeat(view_count, 1, 1), rollout_stream], dim=1)

        return rollout_stream, sorted_lookup, sort_order

    def _inject_guidance(self, layer_index, rollout_stream, guidance_bank):
        if guidance_bank is None or layer_index >= len(guidance_bank.layers):
            return rollout_stream

        if guidance_bank.mode == "crossattn":
            memory_tokens = guidance_bank.layers[layer_index]
            normalized_query = self.guidance_query_norms[layer_index](rollout_stream)
            attended_tokens, _ = self.guidance_memory_readers[layer_index](
                normalized_query,
                memory_tokens,
                memory_tokens,
                need_weights=False,
            )
            residual_gate = torch.tanh(self.guidance_layer_scale[layer_index]).reshape(1, 1, 1)
            return rollout_stream + attended_tokens * residual_gate

        scale_tokens, shift_tokens = guidance_bank.layers[layer_index]
        if guidance_bank.mode == "adaln":
            rollout_stream = self.guidance_prenorms[layer_index](rollout_stream)
        return rollout_stream * (1.0 + scale_tokens.unsqueeze(1)) + shift_tokens.unsqueeze(1)

    @staticmethod
    def _recover_query_stream(rollout_stream, sort_order, visible_token_count):
        inverse_sort = torch.argsort(sort_order, dim=1)
        restored_stream = torch.stack(
            [rollout_stream[row_index, row_order, :] for row_index, row_order in enumerate(inverse_sort)],
            dim=0,
        )
        return restored_stream[:, visible_token_count:]

    def forward(self, x, masks_x, masks_y, mask_index=1, has_cls=False, ext=None):
        if masks_x is None or masks_y is None:
            raise ValueError("Cannot run the ThinkJEPA rollout head without mask indices")

        visible_groups = self._normalize_mask_views(masks_x)
        target_groups = self._normalize_mask_views(masks_y)
        device = x.device
        base_batch = len(x) // len(visible_groups)

        context_stream, cls_token, visible_token_count = self._embed_visible_context(
            x, visible_groups, base_batch, has_cls
        )
        query_stream = self._build_query_stream(
            target_groups,
            base_batch,
            device,
            view_count=len(visible_groups),
            query_template_index=mask_index if self.num_query_templates > 0 else 0,
        )
        rollout_stream, sorted_lookup, sort_order = self._stack_rollout_sequence(
            context_stream,
            query_stream,
            visible_groups,
            target_groups,
            cls_token,
        )

        expanded_batch = rollout_stream.size(0)
        guidance_bank = self._build_layerwise_guidance(ext, base_batch, device)
        if guidance_bank is not None and expanded_batch != base_batch:
            repeat_factor = expanded_batch // base_batch
            guidance_bank = guidance_bank.repeat_for_views(repeat_factor)

        for layer_index, rollout_layer in enumerate(self.rollout_layers):
            rollout_stream = self._inject_guidance(
                layer_index, rollout_stream, guidance_bank
            )
            if self.use_activation_checkpointing:
                rollout_stream = torch.utils.checkpoint.checkpoint(
                    rollout_layer,
                    rollout_stream,
                    sorted_lookup,
                    None,
                    use_reentrant=False,
                )
            else:
                rollout_stream = rollout_layer(
                    rollout_stream, mask=sorted_lookup, attn_mask=None
                )

        rollout_stream = self.rollout_norm(rollout_stream)
        if has_cls:
            rollout_stream = rollout_stream[:, 1:, :]
        if not self.return_all_tokens:
            rollout_stream = self._recover_query_stream(
                rollout_stream, sort_order, visible_token_count
            )
        return self.output_adapter(rollout_stream)


# Backward-compatible alias for older experiment scripts.
ThinkJEPAPredictor = CortexGuidedVideoPredictor
