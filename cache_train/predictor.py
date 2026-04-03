# ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model
# Copyright (c) 2024-2026 Northeastern University.
# Developed in NEU SMILE LAB by Haichao Zhang (https://zhanghaichao.xyz)
# and Yun Raymond Fu (https://www1.ece.neu.edu/~yunfu/).
# SPDX-style identifier: LicenseRef-ThinkJEPA-Attribution
# Original source: https://github.com/Hai-chao-Zhang/ThinkJEPA
# See the root LICENSE, NOTICE, CITATION.cff, and CITATION.bib for attribution and citation requirements.

from functools import partial

import torch
import torch.nn as nn

from vjepa2.src.models.predictor import VisionTransformerPredictor


def build_official_rollout_predictor(
    frame_size=(256, 256),
    patch_size=16,
    clip_frames=64,
    tubelet_size=2,
    encoder_width=1024,
    rollout_width=384,
    rollout_layers=3,
    attention_heads=6,
    mlp_ratio=4.0,
    dropout=0.1,
    attention_dropout=0.0,
    stochastic_depth=0.1,
    use_rotary_positions=True,
    use_query_mask_tokens=True,
    query_mask_token_count=2,
    zero_init_query_mask_tokens=True,
):
    """Build the official V-JEPA predictor used as a rollout baseline."""

    return VisionTransformerPredictor(
        img_size=frame_size,
        patch_size=patch_size,
        num_frames=clip_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder_width,
        predictor_embed_dim=rollout_width,
        depth=rollout_layers,
        num_heads=attention_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        drop_rate=dropout,
        attn_drop_rate=attention_dropout,
        drop_path_rate=stochastic_depth,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_std=0.02,
        use_rope=use_rotary_positions,
        use_mask_tokens=use_query_mask_tokens,
        num_mask_tokens=query_mask_token_count,
        zero_init_mask_tokens=zero_init_query_mask_tokens,
        uniform_power=False,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
        return_all_tokens=False,
        chop_last_n_tokens=0,
    )


class PatchwiseAutoregressiveRolloutHead(nn.Module):
    """Temporal rollout head for dense latent tokens."""

    _LEGACY_KEY_PREFIXES = {
        "proj_in.": "input_adapter.",
        "encoder.": "rollout_backbone.",
        "proj_out.": "output_adapter.",
    }

    def __init__(
        self,
        token_dim=1024,
        hidden_dim=384,
        num_layers=3,
        num_heads=6,
        dropout=0.1,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.input_adapter = nn.Linear(token_dim, hidden_dim)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.rollout_backbone = nn.TransformerEncoder(
            temporal_layer, num_layers=num_layers
        )
        self.output_adapter = nn.Linear(hidden_dim, token_dim)

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

    def _encode_time_major(self, rollout_tokens, temporal_mask=None):
        """Preserve the legacy time-major execution layout."""
        time_major_tokens = rollout_tokens.transpose(0, 1).contiguous()
        encoded_tokens = self.rollout_backbone(time_major_tokens, mask=temporal_mask)
        return encoded_tokens.transpose(0, 1).contiguous()

    def _run_patchwise_rollout(self, token_grid, temporal_mask=None):
        batch_size, num_steps, num_patches, _ = token_grid.shape
        hidden_grid = self.input_adapter(token_grid.contiguous())
        patchwise_sequences = hidden_grid.view(batch_size * num_patches, num_steps, -1)
        encoded_sequences = self._encode_time_major(patchwise_sequences, temporal_mask)
        encoded_grid = encoded_sequences.view(
            batch_size, num_patches, num_steps, -1
        ).transpose(1, 2)
        return self.output_adapter(encoded_grid.contiguous())

    def _run_global_rollout(self, token_sequence, temporal_mask=None):
        hidden_sequence = self.input_adapter(token_sequence.contiguous())
        encoded_sequence = self._encode_time_major(hidden_sequence, temporal_mask)
        return self.output_adapter(encoded_sequence)

    def forward(self, x, attn_mask=None):
        if x.dim() == 4:
            return self._run_patchwise_rollout(x, temporal_mask=attn_mask)
        if x.dim() == 3:
            return self._run_global_rollout(x, temporal_mask=attn_mask)
        raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")


# Backward-compatible public aliases for older experiment scripts and checkpoints.
CompactLatentPredictor = PatchwiseAutoregressiveRolloutHead
create_compact_latent_predictor = build_official_rollout_predictor


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_official_rollout_predictor().to(device)
    predictor.train()
