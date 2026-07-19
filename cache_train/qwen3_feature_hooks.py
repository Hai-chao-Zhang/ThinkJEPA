"""Minimal Qwen3-VL hooks used by the causal ThinkJEPA cache builder.

This module deliberately contains no dataset discovery, supervision loading,
command-line entrypoint, or cache writer.  Cache construction and its temporal
contract live in :mod:`cache_train.rebuild_causal_cache`.
"""

from typing import Dict, List, Optional, Tuple

import torch


def locate_thinker_decoder_layers(module_with_layers: torch.nn.Module):
    """Return the Qwen3-VL language decoder layers and their stable path."""
    if not hasattr(module_with_layers, "model"):
        raise AttributeError("Expected Qwen3-VL model to have `.model`")
    if not hasattr(module_with_layers.model, "language_model"):
        raise AttributeError(
            "Expected Qwen3-VL model to have `.model.language_model`"
        )
    if not hasattr(module_with_layers.model.language_model, "layers"):
        raise AttributeError(
            "Expected Qwen3-VL model to have `.model.language_model.layers`"
        )
    return module_with_layers.model.language_model.layers, "model.language_model.layers"


def register_thinker_decoder_hooks(decoder_layers, layers: List[int]):
    """Capture detached decoder states for the explicitly selected layers."""
    saved: Dict[str, List[torch.Tensor]] = {f"dec_{i}": [] for i in layers}

    def make_hook(name):
        def hook(_module, _inp, out):
            hidden = out
            if isinstance(hidden, (tuple, list)):
                hidden = hidden[0]
                if isinstance(hidden, (tuple, list)):
                    hidden = hidden[0]
            if torch.is_tensor(hidden):
                saved[name].append(hidden.detach().cpu())

        return hook

    for layer_idx in layers:
        decoder_layers[layer_idx].register_forward_hook(
            make_hook(f"dec_{layer_idx}")
        )
    return saved


def _empty_guidance_state() -> torch.Tensor:
    return torch.empty((0,), dtype=torch.float16)


def _slice_sample_state(
    tensor: torch.Tensor, sample_idx: int, batch_size: int
) -> torch.Tensor:
    if tensor.dim() > 0 and tensor.shape[0] == batch_size:
        return tensor[sample_idx : sample_idx + 1]
    return tensor


def _trim_state_to_valid_length(
    tensor: torch.Tensor, valid_len: Optional[int]
) -> torch.Tensor:
    if valid_len is None or tensor.dim() != 3 or tensor.shape[1] <= valid_len:
        return tensor
    if valid_len <= 0:
        return tensor[:, :0, :]
    # Qwen inputs are left padded, so valid tokens are the rightmost tokens.
    return tensor[:, -valid_len:, :]


def stack_pyramid_guidance_states_per_sample(
    saved: Dict[str, List[torch.Tensor]],
    layers: List[int],
    batch_size: int,
    valid_lens: Optional[List[int]] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Stack prompt and generated-token states for every sample and layer."""
    outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for sample_idx in range(batch_size):
        prompt_states = []
        generated_states = []
        for layer_idx in layers:
            layer_states = saved.get(f"dec_{layer_idx}", [])
            if not layer_states:
                continue

            prompt = _slice_sample_state(
                layer_states[0], sample_idx, batch_size
            )
            if valid_lens is not None and sample_idx < len(valid_lens):
                prompt = _trim_state_to_valid_length(
                    prompt, int(valid_lens[sample_idx])
                )
            prompt_states.append(prompt)

            if len(layer_states) > 1:
                try:
                    generated = torch.cat(layer_states[1:], dim=1)
                except Exception:
                    generated = layer_states[-1]
                    if generated.dim() == 3:
                        generated = generated[:, -1:, :]
                generated_states.append(
                    _slice_sample_state(generated, sample_idx, batch_size)
                )

        vlm_old = (
            torch.stack(prompt_states, dim=0)
            if prompt_states
            else _empty_guidance_state()
        )
        if not generated_states:
            vlm_new = _empty_guidance_state()
        else:
            try:
                vlm_new = torch.stack(generated_states, dim=0)
            except Exception:
                last_token_states = [
                    state[:, -1:, :] if state.dim() == 3 else state
                    for state in generated_states
                ]
                vlm_new = torch.stack(last_token_states, dim=0)
        outputs.append((vlm_old, vlm_new))
    return outputs
