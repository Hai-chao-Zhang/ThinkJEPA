#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build leakage-safe ThinkJEPA cache archives from raw MP4 videos.

Schema v2 deliberately has no ``vjepa_feats`` compatibility field.  Each source
video is split once at ``source_total_frames // 2`` into two disjoint raw
frame-index intervals.  Thirty-two indices are sampled from each interval.
This matches the supervision split used by ``trajectory_dataset``; exact source
timestamps are still retained as provenance.

The observed/past frames are the *only* frames supplied to Qwen3-VL.  V-JEPA2
ViT-L receives the observed and target clips in two separate encoder forwards,
so an encoder token stored for the observed clip cannot attend to a target
frame.  The two token tensors retain their native tubelet layout:
``[32 / 2, (256 / 16) ** 2, 1024] == [16, 256, 1024]``.

Normal extraction requires ``--vjepa_checkpoint``.  A lightweight
``--self_test`` mode exercises timestamp splitting, atomic writes, and archive
validation without loading either model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
VJEPA2_ROOT = REPO_ROOT / "vjepa2"
for _import_root in (REPO_ROOT, REPO_ROOT / "cache_train", VJEPA2_ROOT, VJEPA2_ROOT.parent):
    _import_root_text = str(_import_root)
    if _import_root_text not in sys.path:
        sys.path.insert(0, _import_root_text)
SCHEMA_NAME = "thinkjepa.causal_split.v2"
SCHEMA_VERSION = 2
VLM_OBSERVATION_POLICY = "observed_past_exact_frames"
EXTRACTOR_VERSION = "2.0.5"
SAMPLING_STRATEGY = "raw_index_midpoint_disjoint_uniform_v3"
NUM_PAST_FRAMES = 32
NUM_TARGET_FRAMES = 32
NUM_ALL_FRAMES = NUM_PAST_FRAMES + NUM_TARGET_FRAMES
VJEPA_IMAGE_SIZE = 256
VJEPA_PATCH_SIZE = 16
VJEPA_TUBELET_SIZE = 2
VJEPA_EMBED_DIM = 1024
VJEPA_TEMPORAL_TOKENS = NUM_PAST_FRAMES // VJEPA_TUBELET_SIZE
VJEPA_SPATIAL_TOKENS = (VJEPA_IMAGE_SIZE // VJEPA_PATCH_SIZE) ** 2
VJEPA_FEATURE_SHAPE = (
    VJEPA_TEMPORAL_TOKENS,
    VJEPA_SPATIAL_TOKENS,
    VJEPA_EMBED_DIM,
)
FEATURE_CACHE_FORBIDDEN_KEYS = {
    "vjepa_feats",
    "frame_indices",
    "xyz_cam",
    "R_cam",
    "xyz_world",
    "R_world",
    "tfs_in_cam",
    "tfs",
    "cam_ext",
    "cam_int",
    "confs",
    "lang_instruct",
    "path",
    "video_path",
    "raw_video_path",
    "total_frames",
}
DEFAULT_QWEN_LAYERS = (0, 4, 8, 12, 16, 20, 24, 27)
DEFAULT_PROMPT = "Describe this video."
VIDEO_SUFFIXES = {".mp4"}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _string_scalar(value: Any) -> str:
    value = _scalar(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild schema-v2 causal ThinkJEPA caches from raw MP4 files."
    )
    parser.add_argument("--file_dir", required=True, help="Raw MP4 root (recursive).")
    parser.add_argument(
        "--path_manifest",
        default="",
        help="Optional text/JSON/JSONL manifest of raw MP4 paths.",
    )
    parser.add_argument("--output_dir", required=True, help="Schema-v2 cache root.")
    parser.add_argument(
        "--pretrained",
        default="Qwen/Qwen3-VL-2B-Thinking",
        help="Qwen3-VL Hugging Face model id or local directory.",
    )
    parser.add_argument(
        "--qwen_revision",
        default="main",
        help="Hugging Face revision; the resolved commit is saved in provenance.",
    )
    parser.add_argument(
        "--qwen_checkpoint_sha",
        default="",
        help="Optional explicit Qwen commit/content SHA if it cannot be inferred.",
    )
    parser.add_argument(
        "--vjepa_checkpoint",
        required=True,
        help="Required local V-JEPA2 ViT-L checkpoint containing encoder weights.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(DEFAULT_QWEN_LAYERS),
        help="Qwen decoder layers to cache (default: 0 4 8 12 16 20 24 27).",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--dataset_prompt_mode",
        choices=["off"],
        default="off",
        help="Schema v2 defaults to and currently enforces metadata prompts off.",
    )
    parser.add_argument("--max_new_token_num", type=int, default=16)
    parser.add_argument("--qwen_res", type=int, default=256)
    parser.add_argument("--save_dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--save_mode", choices=["compressed", "raw"], default="compressed")
    parser.add_argument(
        "--hf_home",
        default=os.environ.get("HF_HOME", ""),
        help="Hugging Face cache root. Remote model ids require a non-home path.",
    )
    parser.add_argument(
        "--max_videos",
        "--max-videos",
        type=int,
        default=0,
        help="Process at most this many videos assigned to the current rank (0=all).",
    )
    parser.add_argument(
        "--self_test",
        "--self-test",
        action="store_true",
        help="Run pure sampling/validator tests and exit before model loading.",
    )
    return parser.parse_args(argv)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def ensure_write_path_outside_home(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    home = Path.home().resolve()
    if _is_relative_to(resolved, home):
        raise ValueError(
            f"Refusing to write {label} under the home filesystem: {resolved}. "
            "Use /projects, /project, /work, or another approved shared filesystem."
        )
    return resolved


def configure_hf_home(args: argparse.Namespace) -> None:
    pretrained_path = Path(args.pretrained).expanduser()
    is_local_model = pretrained_path.exists()
    if not args.hf_home:
        if is_local_model:
            return
        raise ValueError(
            "Remote --pretrained requires --hf_home (or HF_HOME) on a non-home "
            "shared filesystem; implicit ~/.cache writes are disabled."
        )
    hf_home = ensure_write_path_outside_home(Path(args.hf_home), "Hugging Face cache")
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")


def _manifest_item_path(item: Any) -> str | None:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, Mapping):
        for key in ("video_path", "raw_video_path", "path", "file", "mp4"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def load_manifest_paths(path: Path, dataset_root: Path) -> list[Path]:
    text = path.read_text(encoding="utf-8")
    items: list[Any] = []
    stripped = text.lstrip()
    if stripped.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError(f"JSON manifest must contain a list: {path}")
        items.extend(parsed)
    else:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("{"):
                try:
                    items.append(json.loads(line))
                    continue
                except json.JSONDecodeError:
                    pass
            items.append(line)

    output: list[Path] = []
    for item in items:
        raw = _manifest_item_path(item)
        if not raw:
            continue
        if raw.startswith("file://"):
            raw = raw[7:]
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = dataset_root / candidate
        if candidate.suffix.lower() in VIDEO_SUFFIXES:
            output.append(candidate.resolve())
    return output


def discover_videos(dataset_root: Path, manifest: str = "") -> list[Path]:
    if manifest:
        videos = load_manifest_paths(Path(manifest).expanduser().resolve(), dataset_root)
    else:
        videos = [
            path.resolve()
            for path in dataset_root.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ]
    unique = {str(path): path for path in videos}
    return sorted(unique.values(), key=lambda path: stable_video_key(path, dataset_root))


def stable_video_key(video_path: Path, dataset_root: Path) -> str:
    try:
        return video_path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except ValueError:
        return "external/" + video_path.resolve().as_posix().lstrip("/")


def stable_rank_for_video(video_path: Path, dataset_root: Path, world_size: int) -> int:
    if world_size <= 1:
        return 0
    key = stable_video_key(video_path, dataset_root).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % world_size


def discover_distributed_runtime() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    if world_size < 1 or rank < 0 or rank >= world_size or local_rank < 0:
        raise ValueError(
            f"Invalid distributed runtime: rank={rank} world_size={world_size} "
            f"local_rank={local_rank}"
        )
    return rank, world_size, local_rank


def canonical_output_path(video_path: Path, dataset_root: Path, output_root: Path) -> Path:
    try:
        relative = video_path.resolve().relative_to(dataset_root.resolve())
        # Dataset splitters pair video/cache stems: x.mp4 must map to x.npz.
        return output_root / relative.with_suffix(".npz")
    except ValueError:
        digest = _sha256_text(video_path.resolve().as_posix())[:16]
        return output_root / "_external" / digest / video_path.with_suffix(".npz").name


def source_video_relpath(video_path: Path, dataset_root: Path) -> str:
    """Portable source identity that never records a machine absolute path."""
    try:
        return video_path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except ValueError:
        digest = _sha256_text(video_path.resolve().as_posix())[:16]
        return f"_external/{digest}/{video_path.name}"


@dataclass(frozen=True)
class FrameSelection:
    total_frames: int
    source_fps: float
    source_time_start: float
    source_time_end: float
    time_midpoint: float
    split_raw_index: int
    past_indices: np.ndarray
    target_indices: np.ndarray
    past_times: np.ndarray
    target_times: np.ndarray


def uniform_indices(start: int, stop_exclusive: int, count: int) -> np.ndarray:
    """Uniform, monotonic integer samples from a non-empty half-open interval."""
    if count < 1:
        raise ValueError("count must be positive")
    if stop_exclusive <= start:
        raise ValueError(f"empty source interval [{start}, {stop_exclusive})")
    if stop_exclusive - start == 1:
        return np.full((count,), start, dtype=np.int64)
    # Match trajectory_dataset._sample_dense_jepa_frame_indices exactly so
    # cached visual features and HDF5 supervision address identical raw frames.
    return np.linspace(start, stop_exclusive - 1, num=count, dtype=np.int64)


def split_and_sample_raw_index_midpoint(
    frame_timestamps: np.ndarray,
    count_per_side: int = NUM_PAST_FRAMES,
    source_fps: float = 0.0,
) -> FrameSelection:
    """Split at raw ``T//2`` (supervision policy), then sample both intervals."""
    timestamps = np.asarray(frame_timestamps, dtype=np.float64)
    if timestamps.ndim == 2 and timestamps.shape[1] >= 2:
        starts = timestamps[:, 0]
        ends = timestamps[:, 1]
        centers = (starts + ends) * 0.5
        time_start = float(starts[0])
        time_end = float(ends[-1])
    elif timestamps.ndim == 1:
        centers = timestamps
        time_start = float(centers[0])
        if centers.size > 1:
            last_step = max(float(centers[-1] - centers[-2]), 0.0)
        else:
            last_step = 1.0 / max(float(source_fps), 1.0)
        time_end = float(centers[-1] + last_step)
    else:
        raise ValueError(f"Unsupported timestamp shape: {timestamps.shape}")

    total = int(centers.size)
    if total < 2:
        raise ValueError(f"Need at least two source frames, found {total}")
    if not np.all(np.isfinite(centers)) or np.any(np.diff(centers) < 0):
        raise ValueError("Source frame timestamps must be finite and monotonic")
    if not np.isfinite(time_start) or not np.isfinite(time_end) or time_end <= time_start:
        # A deterministic constant-FPS fallback still splits disjoint raw intervals.
        fps = max(float(source_fps), 1.0)
        centers = np.arange(total, dtype=np.float64) / fps
        time_start = 0.0
        time_end = total / fps

    split_index = total // 2
    # Timestamp at the raw-index boundary is provenance only; it never chooses
    # the split and therefore cannot drift from trajectory supervision.
    midpoint = float((centers[split_index - 1] + centers[split_index]) * 0.5)
    past = uniform_indices(0, split_index, count_per_side)
    target = uniform_indices(split_index, total, count_per_side)
    if np.intersect1d(np.unique(past), np.unique(target)).size:
        raise AssertionError("past and target raw frame-index sets overlap")
    if int(past.max()) >= int(target.min()):
        raise AssertionError("past samples are not strictly before target samples")
    return FrameSelection(
        total_frames=total,
        source_fps=float(source_fps),
        source_time_start=float(time_start),
        source_time_end=float(time_end),
        time_midpoint=float(midpoint),
        split_raw_index=split_index,
        past_indices=past,
        target_indices=target,
        past_times=centers[past].astype(np.float64),
        target_times=centers[target].astype(np.float64),
    )


def inspect_and_decode_video(video_path: Path) -> tuple[FrameSelection, np.ndarray, np.ndarray]:
    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise RuntimeError("decord is required to decode raw MP4 files") from exc

    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    total = int(len(reader))
    if total < 2:
        raise ValueError(f"Need at least two frames: {video_path} has {total}")
    fps = float(reader.get_avg_fps())
    if not np.isfinite(fps) or fps <= 0:
        fps = 0.0
    try:
        timestamps = np.asarray(
            reader.get_frame_timestamp(np.arange(total, dtype=np.int64)),
            dtype=np.float64,
        )
    except Exception:
        fps_safe = fps if fps > 0 else 30.0
        starts = np.arange(total, dtype=np.float64) / fps_safe
        timestamps = np.stack([starts, starts + 1.0 / fps_safe], axis=1)
    selection = split_and_sample_raw_index_midpoint(timestamps, source_fps=fps)

    all_indices = np.concatenate([selection.past_indices, selection.target_indices])
    decoded = np.asarray(reader.get_batch(all_indices).asnumpy())
    if decoded.ndim != 4 or decoded.shape[0] != NUM_ALL_FRAMES or decoded.shape[-1] < 3:
        raise ValueError(f"Unexpected decoded video shape {decoded.shape} for {video_path}")
    decoded = np.ascontiguousarray(decoded[..., :3].astype(np.uint8, copy=False))
    return selection, decoded[:NUM_PAST_FRAMES], decoded[NUM_PAST_FRAMES:]


def preprocess_vjepa_frames(frames: np.ndarray) -> tuple[Any, np.ndarray]:
    """Official deterministic V-JEPA eval geometry and ImageNet normalization."""
    import torch
    from cache_train.video_observation_adapter import resize_short_side_center_crop

    # Match VideoObservationAdapter exactly: convert to unit float before
    # interpolation so cached and online V-JEPA paths do not differ through
    # uint8 interpolation quantization.
    x = (
        torch.from_numpy(np.ascontiguousarray(frames))
        .permute(0, 3, 1, 2)
        .to(dtype=torch.float32)
        .div_(255.0)
    )
    short_side = int(256.0 / 224.0 * VJEPA_IMAGE_SIZE)
    x = resize_short_side_center_crop(
        x,
        short_side=short_side,
        crop_height=VJEPA_IMAGE_SIZE,
        crop_width=VJEPA_IMAGE_SIZE,
        antialias=True,
    )
    imgs = (
        x.mul(255.0)
        .round()
        .clamp_(0, 255)
        .to(dtype=torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
        .cpu()
        .numpy()
    )
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)
    x = x.sub_(mean).div_(std).permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    return x, np.ascontiguousarray(imgs)


def _torch_load_weights(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, weights_only=True, map_location="cpu")
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_encoder_state_dict(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("V-JEPA checkpoint must be a mapping/state_dict")
    state: Any = checkpoint
    for key in ("encoder", "target_encoder", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            state = candidate
            break
    if not isinstance(state, Mapping):
        raise TypeError("Could not locate encoder weights in V-JEPA checkpoint")
    cleaned: dict[str, Any] = {}
    for raw_key, value in state.items():
        key = str(raw_key)
        for prefix in ("module.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def load_vjepa_encoder(checkpoint_path: Path, device: Any) -> Any:
    import torch

    for path in (REPO_ROOT, VJEPA2_ROOT, VJEPA2_ROOT.parent):
        path_string = str(path)
        if path_string not in sys.path:
            sys.path.insert(0, path_string)
    from vjepa2.src.models.vision_transformer import vit_large_rope

    model = vit_large_rope(
        img_size=(VJEPA_IMAGE_SIZE, VJEPA_IMAGE_SIZE),
        patch_size=VJEPA_PATCH_SIZE,
        tubelet_size=VJEPA_TUBELET_SIZE,
        num_frames=NUM_PAST_FRAMES,
        use_sdpa=True,
        uniform_power=False,
        use_silu=False,
        wide_silu=True,
    )
    state = _extract_encoder_state_dict(_torch_load_weights(checkpoint_path))
    incompat = model.load_state_dict(state, strict=False)
    missing = [key for key in incompat.missing_keys if key != "pos_embed"]
    unexpected = [key for key in incompat.unexpected_keys if key != "pos_embed"]
    if missing or unexpected:
        raise RuntimeError(
            "V-JEPA ViT-L checkpoint/model mismatch. "
            f"missing={missing[:20]} unexpected={unexpected[:20]}"
        )
    model.eval().requires_grad_(False)
    if device.type == "cuda":
        model = model.to(device=device, dtype=torch.float16)
    else:
        model = model.to(device=device, dtype=torch.float32)
    return model


def encode_vjepa_clip(model: Any, clip: Any, device: Any) -> np.ndarray:
    import torch

    compute_dtype = torch.float16 if device.type == "cuda" else torch.float32
    clip = clip.to(device=device, dtype=compute_dtype, non_blocking=True)
    with torch.inference_mode():
        output = model(clip)
    expected_flat = VJEPA_TEMPORAL_TOKENS * VJEPA_SPATIAL_TOKENS
    if tuple(output.shape) != (1, expected_flat, VJEPA_EMBED_DIM):
        raise ValueError(
            f"V-JEPA output must be [1,{expected_flat},{VJEPA_EMBED_DIM}], "
            f"got {tuple(output.shape)}"
        )
    return output[0].reshape(VJEPA_FEATURE_SHAPE).float().cpu().numpy()


def _infer_hf_checkpoint_sha(model: Any, processor: Any, explicit_sha: str) -> str:
    if explicit_sha.strip():
        return explicit_sha.strip()
    candidates: list[Any] = [
        getattr(getattr(model, "config", None), "_commit_hash", None),
        getattr(getattr(processor, "tokenizer", None), "_commit_hash", None),
        getattr(processor, "_commit_hash", None),
    ]
    for object_with_path in (model, getattr(model, "config", None), processor):
        candidates.extend(
            [
                getattr(object_with_path, "name_or_path", None),
                getattr(object_with_path, "_name_or_path", None),
            ]
        )
    for candidate in candidates:
        if not candidate:
            continue
        match = re.search(r"(?:snapshots[/\\])([0-9a-fA-F]{40,64})(?:[/\\]|$)", str(candidate))
        if match:
            return match.group(1).lower()
        if re.fullmatch(r"[0-9a-fA-F]{40,64}", str(candidate)):
            return str(candidate).lower()
    raise RuntimeError(
        "Could not infer the resolved Qwen checkpoint commit SHA. "
        "Pass --qwen_checkpoint_sha explicitly."
    )


def load_qwen(args: argparse.Namespace, device: Any) -> tuple[Any, Any, dict[str, list[Any]], str]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    kwargs = {"revision": args.qwen_revision, "device_map": None}
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.pretrained, torch_dtype="auto", **kwargs
        )
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            args.pretrained, dtype="auto", **kwargs
        )
    processor = AutoProcessor.from_pretrained(args.pretrained, revision=args.qwen_revision)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"
    model.eval().requires_grad_(False).to(device)

    from cache_train.qwen3_feature_hooks import (
        locate_thinker_decoder_layers,
        register_thinker_decoder_hooks,
    )

    decoder_layers, layers_path = locate_thinker_decoder_layers(model)
    for layer in args.layers:
        if layer < 0 or layer >= len(decoder_layers):
            raise ValueError(
                f"Qwen layer {layer} is outside [0, {len(decoder_layers) - 1}]"
            )
    saved = register_thinker_decoder_hooks(decoder_layers, list(args.layers))
    checkpoint_sha = _infer_hf_checkpoint_sha(
        model, processor, args.qwen_checkpoint_sha
    )
    print(
        f"[INFO] Qwen hooks={layers_path} layers={list(args.layers)} "
        f"checkpoint_sha={checkpoint_sha}",
        flush=True,
    )
    return model, processor, saved, checkpoint_sha


def _effective_sample_fps(frame_times: np.ndarray, fallback_fps: float) -> float:
    times = np.asarray(frame_times, dtype=np.float64)
    unique = np.unique(times)
    if unique.size >= 2 and unique[-1] > unique[0]:
        return float((unique.size - 1) / (unique[-1] - unique[0]))
    fallback = float(fallback_fps)
    return fallback if np.isfinite(fallback) and fallback > 0 else 1.0


def infer_qwen_past_only(
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    saved: dict[str, list[Any]],
    device: Any,
    past_frames: np.ndarray,
    selection: FrameSelection,
) -> dict[str, Any]:
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from cache_train.qwen3_feature_hooks import (
        stack_pyramid_guidance_states_per_sample,
    )

    if past_frames.shape[0] != NUM_PAST_FRAMES:
        raise ValueError(f"Qwen requires exactly 32 past frames, got {past_frames.shape}")
    for values in saved.values():
        values.clear()

    exact_past_pil = [Image.fromarray(frame) for frame in past_frames]
    effective_fps = _effective_sample_fps(selection.past_times, selection.source_fps)
    video_content = {
        "type": "video",
        "video": exact_past_pil,
        "resized_height": int(args.qwen_res),
        "resized_width": int(args.qwen_res),
        "sample_fps": effective_fps,
        "raw_fps": effective_fps,
    }
    messages = [
        {
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": str(args.prompt)},
            ],
        }
    ]
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if not videos or len(videos) != 1:
        raise RuntimeError("Qwen vision processing did not return exactly one video")
    video_tensors: list[Any] = []
    video_metadata: list[Any] = []
    for value in videos:
        if isinstance(value, tuple):
            video_tensors.append(value[0])
            video_metadata.append(value[1])
        else:
            video_tensors.append(value)
    if int(video_tensors[0].shape[0]) != NUM_PAST_FRAMES:
        raise AssertionError(
            "Qwen vision processing changed the exact past-frame count: "
            f"{tuple(video_tensors[0].shape)}"
        )

    rendered_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    processor_kwargs: dict[str, Any] = {
        "text": [rendered_text],
        "videos": video_tensors,
        "return_tensors": "pt",
        "padding": True,
        "do_resize": False,
    }
    if images:
        processor_kwargs["images"] = images
    if video_metadata:
        processor_kwargs["video_metadata"] = video_metadata
    processor_kwargs.update(video_kwargs or {})
    inputs = processor(**processor_kwargs)
    input_ids_cpu = inputs["input_ids"].detach().cpu()
    attention_mask = inputs.get("attention_mask")
    valid_len = (
        int(attention_mask[0].sum().item())
        if torch.is_tensor(attention_mask)
        else int(input_ids_cpu.shape[1])
    )
    inputs = {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(args.max_new_token_num),
            # The checkpoint generation_config enables multinomial sampling.
            # A cache must be reproducible across ranks, retries, and partial
            # rebuilds, so the safe default is explicit greedy decoding.
            do_sample=False,
        )
    prompt_width = int(inputs["input_ids"].shape[1])
    new_ids = generated[0, prompt_width:]
    text = processor.batch_decode(
        [new_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    states = stack_pyramid_guidance_states_per_sample(
        saved,
        list(args.layers),
        batch_size=1,
        valid_lens=[valid_len],
    )
    if len(states) != 1:
        raise RuntimeError("Qwen hook extraction returned no per-sample state")
    vlm_old, vlm_new = states[0]
    # Remove the singleton inference batch dimension: [L,1,S,D] -> [L,S,D].
    if vlm_old.ndim == 4 and vlm_old.shape[1] == 1:
        vlm_old = vlm_old[:, 0]
    if vlm_new.ndim == 4 and vlm_new.shape[1] == 1:
        vlm_new = vlm_new[:, 0]
    if vlm_old.ndim != 3 or vlm_new.ndim != 3:
        raise ValueError(
            f"Unexpected Qwen states: old={tuple(vlm_old.shape)} new={tuple(vlm_new.shape)}"
        )
    if vlm_old.shape[0] != len(args.layers) or vlm_new.shape[0] != len(args.layers):
        raise ValueError(
            f"Missing Qwen hooked layers: old={tuple(vlm_old.shape)} "
            f"new={tuple(vlm_new.shape)} layers={list(args.layers)}"
        )
    return {
        "text": np.asarray(text),
        "token_ids": new_ids.detach().cpu().numpy().astype(np.int32),
        "input_ids": input_ids_cpu[0].numpy().astype(np.int32),
        "input_valid_len": np.asarray(valid_len, dtype=np.int32),
        "vlm_old_tensor": vlm_old,
        "vlm_new_tensor": vlm_new,
        "effective_sample_fps": np.asarray(effective_fps, dtype=np.float32),
    }


def tensor_to_numpy(tensor: Any, save_dtype: str) -> np.ndarray:
    if save_dtype == "fp16":
        return tensor.detach().float().half().cpu().numpy()
    return tensor.detach().float().cpu().numpy()


def source_stat(video_path: Path, dataset_root: Path) -> dict[str, Any]:
    stat = video_path.stat()
    return {
        "relpath": source_video_relpath(video_path, dataset_root),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def build_global_configuration(
    args: argparse.Namespace,
    qwen_checkpoint_sha: str,
    vjepa_checkpoint_sha: str,
) -> tuple[dict[str, Any], str]:
    config: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "causality_mode": "observed_past",
        "vlm_observation_policy": VLM_OBSERVATION_POLICY,
        "sampling_strategy": SAMPLING_STRATEGY,
        # Make code changes part of the cache identity.  This prevents an
        # extractor implementation change from silently reusing stale files
        # when the human-readable schema/version was not bumped by mistake.
        "extractor_source_sha256": sha256_file(Path(__file__).resolve()),
        "past_frames": NUM_PAST_FRAMES,
        "target_frames": NUM_TARGET_FRAMES,
        "qwen": {
            "model": args.pretrained,
            "revision": args.qwen_revision,
            "checkpoint_sha": qwen_checkpoint_sha,
            "layers": list(args.layers),
            "prompt": args.prompt,
            "dataset_prompt_mode": args.dataset_prompt_mode,
            "res": int(args.qwen_res),
            "max_new_tokens": int(args.max_new_token_num),
            "generation_strategy": "greedy",
            "do_sample": False,
            "input_scope": "exact_decoded_past_indices_only",
            "attention_mask": "model_native_decoder_causal",
        },
        "vjepa": {
            "model": "V-JEPA2 ViT-L RoPE",
            "checkpoint_sha256": vjepa_checkpoint_sha,
            "input_scope": "past_only_separate_forward",
            "target_scope": "target_only_separate_forward",
            "frames_per_forward": NUM_PAST_FRAMES,
            "image_size": VJEPA_IMAGE_SIZE,
            "patch_size": VJEPA_PATCH_SIZE,
            "tubelet_size": VJEPA_TUBELET_SIZE,
            "feature_shape": list(VJEPA_FEATURE_SHAPE),
        },
        "imgs": {
            "layout": "past32_then_target32_THWC_RGB_uint8",
            "geometry": "vjepa_eval_resize_short292_center_crop256",
        },
        "save_dtype": args.save_dtype,
    }
    config_json = _json_dumps(config)
    return config, _sha256_text(config_json)


def _source_fingerprint(stat: Mapping[str, Any]) -> str:
    return _sha256_text(
        _json_dumps(
            {
                "relpath": str(stat["relpath"]),
                "size_bytes": int(stat["size_bytes"]),
                "mtime_ns": int(stat["mtime_ns"]),
            }
        )
    )


def build_payload(
    args: argparse.Namespace,
    video_path: Path,
    dataset_root: Path,
    selection: FrameSelection,
    imgs: np.ndarray,
    vjepa_input_feats: np.ndarray,
    vjepa_target_feats: np.ndarray,
    qwen_result: Mapping[str, Any],
    config: Mapping[str, Any],
    config_fingerprint: str,
    qwen_checkpoint_sha: str,
    vjepa_checkpoint_sha: str,
) -> dict[str, Any]:
    stat = source_stat(video_path, dataset_root)
    provenance = {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "extractor_version": EXTRACTOR_VERSION,
        "created_unix": time.time(),
        "causality": {
            "mode": "observed_past",
            "qwen_input": "exact decoded frames at past_raw_indices only",
            "vjepa_input": "separate encoder forward over past_raw_indices only",
            "vjepa_target": "separate encoder forward over target_raw_indices only",
            "past_target_raw_index_sets_disjoint": True,
        },
        "source": stat,
        "sampling": {
            "strategy": SAMPLING_STRATEGY,
            "source_time_start_seconds": selection.source_time_start,
            "source_time_end_seconds": selection.source_time_end,
            "source_time_midpoint_seconds": selection.time_midpoint,
            "split_raw_index": selection.split_raw_index,
            "past_raw_indices": selection.past_indices.tolist(),
            "target_raw_indices": selection.target_indices.tolist(),
        },
        "configuration": config,
        "configuration_fingerprint": config_fingerprint,
    }
    payload: dict[str, Any] = {
        "schema_name": np.asarray(SCHEMA_NAME),
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
        "cache_schema_name": np.asarray(SCHEMA_NAME),
        "cache_schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
        "vlm_observation_policy": np.asarray(VLM_OBSERVATION_POLICY),
        "extractor_version": np.asarray(EXTRACTOR_VERSION),
        "causality_mode": np.asarray("observed_past"),
        "observed_past": np.asarray(True, dtype=np.bool_),
        "configuration_fingerprint": np.asarray(config_fingerprint),
        "configuration_json": np.asarray(_json_dumps(config)),
        "provenance_json": np.asarray(_json_dumps(provenance)),
        "source_video_relpath": np.asarray(stat["relpath"]),
        "source_size_bytes": np.asarray(stat["size_bytes"], dtype=np.int64),
        "source_mtime_ns": np.asarray(stat["mtime_ns"], dtype=np.int64),
        "source_fingerprint": np.asarray(_source_fingerprint(stat)),
        "source_total_frames": np.asarray(selection.total_frames, dtype=np.int64),
        "source_avg_fps": np.asarray(selection.source_fps, dtype=np.float32),
        "source_time_start_seconds": np.asarray(selection.source_time_start, dtype=np.float64),
        "source_time_end_seconds": np.asarray(selection.source_time_end, dtype=np.float64),
        "source_time_midpoint_seconds": np.asarray(selection.time_midpoint, dtype=np.float64),
        "split_raw_index": np.asarray(selection.split_raw_index, dtype=np.int64),
        "sampling_strategy": np.asarray(SAMPLING_STRATEGY),
        "past_raw_indices": selection.past_indices.astype(np.int64),
        "target_raw_indices": selection.target_indices.astype(np.int64),
        "qwen_raw_indices": selection.past_indices.astype(np.int64),
        "vjepa_input_frame_indices": selection.past_indices.astype(np.int64),
        "vjepa_target_frame_indices": selection.target_indices.astype(np.int64),
        "vlm_observation_frame_indices": selection.past_indices.astype(np.int64),
        "past_frame_times_seconds": selection.past_times.astype(np.float64),
        "target_frame_times_seconds": selection.target_times.astype(np.float64),
        "imgs": np.ascontiguousarray(imgs.astype(np.uint8, copy=False)),
        "vjepa_input_feats": tensor_or_array_to_dtype(vjepa_input_feats, args.save_dtype),
        "vjepa_target_feats": tensor_or_array_to_dtype(vjepa_target_feats, args.save_dtype),
        "vjepa_model": np.asarray("vjepa2_vit_large_rope"),
        "vjepa_checkpoint_sha256": np.asarray(vjepa_checkpoint_sha),
        "vjepa_tubelet_size": np.asarray(VJEPA_TUBELET_SIZE, dtype=np.int32),
        "vjepa_patch_size": np.asarray(VJEPA_PATCH_SIZE, dtype=np.int32),
        "vjepa_image_size": np.asarray(VJEPA_IMAGE_SIZE, dtype=np.int32),
        "qwen_model": np.asarray(args.pretrained),
        "qwen_revision": np.asarray(args.qwen_revision),
        "qwen_checkpoint_sha": np.asarray(qwen_checkpoint_sha),
        "qwen_input_scope": np.asarray("exact_past32_decoded_frames"),
        "qwen_effective_sample_fps": np.asarray(qwen_result["effective_sample_fps"]),
        "vlm_attention_mask": np.asarray("model_native_decoder_causal"),
        "prompt_base": np.asarray(args.prompt),
        "prompt_overlay": np.asarray(""),
        "prompt_full": np.asarray(args.prompt),
        "dataset_prompt_mode": np.asarray("off"),
        "layers": np.asarray(args.layers, dtype=np.int32),
        "cache_config_fingerprint": np.asarray(config_fingerprint),
        "text": np.asarray(qwen_result["text"]),
        "token_ids": np.asarray(qwen_result["token_ids"], dtype=np.int32),
        "vlm_new_token_ids": np.asarray(qwen_result["token_ids"], dtype=np.int32)[
            : int(qwen_result["vlm_new_tensor"].shape[1])
        ],
        "input_ids": np.asarray(qwen_result["input_ids"], dtype=np.int32),
        "input_valid_len": np.asarray(qwen_result["input_valid_len"], dtype=np.int32),
        "vlm_old": tensor_to_numpy(qwen_result["vlm_old_tensor"], args.save_dtype),
        "vlm_new": tensor_to_numpy(qwen_result["vlm_new_tensor"], args.save_dtype),
        "nframes_req": np.asarray(NUM_PAST_FRAMES, dtype=np.int32),
        "nframes_used": np.asarray(NUM_PAST_FRAMES, dtype=np.int32),
        "res": np.asarray(args.qwen_res, dtype=np.int32),
    }
    return payload


def tensor_or_array_to_dtype(value: Any, save_dtype: str) -> np.ndarray:
    array = np.asarray(value)
    if save_dtype == "fp16":
        return array.astype(np.float16, copy=False)
    return array.astype(np.float32, copy=False)


REQUIRED_KEYS = {
    "schema_name",
    "schema_version",
    "cache_schema_name",
    "cache_schema_version",
    "vlm_observation_policy",
    "extractor_version",
    "causality_mode",
    "observed_past",
    "configuration_fingerprint",
    "configuration_json",
    "provenance_json",
    "source_video_relpath",
    "source_size_bytes",
    "source_mtime_ns",
    "source_total_frames",
    "source_time_midpoint_seconds",
    "split_raw_index",
    "sampling_strategy",
    "past_raw_indices",
    "target_raw_indices",
    "qwen_raw_indices",
    "vjepa_input_frame_indices",
    "vjepa_target_frame_indices",
    "vlm_observation_frame_indices",
    "past_frame_times_seconds",
    "target_frame_times_seconds",
    "imgs",
    "vjepa_input_feats",
    "vjepa_target_feats",
    "vjepa_model",
    "vlm_old",
    "vlm_new",
    "vlm_attention_mask",
    "token_ids",
    "vlm_new_token_ids",
    "input_ids",
    "input_valid_len",
    "layers",
    "cache_config_fingerprint",
    "text",
    "qwen_model",
    "qwen_checkpoint_sha",
    "qwen_input_scope",
    "prompt_base",
    "prompt_overlay",
    "prompt_full",
    "dataset_prompt_mode",
    "vjepa_checkpoint_sha256",
    "vjepa_tubelet_size",
    "vjepa_patch_size",
    "vjepa_image_size",
    "nframes_req",
    "nframes_used",
}


def validate_archive(
    path: Path,
    expected_config_fingerprint: str,
    expected_layers: Sequence[int],
    expected_save_dtype: str,
    expected_source: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            missing = sorted(REQUIRED_KEYS - keys)
            if missing:
                return False, f"missing keys: {missing}"
            forbidden = sorted(
                (keys & FEATURE_CACHE_FORBIDDEN_KEYS)
                | {key for key in keys if key.startswith("egoexo_")}
            )
            if forbidden:
                return False, f"supervision/legacy keys are forbidden in feature cache: {forbidden}"
            if int(_scalar(archive["schema_version"])) != SCHEMA_VERSION:
                return False, "schema_version mismatch"
            if _string_scalar(archive["schema_name"]) != SCHEMA_NAME:
                return False, "schema_name mismatch"
            if int(_scalar(archive["cache_schema_version"])) != SCHEMA_VERSION:
                return False, "cache_schema_version mismatch"
            if _string_scalar(archive["cache_schema_name"]) != SCHEMA_NAME:
                return False, "cache_schema_name mismatch"
            if _string_scalar(archive["vlm_observation_policy"]) != VLM_OBSERVATION_POLICY:
                return False, "vlm_observation_policy mismatch"
            if _string_scalar(archive["causality_mode"]) != "observed_past":
                return False, "causality_mode is not observed_past"
            if not bool(_scalar(archive["observed_past"])):
                return False, "observed_past flag is false"
            if _string_scalar(archive["configuration_fingerprint"]) != expected_config_fingerprint:
                return False, "configuration_fingerprint mismatch"
            if _string_scalar(archive["cache_config_fingerprint"]) != expected_config_fingerprint:
                return False, "cache_config_fingerprint mismatch"
            configuration_json = _string_scalar(archive["configuration_json"])
            if _sha256_text(configuration_json) != expected_config_fingerprint:
                return False, "configuration_json does not match its fingerprint"
            configuration = json.loads(configuration_json)
            if configuration.get("causality_mode") != "observed_past":
                return False, "configuration does not enforce observed_past"

            past = np.asarray(archive["past_raw_indices"], dtype=np.int64)
            target = np.asarray(archive["target_raw_indices"], dtype=np.int64)
            qwen = np.asarray(archive["qwen_raw_indices"], dtype=np.int64)
            if past.shape != (NUM_PAST_FRAMES,) or target.shape != (NUM_TARGET_FRAMES,):
                return False, f"bad index shapes: past={past.shape} target={target.shape}"
            if not np.array_equal(qwen, past):
                return False, "Qwen indices are not exactly the past indices"
            if not np.array_equal(
                np.asarray(archive["vjepa_input_frame_indices"], dtype=np.int64), past
            ):
                return False, "V-JEPA input indices are not exactly the past indices"
            if not np.array_equal(
                np.asarray(archive["vjepa_target_frame_indices"], dtype=np.int64), target
            ):
                return False, "V-JEPA target indices do not match target_raw_indices"
            if not np.array_equal(
                np.asarray(archive["vlm_observation_frame_indices"], dtype=np.int64), past
            ):
                return False, "VLM observation indices are not exactly the past indices"
            if np.intersect1d(np.unique(past), np.unique(target)).size:
                return False, "past/target raw index sets overlap"
            if int(past.max()) >= int(target.min()):
                return False, "past indices are not strictly before target indices"
            total = int(_scalar(archive["source_total_frames"]))
            if past.min() < 0 or target.max() >= total:
                return False, "raw indices are outside source bounds"
            if np.any(np.diff(past) < 0) or np.any(np.diff(target) < 0):
                return False, "sampled indices are not monotonic"
            if int(_scalar(archive["split_raw_index"])) <= int(past.max()):
                return False, "split index does not follow the past interval"
            if int(_scalar(archive["split_raw_index"])) > int(target.min()):
                return False, "split index does not begin the target interval"
            if np.asarray(archive["past_frame_times_seconds"]).shape != (NUM_PAST_FRAMES,):
                return False, "bad past timestamp shape"
            if np.asarray(archive["target_frame_times_seconds"]).shape != (NUM_TARGET_FRAMES,):
                return False, "bad target timestamp shape"

            imgs = archive["imgs"]
            if imgs.shape != (NUM_ALL_FRAMES, VJEPA_IMAGE_SIZE, VJEPA_IMAGE_SIZE, 3):
                return False, f"bad imgs shape: {imgs.shape}"
            if imgs.dtype != np.uint8:
                return False, f"imgs dtype must be uint8, got {imgs.dtype}"
            for key in ("vjepa_input_feats", "vjepa_target_feats"):
                value = archive[key]
                if value.shape != VJEPA_FEATURE_SHAPE:
                    return False, f"bad {key} shape: {value.shape}"
                expected_dtype = np.float16 if expected_save_dtype == "fp16" else np.float32
                if value.dtype != expected_dtype:
                    return False, f"bad {key} dtype: {value.dtype}"
                if not np.all(np.isfinite(value)):
                    return False, f"non-finite {key}"

            expected_layers_array = np.asarray(expected_layers, dtype=np.int32)
            if not np.array_equal(np.asarray(archive["layers"], dtype=np.int32), expected_layers_array):
                return False, "layers mismatch"
            for key in ("vlm_old", "vlm_new"):
                value = archive[key]
                if value.ndim != 3 or value.shape[0] != len(expected_layers):
                    return False, f"bad {key} shape: {value.shape}"
                if value.shape[1] < 1 or value.shape[2] < 1:
                    return False, f"empty {key}"
                if not np.all(np.isfinite(value)):
                    return False, f"non-finite {key}"
            if archive["token_ids"].ndim != 1 or archive["input_ids"].ndim != 1:
                return False, "token_ids/input_ids must be 1D"
            if not np.array_equal(
                np.asarray(archive["vlm_new_token_ids"], dtype=np.int32),
                np.asarray(archive["token_ids"], dtype=np.int32)[: archive["vlm_new"].shape[1]],
            ):
                return False, "vlm_new_token_ids are not aligned with vlm_new states"
            if archive["vlm_new"].shape[1] != max(int(archive["token_ids"].size) - 1, 0):
                return False, "vlm_new state/token count mismatch"
            if archive["input_ids"].size < 1:
                return False, "input_ids is empty"
            input_valid_len = int(_scalar(archive["input_valid_len"]))
            if input_valid_len < 1 or input_valid_len > archive["input_ids"].size:
                return False, "invalid input_valid_len"
            if archive["vlm_old"].shape[1] != input_valid_len:
                return False, "vlm_old token count differs from input_valid_len"
            if int(_scalar(archive["vjepa_tubelet_size"])) != VJEPA_TUBELET_SIZE:
                return False, "tubelet size mismatch"
            if int(_scalar(archive["vjepa_patch_size"])) != VJEPA_PATCH_SIZE:
                return False, "patch size mismatch"
            if int(_scalar(archive["vjepa_image_size"])) != VJEPA_IMAGE_SIZE:
                return False, "V-JEPA image size mismatch"
            if int(_scalar(archive["nframes_req"])) != NUM_PAST_FRAMES:
                return False, "nframes_req mismatch"
            if int(_scalar(archive["nframes_used"])) != NUM_PAST_FRAMES:
                return False, "nframes_used mismatch"
            if not re.fullmatch(
                r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}",
                _string_scalar(archive["qwen_checkpoint_sha"]),
            ):
                return False, "invalid Qwen checkpoint SHA"
            if not re.fullmatch(r"[0-9a-fA-F]{64}", _string_scalar(archive["vjepa_checkpoint_sha256"])):
                return False, "invalid V-JEPA checkpoint SHA256"
            if _string_scalar(archive["qwen_input_scope"]) != "exact_past32_decoded_frames":
                return False, "invalid Qwen input scope"
            if _string_scalar(archive["vlm_attention_mask"]) != "model_native_decoder_causal":
                return False, "invalid VLM attention-mask provenance"
            if _string_scalar(archive["dataset_prompt_mode"]) != "off":
                return False, "dataset metadata prompt must be off"
            if _string_scalar(archive["prompt_overlay"]):
                return False, "prompt overlay must be empty"
            if _string_scalar(archive["prompt_base"]) != _string_scalar(archive["prompt_full"]):
                return False, "prompt_full differs from metadata-free prompt_base"
            provenance = json.loads(_string_scalar(archive["provenance_json"]))
            if provenance.get("schema", {}).get("version") != SCHEMA_VERSION:
                return False, "invalid provenance schema"
            if provenance.get("causality", {}).get("qwen_input") != "exact decoded frames at past_raw_indices only":
                return False, "invalid Qwen provenance scope"
            if provenance.get("configuration_fingerprint") != expected_config_fingerprint:
                return False, "provenance configuration fingerprint mismatch"

            if expected_source is not None:
                if _string_scalar(archive["source_video_relpath"]) != str(expected_source["relpath"]):
                    return False, "source relative path changed"
                if int(_scalar(archive["source_size_bytes"])) != int(expected_source["size_bytes"]):
                    return False, "source size changed"
                if int(_scalar(archive["source_mtime_ns"])) != int(expected_source["mtime_ns"]):
                    return False, "source mtime changed"
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def atomic_write_archive(
    output_path: Path,
    payload: Mapping[str, Any],
    save_mode: str,
    config_fingerprint: str,
    layers: Sequence[int],
    save_dtype: str,
    expected_source: Mapping[str, Any] | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / (
        f".{output_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}.npz"
    )
    try:
        with temporary.open("wb") as handle:
            if save_mode == "raw":
                np.savez(handle, **payload)
            else:
                np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        valid, reason = validate_archive(
            temporary,
            expected_config_fingerprint=config_fingerprint,
            expected_layers=layers,
            expected_save_dtype=save_dtype,
            expected_source=expected_source,
        )
        if not valid:
            raise RuntimeError(f"refusing to publish invalid temporary archive: {reason}")
        os.replace(temporary, output_path)
        directory_fd = os.open(str(output_path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def append_failure(
    log_path: Path,
    rank: int,
    video_path: Path,
    dataset_root: Path,
    exc: BaseException,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_text = str(dataset_root.resolve())
    error_text = str(exc).replace(root_text, "<DATASET_ROOT>")
    traceback_text = traceback.format_exc().replace(root_text, "<DATASET_ROOT>")
    event = {
        "time_unix": time.time(),
        "rank": rank,
        "video_relpath": source_video_relpath(video_path, dataset_root),
        "exception_type": type(exc).__name__,
        "error": error_text,
        "traceback": traceback_text,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(_json_dumps(event) + "\n")
        handle.flush()


def _fake_payload_for_self_test(
    path: Path,
    config_fingerprint: str,
    configuration_json: str,
    selection: FrameSelection,
) -> dict[str, Any]:
    layers = np.asarray(DEFAULT_QWEN_LAYERS, dtype=np.int32)
    source = {"relpath": path.name, "size_bytes": 1, "mtime_ns": 2}
    provenance = {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "causality": {"qwen_input": "exact decoded frames at past_raw_indices only"},
        "configuration_fingerprint": config_fingerprint,
    }
    return {
        "schema_name": np.asarray(SCHEMA_NAME),
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
        "cache_schema_name": np.asarray(SCHEMA_NAME),
        "cache_schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
        "vlm_observation_policy": np.asarray(VLM_OBSERVATION_POLICY),
        "extractor_version": np.asarray(EXTRACTOR_VERSION),
        "causality_mode": np.asarray("observed_past"),
        "observed_past": np.asarray(True),
        "configuration_fingerprint": np.asarray(config_fingerprint),
        "cache_config_fingerprint": np.asarray(config_fingerprint),
        "configuration_json": np.asarray(configuration_json),
        "provenance_json": np.asarray(_json_dumps(provenance)),
        "source_video_relpath": np.asarray(source["relpath"]),
        "source_size_bytes": np.asarray(source["size_bytes"], dtype=np.int64),
        "source_mtime_ns": np.asarray(source["mtime_ns"], dtype=np.int64),
        "source_total_frames": np.asarray(selection.total_frames, dtype=np.int64),
        "source_time_midpoint_seconds": np.asarray(selection.time_midpoint),
        "split_raw_index": np.asarray(selection.split_raw_index, dtype=np.int64),
        "sampling_strategy": np.asarray(SAMPLING_STRATEGY),
        "past_raw_indices": selection.past_indices,
        "target_raw_indices": selection.target_indices,
        "qwen_raw_indices": selection.past_indices.copy(),
        "vjepa_input_frame_indices": selection.past_indices.copy(),
        "vjepa_target_frame_indices": selection.target_indices.copy(),
        "vlm_observation_frame_indices": selection.past_indices.copy(),
        "past_frame_times_seconds": selection.past_times,
        "target_frame_times_seconds": selection.target_times,
        "imgs": np.zeros(
            (NUM_ALL_FRAMES, VJEPA_IMAGE_SIZE, VJEPA_IMAGE_SIZE, 3), dtype=np.uint8
        ),
        "vjepa_input_feats": np.zeros(VJEPA_FEATURE_SHAPE, dtype=np.float16),
        "vjepa_target_feats": np.zeros(VJEPA_FEATURE_SHAPE, dtype=np.float16),
        "vjepa_model": np.asarray("vjepa2_vit_large_rope"),
        "vlm_old": np.zeros((len(layers), 2, 4), dtype=np.float16),
        "vlm_new": np.zeros((len(layers), 1, 4), dtype=np.float16),
        "vlm_attention_mask": np.asarray("model_native_decoder_causal"),
        "token_ids": np.asarray([1, 2], dtype=np.int32),
        "vlm_new_token_ids": np.asarray([1], dtype=np.int32),
        "input_ids": np.asarray([2, 3], dtype=np.int32),
        "input_valid_len": np.asarray(2, dtype=np.int32),
        "layers": layers,
        "text": np.asarray("test"),
        "qwen_model": np.asarray("test-qwen"),
        "qwen_checkpoint_sha": np.asarray("a" * 40),
        "qwen_input_scope": np.asarray("exact_past32_decoded_frames"),
        "prompt_base": np.asarray(DEFAULT_PROMPT),
        "prompt_overlay": np.asarray(""),
        "prompt_full": np.asarray(DEFAULT_PROMPT),
        "dataset_prompt_mode": np.asarray("off"),
        "vjepa_checkpoint_sha256": np.asarray("b" * 64),
        "vjepa_tubelet_size": np.asarray(VJEPA_TUBELET_SIZE, dtype=np.int32),
        "vjepa_patch_size": np.asarray(VJEPA_PATCH_SIZE, dtype=np.int32),
        "vjepa_image_size": np.asarray(VJEPA_IMAGE_SIZE, dtype=np.int32),
        "nframes_req": np.asarray(NUM_PAST_FRAMES, dtype=np.int32),
        "nframes_used": np.asarray(NUM_PAST_FRAMES, dtype=np.int32),
    }


def run_self_test() -> None:
    # Deliberately variable timestamps verify that the policy stays aligned to
    # raw T//2 supervision instead of silently reverting to a duration split.
    starts = np.concatenate(
        [np.arange(0, 10, dtype=np.float64) * 0.01, 0.1 + np.arange(54) * 0.2]
    )
    timestamps = np.stack([starts, starts + 0.01], axis=1)
    selection = split_and_sample_raw_index_midpoint(timestamps, source_fps=30.0)
    assert selection.split_raw_index == selection.total_frames // 2
    assert selection.past_indices.shape == (NUM_PAST_FRAMES,)
    assert selection.target_indices.shape == (NUM_TARGET_FRAMES,)
    assert not set(selection.past_indices.tolist()) & set(selection.target_indices.tolist())
    assert int(selection.past_indices.max()) < int(selection.target_indices.min())

    with tempfile.TemporaryDirectory(prefix="thinkjepa-causal-cache-test-") as temp_dir:
        root = Path(temp_dir)
        output = root / "sample.npz"
        configuration_json = _json_dumps(
            {"causality_mode": "observed_past", "self_test": True}
        )
        fingerprint = _sha256_text(configuration_json)
        fake_source = root / "sample.mp4"
        payload = _fake_payload_for_self_test(
            fake_source, fingerprint, configuration_json, selection
        )
        expected_source = {"relpath": fake_source.name, "size_bytes": 1, "mtime_ns": 2}
        atomic_write_archive(
            output,
            payload,
            save_mode="compressed",
            config_fingerprint=fingerprint,
            layers=DEFAULT_QWEN_LAYERS,
            save_dtype="fp16",
            expected_source=expected_source,
        )
        valid, reason = validate_archive(
            output,
            expected_config_fingerprint=fingerprint,
            expected_layers=DEFAULT_QWEN_LAYERS,
            expected_save_dtype="fp16",
            expected_source=expected_source,
        )
        assert valid, reason
        valid, _ = validate_archive(
            output,
            expected_config_fingerprint="0" * 64,
            expected_layers=DEFAULT_QWEN_LAYERS,
            expected_save_dtype="fp16",
            expected_source=expected_source,
        )
        assert not valid
    print("[SELF-TEST] sampling, atomic write, and validator tests passed", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    if args.max_new_token_num < 1:
        raise ValueError("--max_new_token_num must be >= 1")
    if args.max_videos < 0:
        raise ValueError("--max_videos must be >= 0")
    if len(set(args.layers)) != len(args.layers):
        raise ValueError("--layers must not contain duplicates")
    if list(args.layers) != sorted(args.layers):
        raise ValueError("--layers must be sorted in ascending order")

    configure_hf_home(args)
    dataset_root = Path(args.file_dir).expanduser().resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)
    output_root = ensure_write_path_outside_home(Path(args.output_dir), "cache output")
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.vjepa_checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"V-JEPA checkpoint not found: {checkpoint_path}")

    rank, world_size, local_rank = discover_distributed_runtime()
    all_videos = discover_videos(dataset_root, args.path_manifest)
    assigned = [
        path
        for path in all_videos
        if stable_rank_for_video(path, dataset_root, world_size) == rank
    ]
    if args.max_videos:
        assigned = assigned[: args.max_videos]
    if not assigned:
        print(
            f"[DONE] rank={rank}/{world_size} no assigned videos (discovered={len(all_videos)})",
            flush=True,
        )
        return 0

    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    print(
        f"[INFO] rank={rank}/{world_size} device={device} discovered={len(all_videos)} "
        f"assigned={len(assigned)} output={output_root}",
        flush=True,
    )

    print(f"[INFO] hashing V-JEPA checkpoint: {checkpoint_path}", flush=True)
    vjepa_checkpoint_sha = sha256_file(checkpoint_path)
    qwen_model, qwen_processor, qwen_saved, qwen_checkpoint_sha = load_qwen(args, device)
    vjepa_model = load_vjepa_encoder(checkpoint_path, device)
    config, config_fingerprint = build_global_configuration(
        args, qwen_checkpoint_sha, vjepa_checkpoint_sha
    )
    print(f"[INFO] configuration_fingerprint={config_fingerprint}", flush=True)

    failure_log = output_root / "logs" / f"failures_rank{rank:04d}.jsonl"
    saved_count = skipped_count = rebuilt_count = failed_count = 0
    for index, video_path in enumerate(assigned, start=1):
        output_path = canonical_output_path(video_path, dataset_root, output_root)
        try:
            stat = source_stat(video_path, dataset_root)
            if output_path.exists():
                valid, reason = validate_archive(
                    output_path,
                    expected_config_fingerprint=config_fingerprint,
                    expected_layers=args.layers,
                    expected_save_dtype=args.save_dtype,
                    expected_source=stat,
                )
                if valid:
                    skipped_count += 1
                    print(
                        f"[SKIP {index}/{len(assigned)}] valid {output_path}",
                        flush=True,
                    )
                    continue
                rebuilt_count += 1
                print(
                    f"[REBUILD {index}/{len(assigned)}] invalid existing archive: {reason} "
                    f"path={output_path}",
                    flush=True,
                )
            else:
                print(
                    f"[BUILD {index}/{len(assigned)}] {video_path}",
                    flush=True,
                )

            selection, past_raw, target_raw = inspect_and_decode_video(video_path)
            if set(selection.past_indices.tolist()) & set(selection.target_indices.tolist()):
                raise AssertionError("past/target index leakage detected before model inference")

            # Qwen sees this exact decoded past array and never receives target_raw.
            qwen_result = infer_qwen_past_only(
                args,
                qwen_model,
                qwen_processor,
                qwen_saved,
                device,
                past_raw,
                selection,
            )
            past_clip, past_imgs = preprocess_vjepa_frames(past_raw)
            target_clip, target_imgs = preprocess_vjepa_frames(target_raw)
            # Intentionally two independent forwards; do not concatenate clips here.
            vjepa_input_feats = encode_vjepa_clip(vjepa_model, past_clip, device)
            vjepa_target_feats = encode_vjepa_clip(vjepa_model, target_clip, device)
            imgs = np.concatenate([past_imgs, target_imgs], axis=0)
            payload = build_payload(
                args=args,
                video_path=video_path,
                dataset_root=dataset_root,
                selection=selection,
                imgs=imgs,
                vjepa_input_feats=vjepa_input_feats,
                vjepa_target_feats=vjepa_target_feats,
                qwen_result=qwen_result,
                config=config,
                config_fingerprint=config_fingerprint,
                qwen_checkpoint_sha=qwen_checkpoint_sha,
                vjepa_checkpoint_sha=vjepa_checkpoint_sha,
            )
            atomic_write_archive(
                output_path,
                payload,
                save_mode=args.save_mode,
                config_fingerprint=config_fingerprint,
                layers=args.layers,
                save_dtype=args.save_dtype,
                expected_source=stat,
            )
            saved_count += 1
            print(
                f"[SAVED {index}/{len(assigned)}] {output_path} "
                f"past=[{selection.past_indices.min()},{selection.past_indices.max()}] "
                f"target=[{selection.target_indices.min()},{selection.target_indices.max()}]",
                flush=True,
            )
        except Exception as exc:
            failed_count += 1
            append_failure(failure_log, rank, video_path, dataset_root, exc)
            print(
                f"[FAIL {index}/{len(assigned)}] {video_path}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(
        f"[DONE] rank={rank}/{world_size} saved={saved_count} skipped_valid={skipped_count} "
        f"rebuilt_invalid={rebuilt_count} failed={failed_count} output={output_root} "
        f"failure_log={failure_log}",
        flush=True,
    )
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
