#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model
# Copyright (c) 2024-2026 Northeastern University.
# Developed in NEU SMILE LAB by Haichao Zhang (https://zhanghaichao.xyz)
# and Yun Raymond Fu (https://www1.ece.neu.edu/~yunfu/).
# SPDX-style identifier: LicenseRef-ThinkJEPA-Attribution
# Original source: https://github.com/Hai-chao-Zhang/ThinkJEPA
# See the root LICENSE, NOTICE, CITATION.cff, and CITATION.bib for attribution and citation requirements.

"""
Qwen3-VL parallel extractor with torchrun data parallelism.

Key behaviors:
- No manual --num_shards/--shard_idx: split dataset by rank/world_size.
- Batch inference via --batch_size (default 20).
- Per-batch NPZ writes are submitted asynchronously (thread pool).
- Hook path and output naming format remain aligned with the existing qwen3 extractor.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "cache_train"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from cache_train.hf_egodex import configure_huggingface_cache_dirs, resolve_egodex_data_reference, rewrite_manifest_paths_for_release
from egodex.trajectory_dataset import CameraGeometryLoadError, load_supervision_from_hdf5
from cache_train.hand_skeleton_consts import QUERY_TFS
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info


def parse_parallel_thinker_cache_args():
    p = argparse.ArgumentParser("Qwen3-VL extractor (torchrun parallel, batched)")
    p.add_argument("--file_dir", type=str, required=True, help="Root dir containing mp4 files (recursive).")
    p.add_argument(
        "--path_manifest",
        type=str,
        default="",
        help="Optional text manifest listing absolute video paths, one per line. If set, use it instead of recursive scan.",
    )
    p.add_argument("--output_dir", type=str, required=True, help="Where to save npz.")
    p.add_argument("--pretrained", type=str, required=True, help="HF repo id, e.g. Qwen/Qwen3-VL-2B-Thinking")
    p.add_argument("--layers", type=int, nargs="+", required=True, help="Decoder layer indices to export.")
    p.add_argument("--max_frames", type=int, default=32, help="Requested max nframes (retry clamp for short clips).")
    p.add_argument("--max_new_token_num", type=int, default=16, help="Max new tokens.")
    p.add_argument("--batch_size", type=int, default=20, help="Number of videos per forward pass.")
    p.add_argument("--save_dtype", choices=["fp16", "fp32", "bf16"], default="fp16")

    p.add_argument("--res", type=int, default=256, help="Force resize to res x res via resized_height/width.")
    p.add_argument("--prompt", type=str, default="Describe this video.", help="Text query for Qwen3-VL.")
    p.add_argument(
        "--dataset_prompt_mode",
        type=str,
        default=os.environ.get("QWEN3VL_DATASET_PROMPT_MODE", "auto"),
        choices=["auto", "off", "egoexo4d"],
        help="Auto-append dataset-specific prompt context when metadata is available.",
    )
    p.add_argument(
        "--egoexo4d_meta_root",
        type=str,
        default=os.environ.get("EGOEXO4D_META_ROOT", ""),
        help="Optional EgoExo4D metadata root containing takes.json/captures.json/etc. Auto-detected when possible.",
    )
    p.add_argument(
        "--max_prompt_objects",
        type=int,
        default=int(os.environ.get("QWEN3VL_MAX_PROMPT_OBJECTS", "8")),
        help="Maximum object names to inject into the dynamic prompt overlay.",
    )
    p.add_argument(
        "--force_video_backend",
        type=str,
        default="torchcodec",
        choices=["", "torchcodec", "decord", "torchvision"],
        help="Set FORCE_QWENVL_VIDEO_READER. Empty string means do not set.",
    )
    p.add_argument(
        "--save_mode",
        type=str,
        default=os.environ.get("QWEN3VL_SAVE_MODE", "compressed"),
        choices=["compressed", "raw"],
        help="NPZ save mode. 'raw' is faster but larger files.",
    )
    p.add_argument(
        "--io_workers",
        type=int,
        default=int(os.environ.get("QWEN3VL_IO_WORKERS", "8")),
        help="Async writer workers for NPZ output.",
    )
    p.add_argument(
        "--max_pending_writes",
        type=int,
        default=int(os.environ.get("QWEN3VL_MAX_PENDING_WRITES", "20")),
        help="Backpressure limit for async writes.",
    )
    return p.parse_parallel_thinker_cache_args()


def tensor_to_cache_numpy(x: torch.Tensor, save_dtype: str) -> np.ndarray:
    if save_dtype == "fp16":
        return x.detach().float().half().cpu().numpy()
    if save_dtype == "bf16":
        return x.detach().bfloat16().cpu().view(torch.uint16).numpy()
    return x.detach().float().cpu().numpy()


def add_optional_cache_field(npz: Dict[str, Any], key: str, val: Any, save_dtype: str = "fp16"):
    try:
        if torch.is_tensor(val):
            npz[key] = tensor_to_cache_numpy(val, save_dtype)
        else:
            npz[key] = np.asarray(val)
    except Exception:
        pass


def list_video_files(root: str) -> List[str]:
    mp4s = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".mp4"):
                mp4s.append(os.path.join(dp, fn))
    mp4s.sort()
    return mp4s


def load_video_path_manifest(manifest_path: str, dataset_root: str) -> List[str]:
    paths: List[str] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.lower().endswith(".mp4"):
                paths.append(s)
    paths = rewrite_manifest_paths_for_release(paths, dataset_root)
    paths.sort()
    return paths


def make_safe_video_identifier(p: str) -> str:
    base = os.path.basename(p)
    stem = os.path.splitext(base)[0]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return stem


def ensure_output_directory(p: str):
    os.makedirs(p, exist_ok=True)


def relative_video_subdirectory(video_path: str, dataset_root: str) -> str:
    try:
        rel = os.path.relpath(os.path.normpath(video_path), start=os.path.normpath(dataset_root))
        subdir = os.path.dirname(rel)
        if subdir.startswith(".."):
            return ""
        return subdir
    except Exception:
        return ""


def locate_thinker_decoder_layers(module_with_layers: torch.nn.Module):
    if not hasattr(module_with_layers, "model"):
        raise AttributeError("Expected Qwen3-VL model to have `.model`")
    if not hasattr(module_with_layers.model, "language_model"):
        raise AttributeError("Expected Qwen3-VL model to have `.model.language_model`")
    if not hasattr(module_with_layers.model.language_model, "layers"):
        raise AttributeError("Expected Qwen3-VL model to have `.model.language_model.layers`")
    decoder_layers = module_with_layers.model.language_model.layers
    layers_path = "model.language_model.layers"
    return decoder_layers, layers_path


def register_thinker_decoder_hooks(decoder_layers, layers: List[int]):
    saved: Dict[str, List[torch.Tensor]] = {f"dec_{i}": [] for i in layers}

    def make_hook(name):
        def hook(_module, _inp, out):
            h = out
            if isinstance(out, (tuple, list)):
                h = out[0]
                if isinstance(h, (tuple, list)):
                    h = h[0]
            if not torch.is_tensor(h):
                return
            saved[name].append(h.detach().cpu())

        return hook

    for layer_idx in layers:
        decoder_layers[layer_idx].register_forward_hook(make_hook(f"dec_{layer_idx}"))
    return saved


def empty_guidance_state() -> torch.Tensor:
    return torch.empty((0,), dtype=torch.float16)


def slice_sample_state(t: torch.Tensor, sample_idx: int, batch_size: int) -> torch.Tensor:
    if t.dim() > 0 and t.shape[0] == batch_size:
        return t[sample_idx : sample_idx + 1]
    return t


def trim_state_to_valid_length(t: torch.Tensor, valid_len: Optional[int]) -> torch.Tensor:
    if valid_len is None:
        return t
    if t.dim() != 3:
        return t
    if valid_len <= 0:
        return t[:, :0, :]
    if t.shape[1] <= valid_len:
        return t
    # Left padding: keep the rightmost valid_len tokens.
    return t[:, -valid_len:, :]


def stack_pyramid_guidance_states_per_sample(
    saved: Dict[str, List[torch.Tensor]],
    layers: List[int],
    batch_size: int,
    valid_lens: Optional[List[int]] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for b in range(batch_size):
        old_list = []
        new_list = []
        for i in layers:
            key = f"dec_{i}"
            if key not in saved or len(saved[key]) == 0:
                continue

            old = slice_sample_state(saved[key][0], b, batch_size)
            if valid_lens is not None and b < len(valid_lens):
                old = trim_state_to_valid_length(old, int(valid_lens[b]))
            old_list.append(old)

            if len(saved[key]) > 1:
                try:
                    new_cat = torch.cat(saved[key][1:], dim=1)
                except Exception:
                    x = saved[key][-1]
                    if x.dim() == 3:
                        new_cat = x[:, -1:, :]
                    else:
                        new_cat = x
                new_list.append(slice_sample_state(new_cat, b, batch_size))

        vlm_old = torch.stack(old_list, dim=0) if len(old_list) else empty_guidance_state()
        if len(new_list) == 0:
            vlm_new = empty_guidance_state()
        else:
            try:
                vlm_new = torch.stack(new_list, dim=0)
            except Exception:
                new_last = []
                for x in new_list:
                    if x.dim() == 3:
                        new_last.append(x[:, -1:, :])
                    else:
                        new_last.append(x)
                vlm_new = torch.stack(new_last, dim=0)
        out.append((vlm_old, vlm_new))
    return out


def parse_total_frames_from_nframes_error(e: Exception) -> int:
    m = re.search(r"interval\s*\[\s*\d+\s*,\s*(\d+)\s*\]", str(e))
    return int(m.group(1)) if m else -1


def write_cache_archive(out_path: str, npz: Dict[str, Any], save_mode: str) -> None:
    if save_mode == "raw":
        np.savez(out_path, **npz)
    else:
        np.savez_compressed(out_path, **npz)


def clamp_uniform_thinker_frames(max_frames: int, total_frames: int) -> int:
    if total_frames <= 0:
        return 0
    nf = min(int(max_frames), int(total_frames))
    nf = max(2, nf)
    nf = (nf // 2) * 2
    if nf < 2:
        return 0
    return nf


def first_half_video_kwargs(video_path: str) -> Dict[str, float]:
    try:
        import decord

        vr = decord.VideoReader(video_path)
        video_end = (int(len(vr)) / max(float(vr.get_avg_fps()), 1e-6)) * 0.5
    except Exception:
        return {}
    return {"video_start": 0.0, "video_end": video_end} if video_end > 0 else {}


def read_env_int(keys: List[str], default: int) -> int:
    for k in keys:
        v = os.environ.get(k)
        if v is None:
            continue
        try:
            return int(v)
        except ValueError:
            continue
    return default


def discover_distributed_runtime() -> Tuple[int, int, int]:
    rank = read_env_int(["RANK", "SLURM_PROCID"], 0)
    world_size = read_env_int(["WORLD_SIZE", "SLURM_NTASKS"], 1)
    local_rank = read_env_int(["LOCAL_RANK", "SLURM_LOCALID"], 0)
    if world_size < 1:
        world_size = 1
    if rank < 0:
        rank = 0
    if local_rank < 0:
        local_rank = 0
    return rank, world_size, local_rank


def build_thinker_video_messages(
    file_url: str,
    prompt: str,
    res: int,
    nframes: int,
    window_kwargs: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    video_content = {
        "type": "video",
        "video": file_url,
        "nframes": int(nframes),
        "resized_height": int(res),
        "resized_width": int(res),
    }
    if window_kwargs:
        video_content.update(window_kwargs)
    return [
        {
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": prompt},
            ],
        }
    ]


def load_json_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_ancestor_with_file(start_path: str, filename: str) -> Optional[str]:
    cur = os.path.abspath(start_path)
    if os.path.isfile(cur):
        cur = os.path.dirname(cur)
    while True:
        cand = os.path.join(cur, filename)
        if os.path.isfile(cand):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def detect_egoexo4d_metadata_root(args) -> Optional[str]:
    if args.dataset_prompt_mode == "off":
        return None
    if args.egoexo4d_meta_root:
        root = os.path.abspath(args.egoexo4d_meta_root)
        return root if os.path.isfile(os.path.join(root, "takes.json")) else None
    if args.dataset_prompt_mode == "egoexo4d" or ("egoexo4d" in os.path.abspath(args.file_dir).lower()):
        return find_ancestor_with_file(args.file_dir, "takes.json")
    return None


def prettify_metadata_label(x: str) -> str:
    s = str(x).strip()
    s = re.sub(r"[_-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\b\d+\b$", "", s).strip()
    return s


def _build_egoexo4d_prompt_index(args) -> Optional[Dict[str, Any]]:
    meta_root = detect_egoexo4d_metadata_root(args)
    if not meta_root:
        return None

    takes = load_json_file(os.path.join(meta_root, "takes.json"))
    captures = load_json_file(os.path.join(meta_root, "captures.json"))
    metadata = load_json_file(os.path.join(meta_root, "metadata.json"))
    visual_objects = load_json_file(os.path.join(meta_root, "visual_objects.json"))

    take_by_name = {}
    for item in takes:
        take_name = item.get("take_name")
        if isinstance(take_name, str) and take_name:
            take_by_name[take_name] = item

    capture_by_uid = {}
    for item in captures:
        capture_uid = item.get("capture_uid")
        if capture_uid is not None:
            capture_by_uid[capture_uid] = item

    tasks_map = metadata.get("tasks", {}) if isinstance(metadata, dict) else {}

    visual_names_by_physical: Dict[int, List[str]] = {}
    for item in visual_objects:
        physical_uid = item.get("physical_setting_uid")
        name = item.get("name")
        if physical_uid is None or not isinstance(name, str) or not name.strip():
            continue
        visual_names_by_physical.setdefault(int(physical_uid), []).append(prettify_metadata_label(name))

    for k, vals in list(visual_names_by_physical.items()):
        dedup = []
        seen = set()
        for v in vals:
            key = v.lower()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(v)
        visual_names_by_physical[k] = dedup

    return {
        "meta_root": meta_root,
        "take_by_name": take_by_name,
        "capture_by_uid": capture_by_uid,
        "tasks_map": tasks_map,
        "visual_names_by_physical": visual_names_by_physical,
    }


def _infer_egoexo4d_take_name(video_path: str) -> Optional[str]:
    parts = os.path.normpath(video_path).split(os.sep)
    if "takes" not in parts:
        return None
    idx = parts.index("takes")
    if idx + 1 >= len(parts):
        return None
    return parts[idx + 1]


def find_egoexo4d_view_metadata(take_item: Dict[str, Any], basename: str) -> Optional[Dict[str, Any]]:
    fav = take_item.get("frame_aligned_videos", {})
    if not isinstance(fav, dict):
        return None
    for cam_id, streams in fav.items():
        if not isinstance(streams, dict):
            continue
        for stream_name, meta in streams.items():
            if not isinstance(meta, dict):
                continue
            rel = meta.get("relative_path", "")
            if isinstance(rel, str) and rel.endswith(basename):
                out = dict(meta)
                out.setdefault("cam_id", cam_id)
                out.setdefault("readable_stream_id", stream_name)
                return out
    return None


def build_egoexo4d_view_text(view_meta: Optional[Dict[str, Any]], basename: str, take_item: Dict[str, Any]) -> Optional[str]:
    stem = os.path.splitext(basename)[0]
    best_exo = take_item.get("best_exo")
    parts = []
    if view_meta is not None:
        is_ego = bool(view_meta.get("is_ego", False))
        cam_id = view_meta.get("cam_id", stem)
        stream_id = view_meta.get("readable_stream_id", "")
        device_type = view_meta.get("device_type", "")
        if is_ego:
            parts.append(f"viewpoint: ego view from {cam_id}")
        else:
            parts.append(f"viewpoint: exocentric view from {cam_id}")
        if isinstance(stream_id, str) and stream_id and stream_id != "0":
            parts.append(f"stream: {prettify_metadata_label(stream_id)}")
        if isinstance(device_type, str) and device_type:
            parts.append(f"device: {device_type}")
    else:
        if stem.startswith("aria"):
            parts.append("viewpoint: ego wearable view")
        elif stem.startswith("cam") or stem.startswith("gp"):
            parts.append("viewpoint: exocentric static or handheld camera")
        elif stem.startswith("ego_preview"):
            parts.append("viewpoint: ego preview clip")
    if isinstance(best_exo, str) and best_exo and stem == best_exo:
        parts.append("this is the preferred exocentric reference view for the take")
    if not parts:
        return None
    return "; ".join(parts) + "."


def collect_take_object_names(take_item: Dict[str, Any], prompt_index: Dict[str, Any], max_items: int) -> List[str]:
    names: List[str] = []
    objects = take_item.get("objects", [])
    if isinstance(objects, list):
        for obj in objects:
            if isinstance(obj, dict):
                name = obj.get("name")
                if isinstance(name, str) and name.strip():
                    names.append(prettify_metadata_label(name))
            elif isinstance(obj, str) and obj.strip():
                names.append(prettify_metadata_label(obj))

    if not names:
        physical_uid = take_item.get("physical_setting_uid")
        if physical_uid is not None:
            fallback = prompt_index["visual_names_by_physical"].get(int(physical_uid), [])
            if 0 < len(fallback) <= max_items:
                names.extend(fallback)

    dedup = []
    seen = set()
    for name in names:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(name)
        if len(dedup) >= max_items:
            break
    return dedup


def _build_egoexo4d_prompt_overlay(args, prompt_index: Dict[str, Any], video_path: str) -> str:
    take_name = _infer_egoexo4d_take_name(video_path)
    if not take_name:
        return ""
    take_item = prompt_index["take_by_name"].get(take_name)
    if take_item is None:
        return ""

    basename = os.path.basename(video_path)
    task_id = take_item.get("task_id")
    canonical_task = None
    if task_id is not None:
        canonical_task = prompt_index["tasks_map"].get(str(task_id))
        if canonical_task is None:
            canonical_task = prompt_index["tasks_map"].get(int(task_id), None)

    lines = [
        "Use the following EgoExo4D metadata as weak context only. Ground the answer in visible evidence and do not invent actions not supported by the video."
    ]

    parent_task = take_item.get("parent_task_name")
    task_name = take_item.get("task_name")
    if isinstance(parent_task, str) and parent_task.strip():
        lines.append(f"Activity family: {parent_task.strip()}.")
    if isinstance(task_name, str) and task_name.strip():
        lines.append(f"Specific task: {task_name.strip()}.")
    if isinstance(canonical_task, str) and canonical_task.strip() and canonical_task.strip() != str(task_name).strip():
        lines.append(f"Canonical task label: {canonical_task.strip()}.")

    view_meta = find_egoexo4d_view_metadata(take_item, basename)
    view_text = build_egoexo4d_view_text(view_meta, basename, take_item)
    if view_text:
        lines.append(view_text)

    object_names = collect_take_object_names(
        take_item=take_item,
        prompt_index=prompt_index,
        max_items=max(1, int(args.max_prompt_objects)),
    )
    if object_names:
        lines.append(f"Likely relevant objects: {', '.join(object_names)}.")

    capture_item = prompt_index["capture_by_uid"].get(take_item.get("capture_uid"))
    if isinstance(capture_item, dict):
        capture_name = capture_item.get("capture_name")
        university_name = capture_item.get("university_name")
        cap_bits = []
        if isinstance(capture_name, str) and capture_name.strip():
            cap_bits.append(capture_name.strip())
        if isinstance(university_name, str) and university_name.strip():
            cap_bits.append(f"site {university_name.strip()}")
        if cap_bits:
            lines.append("Capture context: " + ", ".join(cap_bits) + ".")

    if bool(take_item.get("is_narrated", False)):
        lines.append("This take is marked as narrated.")

    return " ".join(lines).strip()


def compose_thinker_prompt(args, prompt_index: Optional[Dict[str, Any]], video_path: str) -> Tuple[str, str]:
    base_prompt = str(args.prompt).strip()
    overlay = ""
    if prompt_index is not None:
        overlay = _build_egoexo4d_prompt_overlay(args, prompt_index, video_path)
    full_prompt = base_prompt
    if overlay:
        full_prompt = f"{base_prompt}\n\n{overlay}"
    return full_prompt, overlay


def prepare_single_video_inputs(args, processor, video_path: str, prompt_index: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    nf_req = int(args.max_frames)
    nf_used = nf_req
    total_frames = 0

    abs_path = video_path if os.path.isabs(video_path) else os.path.abspath(video_path)
    file_url = f"file://{abs_path}"
    full_prompt, prompt_overlay = compose_thinker_prompt(args, prompt_index, video_path)
    window_kwargs = first_half_video_kwargs(abs_path)

    last_err = None
    for attempt in range(2):
        messages = build_thinker_video_messages(
            file_url=file_url,
            prompt=full_prompt,
            res=args.res,
            nframes=nf_used,
            window_kwargs=window_kwargs,
        )
        try:
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            video_tensors = None
            video_metadatas = None
            if videos is not None:
                if len(videos) > 0 and isinstance(videos[0], tuple):
                    v, m = zip(*videos)
                    video_tensors = list(v)
                    video_metadatas = list(m)
                else:
                    video_tensors = videos

            return {
                "video_path": video_path,
                "text": text,
                "images": images,
                "videos": video_tensors,
                "video_metadatas": video_metadatas,
                "video_kwargs": video_kwargs or {},
                "nframes_req": nf_req,
                "nframes_used": int(nf_used),
                "total_frames": int(total_frames),
                "prompt_base": np.asarray(str(args.prompt)),
                "prompt_overlay": np.asarray(prompt_overlay),
                "prompt_full": np.asarray(full_prompt),
            }

        except ValueError as e:
            last_err = e
            tf = parse_total_frames_from_nframes_error(e)
            if tf > 0 and attempt == 0:
                total_frames = tf
                nf_used = clamp_uniform_thinker_frames(nf_req, total_frames)
                if nf_used < 2:
                    return None
                continue
            return None
        except Exception as e:
            last_err = e
            break

    if last_err is not None:
        return None
    return None


def derive_hdf5_path_from_video(video_path: str) -> Optional[str]:
    root, ext = os.path.splitext(video_path)
    if ext.lower() != ".mp4":
        return None
    hdf5_path = root + ".hdf5"
    return hdf5_path if os.path.exists(hdf5_path) else None


def load_optional_egodex_supervision_bundle(video_path: str) -> Optional[Dict[str, Any]]:
    hdf5_path = derive_hdf5_path_from_video(video_path)
    if hdf5_path is None:
        return None
    try:
        sup = load_supervision_from_hdf5(hdf5_path, QUERY_TFS)
    except Exception as exc:
        if isinstance(exc, CameraGeometryLoadError) or "[CAM-INVALID]" in str(exc):
            raise
        print(
            f"[WARN] failed to load hdf5 supervision for {video_path}: "
            f"{type(exc).__name__}: {exc}"
        )
        return None
    sup["path"] = np.asarray(str(hdf5_path))
    return sup


def merge_video_processor_kwargs(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for item in items:
        kwargs = item.get("video_kwargs", {})
        for k, v in kwargs.items():
            if k not in merged:
                merged[k] = v
                continue
            cur = merged[k]
            if isinstance(cur, list) and isinstance(v, list):
                merged[k] = cur + v
            elif cur == v:
                continue
            else:
                merged[k] = cur
    return merged


def build_cache_output_path(args, video_path: str, rank: int, world_size: int) -> Tuple[str, str]:
    vid_id = make_safe_video_identifier(video_path)
    shard_tag = f"s{rank:02d}of{world_size:02d}"
    out_name = (
        f"{vid_id}_L{len(args.layers)}_nf{args.max_frames}_res{args.res}"
        f"_new{args.max_new_token_num}_{shard_tag}.npz"
    )
    rel_subdir = relative_video_subdirectory(video_path, args.file_dir)
    out_dir = os.path.join(args.output_dir, rel_subdir) if rel_subdir else args.output_dir
    ensure_output_directory(out_dir)
    out_path = os.path.join(out_dir, out_name)
    rel_out = os.path.relpath(out_path, args.output_dir)
    return out_path, rel_out


def infer_prepared_cache_batch(
    args,
    processor,
    model,
    device: torch.device,
    saved: Dict[str, List[torch.Tensor]],
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if len(items) == 0:
        return []

    for k in saved:
        saved[k].clear()

    texts = [x["text"] for x in items]

    images_flat = []
    videos_flat = []
    metas_flat = []

    for x in items:
        if x["images"] is not None:
            images_flat.extend(x["images"])
        if x["videos"] is not None:
            videos_flat.extend(x["videos"])
        if x["video_metadatas"] is not None:
            metas_flat.extend(x["video_metadatas"])

    proc_kwargs: Dict[str, Any] = {
        "text": texts,
        "return_tensors": "pt",
        "padding": True,
        "do_resize": False,
    }
    if len(images_flat) > 0:
        proc_kwargs["images"] = images_flat
    if len(videos_flat) > 0:
        proc_kwargs["videos"] = videos_flat
    if len(metas_flat) > 0:
        proc_kwargs["video_metadata"] = metas_flat
    proc_kwargs.update(merge_video_processor_kwargs(items))

    inputs = processor(**proc_kwargs)
    input_ids_cpu = inputs["input_ids"].detach().cpu()
    input_attn = inputs.get("attention_mask")
    valid_lens = None
    if torch.is_tensor(input_attn):
        try:
            valid_lens = input_attn.sum(dim=1).detach().cpu().tolist()
        except Exception:
            valid_lens = None
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=int(args.max_new_token_num))

    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    out_texts = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    states = stack_pyramid_guidance_states_per_sample(saved, args.layers, len(items), valid_lens=valid_lens)
    if len(states) < len(items):
        states.extend([(empty_guidance_state(), empty_guidance_state()) for _ in range(len(items) - len(states))])

    outputs: List[Dict[str, Any]] = []
    for i, x in enumerate(items):
        out_text0 = out_texts[i] if i < len(out_texts) else ""
        token_ids = (
            trimmed[i].detach().cpu().numpy().astype(np.int32) if i < len(trimmed) else np.zeros((0,), dtype=np.int32)
        )
        vlm_old, vlm_new = states[i]

        npz: Dict[str, Any] = {}
        npz["text"] = np.asarray(out_text0)
        npz["token_ids"] = token_ids
        npz["input_ids"] = input_ids_cpu[i].numpy().astype(np.int32)
        input_valid_len = (
            int(valid_lens[i])
            if valid_lens is not None and i < len(valid_lens)
            else int(input_ids_cpu.shape[1])
        )
        npz["input_valid_len"] = np.asarray(int(input_valid_len), dtype=np.int32)
        npz["layers"] = np.asarray(args.layers, dtype=np.int32)
        npz["video_path"] = np.asarray(x["video_path"])
        npz["nframes_req"] = np.asarray(int(x["nframes_req"]), dtype=np.int32)
        npz["nframes_used"] = np.asarray(int(x["nframes_used"]), dtype=np.int32)
        npz["total_frames"] = np.asarray(int(x["total_frames"]), dtype=np.int32)
        npz["res"] = np.asarray(int(args.res), dtype=np.int32)
        npz["backend_env"] = np.asarray(os.environ.get("FORCE_QWENVL_VIDEO_READER", ""))
        npz["prompt_base"] = np.asarray(x.get("prompt_base", ""))
        npz["prompt_overlay"] = np.asarray(x.get("prompt_overlay", ""))
        npz["prompt_full"] = np.asarray(x.get("prompt_full", ""))

        if vlm_old.numel():
            npz["vlm_old"] = tensor_to_cache_numpy(vlm_old, args.save_dtype)
        else:
            npz["vlm_old"] = np.zeros((0,), dtype=np.float16 if args.save_dtype == "fp16" else np.float32)

        if vlm_new.numel():
            npz["vlm_new"] = tensor_to_cache_numpy(vlm_new, args.save_dtype)
        else:
            npz["vlm_new"] = np.zeros((0,), dtype=np.float16 if args.save_dtype == "fp16" else np.float32)

        supervision = x.get("supervision", None)
        if isinstance(supervision, dict):
            add_optional_cache_field(npz, "frame_indices", supervision.get("frame_indices"), save_dtype="fp32")
            add_optional_cache_field(npz, "xyz_cam", supervision.get("xyz_cam"), args.save_dtype)
            add_optional_cache_field(npz, "R_cam", supervision.get("R_cam"), args.save_dtype)
            add_optional_cache_field(npz, "xyz_world", supervision.get("xyz_world"), args.save_dtype)
            add_optional_cache_field(npz, "R_world", supervision.get("R_world"), args.save_dtype)
            add_optional_cache_field(npz, "tfs_in_cam", supervision.get("tfs_in_cam"), args.save_dtype)
            add_optional_cache_field(npz, "tfs", supervision.get("tfs"), args.save_dtype)
            add_optional_cache_field(npz, "cam_ext", supervision.get("cam_ext"), args.save_dtype)
            add_optional_cache_field(npz, "cam_int", supervision.get("cam_int"), args.save_dtype)
            add_optional_cache_field(npz, "confs", supervision.get("confs"), args.save_dtype)
            if "lang_instruct" in supervision:
                npz["lang_instruct"] = np.asarray(supervision["lang_instruct"])
            if "path" in supervision:
                npz["path"] = np.asarray(supervision["path"])

        outputs.append(
            {
                "out_path": x["out_path"],
                "rel_out": x["rel_out"],
                "npz": npz,
            }
        )

    return outputs


def load_qwen3vl_thinker_model(pretrained: str):
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            pretrained,
            torch_dtype="auto",
            device_map=None,
        )
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            pretrained,
            dtype="auto",
            device_map=None,
        )
    return model


def main():
    args = parse_parallel_thinker_cache_args()
    configure_huggingface_cache_dirs()
    args.file_dir = resolve_egodex_data_reference(args.file_dir)

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.io_workers < 1:
        raise ValueError("--io_workers must be >= 1")
    if args.max_pending_writes < 1:
        raise ValueError("--max_pending_writes must be >= 1")

    if args.force_video_backend:
        os.environ["FORCE_QWENVL_VIDEO_READER"] = args.force_video_backend

    rank, world_size, local_rank = discover_distributed_runtime()

    ensure_output_directory(args.output_dir)
    ensure_output_directory(os.path.join(args.output_dir, "logs"))

    if args.path_manifest:
        all_videos = load_video_path_manifest(args.path_manifest, args.file_dir)
        print(f"[INFO] Using path manifest: {args.path_manifest} videos={len(all_videos)}")
    else:
        all_videos = list_video_files(args.file_dir)
    rank_videos = [x for i, x in enumerate(all_videos) if (i % world_size) == rank]

    if len(rank_videos) == 0:
        print(f"[WARN] rank={rank}/{world_size}: no videos assigned under {args.file_dir}")
        return

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    model = load_qwen3vl_thinker_model(args.pretrained)
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.pretrained)
    # Decoder-only models require left padding for correct generation.
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
    prompt_index = _build_egoexo4d_prompt_index(args)

    module_with_layers = getattr(model, "module", model)
    decoder_layers, layers_path = locate_thinker_decoder_layers(module_with_layers)

    for layer_idx in args.layers:
        if layer_idx < 0 or layer_idx >= len(decoder_layers):
            raise ValueError(f"--layers contains {layer_idx}, but decoder has {len(decoder_layers)} layers.")

    print(
        f"[INFO] rank={rank}/{world_size} local_rank={local_rank} device={device} "
        f"assigned={len(rank_videos)} batch_size={args.batch_size}"
    )
    print(f"[INFO] Hooking decoder layers at: {layers_path} (len={len(decoder_layers)})")
    if prompt_index is not None:
        print(
            f"[INFO] EgoExo4D prompt overlay enabled: meta_root={prompt_index['meta_root']} "
            f"takes={len(prompt_index['take_by_name'])}"
        )
    saved = register_thinker_decoder_hooks(decoder_layers, args.layers)

    pbar = tqdm(
        total=len(rank_videos),
        desc=f"Qwen3-VL rank {rank}/{world_size}",
        ncols=110,
    )

    saved_count = 0
    skipped_exists = 0
    skipped_prepare = 0
    failed_infer = 0

    with ThreadPoolExecutor(max_workers=args.io_workers) as io_pool:
        pending_writes = []

        for start in range(0, len(rank_videos), args.batch_size):
            chunk = rank_videos[start : start + args.batch_size]
            prepared_batch: List[Dict[str, Any]] = []

            for video_path in chunk:
                out_path, rel_out = build_cache_output_path(args, video_path, rank=rank, world_size=world_size)
                if os.path.exists(out_path):
                    skipped_exists += 1
                    pbar.set_postfix_str(f"skip exists: {rel_out}")
                    continue

                prepared = prepare_single_video_inputs(args, processor, video_path, prompt_index=prompt_index)
                if prepared is None:
                    skipped_prepare += 1
                    pbar.set_postfix_str(f"vision fail; skip: {make_safe_video_identifier(video_path)}")
                    continue

                prepared["out_path"] = out_path
                prepared["rel_out"] = rel_out
                prepared["supervision"] = load_optional_egodex_supervision_bundle(video_path)
                prepared_batch.append(prepared)

            batch_outputs: List[Dict[str, Any]] = []
            if len(prepared_batch) > 0:
                try:
                    batch_outputs = infer_prepared_cache_batch(
                        args=args,
                        processor=processor,
                        model=model,
                        device=device,
                        saved=saved,
                        items=prepared_batch,
                    )
                except RuntimeError as e:
                    emsg = str(e).lower()
                    if "out of memory" in emsg and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    for item in prepared_batch:
                        try:
                            single_out = infer_prepared_cache_batch(
                                args=args,
                                processor=processor,
                                model=model,
                                device=device,
                                saved=saved,
                                items=[item],
                            )
                            batch_outputs.extend(single_out)
                        except Exception as ie:
                            failed_infer += 1
                            print(
                                f"[WARN] rank={rank} fallback single failed: "
                                f"{item['video_path']} ({type(ie).__name__}: {ie})"
                            )
                except Exception as e:
                    for item in prepared_batch:
                        try:
                            single_out = infer_prepared_cache_batch(
                                args=args,
                                processor=processor,
                                model=model,
                                device=device,
                                saved=saved,
                                items=[item],
                            )
                            batch_outputs.extend(single_out)
                        except Exception as ie:
                            failed_infer += 1
                            print(
                                f"[WARN] rank={rank} fallback single failed: "
                                f"{item['video_path']} ({type(ie).__name__}: {ie})"
                            )
                    print(f"[WARN] rank={rank} batch fallback due to {type(e).__name__}: {e}")

            for out in batch_outputs:
                while len(pending_writes) >= args.max_pending_writes:
                    pending_writes[0].result()
                    pending_writes.pop(0)
                pending_writes.append(io_pool.submit(write_cache_archive, out["out_path"], out["npz"], args.save_mode))
                saved_count += 1
                pbar.set_postfix_str(f"saved {out['rel_out']}")

            pbar.update(len(chunk))

        for fut in pending_writes:
            fut.result()

    pbar.close()
    print(
        f"[DONE] rank={rank}/{world_size} saved={saved_count} "
        f"skip_exists={skipped_exists} skip_prepare={skipped_prepare} fail_infer={failed_infer} "
        f"output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
