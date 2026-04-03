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
Qwen3-VL shardable video extractor with intermediate decoder-layer hooks.
Compatible with the provided bash launcher args.

Key behaviors:
- Video input: message {"type":"video","video":"file:///abs/path.mp4", ...}
- Enforce nframes <= --max_frames using retry-on-short-video (NO per-video ffprobe)
- Enforce strict resize to 256x256 via resized_height/resized_width (not max_pixels budget)
- Hook decoder layers at: model.language_model.layers (as requested)
- Save per-video .npz with vlm_old/vlm_new/token_ids/text (+ nframes_used/total_frames)
- Output path preserves subdir structure under --file_dir to avoid overwrites
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "cache_train"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from egodex.trajectory_dataset import CameraGeometryLoadError, load_supervision_from_hdf5
from cache_train.hand_skeleton_consts import QUERY_TFS
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info


# -------------------------
# Args (must match your bash)
# -------------------------
def parse_thinker_cache_extraction_args():
    p = argparse.ArgumentParser("Qwen3-VL extractor (shardable, hook decoder layers)")
    p.add_argument("--file_dir", type=str, required=True, help="Root dir containing mp4 files (recursive).")
    p.add_argument("--output_dir", type=str, required=True, help="Where to save npz.")
    p.add_argument("--pretrained", type=str, required=True, help="HF repo id, e.g. Qwen/Qwen3-VL-2B-Thinking")
    p.add_argument("--layers", type=int, nargs="+", required=True, help="Decoder layer indices to export.")
    p.add_argument("--max_frames", type=int, default=32, help="Requested max nframes (retry clamp for short clips).")
    p.add_argument("--max_new_token_num", type=int, default=16, help="Max new tokens.")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--save_dtype", choices=["fp16", "fp32", "bf16"], default="fp16")

    p.add_argument("--res", type=int, default=256, help="Force resize to res x res via resized_height/width.")
    p.add_argument("--prompt", type=str, default="Describe this video.", help="Text query for Qwen3-VL.")
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
        default=int(os.environ.get("QWEN3VL_IO_WORKERS", "2")),
        help="Async writer workers for NPZ output.",
    )
    p.add_argument(
        "--max_pending_writes",
        type=int,
        default=int(os.environ.get("QWEN3VL_MAX_PENDING_WRITES", "8")),
        help="Backpressure limit for async writes.",
    )
    return p.parse_thinker_cache_extraction_args()


# -------------------------
# Utils
# -------------------------
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


def derive_hdf5_path_from_video(video_path: str) -> str | None:
    root, ext = os.path.splitext(video_path)
    if ext.lower() != ".mp4":
        return None
    hdf5_path = root + ".hdf5"
    return hdf5_path if os.path.exists(hdf5_path) else None


def load_optional_egodex_supervision_bundle(video_path: str) -> Dict[str, Any] | None:
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


def list_video_files(root: str) -> List[str]:
    mp4s = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".mp4"):
                mp4s.append(os.path.join(dp, fn))
    mp4s.sort()
    return mp4s


def select_video_shard(items: List[str], num_shards: int, shard_idx: int) -> List[str]:
    if num_shards <= 1:
        return items
    out = []
    for i, x in enumerate(items):
        if i % num_shards == shard_idx:
            out.append(x)
    return out


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


def stack_pyramid_guidance_states(saved: Dict[str, List[torch.Tensor]], layers: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    old_list = []
    new_list = []

    for i in layers:
        key = f"dec_{i}"
        if key not in saved or len(saved[key]) == 0:
            continue

        old_list.append(saved[key][0])

        if len(saved[key]) > 1:
            try:
                new = torch.cat(saved[key][1:], dim=1)
            except Exception:
                x = saved[key][-1]
                if x.dim() == 3:
                    new = x[:, -1:, :]
                else:
                    new = x
            new_list.append(new)

    vlm_old = torch.stack(old_list, dim=0) if len(old_list) else torch.empty((0,), dtype=torch.float16)

    if len(new_list) == 0:
        vlm_new = torch.empty((0,), dtype=torch.float16)
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

    return vlm_old, vlm_new


def parse_total_frames_from_nframes_error(e: Exception) -> int:
    """
    Parse: "nframes should in interval [2, 21], but got 32."
    Return 21 if found, else -1.
    """
    m = re.search(r"interval\s*\[\s*\d+\s*,\s*(\d+)\s*\]", str(e))
    return int(m.group(1)) if m else -1


def write_cache_archive(out_path: str, npz: Dict[str, Any], save_mode: str) -> None:
    if save_mode == "raw":
        np.savez(out_path, **npz)
    else:
        np.savez_compressed(out_path, **npz)


def clamp_uniform_thinker_frames(max_frames: int, total_frames: int) -> int:
    """
    Mirror qwen_vl_utils constraints:
    nf in [2, total_frames], and FRAME_FACTOR=2 => even.
    """
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


# -------------------------
# Main
# -------------------------
def main():
    try:
        from cache_train.hf_egodex import configure_huggingface_cache_dirs, resolve_egodex_data_reference
    except ModuleNotFoundError:
        from hf_egodex import configure_huggingface_cache_dirs, resolve_egodex_data_reference

    args = parse_thinker_cache_extraction_args()
    configure_huggingface_cache_dirs()
    args.file_dir = resolve_egodex_data_reference(args.file_dir)
    if args.io_workers < 1:
        raise ValueError("--io_workers must be >= 1")
    if args.max_pending_writes < 1:
        raise ValueError("--max_pending_writes must be >= 1")

    if args.force_video_backend:
        os.environ["FORCE_QWENVL_VIDEO_READER"] = args.force_video_backend

    ensure_output_directory(args.output_dir)
    ensure_output_directory(os.path.join(args.output_dir, "logs"))

    all_videos = list_video_files(args.file_dir)
    shard_videos = select_video_shard(all_videos, args.num_shards, args.shard_idx)

    if len(shard_videos) == 0:
        print(f"[WARN] No mp4 found for shard {args.shard_idx}/{args.num_shards} under {args.file_dir}")
        return

    model = AutoModelForImageTextToText.from_pretrained(
        args.pretrained,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.pretrained)
    model.eval()

    module_with_layers = getattr(model, "module", model)
    decoder_layers, layers_path = locate_thinker_decoder_layers(module_with_layers)

    for layer_idx in args.layers:
        if layer_idx < 0 or layer_idx >= len(decoder_layers):
            raise ValueError(f"--layers contains {layer_idx}, but decoder has {len(decoder_layers)} layers.")

    print(f"[INFO] Hooking decoder layers at: {layers_path} (len={len(decoder_layers)})")
    saved = register_thinker_decoder_hooks(decoder_layers, args.layers)

    pbar = tqdm(shard_videos, desc=f"Qwen3-VL shard {args.shard_idx}/{args.num_shards}", ncols=110)

    with ThreadPoolExecutor(max_workers=args.io_workers) as io_pool:
        pending_writes = []
        for video_path in pbar:
            vid_id = make_safe_video_identifier(video_path)

            # ---- output path: preserve subdir under args.file_dir ----
            shard_tag = f"s{args.shard_idx:02d}of{args.num_shards:02d}"
            out_name = (
                f"{vid_id}_L{len(args.layers)}_nf{args.max_frames}_res{args.res}"
                f"_new{args.max_new_token_num}_{shard_tag}.npz"
            )
            rel_subdir = relative_video_subdirectory(video_path, args.file_dir)
            out_dir = os.path.join(args.output_dir, rel_subdir) if rel_subdir else args.output_dir
            ensure_output_directory(out_dir)
            out_path = os.path.join(out_dir, out_name)
            # --------------------------------------------------------

            if os.path.exists(out_path):
                pbar.set_postfix_str(f"skip exists: {os.path.relpath(out_path, args.output_dir)}")
                continue

            for k in saved:
                saved[k].clear()

            abs_path = video_path if os.path.isabs(video_path) else os.path.abspath(video_path)
            file_url = f"file://{abs_path}"
            window_kwargs = first_half_video_kwargs(abs_path)

            def build_messages(nf_: int):
                video_content = {
                    "type": "video",
                    "video": file_url,
                    "nframes": int(nf_),
                    "resized_height": int(args.res),
                    "resized_width": int(args.res),
                }
                if window_kwargs:
                    video_content.update(window_kwargs)
                return [
                    {
                        "role": "user",
                        "content": [
                            video_content,
                            {"type": "text", "text": args.prompt},
                        ],
                    }
                ]

            # --------- Scheme-1: no ffprobe; retry once if short-video nframes error ----------
            nf_req = int(args.max_frames)
            nf_used = nf_req
            total_frames = 0  # unknown unless we parse it from error

            images = None
            videos = None
            video_kwargs = None
            video_metadatas = None

            ok = False
            last_err = None
            text = None

            for attempt in range(2):
                messages = build_messages(nf_used)

                try:
                    images, videos, video_kwargs = process_vision_info(
                        messages,
                        image_patch_size=16,
                        return_video_kwargs=True,
                        return_video_metadata=True,
                    )
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    ok = True
                    break
                except ValueError as e:
                    last_err = e
                    tf = parse_total_frames_from_nframes_error(e)
                    if tf > 0 and attempt == 0:
                        total_frames = tf
                        nf_used = clamp_uniform_thinker_frames(nf_req, total_frames)
                        if nf_used < 2:
                            ok = False
                            break
                        continue
                    ok = False
                    break
                except Exception as e:
                    last_err = e
                    ok = False
                    break

            if not ok:
                pbar.set_postfix_str(f"vision fail; skip ({type(last_err).__name__})")
                continue
            # --------------------------------------------------------------------------------

            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None

            inputs = processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs,
            )
            input_ids_cpu = inputs["input_ids"].detach().cpu()
            input_attn = inputs.get("attention_mask", None)
            input_valid_len = (
                int(input_attn[0].sum().item())
                if torch.is_tensor(input_attn)
                else int(input_ids_cpu.shape[1])
            )

            inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=int(args.max_new_token_num))

            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]

            out_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            out_text0 = out_text[0] if len(out_text) else ""

            vlm_old, vlm_new = stack_pyramid_guidance_states(saved, args.layers)
            supervision = load_optional_egodex_supervision_bundle(video_path)

            npz: Dict[str, Any] = {}
            npz["text"] = np.asarray(out_text0)
            npz["token_ids"] = (
                trimmed[0].detach().cpu().numpy().astype(np.int32) if len(trimmed) else np.zeros((0,), dtype=np.int32)
            )
            npz["input_ids"] = input_ids_cpu[0].numpy().astype(np.int32)
            npz["input_valid_len"] = np.asarray(int(input_valid_len), dtype=np.int32)
            npz["layers"] = np.asarray(args.layers, dtype=np.int32)
            npz["video_path"] = np.asarray(video_path)

            # requested/config
            npz["nframes_req"] = np.asarray(int(nf_req), dtype=np.int32)
            npz["nframes_used"] = np.asarray(int(nf_used), dtype=np.int32)
            npz["total_frames"] = np.asarray(int(total_frames), dtype=np.int32)  # 0 if unknown
            npz["res"] = np.asarray(int(args.res), dtype=np.int32)
            npz["backend_env"] = np.asarray(os.environ.get("FORCE_QWENVL_VIDEO_READER", ""))

            if vlm_old.numel():
                npz["vlm_old"] = tensor_to_cache_numpy(vlm_old, args.save_dtype)
            else:
                npz["vlm_old"] = np.zeros((0,), dtype=np.float16 if args.save_dtype == "fp16" else np.float32)

            if vlm_new.numel():
                npz["vlm_new"] = tensor_to_cache_numpy(vlm_new, args.save_dtype)
            else:
                npz["vlm_new"] = np.zeros((0,), dtype=np.float16 if args.save_dtype == "fp16" else np.float32)

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

            while len(pending_writes) >= args.max_pending_writes:
                pending_writes[0].result()
                pending_writes.pop(0)

            pending_writes.append(io_pool.submit(write_cache_archive, out_path, npz, args.save_mode))
            pbar.set_postfix_str(f"saved {os.path.relpath(out_path, args.output_dir)}")

        for fut in pending_writes:
            fut.result()

    print(f"[DONE] shard {args.shard_idx}/{args.num_shards} finished. saved to {args.output_dir}")


if __name__ == "__main__":
    main()
