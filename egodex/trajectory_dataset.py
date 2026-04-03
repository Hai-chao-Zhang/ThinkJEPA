# ThinkJEPA
# Copyright (c) 2026 Northeastern University, Haichao Zhang, et al.
# This file is part of the ThinkJEPA release associated with:
#
# @article{zhang2026thinkjepa,
#   title={ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model},
#   author={Zhang, Haichao and Li, Yijiang and He, Shwai and Nagarajan, Tushar and Chen, Mingfei and Lu, Jianglin and Li, Ang and Fu, Yun},
#   journal={arXiv preprint arXiv:2603.22281},
#   year={2026}
# }
#
# See LICENSE and NOTICE for release terms.
#
"""
egodex/trajectory_dataset.py
For licensing see accompanying LICENSE.txt file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

A simple PyTorch dataset using torchcodec for MP4 files and h5py for HDF5 files.
"""

import glob
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from decord import cpu, VideoReader
from torch import Tensor
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cache_train.hf_egodex import resolve_egodex_data_reference, rewrite_manifest_paths_for_release

# from torchcodec.decoders import VideoDecoder
from egodex.utils.trajectory_data_utils import index_episode_files
from egodex.utils.skeleton_tfs import LEFT_FINGERS, RIGHT_FINGERS, WRISTS

# loads only the wrist transforms by default. change as desired.
DEFAULT_QUERY_TFS = WRISTS

from torch.utils.data._utils.collate import default_collate

KEY_IMG_IDX = 8
KEY_WORLD_IDX = 2
TIME_FIELD_IDX = [0, 1, 2, 3, 4, 6, 8, 10]  # These fields use time T as their first dimension

K = np.array(
    [
        [736.6339, 0.0, 960.0],
        [0.0, 736.6339, 540.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

_QWEN_CACHE_NAME_RE = re.compile(
    r"^(?P<stem>.+?)_L\d+_nf\d+_res\d+_new\d+_s\d+of\d+$"
)

# Keep NPZ fallback supervision joint order aligned with thinker_train.py QUERY_TFS.
NPZ_TARGET_QUERY_TFS = (
    RIGHT_FINGERS
    + ["rightHand", "rightForearm"]
    + LEFT_FINGERS
    + ["leftHand", "leftForearm"]
)


class CameraGeometryLoadError(RuntimeError):
    """Raised when cache/HDF5 camera matrices are missing or malformed."""


EgoDexCameraGeometryError = CameraGeometryLoadError


def _raise_camera_geometry_error(path: str, context: str, detail: str) -> None:
    raise CameraGeometryLoadError(
        f"[CAM-INVALID] context={context} path={path} detail={detail}"
    )


def _validate_camera_extrinsics_or_raise(cam_ext, *, path: str, context: str) -> np.ndarray:
    arr = np.asarray(cam_ext, dtype=np.float32) if cam_ext is not None else None
    if arr is None:
        _raise_camera_geometry_error(path, context, "cam_ext is None")
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        _raise_camera_geometry_error(path, context, f"expected [T,4,4], got shape={arr.shape}")
    if arr.shape[0] <= 0:
        _raise_camera_geometry_error(path, context, "empty time dimension")
    if not np.isfinite(arr).all():
        _raise_camera_geometry_error(path, context, "contains NaN/Inf")

    last_row = arr[:, 3, :]
    ref_last = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    last_row_err = float(np.max(np.abs(last_row - ref_last)))
    if last_row_err > 1e-3:
        _raise_camera_geometry_error(
            path,
            context,
            f"homogeneous last-row mismatch (max_abs_err={last_row_err:.6g})",
        )

    rot = arr[:, :3, :3].astype(np.float64)
    det = np.linalg.det(rot)
    if not np.isfinite(det).all():
        _raise_camera_geometry_error(path, context, "rotation determinant has NaN/Inf")
    if np.any(np.abs(det) < 1e-6):
        _raise_camera_geometry_error(path, context, "rotation matrix is singular")

    return arr


def _validate_camera_intrinsics_or_raise(cam_int, *, path: str, context: str):
    arr = np.asarray(cam_int, dtype=np.float32) if cam_int is not None else None
    if arr is None:
        _raise_camera_geometry_error(path, context, "cam_int is None")
    keep_2d = arr.ndim == 2
    if keep_2d:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[1:] != (3, 3):
        _raise_camera_geometry_error(path, context, f"expected [T,3,3], got shape={arr.shape}")
    if arr.shape[0] <= 0:
        _raise_camera_geometry_error(path, context, "empty intrinsic time dimension")
    if not np.isfinite(arr).all():
        _raise_camera_geometry_error(path, context, "cam_int contains NaN/Inf")
    if np.any(np.abs(arr[:, 2, 2] - 1.0) > 1e-3):
        _raise_camera_geometry_error(path, context, "cam_int bottom-right entry is not 1")
    out = arr.astype(np.float32)
    return out[0] if keep_2d else out


def _maybe_decode_bfloat16_cache_array(x):
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.dtype != np.uint16:
        return arr
    try:
        return torch.from_numpy(arr.copy()).view(torch.bfloat16).float().cpu().numpy()
    except Exception:
        return arr


def _save_npz_atomic_archive(npz_path: str, payload: dict[str, Any], save_mode: str = "raw") -> None:
    tmp_path = f"{npz_path}.tmp.{os.getpid()}.{random.randint(0, 1_000_000)}.npz"
    if save_mode == "raw":
        np.savez(tmp_path, **payload)
    else:
        np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, npz_path)


def _leading_dimension(x):
    import numpy as np, torch

    return (
        int(x.shape[0])
        if isinstance(x, (np.ndarray, torch.Tensor)) and x.ndim >= 1
        else None
    )


def collate_and_drop_inconsistent_samples(batch):
    if not batch:
        return None

    # Use the first sample's key fields as the target shape
    ref_img = batch[0][KEY_IMG_IDX]
    ref_wld = batch[0][KEY_WORLD_IDX]
    Ti_ref = _leading_dimension(ref_img)
    Tw_ref = _leading_dimension(ref_wld)

    good = []
    drop = 0
    for s in batch:
        try:
            img, wld = s[KEY_IMG_IDX], s[KEY_WORLD_IDX]
            Ti, Tw = _leading_dimension(img), _leading_dimension(wld)
            # Key fields: must exist, have T>0, and agree on T
            if Ti is None or Tw is None or Ti <= 0 or Tw <= 0 or Ti != Tw:
                drop += 1
                continue

            # Temporal fields: first dimension must equal this sample's T
            T_ref = Ti
            ok = True
            for idx in TIME_FIELD_IDX:
                x = s[idx]
                if hasattr(x, "ndim") and x.ndim >= 1:
                    t = x.shape[0]
                    if t != T_ref:
                        ok = False
                        break
            if not ok:
                drop += 1
                continue

            # cam_int may be either (3, 3) or (T, 3, 3)
            cam_int = s[7]
            if not hasattr(cam_int, "ndim"):
                drop += 1
                continue
            if not (
                (cam_int.ndim == 2 and cam_int.shape == (3, 3))
                or (
                    cam_int.ndim == 3
                    and cam_int.shape[0] == T_ref
                    and cam_int.shape[-2:] == (3, 3)
                )
            ):
                drop += 1
                continue

            good.append(s)
        except Exception:
            drop += 1

    if drop:
        print(f"[WARN] dropped {drop}/{len(batch)} (first-dim/shape mismatch)")
    if not good:
        return None
    return default_collate(good)


# ----------------------------- NPZ file utilities -----------------------------


def list_cache_npz_files(
    cache_dir: str, shards: int | None = None, shards_id: int = 0
) -> list[str]:
    """Recursively list .npz files under cache_dir, with optional simple sharding."""
    files = sorted(glob.glob(os.path.join(cache_dir, "**", "*.npz"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .npz found under {cache_dir}")
    if shards is not None:
        if not isinstance(shards, int) or shards <= 0:
            raise ValueError(f"`shards` must be positive int, got {shards}")
        if not (0 <= int(shards_id) < int(shards)):
            raise ValueError(f"`shards_id` must be in [0,{shards-1}]")
        files = files[shards_id::shards]
    return files


def split_cache_npz_files_abt(files: list[str], a: int, b: int):
    """
    Use a and b from (a, b, t) to split:
      - train: files[:a]
      - test : files[a:b]
    Note: the signature matches the HDF5 pipeline; t is unused for NPZ and kept only
    for interface compatibility.
    """
    if a < 0 or b < 0:
        raise ValueError("a,b must be non-negative")
    a = min(a, len(files))
    b = min(b, len(files))
    return files[:a], files[a:b]


def _compute_dataset_split_bounds(
    total: int,
    abt: tuple[int, int, int] | tuple[float, float, float] | None,
    train_ratio: float,
) -> tuple[int, int]:
    """
    Return the train/test split boundaries (a, b):
      - train: [0:a)
      - test : [a:b)
    By default, when total>1 both train and test remain non-empty.
    """
    if total <= 0:
        return 0, 0
    if total == 1:
        return 1, 1

    # Default: split automatically by ratio
    if abt is None:
        a = int(total * float(train_ratio))
        a = max(1, min(a, total - 1))
        return a, total

    if len(abt) != 3:
        raise ValueError(f"`abt` must be a 3-tuple, got {abt}")

    a_raw, b_raw, _ = abt
    # Support ratio-style input such as (0.9, 1.0, 1.0)
    if isinstance(a_raw, float) or isinstance(b_raw, float):
        if not (0.0 <= float(a_raw) <= 1.0 and 0.0 <= float(b_raw) <= 1.0):
            raise ValueError(f"When abt uses float, each value must be in [0,1], got {abt}")
        a = int(total * float(a_raw))
        b = int(total * float(b_raw))
    else:
        a = int(a_raw)
        b = int(b_raw)

    a = max(0, min(a, total))
    b = max(a, min(b, total))

    # When total>1, try to keep both train and test non-empty
    if a == 0:
        a = max(1, min(int(total * float(train_ratio)), total - 1))
    if b <= a:
        b = total
    if a >= b:
        a = b - 1
    a = max(1, min(a, total - 1))
    b = max(a + 1, min(b, total))
    return a, b


def _maybe_shuffle_parallel_lists(
    paths: list[str],
    lengths: list[int] | None = None,
    split_seed: int = 42,
    split_shuffle: bool = True,
):
    if not split_shuffle or len(paths) <= 1:
        return paths, lengths
    idx = list(range(len(paths)))
    rnd = random.Random(int(split_seed))
    rnd.shuffle(idx)
    paths_shuf = [paths[i] for i in idx]
    if lengths is None:
        return paths_shuf, None
    lengths_shuf = [lengths[i] for i in idx]
    return paths_shuf, lengths_shuf


def _expand_dataset_root_candidates(dataset_path) -> list[str]:
    if isinstance(dataset_path, str):
        parts = [x.strip() for x in re.split(r"[,;]", dataset_path) if x.strip()]
        roots = parts if len(parts) > 0 else [dataset_path]
        return [resolve_egodex_data_reference(root) for root in roots]
    if isinstance(dataset_path, (list, tuple)):
        out = [str(x).strip() for x in dataset_path if str(x).strip()]
        if len(out) > 0:
            return [resolve_egodex_data_reference(root) for root in out]
    return [resolve_egodex_data_reference(str(dataset_path))]


def _normalize_thinker_cache_stem(npz_path: str) -> str:
    base = os.path.splitext(os.path.basename(npz_path))[0]
    match = _QWEN_CACHE_NAME_RE.match(base)
    return match.group("stem") if match else base


def _candidate_cache_relative_variants(rel_no_ext: str, cache_leaf: str | None) -> list[str]:
    rel_norm = os.path.normpath(rel_no_ext)
    parts = list(Path(rel_norm).parts)
    variants: list[str] = []

    def add(candidate_parts: list[str] | tuple[str, ...]) -> None:
        if not candidate_parts:
            return
        candidate = os.path.normpath(os.path.join(*candidate_parts))
        if candidate not in variants:
            variants.append(candidate)

    add(parts)
    if len(parts) >= 2 and parts[0] == parts[1]:
        add(parts[1:])
    if cache_leaf and parts and parts[0] == cache_leaf:
        add(parts[1:])
    if cache_leaf and len(parts) >= 3 and parts[0] == cache_leaf and parts[1] == cache_leaf:
        add(parts[2:])
    return variants


def _build_cache_manifest_index(cache_dir: str) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    for npz_path in list_cache_npz_files(cache_dir):
        rel = os.path.relpath(npz_path, start=cache_dir)
        rel_dir = os.path.dirname(rel)
        stem = _normalize_thinker_cache_stem(npz_path)
        index[(os.path.normpath(rel_dir), stem)] = npz_path
    return index


def _rewrite_manifest_paths_to_cache_archives(
    paths: list[str], dataset_path: str | None, cache_dir: str
) -> list[str]:
    if not paths:
        return paths
    if cache_dir is None:
        raise ValueError("cache_dir is required when mapping manifests to npz cache")

    cache_dir = resolve_egodex_data_reference(str(cache_dir))
    cache_index = _build_cache_manifest_index(cache_dir)
    cache_leaf = os.path.basename(os.path.normpath(cache_dir))
    dataset_roots = _expand_dataset_root_candidates(dataset_path) if dataset_path is not None else []

    def resolve_npz_manifest_path(path: str) -> str | None:
        if not path.endswith(".npz"):
            return None
        if not os.path.isabs(path):
            candidate = os.path.join(cache_dir, os.path.normpath(path))
            if os.path.exists(candidate):
                return candidate
        if os.path.exists(path):
            return path
        parts = list(Path(os.path.normpath(path)).parts)
        for idx, part in enumerate(parts):
            if part != cache_leaf:
                continue
            candidate = os.path.join(cache_dir, *parts[idx + 1 :])
            if os.path.exists(candidate):
                return candidate
        return None

    out: list[str] = []
    for path in paths:
        resolved_npz = resolve_npz_manifest_path(path)
        if resolved_npz is not None:
            out.append(resolved_npz)
            continue

        resolved = resolve_egodex_data_reference(path)
        candidates: list[str] = []
        for root in dataset_roots:
            try:
                rel = os.path.relpath(resolved, root)
            except ValueError:
                continue
            if rel == ".." or rel.startswith(f"..{os.sep}"):
                continue
            rel_no_ext = os.path.splitext(rel)[0]
            for rel_variant in _candidate_cache_relative_variants(rel_no_ext, cache_leaf):
                rel_dir = os.path.dirname(rel_variant)
                stem = os.path.basename(rel_variant)
                candidate = cache_index.get((os.path.normpath(rel_dir), stem))
                if candidate is not None:
                    candidates.append(candidate)
        if not candidates:
            raise FileNotFoundError(
                f"manifest path {path} could not be mapped into cache_dir={cache_dir}"
            )
        out.append(candidates[0])
    return out


def _load_path_manifest_file(manifest_path: str, dataset_path: str | None = None) -> list[str]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
    if len(paths) == 0:
        raise ValueError(f"manifest has no usable paths: {manifest_path}")
    if dataset_path is not None:
        paths = rewrite_manifest_paths_for_release(paths, dataset_path)
    return paths


def _index_hdf5_paths_only(dataset_root: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(dataset_root, "**", "*.hdf5"), recursive=True))
    print(f"Found {len(files)} hdf5 files under {dataset_root} (fast index mode)")
    return files


def _expand_roots_for_sidecar_lookup(dataset_path) -> list[str]:
    roots = _expand_dataset_root_candidates(dataset_path)
    return [os.path.abspath(x) for x in roots]


def _resolve_sidecar_path_for_hdf5(
    h5_path: str, sidecar_root: str | None, data_roots: list[str] | None
) -> str | None:
    if not sidecar_root or not data_roots:
        return None
    h5_abs = os.path.abspath(h5_path)
    for root in data_roots:
        try:
            rel = os.path.relpath(h5_abs, root)
        except ValueError:
            continue
        if rel == ".." or rel.startswith(f"..{os.sep}"):
            continue
        rel_noext = os.path.splitext(rel)[0]
        return os.path.join(sidecar_root, rel_noext + ".sup.npz")
    return None


def _scalar_to_string(x):
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    arr = np.asarray(x)
    try:
        return str(arr.item())
    except Exception:
        return None


def _resolve_hdf5_path_from_video(video_path: str) -> str | None:
    root, ext = os.path.splitext(video_path)
    if ext.lower() != ".mp4":
        return None
    hdf5_path = root + ".hdf5"
    return hdf5_path if os.path.exists(hdf5_path) else None


def _coerce_language_annotation(x):
    if isinstance(x, str):
        return x
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return _coerce_language_annotation(x.item())
        if x.size == 1:
            return _coerce_language_annotation(x.reshape(()).item())
    return str(x)


def _read_language_instruction(root) -> str:
    try:
        if root.attrs["llm_type"] == "reversible":
            direction = root.attrs["which_llm_description"]
            lang_instruct = root.attrs[
                "llm_description" if direction == "1" else "llm_description2"
            ]
        else:
            lang_instruct = root.attrs["llm_description"]
    except KeyError:
        lang_instruct = root.attrs["llm_description"]
    return _coerce_language_annotation(lang_instruct)


def load_supervision_from_hdf5(hdf5_file: str, query_tfs) -> dict:
    with h5py.File(hdf5_file, "r") as root:
        T = int(root["/transforms/camera"].shape[0])

        idxs_raw = _sample_dense_jepa_frame_indices(T).astype(np.int64)
        sel, inv = np.unique(idxs_raw, return_inverse=True)
        frame_indices = sel[inv].astype(np.int64)

        tfs = np.stack(
            [root[f"/transforms/{tf_name}"][sel] for tf_name in query_tfs],
            axis=1,
        )
        tfs = tfs[inv]

        cam_ext = root["/transforms/camera"][sel]
        cam_ext = cam_ext[inv]

        cam_int = np.asarray(root["/camera/intrinsic"][:], dtype=np.float32)
        if cam_int.ndim == 3 and cam_int.shape[1:] == (3, 3):
            if cam_int.shape[0] == 1:
                cam_int = cam_int[0]
            elif cam_int.shape[0] == T:
                cam_int = cam_int[sel]
                cam_int = cam_int[inv]
            else:
                _raise_camera_geometry_error(
                    hdf5_file,
                    "egodex_hdf5",
                    f"unsupported intrinsic shape={cam_int.shape}",
                )
        elif cam_int.shape != (3, 3):
            _raise_camera_geometry_error(
                hdf5_file,
                "egodex_hdf5",
                f"unsupported intrinsic shape={cam_int.shape}",
            )

        lang_instruct = _read_language_instruction(root)

        confs = None
        if "confidences" in root:
            conf_list = []
            ok = True
            for tf_name in query_tfs:
                key = f"/confidences/{tf_name}"
                if key not in root:
                    ok = False
                    break
                c = root[key][sel]
                conf_list.append(c[inv])
            if ok and conf_list:
                confs = np.stack(conf_list, axis=1)

    cam_ext = _validate_camera_extrinsics_or_raise(
        cam_ext, path=hdf5_file, context="egodex_hdf5"
    )
    cam_int = _validate_camera_intrinsics_or_raise(
        cam_int, path=hdf5_file, context="egodex_hdf5"
    )
    try:
        cam_inv = np.linalg.inv(cam_ext)
    except np.linalg.LinAlgError as exc:
        _raise_camera_geometry_error(
            hdf5_file, "egodex_hdf5", f"np.linalg.inv failed: {exc}"
        )
    tfs_in_cam = cam_inv[:, None, :, :] @ tfs
    xyz_cam = tfs_in_cam[..., :3, 3]
    R_cam = tfs_in_cam[..., :3, :3]
    xyz_world = tfs[..., :3, 3]
    R_world = tfs[..., :3, :3]

    if confs is None:
        confs = np.ones((xyz_cam.shape[0], xyz_cam.shape[1]), dtype=np.float32)

    return {
        "frame_indices": frame_indices,
        "xyz_cam": xyz_cam,
        "R_cam": R_cam,
        "xyz_world": xyz_world,
        "R_world": R_world,
        "tfs_in_cam": tfs_in_cam,
        "tfs": tfs,
        "cam_ext": cam_ext,
        "cam_int": cam_int,
        "lang_instruct": lang_instruct,
        "confs": confs,
    }


def load_supervision_from_sidecar(sidecar_path: str) -> dict:
    with np.load(sidecar_path, allow_pickle=False) as z:
        confs = z["confs"] if "confs" in z else None
        out = {
            "frame_indices": z["frame_indices"].astype(np.int64),
            "xyz_cam": z["xyz_cam"],
            "R_cam": z["R_cam"],
            "xyz_world": z["xyz_world"],
            "R_world": z["R_world"],
            "tfs_in_cam": z["tfs_in_cam"],
            "tfs": z["tfs"],
            "cam_ext": z["cam_ext"],
            "cam_int": z["cam_int"],
            "lang_instruct": _coerce_language_annotation(z["lang_instruct"])
            if "lang_instruct" in z
            else "",
            "confs": confs,
        }
    out["cam_ext"] = _validate_camera_extrinsics_or_raise(
        out["cam_ext"], path=sidecar_path, context="egodex_sidecar"
    )
    out["cam_int"] = _validate_camera_intrinsics_or_raise(
        out["cam_int"], path=sidecar_path, context="egodex_sidecar"
    )
    if out["confs"] is None:
        out["confs"] = np.ones(
            (out["xyz_cam"].shape[0], out["xyz_cam"].shape[1]), dtype=np.float32
        )
    return out


def read_video_clip(mp4_file: str, frame_indices: np.ndarray) -> Tensor:
    vr = VideoReader(mp4_file, ctx=cpu(0), width=256, height=256)
    if len(vr) == 0:
        raise RuntimeError(f"Empty video: {mp4_file}")
    clip = np.clip(frame_indices, 0, len(vr) - 1).astype(np.int64)
    batch = vr.get_batch(clip)
    return torch.from_numpy(batch.asnumpy()).contiguous()


# --------------------------- HDF5/MP4 pipeline ---------------------------


def build_egodex_dataloaders(
    dataset_path,
    query_tfs,
    if_return_path=False,
    train_batch=8,
    test_batch=8,
    train_manifest=None,
    test_manifest=None,
    # NPZ cache pipeline
    use_npz_cache=False,
    cache_dir=None,
    shards=None,
    shards_id=0,
    # Unified abt split for both HDF5 and NPZ; NPZ ignores t
    abt: tuple[int, int, int] | tuple[float, float, float] | None = None,
    train_ratio: float = 0.9,
    split_seed: int = 42,
    split_shuffle: bool = True,
    num_workers: int = 8,
    fast_index_when_full_scan: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    preload_to_memory: bool = False,
    camera_mode: str = "auto",
    # NPZ padding configuration
    pad_old_to: int = 480,
    pad_new_to: int = 15,
    pad_value: float = 0.0,
    supervision_cache_root: str | None = None,
):
    """
    - use_npz_cache=False: use the original HDF5/MP4 SimpleDataset path
    - use_npz_cache=True: read .npz files from cache_dir
      (list_cache_npz_files + split_cache_npz_files_abt + NpzCacheDataset(files))
    """

    if bool(train_manifest) ^ bool(test_manifest):
        raise ValueError("train_manifest and test_manifest must be provided together")

    dl_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = prefetch_factor

    if train_manifest and test_manifest:
        manifest_dataset_path = None if use_npz_cache else str(dataset_path)
        train_files = _load_path_manifest_file(str(train_manifest), manifest_dataset_path)
        test_files = _load_path_manifest_file(str(test_manifest), manifest_dataset_path)
        dataset_roots = _expand_roots_for_sidecar_lookup(dataset_path)

        if use_npz_cache:
            train_files = _rewrite_manifest_paths_to_cache_archives(
                train_files, dataset_path=dataset_path, cache_dir=cache_dir
            )
            test_files = _rewrite_manifest_paths_to_cache_archives(
                test_files, dataset_path=dataset_path, cache_dir=cache_dir
            )
            train_ds = NpzCacheDataset(
                train_files,
                if_return_path=if_return_path,
                preload_to_memory=preload_to_memory,
                camera_mode=camera_mode,
                pad_old_to=pad_old_to,
                pad_new_to=pad_new_to,
                pad_value=pad_value,
                data_roots=dataset_roots,
                cache_root=cache_dir,
            )
            test_ds = NpzCacheDataset(
                test_files,
                if_return_path=if_return_path,
                preload_to_memory=preload_to_memory,
                camera_mode=camera_mode,
                pad_old_to=pad_old_to,
                pad_new_to=pad_new_to,
                pad_value=pad_value,
                data_roots=dataset_roots,
                cache_root=cache_dir,
            )
        else:
            train_ds = SimpleDataset(
                train_files,
                episode_len=[1] * len(train_files),
                query_tfs=query_tfs,
                if_return_path=if_return_path,
                shards=shards,
                shards_id=shards_id,
                preload_to_memory=preload_to_memory,
                data_roots=dataset_roots,
                supervision_cache_root=supervision_cache_root,
            )
            test_ds = SimpleDataset(
                test_files,
                episode_len=[1] * len(test_files),
                query_tfs=query_tfs,
                if_return_path=if_return_path,
                shards=shards,
                shards_id=shards_id,
                preload_to_memory=preload_to_memory,
                data_roots=dataset_roots,
                supervision_cache_root=supervision_cache_root,
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=train_batch,
            shuffle=True,
            **dl_kwargs,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=test_batch,
            shuffle=False,
            **dl_kwargs,
        )
        print(
            f"[INFO] Using fixed manifests: train={len(train_files)} test={len(test_files)} "
            f"train_manifest={train_manifest} test_manifest={test_manifest}"
        )
        return train_loader, test_loader

    if use_npz_cache:
        assert cache_dir is not None, "cache_dir must be provided when use_npz_cache=True"
        files_all = list_cache_npz_files(cache_dir, shards=shards, shards_id=shards_id)
        files_all, _ = _maybe_shuffle_parallel_lists(
            files_all, None, split_seed=split_seed, split_shuffle=split_shuffle
        )
        a, b = _compute_dataset_split_bounds(len(files_all), abt=abt, train_ratio=train_ratio)
        train_files, test_files = split_cache_npz_files_abt(files_all, a, b)

        dataset_roots = _expand_roots_for_sidecar_lookup(dataset_path)
        train_ds = NpzCacheDataset(
            train_files,
            if_return_path=if_return_path,
            preload_to_memory=preload_to_memory,
            camera_mode=camera_mode,
            pad_old_to=pad_old_to,
            pad_new_to=pad_new_to,
            pad_value=pad_value,
            data_roots=dataset_roots,
            cache_root=cache_dir,
        )
        test_ds = NpzCacheDataset(
            test_files,
            if_return_path=if_return_path,
            preload_to_memory=preload_to_memory,
            camera_mode=camera_mode,
            pad_old_to=pad_old_to,
            pad_new_to=pad_new_to,
            pad_value=pad_value,
            data_roots=dataset_roots,
            cache_root=cache_dir,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=train_batch,
            shuffle=True,
            **dl_kwargs,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=test_batch,
            shuffle=False,
            **dl_kwargs,
        )
        return train_loader, test_loader

    # ===== Original HDF5/MP4 pipeline =====
    if abt is None:
        # Used only as the max traversal length for index_episode_files; None means no early cutoff.
        t = None
    else:
        _, _, t = abt
    dataset_path_list = []
    episode_len = []
    for root in _expand_dataset_root_candidates(dataset_path):
        if fast_index_when_full_scan and t is None:
            _paths = _index_hdf5_paths_only(root)
            _lens = [1] * len(_paths)
        else:
            _paths, _lens = index_episode_files(root, thread=t)
        if len(_paths) != len(_lens):
            n = min(len(_paths), len(_lens))
            print(
                f"[WARN] index_episode_files length mismatch: root={root} "
                f"paths={len(_paths)} lens={len(_lens)} -> trim to {n}"
            )
            _paths = _paths[:n]
            _lens = _lens[:n]
        dataset_path_list.extend(_paths)
        episode_len.extend(_lens)
    dataset_path_list, episode_len = _maybe_shuffle_parallel_lists(
        dataset_path_list,
        episode_len,
        split_seed=split_seed,
        split_shuffle=split_shuffle,
    )
    a, b = _compute_dataset_split_bounds(
        len(dataset_path_list),
        abt=abt,
        train_ratio=train_ratio,
    )
    train_ds = SimpleDataset(
        dataset_path_list[:a],
        episode_len=episode_len[:a],
        query_tfs=query_tfs,
        if_return_path=if_return_path,
        shards=shards,
        shards_id=shards_id,
        preload_to_memory=preload_to_memory,
        data_roots=_expand_roots_for_sidecar_lookup(dataset_path),
        supervision_cache_root=supervision_cache_root,
    )
    test_ds = SimpleDataset(
        dataset_path_list[a:b],
        episode_len=episode_len[a:b],
        query_tfs=query_tfs,
        if_return_path=if_return_path,
        shards=shards,
        shards_id=shards_id,
        preload_to_memory=preload_to_memory,
        data_roots=_expand_roots_for_sidecar_lookup(dataset_path),
        supervision_cache_root=supervision_cache_root,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch,
        shuffle=True,
        **dl_kwargs,
        # collate_fn=collate_and_drop_inconsistent_samples,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch,
        shuffle=False,
        **dl_kwargs,
        # collate_fn=collate_and_drop_inconsistent_samples,
    )
    return train_loader, test_loader


def _sample_dense_jepa_frame_indices(T: int) -> np.ndarray:
    if T <= 0:
        raise ValueError("Empty episode (T=0).")
    # No wrap-around: generate 64 non-decreasing points in [0, T-1];
    # later _h5_take_sorted_unique handles h5py's strict monotonicity requirement
    return np.linspace(0, T - 1, num=64, dtype=np.int64)


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path_list,
        episode_len,
        query_tfs=DEFAULT_QUERY_TFS,
        if_return_path=False,
        shards=None,
        shards_id=0,
        preload_to_memory: bool = False,
        data_roots=None,
        supervision_cache_root: str | None = None,
    ):
        # a list of transforms that the dataset will return
        self.query_tfs = query_tfs

        # --- sharding: take every `shards`-th sample starting from `shards_id` ---
        if shards is not None:
            if not isinstance(shards, int) or shards <= 0:
                raise ValueError(f"`shards` must be positive int, got {shards}")
            if not isinstance(shards_id, int) or not (0 <= shards_id < shards):
                raise ValueError(
                    f"`shards_id` must be in [0, {shards-1}], got {shards_id}"
                )

            idx: list[int] = list(range(shards_id, len(dataset_path_list), shards))
            # Filter paths and lengths in sync
            dataset_path_list = [dataset_path_list[i] for i in idx]
            episode_len = [episode_len[i] for i in idx]

        # get episode paths and lengths
        self.dataset_path_list, self.episode_len = dataset_path_list, episode_len

        self.cumulative_len = np.cumsum(self.episode_len)
        self.if_return_path = if_return_path
        self.preload_to_memory = bool(preload_to_memory)
        self.data_roots = (
            [os.path.abspath(x) for x in data_roots] if data_roots is not None else []
        )
        self.supervision_cache_root = supervision_cache_root
        self._memory_samples = None
        if self.preload_to_memory:
            print(
                f"[INFO] Preloading {len(self.dataset_path_list)} raw EgoDex samples into memory",
                flush=True,
            )
            self._memory_samples = [
                self._load_hdf5_episode(self.dataset_path_list[i])
                for i in range(len(self.dataset_path_list))
            ]

    def __len__(self):
        return len(self.dataset_path_list)

    def _locate_transition_window(self, index):
        # find a particular data point within an episode
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(
            self.cumulative_len > index
        )  # argmax returns first True index
        start_ts = index - (
            self.cumulative_len[episode_index] - self.episode_len[episode_index]
        )
        return episode_index, start_ts

    def _load_hdf5_episode(self, hdf5_file: str):
        mp4_file = hdf5_file[:-5] + ".mp4"
        sup = None
        sidecar_path = _resolve_sidecar_path_for_hdf5(
            hdf5_file, self.supervision_cache_root, self.data_roots
        )
        if sidecar_path and os.path.isfile(sidecar_path):
            try:
                sup = load_supervision_from_sidecar(sidecar_path)
            except Exception as e:
                print(
                    f"[WARN] failed to load supervision sidecar, fallback to hdf5: "
                    f"path={sidecar_path} err={type(e).__name__}: {e}",
                    flush=True,
                )
        if sup is None:
            sup = load_supervision_from_hdf5(hdf5_file, self.query_tfs)

        img = read_video_clip(mp4_file, sup["frame_indices"])
        xyz_cam = sup["xyz_cam"]
        R_cam = sup["R_cam"]
        xyz_world = sup["xyz_world"]
        R_world = sup["R_world"]
        tfs_in_cam = sup["tfs_in_cam"]
        tfs = sup["tfs"]
        cam_ext = sup["cam_ext"]
        cam_int = sup["cam_int"]
        lang_instruct = sup["lang_instruct"]
        confs = sup["confs"]

        if self.if_return_path:
            return (
                xyz_cam,
                R_cam,
                xyz_world,
                R_world,
                tfs_in_cam,
                tfs,
                cam_ext,
                cam_int,
                img,  # (64, H, W, C)
                lang_instruct,
                confs,  # (64, N) or None
                hdf5_file,
            )
        else:
            return (
                xyz_cam,
                R_cam,
                xyz_world,
                R_world,
                tfs_in_cam,
                tfs,
                cam_ext,
                cam_int,
                img,
                lang_instruct,
                confs,
            )

    def __getitem__(self, episode_id):
        if self._memory_samples is not None:
            return self._memory_samples[episode_id]
        hdf5_file = self.dataset_path_list[episode_id]
        return self._load_hdf5_episode(hdf5_file)


# ======================= NPZ cache dataset (consumes only the provided files) =======================

class NpzCacheDataset(torch.utils.data.Dataset):
    """
    Read .npz files from the provided file list and return them in the existing order:
    (xyz_cam, R_cam, xyz_world, R_world, tfs_in_cam, tfs, cam_ext, cam_int, img, lang_instruct, confs, extras, [path])
    Here img is [T, H, W, C] (uint8); all other floating-point arrays are converted
    to float32.
    - Directory scanning/splitting is not handled here; it happens externally
      (list_cache_npz_files + split_cache_npz_files_abt)
    - Optionally pad/truncate to fixed lengths pad_old_to / pad_new_to and return
      the corresponding mask and original length
    """

    def __init__(
        self,
        files: list[str],
        if_return_path: bool = True,
        preload_to_memory: bool = False,
        pad_old_to: int = 480,
        pad_new_to: int = 15,
        pad_value: float = 0.0,
        camera_mode: str = "auto",
        data_roots: list[str] | None = None,
        cache_root: str | None = None,
        npz_save_mode: str = "raw",
    ):
        if not isinstance(files, (list, tuple)) or len(files) == 0:
            raise ValueError("`files` must be a non-empty list of .npz paths")
        self.files = list(files)
        self.if_return_path = if_return_path
        self.preload_to_memory = bool(preload_to_memory)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pad_old_to = int(pad_old_to)
        self.pad_new_to = int(pad_new_to)
        self.pad_value = float(pad_value)
        self.camera_mode = str(camera_mode).strip().lower()
        self.data_roots = [os.path.abspath(x) for x in (data_roots or [])]
        self.cache_root = os.path.abspath(cache_root) if cache_root else None
        self.npz_save_mode = str(npz_save_mode).strip().lower()
        if self.camera_mode not in {"auto", "egodex", "egoexo"}:
            raise ValueError(
                f"`camera_mode` must be one of auto|egodex|egoexo, got {camera_mode}"
            )
        self._memory_samples = None
        if self.preload_to_memory:
            print(
                f"[INFO] Preloading {len(self.files)} npz cache samples into memory",
                flush=True,
            )
            self._memory_samples = [self._load_cache_sample(path) for path in self.files]

    def __len__(self):
        return len(self.files)

    def _cast_tensor_to_float32_cpu(self, x):
        return torch.from_numpy(x).float().contiguous() if x is not None else None

    @staticmethod
    def _to_int_scalar(x):
        if x is None:
            return None
        arr = np.asarray(x)
        if arr.size == 0:
            return None
        try:
            return int(arr.reshape(-1)[0])
        except Exception:
            return None

    @staticmethod
    def _coerce_matrix_4x4(extr):
        if extr is None:
            return None
        arr = np.asarray(_maybe_decode_bfloat16_cache_array(extr), dtype=np.float32)
        if arr.ndim == 2:
            if arr.shape == (4, 4):
                return arr
            if arr.shape == (3, 4):
                out = np.eye(4, dtype=np.float32)
                out[:3, :4] = arr
                return out
            return None
        if arr.ndim == 3:
            if arr.shape[1:] == (4, 4):
                return arr
            if arr.shape[1:] == (3, 4):
                out = np.broadcast_to(
                    np.eye(4, dtype=np.float32), (arr.shape[0], 4, 4)
                ).copy()
                out[:, :3, :4] = arr
                return out
            return None
        return None

    @staticmethod
    def _select_uniform_indices(x, target_len: int):
        assert target_len >= 1
        if x.shape[0] == target_len:
            return x
        if x.shape[0] <= 0:
            return None
        sel = np.linspace(0, x.shape[0] - 1, num=target_len, dtype=np.int64)
        return x[sel]

    @staticmethod
    def _match_nearest_frame_ids(values, src_frame_ids, target_frame_ids):
        if values is None:
            return None
        src = np.asarray(src_frame_ids, dtype=np.float32).reshape(-1)
        tgt = np.asarray(target_frame_ids, dtype=np.float32).reshape(-1)
        if values.shape[0] != src.shape[0] or src.shape[0] == 0 or tgt.shape[0] == 0:
            return None

        order = np.argsort(src)
        src = src[order]
        values = values[order]

        right = np.searchsorted(src, tgt, side="left")
        right = np.clip(right, 0, src.shape[0] - 1)
        left = np.clip(right - 1, 0, src.shape[0] - 1)
        choose_right = np.abs(src[right] - tgt) <= np.abs(src[left] - tgt)
        nn = np.where(choose_right, right, left).astype(np.int64)
        return values[nn]

    @staticmethod
    def _compute_target_frame_ids_for_cache(
        img_t: int, total_frames: int | None, dyn_frame_ids
    ) -> np.ndarray:
        if img_t <= 0:
            return np.zeros((0,), dtype=np.float32)
        if total_frames is not None and total_frames > 1:
            return np.linspace(0, total_frames - 1, num=img_t, dtype=np.float32)
        if dyn_frame_ids is not None:
            arr = np.asarray(dyn_frame_ids, dtype=np.float32).reshape(-1)
            if arr.size >= 2:
                return np.linspace(arr[0], arr[-1], num=img_t, dtype=np.float32)
        return np.linspace(0, max(img_t - 1, 0), num=img_t, dtype=np.float32)

    def _resolve_egoexo_camera_extrinsics(
        self,
        img_t: int,
        total_frames: int | None,
        ego_ext_static,
        ego_ext_dynamic,
        ego_ext_dynamic_frame_ids,
    ):
        """
        EgoExo camera_pose stores camera_extrinsics as cam_from_world.
        Training path expects world_from_cam, so we invert here.
        """
        dyn_4x4 = self._coerce_matrix_4x4(ego_ext_dynamic)
        if dyn_4x4 is not None and dyn_4x4.ndim == 3 and dyn_4x4.shape[0] > 0:
            if dyn_4x4.shape[0] == img_t:
                cam_from_world = dyn_4x4
            else:
                tgt_ids = self._compute_target_frame_ids_for_cache(
                    img_t, total_frames, ego_ext_dynamic_frame_ids
                )
                picked = None
                if ego_ext_dynamic_frame_ids is not None:
                    picked = self._match_nearest_frame_ids(
                        dyn_4x4, ego_ext_dynamic_frame_ids, tgt_ids
                    )
                if picked is None:
                    picked = self._select_uniform_indices(dyn_4x4, target_len=img_t)
                cam_from_world = picked
            if cam_from_world is not None and cam_from_world.shape[0] > 0:
                return np.linalg.inv(cam_from_world.astype(np.float64)).astype(np.float32)

        static_4x4 = self._coerce_matrix_4x4(ego_ext_static)
        if static_4x4 is not None and static_4x4.ndim == 2:
            world_from_cam = np.linalg.inv(static_4x4.astype(np.float64)).astype(
                np.float32
            )
            return np.broadcast_to(world_from_cam[None], (img_t, 4, 4)).copy()
        return None

    def _resolve_egoexo_camera_intrinsics(
        self,
        img_t: int,
        total_frames: int | None,
        ego_int_static,
        ego_int_dynamic,
        ego_int_dynamic_frame_ids,
    ):
        dyn = None if ego_int_dynamic is None else np.asarray(_maybe_decode_bfloat16_cache_array(ego_int_dynamic), dtype=np.float32)
        if dyn is not None:
            if dyn.ndim == 2 and dyn.shape == (3, 3):
                return dyn
            if dyn.ndim == 3 and dyn.shape[1:] == (3, 3) and dyn.shape[0] > 0:
                if dyn.shape[0] == img_t:
                    return dyn
                tgt_ids = self._compute_target_frame_ids_for_cache(
                    img_t, total_frames, ego_int_dynamic_frame_ids
                )
                picked = None
                if ego_int_dynamic_frame_ids is not None:
                    picked = self._match_nearest_frame_ids(
                        dyn, ego_int_dynamic_frame_ids, tgt_ids
                    )
                if picked is None:
                    picked = self._select_uniform_indices(dyn, target_len=img_t)
                return picked

        sta = None if ego_int_static is None else np.asarray(_maybe_decode_bfloat16_cache_array(ego_int_static), dtype=np.float32)
        if sta is not None and sta.shape == (3, 3):
            return sta
        return None

    @staticmethod
    def _normalize_hand_joint_name(name: str) -> str:
        s = str(name).strip()
        if not s:
            return ""
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", s)
        s = s.replace("_", "-").lower()
        s = re.sub(r"-+", "-", s).strip("-")
        return s

    def _align_hand_pose_sequence(
        self,
        values,
        frame_ids,
        *,
        img_t: int,
        total_frames: int | None,
    ):
        if values is None:
            return None
        arr = np.asarray(_maybe_decode_bfloat16_cache_array(values), dtype=np.float32)
        if arr.ndim != 3 or arr.shape[0] <= 0:
            return None
        if arr.shape[0] == img_t:
            return arr
        tgt_ids = self._compute_target_frame_ids_for_cache(img_t, total_frames, frame_ids)
        picked = None
        if frame_ids is not None:
            picked = self._match_nearest_frame_ids(arr, frame_ids, tgt_ids)
        if picked is None:
            picked = self._select_uniform_indices(arr, target_len=img_t)
        if picked is None:
            return None
        return np.asarray(picked, dtype=np.float32)

    @staticmethod
    def _map_query_joint_to_egoexo_candidates(query_name_norm: str) -> list[str]:
        side = "right" if query_name_norm.startswith("right-") else "left"
        pairs = (
            ("little-finger", "pinky"),
            ("ring-finger", "ring"),
            ("middle-finger", "middle"),
            ("index-finger", "index"),
        )
        for q_pat, exo_pat in pairs:
            prefix = f"{side}-{q_pat}-"
            if query_name_norm.startswith(prefix):
                suffix = query_name_norm[len(prefix) :]
                if suffix in {"metacarpal", "knuckle"}:
                    return [f"{side}-{exo_pat}-1"]
                if suffix == "intermediate-base":
                    return [f"{side}-{exo_pat}-2"]
                if suffix == "intermediate-tip":
                    return [f"{side}-{exo_pat}-3"]
                if suffix == "tip":
                    return [f"{side}-{exo_pat}-4"]
                return []

        thumb_prefix = f"{side}-thumb-"
        if query_name_norm.startswith(thumb_prefix):
            suffix = query_name_norm[len(thumb_prefix) :]
            if suffix == "knuckle":
                return [f"{side}-thumb-1"]
            if suffix == "intermediate-base":
                return [f"{side}-thumb-2"]
            if suffix == "intermediate-tip":
                return [f"{side}-thumb-3"]
            if suffix == "tip":
                return [f"{side}-thumb-4"]
            return []

        if query_name_norm == f"{side}-hand":
            return [f"{side}-wrist"]
        if query_name_norm == f"{side}-forearm":
            return [f"{side}-elbow", f"{side}-shoulder", f"{side}-wrist"]
        return []

    def _build_camera_xyz_from_egoexo_pose(
        self,
        *,
        img_t: int,
        total_frames: int | None,
        body_xyz,
        body_joint_names,
        body_frame_ids,
        hand_xyz,
        hand_joint_names,
        hand_frame_ids,
    ):
        if img_t <= 0:
            return None

        body_seq = self._align_hand_pose_sequence(
            body_xyz, body_frame_ids, img_t=img_t, total_frames=total_frames
        )
        hand_seq = self._align_hand_pose_sequence(
            hand_xyz, hand_frame_ids, img_t=img_t, total_frames=total_frames
        )

        sources: dict[str, np.ndarray] = {}
        if body_seq is not None and body_joint_names is not None:
            body_names = [self._normalize_hand_joint_name(x) for x in np.asarray(body_joint_names).tolist()]
            for j, name in enumerate(body_names):
                if name:
                    sources.setdefault(name, body_seq[:, j, :])
        if hand_seq is not None and hand_joint_names is not None:
            hand_names = [self._normalize_hand_joint_name(x) for x in np.asarray(hand_joint_names).tolist()]
            for j, name in enumerate(hand_names):
                if name:
                    sources[name] = hand_seq[:, j, :]

        right_wrist = sources.get("right-wrist", None)
        left_wrist = sources.get("left-wrist", None)
        right_elbow = sources.get("right-elbow", right_wrist)
        left_elbow = sources.get("left-elbow", left_wrist)

        def _fallback(side: str):
            if side == "right":
                if right_elbow is not None:
                    return right_elbow
                if right_wrist is not None:
                    return right_wrist
            else:
                if left_elbow is not None:
                    return left_elbow
                if left_wrist is not None:
                    return left_wrist
            return np.zeros((img_t, 3), dtype=np.float32)

        out = np.zeros((img_t, len(NPZ_TARGET_QUERY_TFS), 3), dtype=np.float32)
        for j, q_name in enumerate(NPZ_TARGET_QUERY_TFS):
            qn = self._normalize_hand_joint_name(q_name)
            cands = self._map_query_joint_to_egoexo_candidates(qn)
            val = None
            for c in cands:
                cur = sources.get(c, None)
                if cur is not None:
                    val = cur
                    break
            if val is None:
                val = _fallback("right" if qn.startswith("right-") else "left")
            out[:, j, :] = np.asarray(val, dtype=np.float32)

        return out

    def _normalize_camera_extrinsics_per_frame(self, cam_ext, img_t: int):
        ext = self._coerce_matrix_4x4(cam_ext)
        if ext is None:
            return None
        if ext.ndim == 2:
            return np.broadcast_to(ext[None], (img_t, 4, 4)).copy().astype(np.float32)
        if ext.ndim == 3:
            if ext.shape[0] == img_t:
                return ext.astype(np.float32)
            picked = self._select_uniform_indices(ext, target_len=img_t)
            return None if picked is None else np.asarray(picked, dtype=np.float32)
        return None

    def _normalize_camera_intrinsics_per_frame(self, cam_int, img_t: int):
        if cam_int is None:
            return None
        arr = np.asarray(_maybe_decode_bfloat16_cache_array(cam_int), dtype=np.float32)
        if arr.ndim == 2 and arr.shape == (3, 3):
            return np.broadcast_to(arr[None], (img_t, 3, 3)).copy()
        if arr.ndim == 3 and arr.shape[1:] == (3, 3):
            if arr.shape[0] == img_t:
                return arr
            picked = self._select_uniform_indices(arr, target_len=img_t)
            return None if picked is None else np.asarray(picked, dtype=np.float32)
        return None

    @staticmethod
    def _project_camera_xyz_to_world_np(xyz_cam: np.ndarray, cam_ext: np.ndarray):
        if xyz_cam is None or cam_ext is None:
            return None
        if xyz_cam.ndim != 3 or cam_ext.ndim != 3:
            return None
        if xyz_cam.shape[0] != cam_ext.shape[0]:
            return None
        ones = np.ones((*xyz_cam.shape[:2], 1), dtype=np.float32)
        xyz_cam_h = np.concatenate([xyz_cam.astype(np.float32), ones], axis=-1)
        xyz_world_h = np.einsum("tij,tkj->tki", cam_ext.astype(np.float32), xyz_cam_h)
        return xyz_world_h[..., :3].astype(np.float32)

    @staticmethod
    def _build_identity_transforms_from_xyz(xyz: np.ndarray):
        if xyz is None or xyz.ndim != 3:
            return None
        T, J, _ = xyz.shape
        tfs = np.broadcast_to(np.eye(4, dtype=np.float32), (T, J, 4, 4)).copy()
        tfs[..., :3, 3] = xyz.astype(np.float32)
        return tfs

    @staticmethod
    def _has_egoexo_extraction_fields(*values) -> bool:
        return any(v is not None for v in values)

    def _resolve_hdf5_path_from_cache_archive(
        self,
        npz_path: str,
        hdf5_path_hint: str | None,
        video_path_hint: str | None,
    ) -> str | None:
        # Cross-cluster runs may carry stale absolute paths inside NPZ metadata.
        # Prefer the explicit dataset/cache roots provided at runtime.
        if self.cache_root is not None and len(self.data_roots) > 0:
            try:
                rel = os.path.relpath(npz_path, start=self.cache_root)
            except ValueError:
                rel = None
            if rel is not None:
                rel_dir = os.path.dirname(rel)
                stem = _normalize_thinker_cache_stem(npz_path)
                for root in self.data_roots:
                    candidate = os.path.join(root, rel_dir, stem + ".hdf5")
                    if os.path.exists(candidate):
                        return os.path.abspath(candidate)
        if hdf5_path_hint and hdf5_path_hint.endswith(".hdf5") and os.path.exists(hdf5_path_hint):
            return os.path.abspath(hdf5_path_hint)
        if video_path_hint:
            hdf5_path = _resolve_hdf5_path_from_video(video_path_hint)
            if hdf5_path is not None:
                return os.path.abspath(hdf5_path)
        return None

    @staticmethod
    def _read_cache_payload_copy(npz_path: str) -> dict[str, Any]:
        with np.load(npz_path, allow_pickle=False) as payload:
            return {key: np.array(payload[key], copy=True) for key in payload.files}

    def _persist_cache_payload_patch(self, npz_path: str, patch: dict[str, Any]) -> None:
        payload = self._read_cache_payload_copy(npz_path)
        payload.update(patch)
        _save_npz_atomic_archive(npz_path, payload, save_mode=self.npz_save_mode)

    def _repair_from_egodex_supervision(
        self,
        npz_path: str,
        *,
        hdf5_path_hint: str | None,
        video_path_hint: str | None,
    ) -> dict[str, Any]:
        hdf5_path = self._resolve_hdf5_path_from_cache_archive(
            npz_path, hdf5_path_hint=hdf5_path_hint, video_path_hint=video_path_hint
        )
        if hdf5_path is None:
            _raise_camera_geometry_error(
                npz_path,
                "egodex_cache_repair",
                "could not resolve source hdf5 for cache repair",
            )
        sup = load_supervision_from_hdf5(hdf5_path, NPZ_TARGET_QUERY_TFS)
        patch = {
            "frame_indices": np.asarray(sup["frame_indices"], dtype=np.int64),
            "xyz_cam": np.asarray(sup["xyz_cam"], dtype=np.float32),
            "R_cam": np.asarray(sup["R_cam"], dtype=np.float32),
            "xyz_world": np.asarray(sup["xyz_world"], dtype=np.float32),
            "R_world": np.asarray(sup["R_world"], dtype=np.float32),
            "tfs_in_cam": np.asarray(sup["tfs_in_cam"], dtype=np.float32),
            "tfs": np.asarray(sup["tfs"], dtype=np.float32),
            "cam_ext": np.asarray(sup["cam_ext"], dtype=np.float32),
            "cam_int": np.asarray(sup["cam_int"], dtype=np.float32),
            "confs": np.asarray(sup["confs"], dtype=np.float32),
            "lang_instruct": np.asarray(str(sup.get("lang_instruct", ""))),
            "path": np.asarray(str(hdf5_path)),
        }
        self._persist_cache_payload_patch(npz_path, patch)
        patch["lang_instruct"] = sup.get("lang_instruct", "")
        return patch

    def _repair_from_egoexo_extraction(
        self,
        npz_path: str,
        *,
        img_t: int,
        total_frames: int | None,
        xyz_cam_np,
        confs_np,
        lang_np,
        ego_body_xyz_np,
        ego_body_joint_names_np,
        ego_body_frame_ids_np,
        ego_hand_xyz_np,
        ego_hand_joint_names_np,
        ego_hand_frame_ids_np,
        ego_ext_static_np,
        ego_int_static_np,
        ego_ext_dynamic_np,
        ego_ext_dynamic_frame_ids_np,
        ego_int_dynamic_np,
        ego_int_dynamic_frame_ids_np,
    ) -> dict[str, Any]:
        ego_cam_ext = self._resolve_egoexo_camera_extrinsics(
            img_t=img_t,
            total_frames=total_frames,
            ego_ext_static=ego_ext_static_np,
            ego_ext_dynamic=ego_ext_dynamic_np,
            ego_ext_dynamic_frame_ids=ego_ext_dynamic_frame_ids_np,
        )
        ego_cam_int = self._resolve_egoexo_camera_intrinsics(
            img_t=img_t,
            total_frames=total_frames,
            ego_int_static=ego_int_static_np,
            ego_int_dynamic=ego_int_dynamic_np,
            ego_int_dynamic_frame_ids=ego_int_dynamic_frame_ids_np,
        )
        ego_cam_ext = self._normalize_camera_extrinsics_per_frame(ego_cam_ext, img_t)
        ego_cam_int = self._normalize_camera_intrinsics_per_frame(ego_cam_int, img_t)
        ego_cam_ext = _validate_camera_extrinsics_or_raise(
            ego_cam_ext, path=npz_path, context="egoexo_extract_repair"
        )
        ego_cam_int = _validate_camera_intrinsics_or_raise(
            ego_cam_int, path=npz_path, context="egoexo_extract_repair"
        )

        xyz_cam_src = xyz_cam_np
        if xyz_cam_src is None:
            xyz_cam_src = self._build_camera_xyz_from_egoexo_pose(
                img_t=img_t,
                total_frames=total_frames,
                body_xyz=ego_body_xyz_np,
                body_joint_names=ego_body_joint_names_np,
                body_frame_ids=ego_body_frame_ids_np,
                hand_xyz=ego_hand_xyz_np,
                hand_joint_names=ego_hand_joint_names_np,
                hand_frame_ids=ego_hand_frame_ids_np,
            )
        if xyz_cam_src is None:
            _raise_camera_geometry_error(
                npz_path,
                "egoexo_extract_repair",
                "missing egoexo pose fields for xyz_cam reconstruction",
            )
        xyz_cam_src = np.asarray(_maybe_decode_bfloat16_cache_array(xyz_cam_src), dtype=np.float32)
        if xyz_cam_src.ndim != 3 or xyz_cam_src.shape[0] != img_t:
            picked = self._select_uniform_indices(xyz_cam_src, target_len=img_t)
            if picked is None:
                _raise_camera_geometry_error(
                    npz_path,
                    "egoexo_extract_repair",
                    f"xyz_cam shape mismatch: {xyz_cam_src.shape}",
                )
            xyz_cam_src = np.asarray(picked, dtype=np.float32)

        xyz_world = self._project_camera_xyz_to_world_np(xyz_cam_src, ego_cam_ext)
        if xyz_world is None:
            _raise_camera_geometry_error(
                npz_path,
                "egoexo_extract_repair",
                "failed to reconstruct xyz_world from xyz_cam and cam_ext",
            )
        tfs_in_cam = self._build_identity_transforms_from_xyz(xyz_cam_src)
        tfs = self._build_identity_transforms_from_xyz(xyz_world)
        if tfs_in_cam is None or tfs is None:
            _raise_camera_geometry_error(
                npz_path,
                "egoexo_extract_repair",
                "failed to rebuild transform tensors from repaired coordinates",
            )

        if confs_np is not None:
            confs = np.asarray(confs_np, dtype=np.float32)
            if confs.ndim != 2 or confs.shape[:2] != xyz_cam_src.shape[:2]:
                confs = np.ones(xyz_cam_src.shape[:2], dtype=np.float32)
            else:
                confs = np.nan_to_num(confs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        else:
            confs = np.ones(xyz_cam_src.shape[:2], dtype=np.float32)

        frame_indices = np.rint(
            self._compute_target_frame_ids_for_cache(img_t, total_frames, ego_ext_dynamic_frame_ids_np)
        ).astype(np.int64)
        patch = {
            "frame_indices": frame_indices,
            "xyz_cam": xyz_cam_src.astype(np.float32),
            "R_cam": np.asarray(tfs_in_cam[..., :3, :3], dtype=np.float32),
            "xyz_world": xyz_world.astype(np.float32),
            "R_world": np.asarray(tfs[..., :3, :3], dtype=np.float32),
            "tfs_in_cam": np.asarray(tfs_in_cam, dtype=np.float32),
            "tfs": np.asarray(tfs, dtype=np.float32),
            "cam_ext": np.asarray(ego_cam_ext, dtype=np.float32),
            "cam_int": np.asarray(ego_cam_int, dtype=np.float32),
            "confs": confs,
        }
        self._persist_cache_payload_patch(npz_path, patch)
        patch["lang_instruct"] = _coerce_language_annotation(lang_np) if lang_np is not None else ""
        return patch

    @staticmethod
    def _pad_or_truncate_sequence_length(
        x: torch.Tensor, target_len: int, dim: int = 1, pad_value: float = 0.0
    ):
        """
        x: [..., S, ...] here primarily used for x.shape == [L, S, D] with pad/trunc on dim=1.
        Returns (y, mask, orig_len):
          y    : tensor padded/truncated to target_len along dim
          mask : bool mask aligned with y, with validity along the target_len axis
          orig_len: original S (int)
        """
        if x is None:
            return None, None, 0
        assert x.dim() >= 2, f"expect >=2D tensor, got {x.shape}"
        S = x.shape[dim]
        tgt = int(target_len)
        if S == tgt:
            mask = torch.ones_like(x, dtype=torch.bool)
            return x.contiguous(), mask, S
        if S > tgt:  # Truncate
            slices = [slice(None)] * x.dim()
            slices[dim] = slice(0, tgt)
            y = x[tuple(slices)].contiguous()
            mask = torch.ones_like(y, dtype=torch.bool)
            return y, mask, S
        # padding
        pad_shape = list(x.shape)
        pad_shape[dim] = tgt
        y = x.new_full(pad_shape, pad_value)
        slices = [slice(None)] * x.dim()
        slices[dim] = slice(0, S)
        y[tuple(slices)] = x
        mask = torch.zeros_like(y, dtype=torch.bool)
        mask[tuple(slices)] = True
        return y.contiguous(), mask, S

    def _load_cache_sample(self, path: str):
        try:
            with np.load(path, allow_pickle=False) as z:
                # Required fields (matching the saved layout)
                imgs_np = z["imgs"]  # [T,H,W,C] uint8

                # ---- Read saved VLM / V-JEPA fields (or None if absent) ----
                vlm_old_np = z.get("vlm_old", None)  # expected [L, S_old, D] or [L, T, S, D]
                vlm_new_np = z.get("vlm_new", None)  # expected [L, S_new, D] or [L, T, S, D]
                token_ids_np = z.get("token_ids", None)  # int32
                layers_np = z.get("layers", None)  # int32
                vjepa_np = z.get("vjepa_feats", None)  # [T,P,D]

                # Supervision / geometry
                xyz_cam_np = z.get("xyz_cam", None)
                R_cam_np = z.get("R_cam", None)
                xyz_world_np = z.get("xyz_world", None)
                R_world_np = z.get("R_world", None)
                tfs_in_cam_np = z.get("tfs_in_cam", None)
                tfs_np = z.get("tfs", None)
                cam_ext_np = z.get("cam_ext", None)
                cam_int_np = z.get("cam_int", None)
                confs_np = z.get("confs", None)
                lang_np = z.get("lang_instruct", "")
                total_frames_np = z.get("total_frames", None)
                hdf5_path_hint_np = z.get("path", None)
                video_path_hint_np = z.get("video_path", None)
                ego_body_xyz_np = z.get("egoexo_body_xyz_3d", None)
                ego_body_joint_names_np = z.get("egoexo_body_joint_names_3d", None)
                ego_body_frame_ids_np = z.get("egoexo_body_frame_indices", None)
                ego_hand_xyz_np = z.get("egoexo_hand_xyz_3d", None)
                ego_hand_joint_names_np = z.get("egoexo_hand_joint_names_3d", None)
                ego_hand_frame_ids_np = z.get("egoexo_hand_frame_indices", None)
                ego_ext_static_np = z.get("egoexo_cam_extrinsics", None)
                ego_int_static_np = z.get("egoexo_cam_intrinsics", None)
                ego_ext_dynamic_np = z.get("egoexo_cam_extrinsics_by_frame", None)
                ego_ext_dynamic_frame_ids_np = z.get(
                    "egoexo_cam_extrinsics_frame_indices", None
                )
                ego_int_dynamic_np = z.get("egoexo_cam_intrinsics_by_frame", None)
                ego_int_dynamic_frame_ids_np = z.get(
                    "egoexo_cam_intrinsics_frame_indices", None
                )
        except Exception as e:
            # Make the error immediately visible inside DataLoader workers
            print(
                f"[NPZ-ERROR] path={path} :: {type(e).__name__}: {e}",
                flush=True,
            )
            # We could return None here and let the upstream collate_fn filter it,
            # but for now we keep the original behavior and re-raise.
            raise

        img_t = int(imgs_np.shape[0]) if hasattr(imgs_np, "shape") and len(imgs_np.shape) >= 1 else 0
        total_frames = self._to_int_scalar(total_frames_np)
        hdf5_path_hint = _scalar_to_string(hdf5_path_hint_np)
        video_path_hint = _scalar_to_string(video_path_hint_np)
        has_egoexo_extract = self._has_egoexo_extraction_fields(
            ego_body_xyz_np,
            ego_hand_xyz_np,
            ego_ext_static_np,
            ego_ext_dynamic_np,
            ego_int_static_np,
            ego_int_dynamic_np,
        )
        if self.camera_mode in {"auto", "egoexo"}:
            ego_cam_ext = self._resolve_egoexo_camera_extrinsics(
                img_t=img_t,
                total_frames=total_frames,
                ego_ext_static=ego_ext_static_np,
                ego_ext_dynamic=ego_ext_dynamic_np,
                ego_ext_dynamic_frame_ids=ego_ext_dynamic_frame_ids_np,
            )
            ego_cam_int = self._resolve_egoexo_camera_intrinsics(
                img_t=img_t,
                total_frames=total_frames,
                ego_int_static=ego_int_static_np,
                ego_int_dynamic=ego_int_dynamic_np,
                ego_int_dynamic_frame_ids=ego_int_dynamic_frame_ids_np,
            )
            if self.camera_mode == "egoexo":
                if ego_cam_ext is None:
                    raise KeyError(
                        f"camera_mode=egoexo but egoexo cam extrinsics unavailable: {path}"
                    )
                if ego_cam_int is None:
                    raise KeyError(
                        f"camera_mode=egoexo but egoexo cam intrinsics unavailable: {path}"
                    )
            if ego_cam_ext is not None:
                cam_ext_np = ego_cam_ext
            if ego_cam_int is not None:
                cam_int_np = ego_cam_int

        cam_ext_np = self._normalize_camera_extrinsics_per_frame(cam_ext_np, img_t)
        cam_int_np = self._normalize_camera_intrinsics_per_frame(cam_int_np, img_t)
        try:
            cam_ext_np = _validate_camera_extrinsics_or_raise(
                cam_ext_np,
                path=path,
                context="egoexo_extract" if has_egoexo_extract else "npz_cache",
            )
            cam_int_np = _validate_camera_intrinsics_or_raise(
                cam_int_np,
                path=path,
                context="egoexo_extract" if has_egoexo_extract else "npz_cache",
            )
        except CameraGeometryLoadError:
            if has_egoexo_extract and self.camera_mode in {"auto", "egoexo"}:
                repaired = self._repair_from_egoexo_extraction(
                    path,
                    img_t=img_t,
                    total_frames=total_frames,
                    xyz_cam_np=xyz_cam_np,
                    confs_np=confs_np,
                    lang_np=lang_np,
                    ego_body_xyz_np=ego_body_xyz_np,
                    ego_body_joint_names_np=ego_body_joint_names_np,
                    ego_body_frame_ids_np=ego_body_frame_ids_np,
                    ego_hand_xyz_np=ego_hand_xyz_np,
                    ego_hand_joint_names_np=ego_hand_joint_names_np,
                    ego_hand_frame_ids_np=ego_hand_frame_ids_np,
                    ego_ext_static_np=ego_ext_static_np,
                    ego_int_static_np=ego_int_static_np,
                    ego_ext_dynamic_np=ego_ext_dynamic_np,
                    ego_ext_dynamic_frame_ids_np=ego_ext_dynamic_frame_ids_np,
                    ego_int_dynamic_np=ego_int_dynamic_np,
                    ego_int_dynamic_frame_ids_np=ego_int_dynamic_frame_ids_np,
                )
            else:
                repaired = self._repair_from_egodex_supervision(
                    path,
                    hdf5_path_hint=hdf5_path_hint,
                    video_path_hint=video_path_hint,
                )
            xyz_cam_np = repaired.get("xyz_cam", xyz_cam_np)
            R_cam_np = repaired.get("R_cam", R_cam_np)
            xyz_world_np = repaired.get("xyz_world", xyz_world_np)
            R_world_np = repaired.get("R_world", R_world_np)
            tfs_in_cam_np = repaired.get("tfs_in_cam", tfs_in_cam_np)
            tfs_np = repaired.get("tfs", tfs_np)
            cam_ext_np = repaired["cam_ext"]
            cam_int_np = repaired["cam_int"]
            confs_np = repaired.get("confs", confs_np)
            repaired_lang = repaired.get("lang_instruct", None)
            if repaired_lang is not None and not lang_np:
                lang_np = repaired_lang

        needs_egodex_supervision_repair = (
            (not has_egoexo_extract)
            and (
                xyz_cam_np is None
                or xyz_world_np is None
                or tfs_in_cam_np is None
                or tfs_np is None
                or R_cam_np is None
                or R_world_np is None
                or (
                    xyz_cam_np is not None
                    and (
                        np.asarray(_maybe_decode_bfloat16_cache_array(xyz_cam_np)).ndim != 3
                        or np.asarray(_maybe_decode_bfloat16_cache_array(xyz_cam_np)).shape[0] != img_t
                    )
                )
                or (
                    xyz_world_np is not None
                    and (
                        np.asarray(_maybe_decode_bfloat16_cache_array(xyz_world_np)).ndim != 3
                        or np.asarray(_maybe_decode_bfloat16_cache_array(xyz_world_np)).shape[0] != img_t
                    )
                )
            )
        )
        if needs_egodex_supervision_repair:
            repaired = self._repair_from_egodex_supervision(
                path,
                hdf5_path_hint=hdf5_path_hint,
                video_path_hint=video_path_hint,
            )
            xyz_cam_np = repaired.get("xyz_cam", xyz_cam_np)
            R_cam_np = repaired.get("R_cam", R_cam_np)
            xyz_world_np = repaired.get("xyz_world", xyz_world_np)
            R_world_np = repaired.get("R_world", R_world_np)
            tfs_in_cam_np = repaired.get("tfs_in_cam", tfs_in_cam_np)
            tfs_np = repaired.get("tfs", tfs_np)
            cam_ext_np = repaired["cam_ext"]
            cam_int_np = repaired["cam_int"]
            confs_np = repaired.get("confs", confs_np)
            repaired_lang = repaired.get("lang_instruct", None)
            if repaired_lang is not None and not lang_np:
                lang_np = repaired_lang

        if xyz_cam_np is None:
            xyz_cam_np = self._build_camera_xyz_from_egoexo_pose(
                img_t=img_t,
                total_frames=total_frames,
                body_xyz=ego_body_xyz_np,
                body_joint_names=ego_body_joint_names_np,
                body_frame_ids=ego_body_frame_ids_np,
                hand_xyz=ego_hand_xyz_np,
                hand_joint_names=ego_hand_joint_names_np,
                hand_frame_ids=ego_hand_frame_ids_np,
            )
        if xyz_cam_np is None:
            xyz_cam_np = np.zeros(
                (img_t, len(NPZ_TARGET_QUERY_TFS), 3), dtype=np.float32
            )

        xyz_cam_np = np.asarray(_maybe_decode_bfloat16_cache_array(xyz_cam_np), dtype=np.float32)
        if xyz_cam_np.ndim != 3 or xyz_cam_np.shape[0] != img_t:
            picked = self._select_uniform_indices(xyz_cam_np, target_len=img_t)
            xyz_cam_np = (
                np.asarray(picked, dtype=np.float32)
                if picked is not None
                else np.zeros((img_t, len(NPZ_TARGET_QUERY_TFS), 3), dtype=np.float32)
            )

        if xyz_world_np is None:
            xyz_world_np = self._project_camera_xyz_to_world_np(xyz_cam_np, cam_ext_np)
        if xyz_world_np is None:
            xyz_world_np = xyz_cam_np.copy()
        xyz_world_np = np.asarray(_maybe_decode_bfloat16_cache_array(xyz_world_np), dtype=np.float32)
        if (
            xyz_world_np.ndim != 3
            or xyz_world_np.shape[0] != img_t
            or xyz_world_np.shape[1] != xyz_cam_np.shape[1]
        ):
            xyz_world_np = xyz_cam_np.copy()

        xyz_finite_mask = np.isfinite(xyz_cam_np).all(axis=-1) & np.isfinite(
            xyz_world_np
        ).all(axis=-1)
        xyz_cam_np = np.nan_to_num(
            xyz_cam_np, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        xyz_world_np = np.nan_to_num(
            xyz_world_np, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)

        if tfs_in_cam_np is None:
            tfs_in_cam_np = self._build_identity_transforms_from_xyz(xyz_cam_np)
        if tfs_np is None:
            tfs_np = self._build_identity_transforms_from_xyz(xyz_world_np)
        if R_cam_np is None and tfs_in_cam_np is not None:
            R_cam_np = np.asarray(tfs_in_cam_np, dtype=np.float32)[..., :3, :3]
        if R_world_np is None and tfs_np is not None:
            R_world_np = np.asarray(tfs_np, dtype=np.float32)[..., :3, :3]

        if confs_np is None:
            confs_np = xyz_finite_mask.astype(np.float32)
        else:
            confs_np = np.asarray(confs_np, dtype=np.float32)
            if (
                confs_np.ndim != 2
                or confs_np.shape[0] != xyz_cam_np.shape[0]
                or confs_np.shape[1] != xyz_cam_np.shape[1]
            ):
                confs_np = xyz_finite_mask.astype(np.float32)
            else:
                confs_np = np.nan_to_num(
                    confs_np, nan=0.0, posinf=0.0, neginf=0.0
                ).astype(np.float32)
                confs_np *= xyz_finite_mask.astype(np.float32)

        # ===== Keep everything on CPU; do not call .to(device) here =====
        # Images remain uint8; other floating-point arrays are converted to float32
        img: Tensor = torch.from_numpy(imgs_np).contiguous()  # uint8 [T,H,W,C]

        xyz_cam = self._cast_tensor_to_float32_cpu(xyz_cam_np)
        R_cam = self._cast_tensor_to_float32_cpu(R_cam_np)
        xyz_world = self._cast_tensor_to_float32_cpu(xyz_world_np)
        R_world = self._cast_tensor_to_float32_cpu(R_world_np)
        tfs_in_cam = self._cast_tensor_to_float32_cpu(tfs_in_cam_np)
        tfs = self._cast_tensor_to_float32_cpu(tfs_np)
        cam_ext = self._cast_tensor_to_float32_cpu(cam_ext_np)
        cam_int = self._cast_tensor_to_float32_cpu(cam_int_np)
        confs = (
            torch.from_numpy(confs_np).float().contiguous()
            if confs_np is not None
            else None
        )
        if confs is None:
            if xyz_cam is not None and xyz_cam.ndim >= 2:
                confs = torch.ones(
                    (xyz_cam.shape[0], xyz_cam.shape[1]), dtype=torch.float32
                )
            else:
                confs = torch.ones((img.shape[0], 1), dtype=torch.float32)

        lang_instr = lang_np if isinstance(lang_np, str) else ""

        # VLM / V-JEPA tensors are also kept on CPU
        vlm_old = (
            torch.from_numpy(vlm_old_np).contiguous()
            if vlm_old_np is not None
            else None
        )
        vlm_new = (
            torch.from_numpy(vlm_new_np).contiguous()
            if vlm_new_np is not None
            else None
        )
        vjepa_feats = (
            torch.from_numpy(vjepa_np).contiguous() if vjepa_np is not None else None
        )

        token_ids = (
            torch.from_numpy(token_ids_np).to(torch.int32).contiguous()
            if token_ids_np is not None
            else None
        )
        layers = (
            torch.from_numpy(layers_np).to(torch.int32).contiguous()
            if layers_np is not None
            else None
        )

        # === Minimal change 1: if shape is [L, T, S, D], collapse time to the last frame -> [L, S, D] ===
        if vlm_old is not None and vlm_old.dim() == 4:
            vlm_old = vlm_old[:, -1, :, :]
        if vlm_new is not None and vlm_new.dim() == 4:
            vlm_new = vlm_new[:, -1, :, :]

        # === Normalize vlm_old / vlm_new to fixed sequence lengths and return masks/original lengths ===
        vlm_old_pad = vlm_old_mask = vlm_new_pad = vlm_new_mask = None
        vlm_old_len = 0
        vlm_new_len = 0

        if vlm_old is not None:
            vlm_old = vlm_old.float()
            vlm_old_pad, vlm_old_mask, vlm_old_len = self._pad_or_truncate_sequence_length(
                vlm_old, self.pad_old_to, dim=1, pad_value=self.pad_value
            )
        if vlm_new is not None:
            vlm_new = vlm_new.float()
            vlm_new_pad, vlm_new_mask, vlm_new_len = self._pad_or_truncate_sequence_length(
                vlm_new, self.pad_new_to, dim=1, pad_value=self.pad_value
            )

        # === Minimal change 2: reduce mask from [L, S, D] to [L, S] for token-level validity ===
        if vlm_old_mask is not None and vlm_old_mask.dim() == 3:
            vlm_old_mask = vlm_old_mask.any(dim=-1)  # [L,S]
        if vlm_new_mask is not None and vlm_new_mask.dim() == 3:
            vlm_new_mask = vlm_new_mask.any(dim=-1)  # [L,S]

        extras = {
            "vlm_old": vlm_old_pad if vlm_old_pad is not None else vlm_old,  # [L,S,D]
            "vlm_new": vlm_new_pad if vlm_new_pad is not None else vlm_new,  # [L,S,D]
            "vlm_old_mask": vlm_old_mask,  # [L,S] bool
            "vlm_new_mask": vlm_new_mask,  # [L,S] bool
            "vlm_old_len": vlm_old_len,  # Original length
            "vlm_new_len": vlm_new_len,  # Original length
            "token_ids": token_ids,  # int32
            "layers": layers,  # int32
            "vjepa_feats": vjepa_feats,  # [T,P,D]
        }

        tup = (
            xyz_cam,
            R_cam,
            xyz_world,
            R_world,
            tfs_in_cam,
            tfs,
            cam_ext,
            cam_int,
            img,  # [T,H,W,C] uint8 (CPU)
            lang_instr,  # str
            confs,  # May be None
            extras,  # Extended fields (including fixed-length features and masks)
        )
        if self.if_return_path:
            tup = tup + (path,)
        return tup

    def __getitem__(self, idx):
        if self._memory_samples is not None:
            return self._memory_samples[idx]
        path = self.files[idx]
        return self._load_cache_sample(path)
