# ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model
# Copyright (c) 2024-2026 Northeastern University.
# Developed in NEU SMILE LAB by Haichao Zhang (https://zhanghaichao.xyz)
# and Yun Raymond Fu (https://www1.ece.neu.edu/~yunfu/).
# SPDX-style identifier: LicenseRef-ThinkJEPA-Attribution
# Original source: https://github.com/Hai-chao-Zhang/ThinkJEPA
# See the root LICENSE, NOTICE, CITATION.cff, and CITATION.bib for attribution and citation requirements.

import argparse
import glob
import json
import os, random
import re
import sys
import time
from functools import partial  # NEW
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
VJEPA2_ROOT = Path(
    os.environ.get("VJEPA2_ROOT", str(REPO_ROOT / "external" / "vjepa2"))
).resolve()
for _path in (
    REPO_ROOT,
    REPO_ROOT / "cache_train",
    REPO_ROOT / "cache_train" / "egodex",
    VJEPA2_ROOT,
    VJEPA2_ROOT.parent,
    REPO_ROOT / "vjepa2",
):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import vjepa2.src.datasets.utils.video.transforms as video_transforms
import vjepa2.src.datasets.utils.video.volume_transforms as volume_transforms

from cache_train.models import (
    TrajectoryReadoutMLP,
)
from cache_train.hf_egodex import (
    DEFAULT_EGODEX_PART2_HF_DIR,
    configure_huggingface_cache_dirs,
    is_huggingface_cache_path,
    resolve_egodex_data_reference,
)
from cache_train.checkpoint_paths import resolve_dense_jepa_checkpoint
from cache_train.thinker_predictor import CortexGuidedVideoPredictor

from egodex.trajectory_dataset import CameraGeometryLoadError, build_egodex_dataloaders
from egodex.utils.draw_utils import write_video_frames_to_mp4
from egodex.visualize_2d import render_hand_projection

from cache_train.video_observation_adapter import VideoObservationAdapter

from hand_skeleton_consts import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    left_dict,
    left_idx,
    LEFT_JOINT_MASK_T,
    QUERY_TFS,
    right_dict,
    right_idx,
    RIGHT_JOINT_MASK_T,
    tf2idx,
    visualize_flag,
)

from cache_train.predictor import PatchwiseAutoregressiveRolloutHead
from vjepa2.src.models.predictor import VisionTransformerPredictor  # NEW
from vjepa2.src.models.vision_transformer import vit_large_rope


def configure_reproducibility_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_dense_jepa_cudnn():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


def build_amp_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def build_amp_autocast(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def initialize_distributed_runtime(ddp: bool):
    if not ddp:
        return 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size


def shutdown_distributed_runtime(ddp: bool):
    if ddp and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_primary_process(rank: int):
    return rank == 0


def append_hand_reference_offsets(pred_traj, right_ref, left_ref):
    B, T, J, _ = pred_traj.shape
    device = pred_traj.device
    dtype = pred_traj.dtype
    right_mask = RIGHT_JOINT_MASK_T.to(device=device, dtype=dtype)
    left_mask = LEFT_JOINT_MASK_T.to(device=device, dtype=dtype)
    right_ref = right_ref.to(device=device, dtype=dtype).unsqueeze(2)
    left_ref = left_ref.to(device=device, dtype=dtype).unsqueeze(2)
    pred_traj = pred_traj + right_ref * right_mask + left_ref * left_mask
    return pred_traj


@torch.no_grad()
def select_joint_subset_if_needed(x, joint_keep_idx):
    if joint_keep_idx is None:
        return x
    return x[..., joint_keep_idx, :]


def compute_trajectory_loss_and_accuracy(pred, target, loss_fn, thr=0.05, joint_keep_idx=None):
    if pred.shape != target.shape:
        assert (
            pred.shape[-1] == target.shape[-1] == 3
        ), f"{pred.shape} vs {target.shape}"
        pred, target = torch.broadcast_tensors(pred, target)
    pred = select_joint_subset_if_needed(pred, joint_keep_idx)
    target = select_joint_subset_if_needed(target, joint_keep_idx)
    loss = loss_fn(pred, target)
    err = torch.linalg.norm(pred - target, dim=-1)
    avg_dist = err.mean(dim=(1, 2))
    final_dist = err[:, -1, :].mean(dim=1)
    acc = (err < thr).float().mean().item()
    return loss, avg_dist, final_dist, acc


def initialize_latent_metric_totals():
    return {
        "pred_loss": 0.0,
        "pred_latent_dist": 0.0,
        "pred_latent_smooth_l1": 0.0,
        "pred_latent_cosine_distance": 0.0,
    }


def compute_predicted_latent_metrics(pred_latent, target_latent):
    pred_latent = pred_latent.float()
    target_latent = target_latent.float()
    cosine_sim = F.cosine_similarity(
        pred_latent, target_latent, dim=-1, eps=1e-8
    ).mean()
    return {
        "pred_loss": F.mse_loss(pred_latent, target_latent),
        "pred_latent_dist": torch.linalg.norm(
            pred_latent - target_latent, dim=-1
        ).mean(),
        "pred_latent_smooth_l1": F.smooth_l1_loss(pred_latent, target_latent),
        "pred_latent_cosine_distance": 1.0 - cosine_sim,
    }


def write_json_atomic(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        tmp_path.replace(path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def write_text_file(text: str, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(text)


def maybe_import_matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def maybe_import_pil_modules():
    try:
        from PIL import Image, ImageDraw

        return Image, ImageDraw
    except Exception:
        return None, None


def draw_metric_line_panel(draw, box, xs, series, title):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline="#cfcfcf", width=1)
    draw.text((x0 + 6, y0 + 6), title, fill="#111111")

    plot_x0 = x0 + 52
    plot_y0 = y0 + 28
    plot_x1 = x1 - 12
    plot_y1 = y1 - 28
    draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill="#444444", width=1)
    draw.line((plot_x0, plot_y0, plot_x0, plot_y1), fill="#444444", width=1)

    values = []
    for _, ys, _ in series:
        values.extend([float(v) for v in ys if np.isfinite(v)])
    if not values:
        draw.text((plot_x0 + 12, plot_y0 + 12), "No finite values", fill="#666666")
        return

    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0

    for frac in (0.25, 0.5, 0.75):
        yy = plot_y1 - frac * (plot_y1 - plot_y0)
        draw.line((plot_x0, yy, plot_x1, yy), fill="#ebebeb", width=1)

    draw.text((x0 + 4, plot_y0 - 6), f"{vmax:.3f}", fill="#666666")
    draw.text((x0 + 4, plot_y1 - 6), f"{vmin:.3f}", fill="#666666")
    draw.text((plot_x0, plot_y1 + 4), str(int(xs[0])), fill="#666666")
    draw.text((plot_x1 - 22, plot_y1 + 4), str(int(xs[-1])), fill="#666666")

    def project(ix, val):
        if len(xs) == 1:
            xp = 0.5
        else:
            xp = (ix - xs[0]) / max(xs[-1] - xs[0], 1)
        yp = (float(val) - vmin) / (vmax - vmin)
        px = plot_x0 + xp * (plot_x1 - plot_x0)
        py = plot_y1 - yp * (plot_y1 - plot_y0)
        return px, py

    legend_x = plot_x0
    for label, ys, color in series:
        draw.text((legend_x, y0 + 6), label, fill=color)
        legend_x += 90
        prev = None
        for x, y in zip(xs, ys):
            if not np.isfinite(y):
                prev = None
                continue
            pt = project(x, y)
            if prev is not None:
                draw.line((prev[0], prev[1], pt[0], pt[1]), fill=color, width=2)
            prev = pt


def plot_latent_metric_curves(logs: dict, out_path: Path):
    epochs = logs.get("epochs", []) if isinstance(logs, dict) else []
    if not epochs:
        return None
    plt = maybe_import_matplotlib_pyplot()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xs = [int(e.get("epoch", idx + 1)) for idx, e in enumerate(epochs)]
    panels = [
        ("Latent MSE", "train_pred_loss", "val_pred_loss"),
        ("Latent L2", "train_pred_latent_dist", "val_pred_latent_dist"),
        (
            "Latent SmoothL1",
            "train_pred_latent_smooth_l1",
            "val_pred_latent_smooth_l1",
        ),
        (
            "Latent Cosine Distance",
            "train_pred_latent_cosine_distance",
            "val_pred_latent_cosine_distance",
        ),
    ]

    if plt is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes = axes.flatten()
        for ax, (title, train_key, val_key) in zip(axes, panels):
            train_vals = [
                float(e[train_key]) if e.get(train_key) is not None else np.nan for e in epochs
            ]
            val_vals = [
                float(e[val_key]) if e.get(val_key) is not None else np.nan for e in epochs
            ]
            ax.plot(xs, train_vals, label="train", linewidth=1.5)
            ax.plot(xs, val_vals, label="val", linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return out_path

    Image, ImageDraw = maybe_import_pil_modules()
    if Image is None or ImageDraw is None:
        return None

    img = Image.new("RGB", (1200, 820), "white")
    draw = ImageDraw.Draw(img)
    panel_boxes = [
        (20, 20, 590, 400),
        (610, 20, 1180, 400),
        (20, 420, 590, 800),
        (610, 420, 1180, 800),
    ]
    colors = [("#1f77b4", "#d62728")] * 4
    for box, (title, train_key, val_key), (train_color, val_color) in zip(
        panel_boxes, panels, colors
    ):
        train_vals = [
            float(e[train_key]) if e.get(train_key) is not None else np.nan for e in epochs
        ]
        val_vals = [
            float(e[val_key]) if e.get(val_key) is not None else np.nan for e in epochs
        ]
        draw_metric_line_panel(
            draw,
            box,
            xs,
            [
                ("train", train_vals, train_color),
                ("val", val_vals, val_color),
            ],
            title,
        )
    img.save(out_path)
    return out_path


def load_json_or_default(path, default):
    p = Path(path)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default


def unwrap_ddp_module(module):
    if isinstance(module, torch.nn.parallel.DistributedDataParallel):
        return module.module
    return module


def save_training_checkpoint(
    path,
    epoch,
    cls_model,
    optimizer,
    best_blob,
    *,
    predictor=None,
    optimizer_pred=None,
    scheduler=None,
    scaler=None,
    args=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "cls_model": unwrap_ddp_module(cls_model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "best": best_blob,
    }
    if predictor is not None:
        payload["predictor"] = unwrap_ddp_module(predictor).state_dict()
    if optimizer_pred is not None:
        payload["optimizer_pred"] = optimizer_pred.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if args is not None:
        payload["args"] = vars(args)
        payload["target_epochs"] = int(getattr(args, "epochs", epoch))
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    last_ex = None
    for attempt in range(3):
        try:
            torch.save(payload, tmp_path)
            tmp_path.replace(path)
            return
        except Exception as ex:
            last_ex = ex
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    if last_ex is not None:
        raise last_ex


def torch_load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_checkpoint_epoch_from_path(path):
    m = re.search(r"ckpt_epoch(\d+)\.pt$", str(path))
    return int(m.group(1)) if m else -1


def cleanup_legacy_epoch_checkpoints(out_dir, *, keep_paths=()):
    out_dir = Path(out_dir)
    keep = {Path(p).resolve() for p in keep_paths}
    for ckpt_path in out_dir.glob("ckpt_epoch*.pt"):
        try:
            if ckpt_path.resolve() in keep:
                continue
            ckpt_path.unlink()
        except FileNotFoundError:
            continue
        except Exception as ex:
            rank = int(os.environ.get("RANK", "0"))
            if rank == 0:
                print(
                    f"[WARN] failed to remove legacy checkpoint {ckpt_path}: {ex}",
                    flush=True,
                )


def find_latest_valid_checkpoint(out_dir):
    out_dir = Path(out_dir)
    latest_path = out_dir / "ckpt_latest.pt"
    candidates = []
    if latest_path.exists():
        candidates.append(latest_path)
    candidates.extend(out_dir.glob("ckpt_epoch*.pt"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: (parse_checkpoint_epoch_from_path(p), p.name), reverse=True)
    rank = int(os.environ.get("RANK", "0"))
    for ckpt_path in candidates:
        try:
            blob = torch_load_checkpoint(ckpt_path, map_location="cpu")
            if isinstance(blob, dict) and ("cls_model" in blob):
                return ckpt_path
            if rank == 0:
                print(
                    f"[WARN] skip invalid auto-resume checkpoint (missing cls_model): {ckpt_path}",
                    flush=True,
                )
        except Exception as ex:
            if rank == 0:
                print(
                    f"[WARN] skip unreadable auto-resume checkpoint: {ckpt_path} ({ex})",
                    flush=True,
                )
    return None


def resolve_resume_checkpoint(*, out_dir, resume_ckpt="", auto_resume=False):
    resume_ckpt = str(resume_ckpt or "").strip()
    if resume_ckpt:
        path = Path(resume_ckpt)
        if not path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {path}")
        return path
    if not auto_resume:
        return None
    return find_latest_valid_checkpoint(out_dir)


def load_pretrained_dense_jepa_weights(model, pretrained_weights):
    pretrained_dict = torch.load(
        pretrained_weights, weights_only=True, map_location="cpu"
    )["encoder"]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {
        k.replace("backbone.", ""): v for k, v in pretrained_dict.items()
    }
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print(
        f"Pretrained weights found at {pretrained_weights} and loaded with msg: {msg}"
    )


def build_dense_jepa_video_transform(img_size):
    short_side_size = int(256.0 / 224 * img_size)
    eval_transform = video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
            ),
        ]
    )
    return eval_transform


def load_dense_jepa_encoder(pt_model_path=None):
    if pt_model_path is None:
        pt_model_path = resolve_dense_jepa_checkpoint()
    img_size = 256
    model_pt = vit_large_rope(img_size=(img_size, img_size), num_frames=64)
    model_pt.cuda().eval()
    load_pretrained_dense_jepa_weights(model_pt, pt_model_path)
    pt_video_transform = build_dense_jepa_video_transform(img_size=img_size)
    return model_pt, pt_video_transform


def encode_dense_jepa_video(video, model_pt):
    with torch.no_grad():
        B, T, C, H, W = video.shape
        video = video.permute(0, 2, 1, 3, 4)
        out = model_pt(video)
        out = out.contiguous().view(B, T, -1, out.shape[-1])  # [B, T, P, D]
    return out


def predict_trajectory_from_latents(cls_model, out_patch_features_pt, ref_pts=None, n_tokens=128):
    x = out_patch_features_pt
    try:
        if ref_pts is not None:
            pred = cls_model(x, ref_pts)
        else:
            pred = cls_model(x)
    except TypeError:
        # Support heads that accept only [B, D] or [B, T, D]
        if x.ndim == 4:
            x = x.mean(dim=2)
        if x.ndim == 3:
            x = x.mean(dim=1)
        pred = cls_model(x)
    return pred


def distributed_mean_scalar(x: float, *, ddp: bool, world_size: int):
    # Avoid forcing cuda() in CPU-only environments
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.tensor([x], dtype=torch.float32, device=dev)
    if ddp and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        t /= max(world_size, 1)
    return float(t.item())


def distributed_average_from_sum_count(sum_x: float, count_x: int, *, ddp: bool):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.tensor([sum_x, float(count_x)], dtype=torch.float64, device=dev)
    if ddp and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    total_sum = float(t[0].item())
    total_count = int(round(float(t[1].item())))
    if total_count <= 0:
        return None
    return total_sum / total_count


def project_camera_points_to_world(xyz_cam, cam_ext):
    B, T, J, _ = xyz_cam.shape
    ones = torch.ones_like(xyz_cam[..., :1])
    xyz_cam_h = torch.cat([xyz_cam, ones], dim=-1)
    xyz_world_h = torch.matmul(cam_ext.unsqueeze(2), xyz_cam_h.unsqueeze(-1)).squeeze(
        -1
    )
    return xyz_world_h[..., :3]


# ================= cache helpers =================
def _is_fatal_camera_geometry_error(exc: Exception) -> bool:
    return isinstance(exc, CameraGeometryLoadError) or "[CAM-INVALID]" in str(exc)


def _identity_transform(x):
    return x


_QWEN_CACHE_NAME_RE = re.compile(
    r"^(?P<stem>.+?)_L\d+_nf\d+_res\d+_new\d+_s\d+of\d+$"
)


def _normalize_thinker_cache_stem(npz_path: str) -> str:
    base = os.path.splitext(os.path.basename(npz_path))[0]
    m = _QWEN_CACHE_NAME_RE.match(base)
    return m.group("stem") if m else base


def build_thinker_cache_index(cache_root: str):
    index = {}
    files = sorted(glob.glob(os.path.join(cache_root, "**", "*.npz"), recursive=True))
    for p in files:
        rel = os.path.relpath(p, cache_root)
        rel_dir = os.path.normpath(os.path.dirname(rel))
        stem = _normalize_thinker_cache_stem(p)
        index.setdefault((rel_dir, stem), []).append(p)
    return index


def resolve_cache_preload_policy(args) -> bool:
    explicit = getattr(args, "preload_cache_to_memory", None)
    if explicit is not None:
        return bool(explicit)
    value = getattr(args, "cache_dir", None)
    return isinstance(value, str) and value and is_huggingface_cache_path(value)


def preload_npz_archives(cache_root: str):
    files = sorted(glob.glob(os.path.join(cache_root, "**", "*.npz"), recursive=True))
    if not files:
        return {}
    print(
        f"[INFO] Preloading {len(files)} cache npz files into memory from {cache_root}",
        flush=True,
    )
    archives = {}
    for p in files:
        with np.load(p, allow_pickle=False) as z:
            archives[p] = {k: np.array(z[k], copy=True) for k in z.files}
    return archives


def infer_thinker_guidance_dims_from_cache(
    cache_root: str, max_scan: int = 256, preloaded_archives: dict | None = None
):
    """
    Sample .npz files from cache_root and infer the last-dimension size D of
    vlm_old/vlm_new automatically.
    Returns (old_dim, new_dim, sample_path); returns (None, None, None) on failure.
    """
    if not cache_root or (not os.path.isdir(cache_root)):
        return None, None, None
    files = (
        sorted(preloaded_archives.keys())
        if preloaded_archives is not None
        else sorted(glob.glob(os.path.join(cache_root, "**", "*.npz"), recursive=True))
    )
    if not files:
        return None, None, None

    for p in files[: max(1, int(max_scan))]:
        try:
            if preloaded_archives is not None:
                payload = preloaded_archives.get(p, {})
                old = payload.get("vlm_old", None)
                new = payload.get("vlm_new", None)
            else:
                with np.load(p, allow_pickle=False) as z:
                    old = z["vlm_old"] if "vlm_old" in z else None
                    new = z["vlm_new"] if "vlm_new" in z else None
            old_dim = int(old.shape[-1]) if old is not None and old.ndim >= 1 else None
            new_dim = int(new.shape[-1]) if new is not None and new.ndim >= 1 else None
            if old_dim is not None or new_dim is not None:
                return old_dim, new_dim, p
        except Exception:
            continue
    return None, None, None


def _candidate_relative_dirs_for_hdf5(h5_path: str, data_root: str):
    rel = os.path.splitext(os.path.relpath(h5_path, data_root))[0]
    rel_dir = os.path.normpath(os.path.dirname(rel))
    cands = [rel_dir]

    # If rel_dir starts with partX/, also try the version without partX
    parts = rel_dir.split(os.sep)
    if len(parts) >= 2 and re.fullmatch(r"part\d+", parts[0]):
        cands.append(os.path.normpath(os.path.join(*parts[1:])))

    # If data_root is already partX, also add a candidate that explicitly keeps partX
    m = re.search(r"(?:^|/)(part\d+)/(.*?)/[^/]+\.hdf5$", h5_path)
    if m:
        rel_with_part = os.path.normpath(os.path.join(m.group(1), m.group(2)))
        rel_wo_part = os.path.normpath(m.group(2))
        cands.extend([rel_with_part, rel_wo_part])

    out = []
    seen = set()
    for x in cands:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def resolve_thinker_cache_for_hdf5(
    h5_path: str,
    data_root: str,
    cache_root: str,
    cache_index: dict,
    path_cache: dict,
):
    if h5_path in path_cache:
        return path_cache[h5_path]

    stem = os.path.splitext(os.path.basename(h5_path))[0]
    for rel_dir in _candidate_relative_dirs_for_hdf5(h5_path, data_root):
        key = (os.path.normpath(rel_dir), stem)
        hits = cache_index.get(key, [])
        if hits:
            path_cache[h5_path] = hits[0]
            return hits[0]

    # Final fallback: recursive search by stem, preferring the same task directory
    parent_dir = os.path.basename(os.path.dirname(h5_path))
    pattern = os.path.join(cache_root, "**", f"{stem}_L*_nf*_res*_new*_s*of*.npz")
    hits = sorted(glob.glob(pattern, recursive=True))
    if hits:
        hits = sorted(
            hits,
            key=lambda p: (
                os.path.basename(os.path.dirname(p)) != parent_dir,
                p,
            ),
        )
        path_cache[h5_path] = hits[0]
        return hits[0]

    path_cache[h5_path] = None
    return None


def pad_or_truncate_guidance_tensor(x: torch.Tensor, target_len: int, pad_value: float = 0.0):
    """
    x: [L, S, D] -> y: [L, target_len, D], mask: [L, target_len] bool, orig_len: int
    """
    if x is None:
        return None, None, 0
    if x.dim() != 3:
        raise ValueError(f"Expected [L,S,D], got {tuple(x.shape)}")
    L, S, D = x.shape
    tgt = int(target_len)
    if S >= tgt:
        y = x[:, :tgt, :].contiguous()
        mask = torch.ones((L, tgt), dtype=torch.bool)
        return y, mask, S
    y = x.new_full((L, tgt, D), pad_value)
    y[:, :S, :] = x
    mask = torch.zeros((L, tgt), dtype=torch.bool)
    mask[:, :S] = True
    return y.contiguous(), mask, S


def pad_or_truncate_token_ids(
    ids: torch.Tensor, target_len: int, pad_id: int = -1
) -> tuple[torch.Tensor, int]:
    """
    ids: [S] int -> [target_len] int, orig_len
    """
    if ids is None:
        return None, 0
    if ids.dim() != 1:
        ids = ids.reshape(-1)
    ids = ids.to(torch.int32).contiguous()
    S = int(ids.numel())
    tgt = int(target_len)
    if S >= tgt:
        return ids[:tgt].contiguous(), S
    out = ids.new_full((tgt,), int(pad_id))
    out[:S] = ids
    return out.contiguous(), S


def load_thinker_guidance_from_npz_batch(
    npz_list,
    device,
    pad_old_to: int,
    pad_new_to: int,
    preloaded_archives: dict | None = None,
):
    samples = []
    for p in npz_list:
        item = {
            "vlm_old": None,
            "vlm_new": None,
            "vlm_old_mask": None,
            "vlm_new_mask": None,
            "vlm_old_len": 0,
            "vlm_new_len": 0,
            "token_ids": None,
            "token_ids_len": 0,
            "layers": None,
        }
        if p is None or (not os.path.exists(p)):
            samples.append(item)
            continue
        try:
            if preloaded_archives is not None:
                payload = preloaded_archives.get(p, {})
                old = payload.get("vlm_old", None)
                new = payload.get("vlm_new", None)
                token_ids = payload.get("token_ids", None)
                layers = payload.get("layers", None)
            else:
                with np.load(p, allow_pickle=False) as z:
                    old = z["vlm_old"] if "vlm_old" in z else None
                    new = z["vlm_new"] if "vlm_new" in z else None
                    token_ids = z["token_ids"] if "token_ids" in z else None
                    layers = z["layers"] if "layers" in z else None

            if old is not None:
                t_old = torch.from_numpy(old)
                if t_old.dim() == 4:  # [L,T,S,D] -> [L,S,D]
                    t_old = t_old[:, -1, :, :]
                t_old = t_old.float().contiguous()
                old_pad, old_mask, old_len = pad_or_truncate_guidance_tensor(
                    t_old, target_len=pad_old_to, pad_value=0.0
                )
                item["vlm_old"] = old_pad
                item["vlm_old_mask"] = old_mask
                item["vlm_old_len"] = int(old_len)

            if new is not None:
                t_new = torch.from_numpy(new)
                if t_new.dim() == 4:  # [L,T,S,D] -> [L,S,D]
                    t_new = t_new[:, -1, :, :]
                t_new = t_new.float().contiguous()
                new_pad, new_mask, new_len = pad_or_truncate_guidance_tensor(
                    t_new, target_len=pad_new_to, pad_value=0.0
                )
                item["vlm_new"] = new_pad
                item["vlm_new_mask"] = new_mask
                item["vlm_new_len"] = int(new_len)

            if token_ids is not None:
                t_ids = torch.from_numpy(token_ids)
                if t_ids.dim() > 1:
                    t_ids = t_ids.reshape(-1)
                t_ids_pad, t_ids_len = pad_or_truncate_token_ids(
                    t_ids, target_len=pad_new_to, pad_id=-1
                )
                item["token_ids"] = t_ids_pad
                item["token_ids_len"] = int(t_ids_len)

            if layers is not None:
                item["layers"] = torch.from_numpy(layers).to(torch.int32).contiguous()

        except Exception:
            pass
        samples.append(item)

    def _first_shape(key):
        for s in samples:
            v = s.get(key, None)
            if isinstance(v, torch.Tensor):
                return tuple(v.shape)
        return None

    old_shape = _first_shape("vlm_old")
    new_shape = _first_shape("vlm_new")
    token_shape = _first_shape("token_ids")
    lay_shape = _first_shape("layers")

    if old_shape is None and new_shape is None:
        return None

    def _stack_with_fill(key, shape, dtype):
        if shape is None:
            return None
        vals = []
        for s in samples:
            v = s.get(key, None)
            if v is None:
                if dtype == torch.bool:
                    v = torch.zeros(shape, dtype=torch.bool)
                elif dtype in (torch.int32, torch.int64):
                    v = torch.zeros(shape, dtype=dtype)
                else:
                    v = torch.zeros(shape, dtype=dtype)
            vals.append(v)
        return torch.stack(vals, dim=0).to(device=device, non_blocking=True)

    extras = {
        "vlm_old": _stack_with_fill("vlm_old", old_shape, torch.float32),
        "vlm_new": _stack_with_fill("vlm_new", new_shape, torch.float32),
        "vlm_old_mask": _stack_with_fill(
            "vlm_old_mask",
            old_shape[:2] if old_shape is not None else None,
            torch.bool,
        ),
        "vlm_new_mask": _stack_with_fill(
            "vlm_new_mask",
            new_shape[:2] if new_shape is not None else None,
            torch.bool,
        ),
        "vlm_old_len": torch.tensor(
            [int(s.get("vlm_old_len", 0)) for s in samples],
            dtype=torch.int32,
            device=device,
        ),
        "vlm_new_len": torch.tensor(
            [int(s.get("vlm_new_len", 0)) for s in samples],
            dtype=torch.int32,
            device=device,
        ),
        "token_ids": _stack_with_fill("token_ids", token_shape, torch.int32),
        "token_ids_len": torch.tensor(
            [int(s.get("token_ids_len", 0)) for s in samples],
            dtype=torch.int32,
            device=device,
        ),
        "layers": _stack_with_fill("layers", lay_shape, torch.int32),
    }
    return extras


def ensure_thinker_guidance_payload(
    *,
    extras,
    paths,
    args,
    device,
    cache_index,
    path_cache,
    preloaded_archives=None,
):
    if not bool(getattr(args, "thinkjepa_use_vlm_merge", True)):
        return extras

    # Return immediately if usable extras are already available
    if isinstance(extras, dict) and (
        extras.get("vlm_old", None) is not None or extras.get("vlm_new", None) is not None
    ):
        return apply_guidance_ablation_policy(extras, args)

    if not bool(getattr(args, "thinkjepa_use_cache_ext", True)):
        return extras
    cache_root = getattr(args, "cache_dir", None)
    if cache_root is None:
        return extras
    if paths is None:
        return extras

    npz_list = resolve_cache_archives_for_paths(
        paths,
        data_root=getattr(args, "data_dir"),
        cache_root=cache_root,
        cache_index=cache_index,
        path_cache=path_cache,
    )

    loaded = load_thinker_guidance_from_npz_batch(
        npz_list=npz_list,
        device=device,
        pad_old_to=int(getattr(args, "vlm_pad_old_to", 480)),
        pad_new_to=int(getattr(args, "vlm_pad_new_to", 15)),
        preloaded_archives=preloaded_archives,
    )
    if loaded is None:
        return extras
    return apply_guidance_ablation_policy(loaded, args)


def _ensure_thinkjepa_extras(
    *,
    extras,
    paths,
    args,
    device,
    cache_index,
    path_cache,
    preloaded_archives=None,
):
    return ensure_thinker_guidance_payload(
        extras=extras,
        paths=paths,
        args=args,
        device=device,
        cache_index=cache_index,
        path_cache=path_cache,
        preloaded_archives=preloaded_archives,
    )


def select_pyramid_guidance_tensor(x, args):
    if x is None:
        return None
    mode = str(getattr(args, "thinkjepa_vlm_layer_selector", "last")).lower()
    idx = int(getattr(args, "thinkjepa_vlm_layer_index", -1))
    if x.dim() == 4:
        # [B,L,S,D]
        L = x.size(1)
        if mode == "all":
            return x
        if mode == "mid":
            mid = L // 2
            return x[:, mid : mid + 1, ...]
        if mode == "index":
            j = idx if idx >= 0 else (L + idx)
            j = max(0, min(L - 1, j))
            return x[:, j : j + 1, ...]
        return x[:, -1:, ...]
    if x.dim() == 3:
        # [L,S,D]
        L = x.size(0)
        if mode == "all":
            return x
        if mode == "mid":
            mid = L // 2
            return x[mid : mid + 1, ...]
        if mode == "index":
            j = idx if idx >= 0 else (L + idx)
            j = max(0, min(L - 1, j))
            return x[j : j + 1, ...]
        return x[-1:, ...]
    return x


def select_pyramid_guidance_mask(m, args):
    if m is None:
        return None
    mode = str(getattr(args, "thinkjepa_vlm_layer_selector", "last")).lower()
    idx = int(getattr(args, "thinkjepa_vlm_layer_index", -1))
    if m.dim() == 4:
        m = m.any(dim=-1)
    if m.dim() == 3:
        # [B,L,S]
        L = m.size(1)
        if mode == "all":
            return m
        if mode == "mid":
            mid = L // 2
            return m[:, mid : mid + 1, ...]
        if mode == "index":
            j = idx if idx >= 0 else (L + idx)
            j = max(0, min(L - 1, j))
            return m[:, j : j + 1, ...]
        return m[:, -1:, ...]
    if m.dim() == 2:
        # [L,S]
        L = m.size(0)
        if mode == "all":
            return m
        if mode == "mid":
            mid = L // 2
            return m[mid : mid + 1, ...]
        if mode == "index":
            j = idx if idx >= 0 else (L + idx)
            j = max(0, min(L - 1, j))
            return m[j : j + 1, ...]
        return m[-1:, ...]
    return m


def parse_token_id_set(spec) -> set[int]:
    if spec is None:
        return set()
    if isinstance(spec, (list, tuple, set)):
        out = set()
        for x in spec:
            try:
                out.add(int(x))
            except Exception:
                continue
        return out
    text = str(spec).strip()
    if text == "":
        return set()
    out = set()
    for part in re.split(r"[,\s]+", text):
        if not part:
            continue
        try:
            out.add(int(part))
        except Exception:
            continue
    return out


def coerce_token_ids_matrix(token_ids):
    if token_ids is None:
        return None
    if isinstance(token_ids, np.ndarray):
        token_ids = torch.from_numpy(token_ids)
    if not isinstance(token_ids, torch.Tensor):
        return None
    if token_ids.dim() == 1:
        return token_ids.unsqueeze(0).to(torch.int64)
    if token_ids.dim() == 2:
        return token_ids.to(torch.int64)
    if token_ids.dim() == 3 and token_ids.size(1) == 1:
        return token_ids[:, 0, :].to(torch.int64)
    if token_ids.dim() >= 2:
        return token_ids.reshape(token_ids.shape[0], -1).to(torch.int64)
    return None


def align_token_drop_mask(drop_2d: torch.Tensor, batch: int, seq_len: int):
    if drop_2d is None:
        return None
    if drop_2d.size(-1) < seq_len:
        pad = torch.zeros(
            (drop_2d.size(0), seq_len - drop_2d.size(-1)),
            dtype=torch.bool,
            device=drop_2d.device,
        )
        drop_2d = torch.cat([drop_2d, pad], dim=-1)
    elif drop_2d.size(-1) > seq_len:
        drop_2d = drop_2d[:, :seq_len]

    if drop_2d.size(0) == batch:
        return drop_2d
    if drop_2d.size(0) == 1 and batch > 1:
        return drop_2d.expand(batch, -1)

    aligned = torch.zeros((batch, seq_len), dtype=torch.bool, device=drop_2d.device)
    n = min(batch, int(drop_2d.size(0)))
    aligned[:n, :] = drop_2d[:n, :]
    return aligned


def compute_reasoning_token_drop_mask(token_ids_2d: torch.Tensor, args):
    if token_ids_2d is None:
        return None
    if not bool(getattr(args, "thinkjepa_drop_thinking_tokens", False)):
        return None

    start_ids = parse_token_id_set(getattr(args, "thinkjepa_think_start_ids", ""))
    end_ids = parse_token_id_set(getattr(args, "thinkjepa_think_end_ids", ""))
    drop_ids = parse_token_id_set(getattr(args, "thinkjepa_think_drop_ids", ""))
    prefix_drop_len = max(0, int(getattr(args, "thinkjepa_think_drop_prefix_len", 0)))
    suffix_drop_len = max(0, int(getattr(args, "thinkjepa_think_drop_suffix_len", 0)))
    has_positional_drop = prefix_drop_len > 0 or suffix_drop_len > 0
    if (not start_ids) and (not end_ids) and (not drop_ids) and (not has_positional_drop):
        return None

    if bool(getattr(args, "thinkjepa_verbose", False)):
        print(
            f"[INFO] thinking-token filter ids: "
            f"start={sorted(start_ids)} end={sorted(end_ids)} drop={sorted(drop_ids)} "
            f"prefix_len={prefix_drop_len} suffix_len={suffix_drop_len}",
            flush=True,
        )

    ids_cpu = token_ids_2d.detach().to(device="cpu", dtype=torch.int64)
    B, S = ids_cpu.shape
    drop_cpu = torch.zeros((B, S), dtype=torch.bool)

    if prefix_drop_len > 0:
        n = min(int(prefix_drop_len), int(S))
        if n > 0:
            drop_cpu[:, :n] = True

    if suffix_drop_len > 0:
        n = min(int(suffix_drop_len), int(S))
        if n > 0:
            drop_cpu[:, (S - n) :] = True

    if drop_ids:
        drop_ids_t = torch.tensor(sorted(drop_ids), dtype=torch.int64)
        drop_cpu |= torch.isin(ids_cpu, drop_ids_t)

    pad_id = int(getattr(args, "thinkjepa_think_token_pad_id", -1))
    think_prefix_open = bool(getattr(args, "thinkjepa_think_prefix_open", False))
    if end_ids and (start_ids or think_prefix_open):
        for b in range(B):
            depth = 1 if think_prefix_open else 0
            for i in range(S):
                tid = int(ids_cpu[b, i].item())
                if tid == pad_id:
                    continue
                is_start = tid in start_ids if start_ids else False
                is_end = tid in end_ids
                if is_start:
                    depth += 1
                    drop_cpu[b, i] = True
                    if is_end:
                        depth = max(depth - 1, 0)
                    continue
                if depth > 0:
                    drop_cpu[b, i] = True
                    if is_end:
                        depth = max(depth - 1, 0)

    return drop_cpu.to(device=token_ids_2d.device)


def filter_reasoning_tokens_from_guidance(extras, args):
    if extras is None or not isinstance(extras, dict):
        return extras
    if not bool(getattr(args, "thinkjepa_drop_thinking_tokens", False)):
        return extras

    out = dict(extras)
    vlm_new = out.get("vlm_new", None)
    token_ids = coerce_token_ids_matrix(out.get("token_ids", None))
    if vlm_new is None or token_ids is None:
        return out

    drop_2d = compute_reasoning_token_drop_mask(token_ids, args)
    if drop_2d is None:
        return out
    if bool(getattr(args, "thinkjepa_verbose", False)) and not bool(
        getattr(args, "_thinkjepa_think_drop_logged", False)
    ):
        drop_counts = drop_2d.detach().to(device="cpu").sum(dim=-1).tolist()
        if len(drop_counts) > 8:
            drop_counts = drop_counts[:8] + ["..."]
        print(
            f"[INFO] thinking-token drop count per sample (pre-align): {drop_counts}",
            flush=True,
        )
        setattr(args, "_thinkjepa_think_drop_logged", True)

    vlm_new_mask = out.get("vlm_new_mask", None)
    if vlm_new_mask is None and isinstance(vlm_new, torch.Tensor):
        if vlm_new.dim() == 4:  # [B,L,S,D]
            vlm_new_mask = torch.ones(
                vlm_new.shape[:3], dtype=torch.bool, device=vlm_new.device
            )
        elif vlm_new.dim() == 3:  # [L,S,D]
            vlm_new_mask = torch.ones(
                vlm_new.shape[:2], dtype=torch.bool, device=vlm_new.device
            )
    if isinstance(vlm_new_mask, np.ndarray):
        vlm_new_mask = torch.from_numpy(vlm_new_mask)
    if not isinstance(vlm_new_mask, torch.Tensor):
        return out

    vlm_new_mask = vlm_new_mask.to(dtype=torch.bool)
    if vlm_new_mask.dim() == 3:  # [B,L,S]
        B, L, S = vlm_new_mask.shape
        drop_2d = align_token_drop_mask(
            drop_2d.to(vlm_new_mask.device), batch=B, seq_len=S
        )
        drop_3d = drop_2d.unsqueeze(1).expand(B, L, S)
        vlm_new_mask = vlm_new_mask & (~drop_3d)
    elif vlm_new_mask.dim() == 2:  # [L,S]
        L, S = vlm_new_mask.shape
        drop_2d = align_token_drop_mask(
            drop_2d.to(vlm_new_mask.device), batch=1, seq_len=S
        )
        drop_2d = drop_2d.expand(L, S)
        vlm_new_mask = vlm_new_mask & (~drop_2d)
    else:
        return out

    out["vlm_new_mask"] = vlm_new_mask

    # Optionally zero the dropped token features to avoid misuse in branches that do not check the mask.
    if bool(getattr(args, "thinkjepa_zero_dropped_think_tokens", True)):
        if isinstance(vlm_new, np.ndarray):
            vlm_new = torch.from_numpy(vlm_new)
        if isinstance(vlm_new, torch.Tensor):
            if vlm_new.dim() == 4:  # [B,L,S,D]
                B, L, S, _ = vlm_new.shape
                d2 = align_token_drop_mask(drop_2d.to(vlm_new.device), batch=B, seq_len=S)
                d4 = d2.unsqueeze(1).unsqueeze(-1).expand(B, L, S, 1)
                vlm_new = vlm_new.masked_fill(d4, 0.0)
            elif vlm_new.dim() == 3:  # [L,S,D]
                L, S, _ = vlm_new.shape
                d2 = align_token_drop_mask(drop_2d.to(vlm_new.device), batch=1, seq_len=S)
                d3 = d2.expand(L, S).unsqueeze(-1)
                vlm_new = vlm_new.masked_fill(d3, 0.0)
            out["vlm_new"] = vlm_new

    return out


def apply_guidance_ablation_policy(extras, args):
    if extras is None or not isinstance(extras, dict):
        return extras
    out = dict(extras)
    src = str(getattr(args, "thinkjepa_vlm_source", "both")).lower()
    if src == "old":
        out["vlm_new"] = None
        out["vlm_new_mask"] = None
        out["vlm_new_len"] = None
    elif src == "new":
        out["vlm_old"] = None
        out["vlm_old_mask"] = None
        out["vlm_old_len"] = None
    elif src == "none":
        out["vlm_old"] = None
        out["vlm_new"] = None
        out["vlm_old_mask"] = None
        out["vlm_new_mask"] = None
        out["vlm_old_len"] = None
        out["vlm_new_len"] = None

    out["vlm_old"] = select_pyramid_guidance_tensor(out.get("vlm_old", None), args)
    out["vlm_new"] = select_pyramid_guidance_tensor(out.get("vlm_new", None), args)
    out["vlm_old_mask"] = select_pyramid_guidance_mask(out.get("vlm_old_mask", None), args)
    out["vlm_new_mask"] = select_pyramid_guidance_mask(out.get("vlm_new_mask", None), args)
    out = filter_reasoning_tokens_from_guidance(out, args)
    return out


def build_thinkjepa_guidance_inputs(extras, args, device):
    if extras is None:
        return None
    if not bool(getattr(args, "thinkjepa_use_vlm_merge", True)):
        return None
    extras = apply_guidance_ablation_policy(extras, args)

    def _to_tensor(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.to(device).float()
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device).float()
        raise TypeError(f"Unsupported type {type(x)} for VLM features")

    def _collapse_feat(x):
        if x is None:
            return None
        x = _to_tensor(x)
        if x.dim() == 4:
            x = x.mean(dim=0, keepdim=False)
        return x

    def _collapse_mask(m):
        if m is None:
            return None
        m = _to_tensor(m)
        if m.dim() == 4:
            m = m.any(dim=-1)
        if m.dim() == 3:
            m = m.any(dim=0)
        return m

    return {
        "vlm_old": _collapse_feat(extras.get("vlm_old", None)),
        "vlm_new": _collapse_feat(extras.get("vlm_new", None)),
        "vlm_old_mask": _collapse_mask(extras.get("vlm_old_mask", None)),
        "vlm_new_mask": _collapse_mask(extras.get("vlm_new_mask", None)),
    }


def _build_thinkjepa_ext_from_extras(extras, args, device):
    return build_thinkjepa_guidance_inputs(extras, args, device)


def write_markdown_experiment_report(
    md_path: Path,
    args,
    logs: dict,
    train_size: int,
    test_size: int,
    artifact_paths: dict | None = None,
):
    epochs = logs.get("epochs", []) if isinstance(logs, dict) else []
    best = logs.get("best", {}) if isinstance(logs, dict) else {}

    lines = []
    lines.append("# Train/Eval Summary")
    lines.append("")
    lines.append("## Run Config")
    lines.append("")
    lines.append(f"- `data_dir`: `{getattr(args, 'data_dir', '')}`")
    lines.append(f"- `cache_dir`: `{getattr(args, 'cache_dir', '')}`")
    lines.append(f"- `backbone`: `{getattr(args, 'backbone', '')}`")
    lines.append(f"- `predictor`: `{getattr(args, 'predictor', '')}`")
    lines.append(f"- `epochs`: `{getattr(args, 'epochs', '')}`")
    lines.append(f"- `seed`: `{getattr(args, 'seed', '')}`")
    lines.append(f"- `trajmode`: `{getattr(args, 'trajmode', '')}`")
    lines.append(f"- `past_T`: `{getattr(args, 'past_T', '')}`")
    lines.append(f"- `future_T`: `{getattr(args, 'future_T', '')}`")
    lines.append(f"- `train_ratio`: `{getattr(args, 'train_ratio', '')}`")
    lines.append(f"- `split_seed`: `{getattr(args, 'split_seed', '')}`")
    lines.append(f"- `train_manifest`: `{getattr(args, 'train_manifest', '')}`")
    lines.append(f"- `test_manifest`: `{getattr(args, 'test_manifest', '')}`")
    lines.append(f"- `train_size`: `{train_size}`")
    lines.append(f"- `test_size`: `{test_size}`")
    if artifact_paths:
        for key, value in artifact_paths.items():
            if value:
                lines.append(f"- `{key}`: `{value}`")
    lines.append("")

    if len(epochs) > 0:
        last = epochs[-1]
        lines.append("## Best Validation")
        lines.append("")
        lines.append(f"- `best_epoch`: `{best.get('epoch', 'NA')}`")
        lines.append(f"- `best_val_avg_dist (ADE)`: `{best.get('ade', 'NA')}`")
        lines.append(f"- `best_val_loss`: `{best.get('loss', 'NA')}`")
        lines.append(f"- `best_val_pred_loss`: `{best.get('pred_loss', 'NA')}`")
        lines.append(
            f"- `best_val_pred_latent_dist`: `{best.get('pred_latent_dist', 'NA')}`"
        )
        lines.append(
            f"- `best_val_pred_latent_smooth_l1`: `{best.get('pred_latent_smooth_l1', 'NA')}`"
        )
        lines.append(
            f"- `best_val_pred_latent_cosine_distance`: `{best.get('pred_latent_cosine_distance', 'NA')}`"
        )
        lines.append(f"- `best_ckpt`: `{best.get('ckpt', 'NA')}`")
        lines.append("")
        lines.append("## Last Epoch")
        lines.append("")
        lines.append(f"- `epoch`: `{last.get('epoch', 'NA')}`")
        lines.append(f"- `train_loss`: `{last.get('train_loss', 'NA')}`")
        lines.append(f"- `train_pred_loss`: `{last.get('train_pred_loss', 'NA')}`")
        lines.append(
            f"- `train_pred_latent_dist`: `{last.get('train_pred_latent_dist', 'NA')}`"
        )
        lines.append(
            f"- `train_pred_latent_smooth_l1`: `{last.get('train_pred_latent_smooth_l1', 'NA')}`"
        )
        lines.append(
            f"- `train_pred_latent_cosine_distance`: `{last.get('train_pred_latent_cosine_distance', 'NA')}`"
        )
        lines.append(f"- `train_avg_dist`: `{last.get('train_avg_dist', 'NA')}`")
        lines.append(f"- `train_final_dist`: `{last.get('train_final_dist', 'NA')}`")
        lines.append(f"- `val_loss`: `{last.get('val_loss', 'NA')}`")
        lines.append(f"- `val_pred_loss`: `{last.get('val_pred_loss', 'NA')}`")
        lines.append(
            f"- `val_pred_latent_dist`: `{last.get('val_pred_latent_dist', 'NA')}`"
        )
        lines.append(
            f"- `val_pred_latent_smooth_l1`: `{last.get('val_pred_latent_smooth_l1', 'NA')}`"
        )
        lines.append(
            f"- `val_pred_latent_cosine_distance`: `{last.get('val_pred_latent_cosine_distance', 'NA')}`"
        )
        lines.append(f"- `val_avg_dist`: `{last.get('val_avg_dist', 'NA')}`")
        lines.append(f"- `val_final_dist`: `{last.get('val_final_dist', 'NA')}`")
        lines.append("")

        lines.append("## Epoch Table")
        lines.append("")
        lines.append(
            "| epoch | train_loss | train_pred_loss | train_pred_latent_dist | "
            "train_pred_latent_smooth_l1 | train_pred_latent_cosine_distance | "
            "val_loss | val_pred_loss | val_pred_latent_dist | "
            "val_pred_latent_smooth_l1 | val_pred_latent_cosine_distance | "
            "val_avg_dist | val_final_dist | is_best |"
        )
        lines.append(
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|"
        )
        for e in epochs:
            lines.append(
                f"| {e.get('epoch','')} | "
                f"{e.get('train_loss','')} | "
                f"{e.get('train_pred_loss','')} | "
                f"{e.get('train_pred_latent_dist','')} | "
                f"{e.get('train_pred_latent_smooth_l1','')} | "
                f"{e.get('train_pred_latent_cosine_distance','')} | "
                f"{e.get('val_loss','')} | "
                f"{e.get('val_pred_loss','')} | "
                f"{e.get('val_pred_latent_dist','')} | "
                f"{e.get('val_pred_latent_smooth_l1','')} | "
                f"{e.get('val_pred_latent_cosine_distance','')} | "
                f"{e.get('val_avg_dist','')} | "
                f"{e.get('val_final_dist','')} | "
                f"{'Y' if e.get('is_best', False) else ''} |"
            )
        lines.append("")
    else:
        lines.append("No epoch metrics were recorded.")
        lines.append("")

    write_text_file("\n".join(lines), md_path)


def stack_cached_feature_tensors(npz_paths, device, preloaded_archives: dict | None = None):
    """Stack vjepa_feats from multiple samples into [B, T, P, D]; supports bf16(uint16 view)."""
    feats = []
    for p in npz_paths:
        if preloaded_archives is not None:
            payload = preloaded_archives.get(p, None)
            if payload is None or "vjepa_feats" not in payload:
                raise KeyError(f"{p} has no key 'vjepa_feats'")
            f = payload["vjepa_feats"]
        else:
            with np.load(p, allow_pickle=False, mmap_mode="r") as z:
                if "vjepa_feats" not in z:
                    raise KeyError(f"{p} has no key 'vjepa_feats'")
                f = z["vjepa_feats"]
        x = torch.from_numpy(f)
        if x.dtype == torch.uint16:
            x = x.view(torch.bfloat16)
            x = x.float()
            feats.append(x)
    return torch.stack(feats, dim=0).to(device, non_blocking=True)  # [B,T,P,D]


def resolve_cache_archives_for_paths(
    paths,
    *,
    data_root: str,
    cache_root: str,
    cache_index: dict,
    path_cache: dict,
):
    if paths is None:
        return None

    sample_paths = list(paths) if isinstance(paths, (list, tuple)) else [paths]
    npz_list = []
    for p in sample_paths:
        sp = str(p)
        if sp.endswith(".npz"):
            npz_list.append(sp)
        elif sp.endswith(".hdf5") or sp.endswith(".h5"):
            npz_list.append(
                resolve_thinker_cache_for_hdf5(
                    sp,
                    data_root=data_root,
                    cache_root=cache_root,
                    cache_index=cache_index,
                    path_cache=path_cache,
                )
            )
        else:
            npz_list.append(None)
    return npz_list


def try_load_cached_dense_jepa_features(
    *,
    paths,
    args,
    device,
    cache_index: dict,
    path_cache: dict,
    preloaded_archives: dict | None = None,
):
    cache_root = getattr(args, "cache_dir", None)
    if cache_root is None or paths is None:
        return None

    npz_list = resolve_cache_archives_for_paths(
        paths,
        data_root=getattr(args, "data_dir"),
        cache_root=cache_root,
        cache_index=cache_index,
        path_cache=path_cache,
    )
    if not npz_list or any(p is None for p in npz_list):
        return None

    try:
        return stack_cached_feature_tensors(
            npz_list, device=device, preloaded_archives=preloaded_archives
        )
    except Exception:
        return None


def parse_batch_extras_and_paths(batch):
    """
    Parse the dataloader tail in a unified way:
      - NpzCacheDataset: [..., confs, extras(dict), paths]
      - SimpleDataset(if_return_path=True): [..., confs, paths]
    """
    extras = None
    paths = None
    if len(batch) >= 12:
        tail = batch[11]
        if isinstance(tail, dict):
            extras = tail
            if len(batch) >= 13:
                paths = batch[12]
        else:
            paths = tail
    return extras, paths


def summarize_batch_paths(paths, max_items: int = 2) -> str:
    if paths is None:
        return ""

    if isinstance(paths, (str, os.PathLike)):
        items = [os.fspath(paths)]
    elif isinstance(paths, np.ndarray):
        items = [str(x) for x in paths.reshape(-1).tolist()]
    elif isinstance(paths, (list, tuple)):
        items = [str(x) for x in paths]
    else:
        try:
            items = [str(x) for x in list(paths)]
        except TypeError:
            items = [str(paths)]

    items = [x for x in items if x and x != "None"]
    if not items:
        return ""

    shown = items[:max_items]
    suffix = "" if len(items) <= max_items else f" ... (+{len(items) - max_items} more)"
    return ", ".join(shown) + suffix


def scalar_debug_string(x) -> str:
    try:
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return f"{float(x.detach().float().cpu().item()):.6g}"
            return f"{float(x.detach().float().mean().cpu().item()):.6g}"
        return f"{float(x):.6g}"
    except Exception:
        return str(x)


def guidance_feature_dim_from_payload(extras, key: str):
    if not isinstance(extras, dict):
        return None
    x = extras.get(key, None)
    if isinstance(x, torch.Tensor) and x.dim() >= 1:
        return int(x.shape[-1])
    if isinstance(x, np.ndarray) and x.ndim >= 1:
        return int(x.shape[-1])
    return None


# ================= visualization helpers =================
def concatenate_three_segments(a, b, c):
    H, W, C = a.shape
    return np.concatenate([a, b, c], axis=1)


def compose_left_then_right_panels(
    past_base, past_draw, future_base, future_draw_gt, future_draw_pred, left_hold=None
):
    """
    First play the history window: each frame is [past_draw_or_base, black, black]
    Then play the future window: each frame is [left_hold(fixed), future_draw_gt, future_draw_pred]
    - If left_hold is None, use a black frame on the left during the future segment;
      otherwise use the provided frame (typically the last frame of the past segment)
    """
    H, W, C = past_base.shape[1:]
    black = np.zeros((H, W, C), dtype=past_base.dtype)
    frames = []
    # Left panel animates during history
    for i in range(past_base.shape[0]):
        frames.append(concatenate_three_segments(past_draw[i], black, black))
    # Right panels animate during the future segment (GT & Pred)
    if left_hold is None:
        for i in range(future_base.shape[0]):
            frames.append(concatenate_three_segments(black, future_draw_gt[i], future_draw_pred[i]))
    else:
        for i in range(future_base.shape[0]):
            frames.append(
                concatenate_three_segments(left_hold, future_draw_gt[i], future_draw_pred[i])
            )
    return frames


def render_bimanual_rollout_panel(
    *,
    img_ori: torch.Tensor,  # [B, T, H, W, C]
    tfs_in_cam: torch.Tensor,  # [B, T, J, 4, 4]
    cam_int: torch.Tensor,  # [3,3] or [T,3,3]
    pred_cam: torch.Tensor,  # [B, Tf, J, 3]
    p0: int,
    p1: int,
    f0: int,
    f1: int,
    right_dict,
    left_dict,
    tf2idx,
    draw_past: bool = False,  # Whether to overlay skeletons on the history segment
    max_batches: int = 8,  # Maximum number of samples to visualize from each batch
) -> np.ndarray:
    """
    Visualize one dataloader batch:
      - First crop the batch dimension to at most max_batches samples
      - For each sample b:
          play its own past (observe) -> then its own future (GT & Pred)
      - Concatenate all sample sequences in order
    """
    if img_ori.ndim != 5:
        raise ValueError(f"img_ori should be [B,T,H,W,C], got {img_ori.shape}")

    B_total = img_ori.shape[0]
    B_keep = (
        min(B_total, int(max_batches))
        if (max_batches is not None and max_batches > 0)
        else B_total
    )

    # Crop along the batch dimension
    if B_keep < B_total:
        img_ori = img_ori[:B_keep, ...]
        if isinstance(tfs_in_cam, torch.Tensor) and tfs_in_cam.shape[0] == B_total:
            tfs_in_cam = tfs_in_cam[:B_keep, ...]
        if isinstance(pred_cam, torch.Tensor) and pred_cam.shape[0] == B_total:
            pred_cam = pred_cam[:B_keep, ...]
        # cam_int is either (3,3) or (T,3,3) and has no batch dimension

    # ====== Scale camera intrinsics to the current canvas resolution ======
    cam_int_np = np.asarray(cam_int.detach().cpu().numpy())
    K = cam_int_np
    while K.ndim > 2:
        K = K[0]
    if K.shape != (3, 3):
        raise ValueError(
            f"Expected camera intrinsics to reduce to shape (3, 3), got {cam_int_np.shape}"
        )
    K = K.astype(np.float32).copy()

    # Estimate canvas resolution using one frame from one sample
    _, _, Hc, Wc, _ = img_ori.shape

    # Approximate the original resolution as W0 ≈ 2*cx, H0 ≈ 2*cy
    # (e.g. 1920x1080 -> cx=960, cy=540)
    cx0, cy0 = float(K[0, 2]), float(K[1, 2])
    W0 = max(1, int(round(2 * cx0)))
    H0 = max(1, int(round(2 * cy0)))

    sx = Wc / float(W0)
    sy = Hc / float(H0)

    K_adj = K.copy()
    K_adj[0, 0] *= sx  # fx
    K_adj[1, 1] *= sy  # fy
    K_adj[0, 2] *= sx  # cx
    K_adj[1, 2] *= sy  # cy

    # ====== Per-sample composition: each sample's own past -> own future ======
    frames_all = []

    device, dtype = pred_cam.device, pred_cam.dtype
    _, Tf_pred, J, _ = pred_cam.shape

    for b in range(B_keep):
        # ---------- History window (this sample's observe segment) ----------
        img_past_b = img_ori[b, p0:p1, ...].contiguous()  # [Tp,H,W,C]
        Tp = img_past_b.shape[0]
        base_past = img_past_b.detach().cpu().numpy()

        if draw_past:
            tfs_in_cam_past_b = tfs_in_cam[b, p0:p1, ...].contiguous()  # [Tp,J,4,4]
            tfs_in_cam_past_np = (
                tfs_in_cam_past_b.view(Tp, *tfs_in_cam_past_b.shape[1:])
                .detach()
                .cpu()
                .numpy()
            )
            draw_past_imgs = base_past.copy()
            render_hand_projection(
                right_dict,
                tfs_in_cam_past_np,
                draw_past_imgs,
                K_adj,
                tf2idx=tf2idx,
                right=True,
            )
            render_hand_projection(
                left_dict,
                tfs_in_cam_past_np,
                draw_past_imgs,
                K_adj,
                tf2idx=tf2idx,
                right=False,
            )
        else:
            draw_past_imgs = base_past.copy()

        # ---------- Future window GT (this sample's future GT) ----------
        img_fut_b = img_ori[b, f0:f1, ...].contiguous()  # [Tf,H,W,C]
        Tf = img_fut_b.shape[0]
        base_fut = img_fut_b.detach().cpu().numpy()

        tfs_in_cam_fut_b = tfs_in_cam[b, f0:f1, ...].contiguous()  # [Tf,J,4,4]
        tfs_in_cam_fut_np = (
            tfs_in_cam_fut_b.view(Tf, *tfs_in_cam_fut_b.shape[1:])
            .detach()
            .cpu()
            .numpy()
        )
        draw_fut_gt = base_fut.copy()
        render_hand_projection(
            right_dict, tfs_in_cam_fut_np, draw_fut_gt, K_adj, tf2idx=tf2idx, right=True
        )
        render_hand_projection(
            left_dict, tfs_in_cam_fut_np, draw_fut_gt, K_adj, tf2idx=tf2idx, right=False
        )

        # ---------- Future window Pred (this sample's pred_cam -> tfs) ----------
        pred_cam_b = pred_cam[b : b + 1, ...]  # [1,Tf_pred,J,3]
        eye4 = torch.eye(4, device=device, dtype=dtype).view(1, 1, 1, 4, 4)
        pred_tfs_cam = eye4.repeat(1, Tf_pred, J, 1, 1).clone()
        pred_tfs_cam[..., :3, 3] = pred_cam_b
        pred_tfs_cam_np = (
            pred_tfs_cam.view(Tf_pred, *pred_tfs_cam.shape[2:]).detach().cpu().numpy()
        )

        draw_fut_pred = base_fut.copy()
        render_hand_projection(
            right_dict, pred_tfs_cam_np, draw_fut_pred, K_adj, tf2idx=tf2idx, right=True
        )
        render_hand_projection(
            left_dict, pred_tfs_cam_np, draw_fut_pred, K_adj, tf2idx=tf2idx, right=False
        )

        # Keep the left column fixed during the future segment using the last history frame
        left_hold = draw_past_imgs[-1] if draw_past else base_past[-1]

        frames_b = compose_left_then_right_panels(
            base_past,
            draw_past_imgs,
            base_fut,
            draw_fut_gt,
            draw_fut_pred,
            left_hold=left_hold,
        )
        frames_all.extend(frames_b)

    return np.stack(frames_all, axis=0)  # [M_total, H, 3W, C]


# ===== helper funcs for official predictor =====
def split_context_and_future_windows(T, past_T, future_T):
    past_T = min(past_T, T)
    if future_T is None:
        future_T = max(0, T - past_T)
    future_T = min(future_T, T - past_T)
    past_idx = (0, past_T)
    fut_idx = (past_T, past_T + future_T)
    return past_idx, fut_idx


def stride_time_tensor(x, stride):
    if (x is None) or (not torch.is_tensor(x)) or int(stride) <= 1:
        return x
    if x.dim() < 2:
        return x
    return x[:, :: int(stride), ...].contiguous()


def build_future_causal_mask(T, device):
    m = torch.ones(T, T, dtype=torch.bool, device=device).tril()
    return m


def flatten_temporal_patch_tokens(x_bt_p_d):
    """[B, T, P, D] -> [B, N=T*P, D]"""
    B, T, P, D = x_bt_p_d.shape
    return x_bt_p_d.reshape(B, T * P, D)


def build_temporal_patch_indices(P, t0, t1):
    """Map (t0:t1) x P into flattened sequence indices [0..T*P); index = t*P + p."""
    t = torch.arange(t0, t1)
    p = torch.arange(P)
    idx = (t[:, None] * P + p[None, :]).reshape(-1)
    return idx


def repeat_indices_for_batch(idx_1d, B, device):
    return idx_1d.to(device).unsqueeze(0).repeat(B, 1)


def main(args):
    configure_huggingface_cache_dirs()
    args.data_dir = resolve_egodex_data_reference(str(args.data_dir))
    if getattr(args, "cache_dir", None):
        args.cache_dir = resolve_egodex_data_reference(str(args.cache_dir))
    args.preload_cache_to_memory = resolve_cache_preload_policy(args)

    ddp = bool(getattr(args, "ddp", False))
    rank, world_size = initialize_distributed_runtime(ddp)
    device = torch.device(
        f"cuda:{int(os.environ['LOCAL_RANK'])}"
        if (ddp and torch.cuda.is_available())
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    if is_primary_process(rank):
        if bool(getattr(args, "preload_cache_to_memory", False)):
            print("[INFO] preload_cache_to_memory enabled", flush=True)
    configure_reproducibility_seed(int(getattr(args, "seed", 42)))
    configure_dense_jepa_cudnn()

    num_epoch = int(getattr(args, "epochs", 300))
    use_amp = (not getattr(args, "no_amp", False)) and device.type == "cuda"
    grad_accum_steps = int(getattr(args, "grad_accum", 1))
    temporal_stride = max(1, int(getattr(args, "temporal_stride", 1)))
    if is_primary_process(rank):
        print(f"[INFO] temporal_stride={temporal_stride}", flush=True)

    Crit = nn.MSELoss()

    # === Dataset ===
    # If features come from NPZ or path remapping is needed, make sure the dataloader returns path
    need_paths = bool(
        getattr(args, "use_npz_cache", False)
        or getattr(args, "skip_vjepa", False)
        or (
            getattr(args, "predictor", "none") == "thinkjepa"
            and bool(getattr(args, "thinkjepa_use_cache_ext", True))
            and bool(getattr(args, "thinkjepa_use_vlm_merge", True))
        )
    )
    train_loader, test_loader = build_egodex_dataloaders(
        args.data_dir,
        query_tfs=QUERY_TFS,
        if_return_path=need_paths,
        camera_mode=str(getattr(args, "camera_mode", "auto")).lower(),
        train_batch=int(getattr(args, "train_batch_size", 8)),
        test_batch=int(getattr(args, "test_batch_size", 8)),
        train_manifest=getattr(args, "train_manifest", None),
        test_manifest=getattr(args, "test_manifest", None),
        use_npz_cache=bool(getattr(args, "use_npz_cache", False)),
        cache_dir=getattr(args, "cache_dir", None),
        shards=getattr(args, "shards", None) if hasattr(args, "shards") else None,
        shards_id=(
            int(getattr(args, "shards_id", 0)) if hasattr(args, "shards_id") else 0
        ),
        train_ratio=float(getattr(args, "train_ratio", 0.9)),
        split_seed=int(getattr(args, "split_seed", 42)),
        split_shuffle=not bool(getattr(args, "no_split_shuffle", False)),
        num_workers=int(getattr(args, "num_workers", 8)),
        fast_index_when_full_scan=not bool(
            getattr(args, "no_fast_hdf5_index", False)
        ),
        pin_memory=bool(getattr(args, "pin_memory", False)),
        persistent_workers=bool(getattr(args, "persistent_workers", False)),
        prefetch_factor=getattr(args, "prefetch_factor", 1),
        preload_to_memory=bool(
            getattr(args, "preload_cache_to_memory", False)
            and getattr(args, "use_npz_cache", False)
        ),
    )
    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)

    # Cache: build the index once to accelerate per-batch h5->npz lookup
    cache_index = {}
    cache_path_cache = {}
    preloaded_cache_archives = None
    thinkjepa_vlm_old_dim = int(getattr(args, "thinkjepa_vlm_old_dim", 0))
    thinkjepa_vlm_new_dim = int(getattr(args, "thinkjepa_vlm_new_dim", 0))
    if getattr(args, "cache_dir", None) and (
        bool(getattr(args, "skip_vjepa", False))
        or (
            getattr(args, "predictor", "none") == "thinkjepa"
            and bool(getattr(args, "thinkjepa_use_cache_ext", True))
            and bool(getattr(args, "thinkjepa_use_vlm_merge", True))
        )
    ):
        cache_index = build_thinker_cache_index(getattr(args, "cache_dir"))
        if bool(getattr(args, "preload_cache_to_memory", False)):
            preloaded_cache_archives = preload_npz_archives(
                getattr(args, "cache_dir")
            )
        # If dimensions are not set explicitly, infer old/new token dims from cached samples
        if thinkjepa_vlm_old_dim <= 0 or thinkjepa_vlm_new_dim <= 0:
            inf_old, inf_new, inf_src = infer_thinker_guidance_dims_from_cache(
                getattr(args, "cache_dir"),
                preloaded_archives=preloaded_cache_archives,
            )
            if thinkjepa_vlm_old_dim <= 0 and inf_old is not None:
                thinkjepa_vlm_old_dim = int(inf_old)
            if thinkjepa_vlm_new_dim <= 0 and inf_new is not None:
                thinkjepa_vlm_new_dim = int(inf_new)
            if is_primary_process(rank) and (inf_old is not None or inf_new is not None):
                print(
                    f"[INFO] ThinkJEPA VLM dims inferred from cache: "
                    f"old={inf_old} new={inf_new} sample={inf_src}"
                )
        if is_primary_process(rank):
            print(
                f"[INFO] ThinkJEPA VLM cache index built: {len(cache_index)} keys "
                f"from {getattr(args, 'cache_dir')}"
            )

    # === Conditionally load the backbone (public release keeps only V-JEPA) ===
    backbone = getattr(args, "backbone", "vjepa").lower()
    if backbone != "vjepa":
        raise ValueError(
            f"Unsupported backbone in public release: {backbone}. Only 'vjepa' is available."
        )

    model_pt = None
    fast_tx = _identity_transform

    def _ensure_vjepa_runtime():
        nonlocal model_pt, fast_tx
        if model_pt is None:
            model_pt, pt_video_transform = load_dense_jepa_encoder()
            for p in model_pt.parameters():
                p.requires_grad_(False)
            fast_tx = VideoObservationAdapter(pt_video_transform, antialias=False)
        return model_pt, fast_tx

    if not bool(getattr(args, "skip_vjepa", False)):
        _ensure_vjepa_runtime()

    # Keep the head constructor signature unchanged; joint_pred only changes input concatenation
    cls_model = TrajectoryReadoutMLP(downsample=args.joint_pred).cuda()
    if ddp:
        cls_model = torch.nn.parallel.DistributedDataParallel(
            cls_model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=bool(
                getattr(args, "ddp_find_unused_parameters", False)
            ),
        )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, cls_model.parameters()),
        lr=float(getattr(args, "lr", 1e-3)),
        weight_decay=1e-4,
    )

    predictor = None
    optimizer_pred = None

    total_epochs = num_epoch
    milestone1 = int(total_epochs * 1.0 / 3)
    milestone2 = int(total_epochs * 2.0 / 3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[milestone1, milestone2], gamma=0.1
    )
    scaler = build_amp_grad_scaler(enabled=use_amp)

    if ddp:
        from torch.utils.data.distributed import DistributedSampler

        train_ds, test_ds = train_loader.dataset, test_loader.dataset
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        test_sampler = DistributedSampler(
            test_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
        train_dl_kwargs = {
            "batch_size": train_loader.batch_size,
            "num_workers": train_loader.num_workers,
            "pin_memory": bool(getattr(args, "pin_memory", False)),
            "sampler": train_sampler,
            "drop_last": False,
        }
        test_dl_kwargs = {
            "batch_size": test_loader.batch_size,
            "num_workers": test_loader.num_workers,
            "pin_memory": bool(getattr(args, "pin_memory", False)),
            "sampler": test_sampler,
            "drop_last": False,
        }
        if train_loader.num_workers > 0:
            train_dl_kwargs["persistent_workers"] = bool(
                getattr(args, "persistent_workers", False)
            )
            train_dl_kwargs["prefetch_factor"] = int(
                getattr(args, "prefetch_factor", 1)
            )
        if test_loader.num_workers > 0:
            test_dl_kwargs["persistent_workers"] = bool(
                getattr(args, "persistent_workers", False)
            )
            test_dl_kwargs["prefetch_factor"] = int(
                getattr(args, "prefetch_factor", 1)
            )
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            **train_dl_kwargs,
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            **test_dl_kwargs,
        )
    else:
        train_sampler = test_sampler = None

    out_dir_arg = getattr(args, "output_dir", None)
    if out_dir_arg:
        out_dir = Path(out_dir_arg)
    else:
        out_dir = REPO_ROOT / "outputs" / f"thinkjepa_run_{int(time.time())}"
    if is_primary_process(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
    if ddp:
        try:
            torch.distributed.barrier(device_ids=[device.index])
        except TypeError:
            torch.distributed.barrier()
    metrics_json_path = out_dir / "metrics.json"
    logs = load_json_or_default(
        metrics_json_path,
        {
            "epochs": [],
            "best": {
                "epoch": -1,
                "ade": float("inf"),
                "loss": float("inf"),
                "pred_loss": float("inf"),
                "pred_latent_dist": float("inf"),
                "pred_latent_smooth_l1": float("inf"),
                "pred_latent_cosine_distance": float("inf"),
                "ckpt": "",
            },
        },
    )
    resume_ckpt_path = resolve_resume_checkpoint(
        out_dir=out_dir,
        resume_ckpt=getattr(args, "resume_ckpt", ""),
        auto_resume=bool(getattr(args, "auto_resume", False)),
    )
    resume_predictor_state = None
    resume_optimizer_pred_state = None
    resume_predictor_loaded = False
    start_epoch = 0
    if resume_ckpt_path is not None:
        resume_blob = torch_load_checkpoint(resume_ckpt_path, map_location="cpu")
        ckpt_epoch = int(resume_blob.get("epoch", 0))
        if ckpt_epoch < 0:
            raise ValueError(
                f"invalid checkpoint epoch {ckpt_epoch} in {resume_ckpt_path}"
            )
        unwrap_ddp_module(cls_model).load_state_dict(
            resume_blob["cls_model"], strict=True
        )
        if "optimizer" in resume_blob:
            optimizer.load_state_dict(resume_blob["optimizer"])
        if "scheduler" in resume_blob and resume_blob["scheduler"] is not None:
            scheduler.load_state_dict(resume_blob["scheduler"])
        if (
            "scaler" in resume_blob
            and resume_blob["scaler"] is not None
            and scaler is not None
        ):
            scaler.load_state_dict(resume_blob["scaler"])
        if args.predictor != "none" and ("predictor" not in resume_blob):
            raise KeyError(
                f"resume checkpoint is missing predictor state for predictor={args.predictor}: "
                f"{resume_ckpt_path}"
            )
        resume_predictor_state = resume_blob.get("predictor")
        resume_optimizer_pred_state = resume_blob.get("optimizer_pred")
        start_epoch = ckpt_epoch
        if isinstance(resume_blob.get("best"), dict):
            logs["best"] = resume_blob["best"]
        if is_primary_process(rank):
            print(
                f"[INFO] resuming training from checkpoint {resume_ckpt_path} "
                f"(completed_epochs={ckpt_epoch}, target_epochs={num_epoch})"
            )
    if is_primary_process(rank):
        write_json_atomic(logs, metrics_json_path)
    best_ade = float(logs["best"]["ade"])

    for epoch in range(start_epoch, num_epoch):
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if ddp and test_sampler is not None:
            test_sampler.set_epoch(epoch)
        t0 = time.time()

        cls_model.train()
        if model_pt is not None:
            model_pt.eval()
        optimizer.zero_grad(set_to_none=True)
        if optimizer_pred is not None:
            optimizer_pred.zero_grad(set_to_none=True)
        train_loss_sum = train_acc_sum = train_avgdist_sum = train_finaldist_sum = 0.0
        train_lat_metric_sums = initialize_latent_metric_totals()
        train_count = 0
        train_pred_count = 0
        nonfinite_skip_count = 0
        accum_steps = 0

        def _any_rank(flag: int, ddp: bool) -> bool:
            if not ddp:
                return bool(flag)
            t = torch.tensor([flag], device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            return t.item() > 0

        if is_primary_process(rank):
            pbar = tqdm.tqdm(
                total=len(train_loader), dynamic_ncols=True, leave=False, disable=False
            )
        else:

            class _NoBar:
                def update(self, *a, **k): ...
                def set_postfix(self, *a, **k): ...
                def write(self, *a, **k): ...
                def close(self): ...

            pbar = _NoBar()

        itr = iter(train_loader)
        step = 0
        while True:
            try:
                batch = next(itr)
            except StopIteration:
                break
            except Exception as e:
                if _is_fatal_camera_geometry_error(e):
                    raise
                if is_primary_process(rank):
                    pbar.write(
                        f"[DataLoader-ERROR][epoch {epoch}] {type(e).__name__}: {e}"
                    )
                pbar.update(1)
                continue

            step += 1
            if batch is None:
                pbar.update(1)
                continue

            # === Handle variable-length tails (may include extras/path) ===
            if len(batch) < 11:
                if is_primary_process(rank):
                    pbar.write(f"[WARN] Unexpected batch length: {len(batch)}")
                pbar.update(1)
                continue

            (
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
            ) = batch[:11]
            extras, paths = parse_batch_extras_and_paths(batch)

            # === Move to GPU / preprocess ===

            xyz_world = xyz_world.cuda(non_blocking=True)
            xyz_cam = xyz_cam.cuda(non_blocking=True)
            cam_ext = cam_ext.cuda(non_blocking=True)

            # === Get token features: prefer extras cache; otherwise use online extraction or NPZ (V-JEPA only) ===
            use_cached = False
            if (
                extras is not None
                and isinstance(extras, dict)
                and ("vjepa_feats" in extras)
                and (extras["vjepa_feats"] is not None)
                and backbone == "vjepa"
            ):
                out_feats_full = extras["vjepa_feats"].to(dtype=torch.float32).cuda()
                use_cached = True
                vlm_old = extras.get("vlm_old", None)
                vlm_new = extras.get("vlm_new", None)
                layers = extras.get("layers", None)

            if not use_cached:
                img = img.cuda(non_blocking=True)
                if backbone == "vjepa":
                    out_feats_full = None
                    if bool(getattr(args, "skip_vjepa", False)):
                        out_feats_full = try_load_cached_dense_jepa_features(
                            paths=paths,
                            args=args,
                            device=device,
                            cache_index=cache_index,
                            path_cache=cache_path_cache,
                            preloaded_archives=preloaded_cache_archives,
                        )
                    if out_feats_full is None:
                        model_pt, fast_tx = _ensure_vjepa_runtime()
                        img = fast_tx(img)
                        if img.ndim == 4:
                            img = img.unsqueeze(0)
                        with torch.no_grad():
                            with build_amp_autocast(enabled=use_amp):
                                out_feats_full = encode_dense_jepa_video(
                                    img, model_pt
                                )  # [B,T,P,D]
            if temporal_stride > 1:
                xyz_world = stride_time_tensor(xyz_world, temporal_stride)
                xyz_cam = stride_time_tensor(xyz_cam, temporal_stride)
                cam_ext = stride_time_tensor(cam_ext, temporal_stride)
                out_feats_full = stride_time_tensor(out_feats_full, temporal_stride)

            target_world_full = xyz_world
            B, Tall, J, Dj = target_world_full.shape

            # === Temporal slicing ===
            if args.trajmode == "traj":
                (p0, p1), (f0, f1) = split_context_and_future_windows(
                    Tall, args.past_T, args.future_T
                )
            else:
                total = min(Tall, args.past_T + args.future_T)
                (p0, p1), (f0, f1) = (0, total), (0, total)
            Tpred = f1 - f0
            feats_gt_full = out_feats_full.contiguous()
            feats_teacher_full = (
                torch.zeros_like(feats_gt_full)
                if bool(getattr(args, "zero_visual_input", False))
                else feats_gt_full
            )

            if args.predictor == "thinkjepa":
                extras = ensure_thinker_guidance_payload(
                    extras=extras,
                    paths=paths,
                    args=args,
                    device=device,
                    cache_index=cache_index,
                    path_cache=cache_path_cache,
                    preloaded_archives=preloaded_cache_archives,
                )

            # ==== (re)init predictor lazily ====
            use_pred = args.predictor != "none"
            if use_pred and (predictor is None):
                D = feats_teacher_full.shape[-1]
                P = feats_teacher_full.shape[2]
                if args.trajmode == "traj":
                    total_frames = (p1 - p0) + (f1 - f0)  # Use only [p0:p1] + [f0:f1]
                else:
                    total = min(Tall, args.past_T + args.future_T)
                    total_frames = total

                if args.predictor == "tiny":
                    predictor = PatchwiseAutoregressiveRolloutHead(
                        in_dim=D, model_dim=384, depth=2, nhead=6, dropout=0.1
                    ).cuda()
                elif args.predictor == "official":
                    predictor = VisionTransformerPredictor(
                        img_size=(P, 1),
                        patch_size=1,
                        num_frames=total_frames,
                        tubelet_size=1,
                        embed_dim=D,
                        predictor_embed_dim=384,
                        depth=12,
                        num_heads=6,
                        mlp_ratio=4.0,
                        drop_rate=0.1,
                        attn_drop_rate=0.0,
                        drop_path_rate=0.1,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        init_std=0.02,
                        uniform_power=False,
                        use_mask_tokens=True,
                        num_mask_tokens=1,
                        zero_init_mask_tokens=True,
                        use_silu=False,
                        wide_silu=True,
                        use_activation_checkpointing=False,
                        return_all_tokens=False,
                        chop_last_n_tokens=0,
                        use_rope=True,
                    ).cuda()
                elif args.predictor == "thinkjepa":
                    thinkjepa_use_vlm_merge = bool(
                        getattr(args, "thinkjepa_use_vlm_merge", True)
                    )
                    thinkjepa_vlm_cond_mode = str(
                        getattr(args, "thinkjepa_vlm_cond_mode", "film")
                    ).lower()
                    if thinkjepa_vlm_cond_mode not in {"film", "crossattn", "adaln"}:
                        thinkjepa_vlm_cond_mode = "film"
                    if thinkjepa_use_vlm_merge:
                        old_dim_from_extras = guidance_feature_dim_from_payload(extras, "vlm_old")
                        new_dim_from_extras = guidance_feature_dim_from_payload(extras, "vlm_new")
                        eff_old_dim = (
                            int(thinkjepa_vlm_old_dim)
                            if int(thinkjepa_vlm_old_dim) > 0
                            else (
                                int(old_dim_from_extras)
                                if old_dim_from_extras is not None
                                else 3584
                            )
                        )
                        eff_new_dim = (
                            int(thinkjepa_vlm_new_dim)
                            if int(thinkjepa_vlm_new_dim) > 0
                            else (
                                int(new_dim_from_extras)
                                if new_dim_from_extras is not None
                                else 3584
                            )
                        )
                        thinkjepa_vlm_old_dim, thinkjepa_vlm_new_dim = eff_old_dim, eff_new_dim
                        if is_primary_process(rank):
                            print(
                                f"[INFO] ThinkJEPA predictor VLM proj dims: "
                                f"old={thinkjepa_vlm_old_dim} new={thinkjepa_vlm_new_dim} "
                                f"mode={thinkjepa_vlm_cond_mode}"
                            )
                    else:
                        thinkjepa_vlm_old_dim, thinkjepa_vlm_new_dim = 1, 1
                        if is_primary_process(rank):
                            print(
                                "[INFO] ThinkJEPA predictor VLM merge disabled; "
                                "running direct ViT conditioning baseline."
                            )
                    predictor = CortexGuidedVideoPredictor(
                        img_size=(P, 1),
                        patch_size=1,
                        num_frames=total_frames,
                        tubelet_size=1,
                        embed_dim=D,
                        predictor_embed_dim=384,
                        depth=12,
                        num_heads=6,
                        mlp_ratio=4.0,
                        drop_rate=0.1,
                        attn_drop_rate=0.0,
                        drop_path_rate=0.1,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        init_std=0.02,
                        uniform_power=False,
                        use_mask_tokens=True,
                        num_mask_tokens=2,
                        zero_init_mask_tokens=True,
                        use_silu=False,
                        wide_silu=True,
                        use_activation_checkpointing=False,
                        return_all_tokens=False,
                        chop_last_n_tokens=0,
                        use_rope=True,
                        use_vlm_merge=thinkjepa_use_vlm_merge,
                        vlm_cond_mode=thinkjepa_vlm_cond_mode,
                        vlm_old_dim=thinkjepa_vlm_old_dim,
                        vlm_new_dim=thinkjepa_vlm_new_dim,
                    ).cuda()
                else:
                    predictor = None

                if ddp and (predictor is not None):
                    predictor_find_unused = bool(
                        getattr(args, "ddp_find_unused_parameters", False)
                    )
                    if args.predictor == "thinkjepa":
                        predictor_find_unused = True
                        if is_primary_process(rank):
                            print(
                                "[INFO] enabling DDP find_unused_parameters for ThinkJEPA predictor"
                            )
                    predictor = torch.nn.parallel.DistributedDataParallel(
                        predictor,
                        device_ids=[device.index],
                        output_device=device.index,
                        find_unused_parameters=predictor_find_unused,
                    )

                if predictor is not None:
                    optimizer_pred = torch.optim.AdamW(
                        (p for p in predictor.parameters() if p.requires_grad),
                        lr=float(getattr(args, "lr_pred", 1e-4)),
                        weight_decay=1e-4,
                    )
                    if not resume_predictor_loaded and (
                        resume_predictor_state is not None
                    ):
                        unwrap_ddp_module(predictor).load_state_dict(
                            resume_predictor_state, strict=True
                        )
                        if resume_optimizer_pred_state is not None:
                            optimizer_pred.load_state_dict(
                                resume_optimizer_pred_state
                            )
                        resume_predictor_loaded = True
                        if is_primary_process(rank):
                            print(
                                "[INFO] loaded predictor/optimizer_pred state from resume checkpoint"
                            )

            # ==== predictor forward (build pred_loss & feats_task_in) ====
            pred_metrics = {
                "pred_loss": torch.tensor(0.0, device=device),
                "pred_latent_dist": torch.tensor(0.0, device=device),
                "pred_latent_smooth_l1": torch.tensor(0.0, device=device),
                "pred_latent_cosine_distance": torch.tensor(0.0, device=device),
            }
            pred_metric_valid = False
            if use_pred and (predictor is not None):
                if args.predictor == "tiny":
                    # Strictly use the history window (p0, p1) as feats_ar_in (left side)
                    feats_ar_in = feats_teacher_full[:, p0:p1, ...].contiguous()
                    with build_amp_autocast(enabled=use_amp):
                        try:
                            pred_feats_past = predictor(feats_ar_in)  # [B, p1-p0, P, D]
                        except TypeError:
                            pred_feats_past = predictor(
                                feats_ar_in,
                                attn_mask=build_future_causal_mask(
                                    feats_ar_in.shape[1], feats_ar_in.device
                                ),
                            )
                        tgt_feats_future = feats_gt_full[:, f0:f1, ...].detach()
                        if Tpred > 0:
                            pred_feats_future = pred_feats_past[
                                :, -Tpred:, ...
                            ].contiguous()
                            pred_metrics = compute_predicted_latent_metrics(
                                pred_feats_future, tgt_feats_future
                            )
                            pred_metric_valid = True
                        else:
                            pred_metrics["pred_loss"] = pred_feats_past.new_zeros(())
                    feats_task_in = pred_feats_past.detach()

                elif args.predictor == "official":
                    # Use [p0:p1] + [f0:f1] as the full window and build relative time indices
                    total_frames = (
                        (p1 - p0) + max(0, f1 - f0)
                        if args.trajmode == "traj"
                        else min(Tall, args.past_T + args.future_T)
                    )
                    feats_total = feats_teacher_full[:, :total_frames, ...].contiguous()
                    B_, total_, P_, D_ = feats_total.shape
                    x_seq = flatten_temporal_patch_tokens(feats_total)

                    # History context occupies [0, p1-p0) in the concatenated sequence
                    ctx_len = p1 - p0
                    idx_ctx_1d = build_temporal_patch_indices(P_, 0, ctx_len)

                    if args.trajmode == "traj":
                        # Target interval is [ctx_len, ctx_len + (f1-f0))
                        T_tgt = f1 - f0
                        idx_tgt_1d = build_temporal_patch_indices(P_, ctx_len, ctx_len + T_tgt)
                    else:
                        T_tgt = ctx_len
                        idx_tgt_1d = idx_ctx_1d

                    masks_x = repeat_indices_for_batch(
                        idx_ctx_1d, B_, device=x_seq.device
                    )
                    masks_y = repeat_indices_for_batch(
                        idx_tgt_1d, B_, device=x_seq.device
                    )

                    x_ctxt = x_seq.gather(
                        dim=1, index=masks_x.unsqueeze(-1).expand(-1, -1, D_)
                    )

                    with build_amp_autocast(enabled=use_amp):
                        y_future_seq = predictor(
                            x_ctxt, masks_x, masks_y
                        )  # [B, N_tgt, D]
                        y_future = y_future_seq.view(B_, T_tgt, P_, D_)
                        tgt_future = (
                            feats_gt_full[:, f0:f1, ...].detach()
                            if args.trajmode == "traj"
                            else feats_gt_full[:, p0:p1, ...].detach()
                        )
                        pred_metrics = compute_predicted_latent_metrics(y_future, tgt_future)
                        pred_metric_valid = True

                    # Feed only future features into the downstream head
                    feats_task_in = y_future.detach()

                elif args.predictor == "thinkjepa":
                    # 1) Build the V-JEPA total window and flatten the sequence
                    total_frames = (
                        ((p1 - p0) + max(0, f1 - f0))
                        if args.trajmode == "traj"
                        else min(Tall, args.past_T + args.future_T)
                    )
                    feats_total = feats_teacher_full[:, :total_frames, ...].contiguous()
                    B_, total_, P_, D_ = feats_total.shape
                    x_seq = flatten_temporal_patch_tokens(feats_total)  # [B, total*P, D]

                    # 2) Build masks_x / masks_y (global indices)
                    ctx_len = p1 - p0
                    idx_ctx_1d = build_temporal_patch_indices(P_, 0, ctx_len)  # [ctx_len*P]
                    if args.trajmode == "traj":
                        T_tgt = f1 - f0
                        idx_tgt_1d = build_temporal_patch_indices(P_, ctx_len, ctx_len + T_tgt)
                    else:
                        T_tgt = ctx_len
                        idx_tgt_1d = idx_ctx_1d
                    masks_x = repeat_indices_for_batch(
                        idx_ctx_1d.long(), B_, device=x_seq.device
                    )
                    masks_y = repeat_indices_for_batch(
                        idx_tgt_1d.long(), B_, device=x_seq.device
                    )

                    # 3) Gather context tokens
                    x_ctxt = x_seq.gather(
                        dim=1, index=masks_x.unsqueeze(-1).expand(-1, -1, D_)
                    )

                    # 4) Assemble ext (VLM conditioning)
                    ext = build_thinkjepa_guidance_inputs(
                        extras=extras,
                        args=args,
                        device=x_seq.device,
                    )

                    with build_amp_autocast(enabled=use_amp):
                        y_future_seq = predictor(
                            x_ctxt, masks_x, masks_y, ext=ext
                        )  # [B, N_tgt, D]
                        y_future = y_future_seq.view(B_, T_tgt, P_, D_)
                        tgt_future = (
                            feats_gt_full[:, f0:f1, ...].detach()
                            if args.trajmode == "traj"
                            else feats_gt_full[:, p0:p1, ...].detach()
                        )
                        pred_metrics = compute_predicted_latent_metrics(y_future, tgt_future)
                        pred_metric_valid = True

                    feats_task_in = y_future.detach()

            else:
                feats_task_in = feats_teacher_full[:, p0:p1, ...].contiguous()

            # === (NEW) joint_pred: feed both past and future teacher/predicted features into the downstream head
            # === only in traj mode with Tpred > 0 ===
            if (
                getattr(args, "joint_pred", False)
                and args.trajmode == "traj"
                and Tpred > 0
            ):
                feats_task_in = torch.cat(
                    [
                        feats_teacher_full[:, p0:p1, ...].contiguous(),
                        feats_task_in,
                    ],
                    dim=1,
                ).contiguous()

            # === targets & refs ===
            target_world = target_world_full[:, f0:f1, ...].contiguous()
            cam_ext_slice = cam_ext[:, f0:f1, ...].contiguous()

            xyz_cam_slice = xyz_cam[:, f0:f1, ...].contiguous()
            right_ref_cam = xyz_cam_slice[..., right_idx, :]
            left_ref_cam = xyz_cam_slice[..., left_idx, :]

            with build_amp_autocast(enabled=use_amp):
                pred_cam = (
                    predict_trajectory_from_latents(cls_model, feats_task_in)
                    .view(B, Tpred, J, 3)
                    .contiguous()
                )

                if args.ref_mode == "perhand":
                    pred_cam = append_hand_reference_offsets(pred_cam, right_ref_cam, left_ref_cam)
                else:
                    ref_offset_cam = None
                    if args.ref_mode == "right":
                        ref_offset_cam = right_ref_cam
                    elif args.ref_mode == "left":
                        ref_offset_cam = left_ref_cam
                    elif args.ref_mode == "both":
                        ref_offset_cam = right_ref_cam + left_ref_cam
                    elif args.ref_mode == "avg":
                        ref_offset_cam = 0.5 * (right_ref_cam + left_ref_cam)
                    if ref_offset_cam is not None:
                        pred_cam = pred_cam + ref_offset_cam.unsqueeze(2)

                pred_world = project_camera_points_to_world(pred_cam, cam_ext_slice)

                task_loss, avg_dist, final_dist, acc = compute_trajectory_loss_and_accuracy(
                    pred_world, target_world, Crit, thr=0.05
                )

            nonfinite_reasons = []
            if not torch.isfinite(task_loss).all():
                nonfinite_reasons.append(f"task_loss={scalar_debug_string(task_loss)}")
            if (
                use_pred
                and (predictor is not None)
                and pred_metric_valid
                and (not torch.isfinite(pred_metrics["pred_loss"]).all())
            ):
                nonfinite_reasons.append(
                    f"pred_loss={scalar_debug_string(pred_metrics['pred_loss'])}"
                )
            any_nonfinite = _any_rank(int(bool(nonfinite_reasons)), ddp)
            if any_nonfinite:
                batch_paths_msg = summarize_batch_paths(paths)
                if not getattr(args, "skip_nonfinite_loss", False):
                    if nonfinite_reasons:
                        reason_str = ", ".join(nonfinite_reasons)
                        path_str = f" paths={batch_paths_msg}" if batch_paths_msg else ""
                        raise FloatingPointError(
                            f"Non-finite loss at epoch={epoch+1} step={step}{path_str}: "
                            f"{reason_str}"
                        )
                    raise FloatingPointError(
                        f"Non-finite loss detected on another rank at epoch={epoch+1} step={step}"
                    )

                optimizer.zero_grad(set_to_none=True)
                if optimizer_pred is not None:
                    optimizer_pred.zero_grad(set_to_none=True)
                accum_steps = 0
                nonfinite_skip_count += 1
                if is_primary_process(rank):
                    path_str = f" paths={batch_paths_msg}" if batch_paths_msg else ""
                    if nonfinite_reasons:
                        pbar.write(
                            f"[WARN][epoch {epoch+1:03d} step {step}] "
                            f"skip non-finite batch{path_str}: {', '.join(nonfinite_reasons)}"
                        )
                    else:
                        pbar.write(
                            f"[WARN][epoch {epoch+1:03d} step {step}] "
                            "skip batch because another rank reported non-finite loss"
                        )
                pbar.update(1)
                continue

            # === backward ===
            if (
                use_pred
                and args.optimize_together_downstream
                and (predictor is not None)
            ):
                total_loss = (
                    args.lambda_task * task_loss
                    + args.lambda_pred * pred_metrics["pred_loss"]
                ) / max(grad_accum_steps, 1)
                if use_amp:
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
                display_loss = total_loss.detach()
                accum_steps += 1
            elif (
                use_pred
                and (predictor is not None)
                and not args.optimize_together_downstream
            ):
                if optimizer_pred is not None:
                    optimizer_pred.zero_grad(set_to_none=True)
                    loss_pred_norm = pred_metrics["pred_loss"] / max(
                        grad_accum_steps, 1
                    )
                    if use_amp:
                        scaler.scale(loss_pred_norm).backward(retain_graph=True)
                    else:
                        loss_pred_norm.backward(retain_graph=True)

                loss_task_norm = task_loss / max(grad_accum_steps, 1)
                if use_amp:
                    scaler.scale(loss_task_norm).backward()
                else:
                    loss_task_norm.backward()
                display_loss = task_loss.detach()
                accum_steps += 1
            else:
                loss_task_norm = task_loss / max(grad_accum_steps, 1)
                if use_amp:
                    scaler.scale(loss_task_norm).backward()
                else:
                    loss_task_norm.backward()
                display_loss = task_loss.detach()
                accum_steps += 1

            if _any_rank(0, ddp):  # skip_this is always 0 here; keep the structure unchanged
                pbar.update(1)
                continue

            train_loss_sum += float(display_loss.item() * max(grad_accum_steps, 1))
            train_acc_sum += float(acc)
            train_avgdist_sum += float(avg_dist.mean().item())
            train_finaldist_sum += float(final_dist.mean().item())
            train_count += 1
            if pred_metric_valid:
                for key in train_lat_metric_sums:
                    train_lat_metric_sums[key] += float(pred_metrics[key].item())
                train_pred_count += 1

            if accum_steps >= max(grad_accum_steps, 1):
                if use_amp:
                    scaler.step(optimizer)
                    if use_pred and (optimizer_pred is not None):
                        scaler.step(optimizer_pred)
                    scaler.update()
                else:
                    optimizer.step()
                    if use_pred and (optimizer_pred is not None):
                        optimizer_pred.step()
                optimizer.zero_grad(set_to_none=True)
                if optimizer_pred is not None:
                    optimizer_pred.zero_grad(set_to_none=True)
                accum_steps = 0

            if step % 2 == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{float(display_loss.item()):.4f}",
                        "pred_loss": (
                            f"{float(pred_metrics['pred_loss'].item()):.4f}"
                            if (use_pred and predictor is not None)
                            else "NA"
                        ),
                        "pred_lat": (
                            f"{float(pred_metrics['pred_latent_dist'].item()):.4f}"
                            if pred_metric_valid
                            else "NA"
                        ),
                        "pred_s1": (
                            f"{float(pred_metrics['pred_latent_smooth_l1'].item()):.4f}"
                            if pred_metric_valid
                            else "NA"
                        ),
                        "pred_cos": (
                            f"{float(pred_metrics['pred_latent_cosine_distance'].item()):.4f}"
                            if pred_metric_valid
                            else "NA"
                        ),
                        "acc": f"{float(acc):.3f}",
                        "ADE": f"{float(avg_dist.mean().item()):.3f}",
                        "FDE": f"{float(final_dist.mean().item()):.3f}",
                        "step": step,
                    }
                )
            pbar.update(1)

        if is_primary_process(rank):
            pbar.close()

        if accum_steps > 0:
            if use_amp:
                scaler.step(optimizer)
                if (args.predictor != "none") and (optimizer_pred is not None):
                    scaler.step(optimizer_pred)
                scaler.update()
            else:
                optimizer.step()
                if (args.predictor != "none") and (optimizer_pred is not None):
                    optimizer_pred.step()
            optimizer.zero_grad(set_to_none=True)
            if optimizer_pred is not None:
                optimizer_pred.zero_grad(set_to_none=True)

        # ---------------- Eval ----------------
        cls_model.eval()
        if model_pt is not None:
            model_pt.eval()
        test_loss_sum = test_acc_sum = test_avgdist_sum = test_finaldist_sum = 0.0
        test_lat_metric_sums = initialize_latent_metric_totals()
        test_count = 0
        test_pred_count = 0

        with torch.no_grad():
            itr_test = iter(test_loader)
            composed_pool = []  # for visualization across batches
            vis_batch_count = 0  # Track how many batches have already been visualized
            max_vis_batches = getattr(args, "max_visual_batches", 10)
            while True:
                try:
                    batch = next(itr_test)
                except StopIteration:
                    break
                except Exception as e:
                    if _is_fatal_camera_geometry_error(e):
                        raise
                    if is_primary_process(rank):
                        print(
                            f"[DataLoader-ERROR][EVAL][epoch {epoch}] {type(e).__name__}: {e}"
                        )
                    continue

                if batch is None or len(batch) < 11:
                    continue

                (
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
                ) = batch[:11]
                extras, paths = parse_batch_extras_and_paths(batch)

                xyz_world = xyz_world.to(device, non_blocking=True)
                xyz_cam = xyz_cam.to(device, non_blocking=True)
                cam_ext = cam_ext.to(device, non_blocking=True)
                img_ori = img.detach().clone()

                # === Get token features: prefer extras (V-JEPA only), otherwise use online extraction or NPZ ===
                use_cached = False
                if (
                    extras is not None
                    and isinstance(extras, dict)
                    and ("vjepa_feats" in extras)
                    and (extras["vjepa_feats"] is not None)
                    and backbone == "vjepa"
                ):
                    out_feats_full = extras["vjepa_feats"].to(
                        device=device, dtype=torch.float32
                    )
                    use_cached = True

                if not use_cached:
                    img = img.to(device, non_blocking=True)

                    if backbone == "vjepa":
                        out_feats_full = None
                        if bool(getattr(args, "skip_vjepa", False)):
                            out_feats_full = try_load_cached_dense_jepa_features(
                                paths=paths,
                                args=args,
                                device=device,
                                cache_index=cache_index,
                                path_cache=cache_path_cache,
                                preloaded_archives=preloaded_cache_archives,
                            )
                        if out_feats_full is None:
                            model_pt, fast_tx = _ensure_vjepa_runtime()
                            img = fast_tx(img)
                            if img.ndim == 4:
                                img = img.unsqueeze(0)
                            out_feats_full = encode_dense_jepa_video(img, model_pt)
                if temporal_stride > 1:
                    xyz_world = stride_time_tensor(xyz_world, temporal_stride)
                    xyz_cam = stride_time_tensor(xyz_cam, temporal_stride)
                    cam_ext = stride_time_tensor(cam_ext, temporal_stride)
                    out_feats_full = stride_time_tensor(
                        out_feats_full, temporal_stride
                    )

                B, Tall, J, Dj = xyz_world.shape

                if args.trajmode == "traj":
                    (p0, p1), (f0, f1) = split_context_and_future_windows(
                        Tall, args.past_T, args.future_T
                    )
                else:
                    total = min(Tall, args.past_T + args.future_T)
                    (p0, p1), (f0, f1) = (0, total), (0, total)

                Tpred = f1 - f0
                feats_gt_full = out_feats_full.contiguous()
                feats_eval_full = (
                    torch.zeros_like(feats_gt_full)
                    if bool(getattr(args, "zero_visual_input", False))
                    else feats_gt_full
                )
                xyz_world_slice = xyz_world[:, f0:f1, ...].contiguous()
                cam_ext_slice = cam_ext[:, f0:f1, ...].contiguous()

                if args.predictor == "thinkjepa":
                    extras = ensure_thinker_guidance_payload(
                        extras=extras,
                        paths=paths,
                        args=args,
                        device=device,
                        cache_index=cache_index,
                        path_cache=cache_path_cache,
                        preloaded_archives=preloaded_cache_archives,
                    )

                use_pred = (args.predictor != "none") and (predictor is not None)
                pred_metrics_eval = {
                    "pred_loss": torch.tensor(0.0, device=device),
                    "pred_latent_dist": torch.tensor(0.0, device=device),
                    "pred_latent_smooth_l1": torch.tensor(0.0, device=device),
                    "pred_latent_cosine_distance": torch.tensor(0.0, device=device),
                }
                pred_metric_valid = False
                if use_pred:
                    if args.predictor == "tiny":
                        feats_ar_in = feats_eval_full[:, p0:p1, ...].contiguous()
                        try:
                            pred_feats_past = predictor(feats_ar_in)
                        except TypeError:
                            pred_feats_past = predictor(
                                feats_ar_in,
                                attn_mask=build_future_causal_mask(
                                    feats_ar_in.shape[1], feats_ar_in.device
                                ),
                            )
                        tgt_future = feats_gt_full[:, f0:f1, ...].detach()
                        if Tpred > 0:
                            pred_feats_future = pred_feats_past[:, -Tpred:, ...].contiguous()
                            pred_metrics_eval = compute_predicted_latent_metrics(
                                pred_feats_future, tgt_future
                            )
                            pred_metric_valid = True
                        feats_task_in = pred_feats_past.detach()

                    elif args.predictor == "official":
                        ctx_len = p1 - p0
                        feats_total = torch.cat(
                            [
                                feats_eval_full[:, p0:p1, ...].contiguous(),
                                (
                                    feats_eval_full[:, f0:f1, ...].contiguous()
                                    if args.trajmode == "traj"
                                    else feats_eval_full[:, p0:p1, ...].contiguous()
                                ),
                            ],
                            dim=1,
                        )
                        B_, total_, P_, D_ = feats_total.shape
                        x_seq = flatten_temporal_patch_tokens(feats_total)

                        # History context interval [0, ctx_len)
                        idx_ctx_1d = build_temporal_patch_indices(P_, 0, ctx_len)
                        masks_x = repeat_indices_for_batch(
                            idx_ctx_1d, B_, device=x_seq.device
                        )
                        x_ctxt = x_seq.gather(
                            dim=1, index=masks_x.unsqueeze(-1).expand(-1, -1, D_)
                        )

                        # Target (future) interval [ctx_len, ctx_len + T_tgt)
                        T_tgt = (f1 - f0) if args.trajmode == "traj" else ctx_len
                        idx_tgt_1d = build_temporal_patch_indices(P_, ctx_len, ctx_len + T_tgt)
                        masks_y = repeat_indices_for_batch(
                            idx_tgt_1d, B_, device=x_seq.device
                        )

                        with build_amp_autocast(enabled=use_amp):
                            y_future_seq = predictor(
                                x_ctxt, masks_x, masks_y
                            )  # [B, N_tgt, D]
                            y_future = y_future_seq.view(B_, T_tgt, P_, D_)
                        tgt_future = (
                            feats_gt_full[:, f0:f1, ...].detach()
                            if args.trajmode == "traj"
                            else feats_gt_full[:, p0:p1, ...].detach()
                        )
                        pred_metrics_eval = compute_predicted_latent_metrics(
                            y_future, tgt_future
                        )
                        pred_metric_valid = True

                        # Feed only future features into the downstream head
                        feats_task_in = y_future.detach()

                    elif args.predictor == "thinkjepa":
                        # Keep evaluation aligned with training: use x + masks and route VLM through ext conditioning
                        total_frames = (
                            ((p1 - p0) + max(0, f1 - f0))
                            if args.trajmode == "traj"
                            else min(Tall, args.past_T + args.future_T)
                        )
                        feats_total = feats_eval_full[:, :total_frames, ...].contiguous()
                        B_, total_, P_, D_ = feats_total.shape
                        x_seq = flatten_temporal_patch_tokens(feats_total)

                        ctx_len = p1 - p0
                        idx_ctx_1d = build_temporal_patch_indices(P_, 0, ctx_len)
                        if args.trajmode == "traj":
                            T_tgt = f1 - f0
                            idx_tgt_1d = build_temporal_patch_indices(P_, ctx_len, ctx_len + T_tgt)
                        else:
                            T_tgt = ctx_len
                            idx_tgt_1d = idx_ctx_1d

                        masks_x = repeat_indices_for_batch(
                            idx_ctx_1d.long(), B_, device=x_seq.device
                        )
                        masks_y = repeat_indices_for_batch(
                            idx_tgt_1d.long(), B_, device=x_seq.device
                        )

                        x_ctxt = x_seq.gather(
                            dim=1, index=masks_x.unsqueeze(-1).expand(-1, -1, D_)
                        )

                        ext = build_thinkjepa_guidance_inputs(
                            extras=extras,
                            args=args,
                            device=x_seq.device,
                        )

                        with build_amp_autocast(enabled=use_amp):
                            y_future_seq = predictor(x_ctxt, masks_x, masks_y, ext=ext)
                            y_future = y_future_seq.view(B_, T_tgt, P_, D_)
                        tgt_future = (
                            feats_gt_full[:, f0:f1, ...].detach()
                            if args.trajmode == "traj"
                            else feats_gt_full[:, p0:p1, ...].detach()
                        )
                        pred_metrics_eval = compute_predicted_latent_metrics(
                            y_future, tgt_future
                        )
                        pred_metric_valid = True

                        feats_task_in = y_future.detach()
                else:
                    feats_task_in = feats_eval_full[:, p0:p1, ...].contiguous()

                # joint_pred at eval
                if (
                    getattr(args, "joint_pred", False)
                    and args.trajmode == "traj"
                    and Tpred > 0
                ):
                    feats_task_in = torch.cat(
                        [
                            feats_eval_full[:, p0:p1, ...].contiguous(),
                            feats_task_in,
                        ],
                        dim=1,
                    ).contiguous()

                xyz_cam_slice = xyz_cam[:, f0:f1, ...].contiguous()
                right_ref_cam = xyz_cam_slice[..., right_idx, :]
                left_ref_cam = xyz_cam_slice[..., left_idx, :]

                pred_cam = (
                    predict_trajectory_from_latents(cls_model, feats_task_in)
                    .view(B, Tpred, J, 3)
                    .contiguous()
                )

                if args.ref_mode == "perhand":
                    pred_cam = append_hand_reference_offsets(pred_cam, right_ref_cam, left_ref_cam)
                else:
                    ref_offset_cam = None
                    if args.ref_mode == "right":
                        ref_offset_cam = right_ref_cam
                    elif args.ref_mode == "left":
                        ref_offset_cam = left_ref_cam
                    elif args.ref_mode == "both":
                        ref_offset_cam = right_ref_cam + left_ref_cam
                    elif args.ref_mode == "avg":
                        ref_offset_cam = 0.5 * (right_ref_cam + left_ref_cam)
                    if ref_offset_cam is not None:
                        pred_cam = pred_cam + ref_offset_cam.unsqueeze(2)

                pred_world = project_camera_points_to_world(pred_cam, cam_ext_slice)

                loss, avg_dist, final_dist, acc = compute_trajectory_loss_and_accuracy(
                    pred_world, xyz_world_slice, Crit, thr=0.05
                )
                test_loss_sum += float(loss.item())
                test_acc_sum += float(acc)
                test_avgdist_sum += float(avg_dist.mean().item())
                test_finaldist_sum += float(final_dist.mean().item())
                test_count += 1
                if pred_metric_valid:
                    for key in test_lat_metric_sums:
                        test_lat_metric_sums[key] += float(
                            pred_metrics_eval[key].item()
                        )
                    test_pred_count += 1

                # ---------- Visualization ----------
                if (
                    visualize_flag
                    and epoch % 10 == 0
                    and is_primary_process(rank)
                    and vis_batch_count < max_vis_batches
                ):
                    composed = render_bimanual_rollout_panel(
                        img_ori=img_ori,
                        tfs_in_cam=tfs_in_cam,
                        cam_int=cam_int,
                        pred_cam=pred_cam,
                        p0=p0,
                        p1=p1,
                        f0=f0,
                        f1=f1,
                        right_dict=right_dict,
                        left_dict=left_dict,
                        tf2idx=tf2idx,
                    )
                    composed_pool.append(composed)
                    vis_batch_count += 1

        # ---------- Save composed video (both modes) ----------
        if "composed_pool" in locals() and len(composed_pool) > 0:
            os.makedirs(args.output_mp4, exist_ok=True)  # Ensure the directory exists
            composed_all = np.concatenate(composed_pool, axis=0)  # [M, H, 3W, C]
            write_video_frames_to_mp4(
                [composed_all], f"{args.output_mp4}/output_{epoch}_ori_gt_pred.mp4"
            )
            print(
                f"Done. Video saved to: {args.output_mp4}/output_{epoch}_ori_gt_pred.mp4"
            )
            del composed_pool

        dt = time.time() - t0
        avg_train_loss = train_loss_sum / max(train_count, 1)
        avg_train_acc = train_acc_sum / max(train_count, 1)
        avg_train_avgdist = train_avgdist_sum / max(train_count, 1)
        avg_train_finaldist = train_finaldist_sum / max(train_count, 1)
        avg_test_loss = test_loss_sum / max(test_count, 1)
        avg_test_acc = test_acc_sum / max(test_count, 1)
        avg_test_avgdist = test_avgdist_sum / max(test_count, 1)
        avg_test_finaldist = test_finaldist_sum / max(test_count, 1)
        avg_train_lat_metrics = {
            key: distributed_average_from_sum_count(val, train_pred_count, ddp=ddp)
            for key, val in train_lat_metric_sums.items()
        }
        avg_test_lat_metrics = {
            key: distributed_average_from_sum_count(val, test_pred_count, ddp=ddp)
            for key, val in test_lat_metric_sums.items()
        }

        avg_train_loss = distributed_mean_scalar(
            avg_train_loss, ddp=ddp, world_size=world_size
        )
        avg_train_acc = distributed_mean_scalar(avg_train_acc, ddp=ddp, world_size=world_size)
        avg_train_avgdist = distributed_mean_scalar(
            avg_train_avgdist, ddp=ddp, world_size=world_size
        )
        avg_train_finaldist = distributed_mean_scalar(
            avg_train_finaldist, ddp=ddp, world_size=world_size
        )
        avg_test_loss = distributed_mean_scalar(avg_test_loss, ddp=ddp, world_size=world_size)
        avg_test_acc = distributed_mean_scalar(avg_test_acc, ddp=ddp, world_size=world_size)
        avg_test_avgdist = distributed_mean_scalar(
            avg_test_avgdist, ddp=ddp, world_size=world_size
        )
        avg_test_finaldist = distributed_mean_scalar(
            avg_test_finaldist, ddp=ddp, world_size=world_size
        )

        if is_primary_process(rank):
            train_pred_loss_str = (
                f"{avg_train_lat_metrics['pred_loss']:.4f}"
                if avg_train_lat_metrics["pred_loss"] is not None
                else "NA"
            )
            train_pred_latent_str = (
                f"{avg_train_lat_metrics['pred_latent_dist']:.4f}"
                if avg_train_lat_metrics["pred_latent_dist"] is not None
                else "NA"
            )
            train_pred_s1_str = (
                f"{avg_train_lat_metrics['pred_latent_smooth_l1']:.4f}"
                if avg_train_lat_metrics["pred_latent_smooth_l1"] is not None
                else "NA"
            )
            train_pred_cos_str = (
                f"{avg_train_lat_metrics['pred_latent_cosine_distance']:.4f}"
                if avg_train_lat_metrics["pred_latent_cosine_distance"] is not None
                else "NA"
            )
            val_pred_loss_str = (
                f"{avg_test_lat_metrics['pred_loss']:.4f}"
                if avg_test_lat_metrics["pred_loss"] is not None
                else "NA"
            )
            val_pred_latent_str = (
                f"{avg_test_lat_metrics['pred_latent_dist']:.4f}"
                if avg_test_lat_metrics["pred_latent_dist"] is not None
                else "NA"
            )
            val_pred_s1_str = (
                f"{avg_test_lat_metrics['pred_latent_smooth_l1']:.4f}"
                if avg_test_lat_metrics["pred_latent_smooth_l1"] is not None
                else "NA"
            )
            val_pred_cos_str = (
                f"{avg_test_lat_metrics['pred_latent_cosine_distance']:.4f}"
                if avg_test_lat_metrics["pred_latent_cosine_distance"] is not None
                else "NA"
            )
            print(
                f"[Epoch {epoch+1:03d}] "
                f"train_loss={avg_train_loss:.4f} acc={avg_train_acc:.4f} "
                f"pred_loss={train_pred_loss_str} pred_latent_dist={train_pred_latent_str} "
                f"pred_smooth_l1={train_pred_s1_str} pred_cos={train_pred_cos_str} "
                f"avg_dist={avg_train_avgdist:.4f} final_dist={avg_train_finaldist:.4f} | "
                f"val_loss={avg_test_loss:.4f} acc={avg_test_acc:.4f} "
                f"pred_loss={val_pred_loss_str} pred_latent_dist={val_pred_latent_str} "
                f"pred_smooth_l1={val_pred_s1_str} pred_cos={val_pred_cos_str} "
                f"avg_dist={avg_test_avgdist:.4f} final_dist={avg_test_finaldist:.4f} | "
                f"time={dt:.1f}s"
                + (
                    f" skipped_nonfinite={int(nonfinite_skip_count)}"
                    if nonfinite_skip_count > 0
                    else ""
                )
            )

        latest_path = out_dir / "ckpt_latest.pt"
        if is_primary_process(rank):
            try:
                save_training_checkpoint(
                    latest_path,
                    epoch + 1,
                    cls_model,
                    optimizer,
                    logs["best"],
                    predictor=predictor,
                    optimizer_pred=optimizer_pred,
                    scheduler=scheduler,
                    scaler=scaler,
                    args=args,
                )
                cleanup_legacy_epoch_checkpoints(out_dir)
            except Exception as ex:
                print(
                    f"[WARN] failed to save latest checkpoint {latest_path}: {ex}",
                    flush=True,
                )

        is_best = avg_test_avgdist < best_ade
        if is_primary_process(rank) and is_best:
            best_ade = avg_test_avgdist
            best_epoch = epoch + 1
            best_path = out_dir / "ckpt_best.pt"
            best_blob = {
                "epoch": best_epoch,
                "ade": best_ade,
                "loss": avg_test_loss,
                "pred_loss": avg_test_lat_metrics["pred_loss"],
                "pred_latent_dist": avg_test_lat_metrics["pred_latent_dist"],
                "pred_latent_smooth_l1": avg_test_lat_metrics[
                    "pred_latent_smooth_l1"
                ],
                "pred_latent_cosine_distance": avg_test_lat_metrics[
                    "pred_latent_cosine_distance"
                ],
                "ckpt": str(best_path),
            }
            try:
                save_training_checkpoint(
                    best_path,
                    best_epoch,
                    cls_model,
                    optimizer,
                    best_blob,
                    predictor=predictor,
                    optimizer_pred=optimizer_pred,
                    scheduler=scheduler,
                    scaler=scaler,
                    args=args,
                )
                logs["best"] = best_blob
            except Exception as ex:
                print(
                    f"[WARN] failed to save best checkpoint {best_path}: {ex}",
                    flush=True,
                )

        if is_primary_process(rank):
            logs["epochs"].append(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "train_pred_loss": avg_train_lat_metrics["pred_loss"],
                    "train_pred_latent_dist": avg_train_lat_metrics[
                        "pred_latent_dist"
                    ],
                    "train_pred_latent_smooth_l1": avg_train_lat_metrics[
                        "pred_latent_smooth_l1"
                    ],
                    "train_pred_latent_cosine_distance": avg_train_lat_metrics[
                        "pred_latent_cosine_distance"
                    ],
                    "train_nonfinite_skips": int(nonfinite_skip_count),
                    "train_acc": avg_train_acc,
                    "train_avg_dist": avg_train_avgdist,
                    "train_final_dist": avg_train_finaldist,
                    "val_loss": avg_test_loss,
                    "val_pred_loss": avg_test_lat_metrics["pred_loss"],
                    "val_pred_latent_dist": avg_test_lat_metrics["pred_latent_dist"],
                    "val_pred_latent_smooth_l1": avg_test_lat_metrics[
                        "pred_latent_smooth_l1"
                    ],
                    "val_pred_latent_cosine_distance": avg_test_lat_metrics[
                        "pred_latent_cosine_distance"
                    ],
                    "val_acc": avg_test_acc,
                    "val_avg_dist": avg_test_avgdist,
                    "val_final_dist": avg_test_finaldist,
                    "time_sec": dt,
                    "ckpt": str(latest_path),
                    "is_best": is_best,
                }
            )
            try:
                write_json_atomic(logs, metrics_json_path)
            except Exception as ex:
                print(
                    f"[WARN] failed to save metrics json {metrics_json_path}: {ex}",
                    flush=True,
                )
            try:
                write_json_atomic(
                    {
                        "current_epoch": epoch + 1,
                        "target_epochs": int(num_epoch),
                        "latest_ckpt": str(latest_path),
                        "best_epoch": int(best_epoch),
                        "best_ckpt": str(out_dir / "ckpt_best.pt"),
                    },
                    out_dir / "training_state.json",
                )
                write_text_file(f"{epoch + 1}\n", out_dir / "latest_epoch.txt")
            except Exception as ex:
                print(
                    f"[WARN] failed to save training state under {out_dir}: {ex}",
                    flush=True,
                )
        if ddp:
            torch.distributed.barrier()
        scheduler.step()

    if is_primary_process(rank):
        latent_plot_path = plot_latent_metric_curves(
            logs=logs, out_path=out_dir / "latent_metric_curves.png"
        )
        md_path = (
            Path(getattr(args, "results_md"))
            if getattr(args, "results_md", None)
            else (out_dir / "test_results.md")
        )
        write_markdown_experiment_report(
            md_path=md_path,
            args=args,
            logs=logs,
            train_size=train_size,
            test_size=test_size,
            artifact_paths={
                "latent_metric_plot": str(latent_plot_path) if latent_plot_path else ""
            },
        )
        print(f"[REPORT] markdown saved to: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default=DEFAULT_EGODEX_PART2_HF_DIR
    )
    parser.add_argument("--num_episodes", default=1)
    parser.add_argument("--output_mp4", default="visualize/output")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="directory to save checkpoints / metrics / markdown summary",
    )
    parser.add_argument(
        "--results_md",
        type=str,
        default=None,
        help="optional markdown summary path; default: <output_dir>/test_results.md",
    )
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help="resume from ckpt_latest.pt under --output_dir, or fall back to the latest legacy ckpt_epoch*.pt, before continuing to --epochs",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default="",
        help="resume from an explicit checkpoint path instead of auto-discovering the latest epoch checkpoint",
    )
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--skip_nonfinite_loss",
        action="store_true",
        help="skip train batches with non-finite task/predictor loss instead of aborting the run",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument(
        "--trajmode", type=str, default="track", choices=["track", "traj"]
    )
    parser.add_argument("--past_T", type=int, default=32)
    parser.add_argument("--future_T", type=int, default=32)
    parser.add_argument(
        "--temporal_stride",
        type=int,
        default=1,
        help="subsample temporal dimension by this stride before past/future slicing",
    )
    parser.add_argument("--temporal_causal_attn", action="store_true")

    parser.add_argument(
        "--backbone",
        type=str,
        default="vjepa",
        choices=["vjepa"],
        help="choose backbone for the public release; only vjepa is supported",
    )

    # old flag (kept for compatibility)
    parser.add_argument(
        "--vjepa_predictor",
        action="store_true",
        help="(deprecated) kept for compatibility; use --predictor instead.",
    )
    # NEW: unified predictor selector
    parser.add_argument(
        "--predictor",
        type=str,
        default="none",
        choices=["none", "tiny", "official", "thinkjepa"],
        help="choose predictor head: none | tiny (paper) | official (V-JEPA predictor) | thinkjepa (cortex-guided ThinkJEPA rollout head)",
    )

    parser.add_argument("--optimize_together_downstream", action="store_true")
    parser.add_argument("--lambda_pred", type=float, default=1.0)
    parser.add_argument("--lambda_task", type=float, default=1.0)
    parser.add_argument("--lr_pred", type=float, default=1e-4)
    parser.add_argument(
        "--ref_mode",
        type=str,
        default="none",
        help="how to add reference joint back to prediction (none|right|left|both|avg|perhand)",
    )
    # (NEW) joint_pred flag
    parser.add_argument(
        "--joint_pred",
        action="store_true",
        help="feed both past and future features into the MLP head (enabled only in traj mode with Tpred>0)",
    )
    # (NEW) NPZ / cache settings (used only for V-JEPA)
    parser.add_argument(
        "--camera_mode",
        type=str,
        default="auto",
        choices=["auto", "egodex", "egoexo"],
        help="camera source for cache loader: auto prefers egoexo_cam_* when present; egoexo forces EgoExo intrinsics/extrinsics; egodex uses legacy cam_ext/cam_int",
    )
    parser.add_argument(
        "--use_npz_cache",
        action="store_true",
        help="read samples from cached .npz files (including vjepa_feats, etc.; V-JEPA only)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_EGODEX_PART2_HF_DIR,
        help="root directory of the NPZ cache (should match your extraction output; V-JEPA only)",
    )
    parser.add_argument(
        "--preload_cache_to_memory",
        dest="preload_cache_to_memory",
        action="store_true",
        help="eagerly load the full resolved dataset/cache into RAM before training",
    )
    parser.add_argument(
        "--no_preload_cache_to_memory",
        dest="preload_cache_to_memory",
        action="store_false",
        help="disable eager full-cache RAM preload even when using Hugging Face cache paths",
    )
    parser.set_defaults(preload_cache_to_memory=None)
    parser.add_argument(
        "--skip_vjepa",
        action="store_true",
        help="skip V-JEPA initialization and forward when cached vjepa_feats are available",
    )
    # Optional sharding
    parser.add_argument("--shards", type=int, default=None, help="total number of data shards")
    parser.add_argument("--shards_id", type=int, default=0, help="current shard id")
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="auto split ratio for train set (test uses remaining samples)",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="random seed for train/test split shuffling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for model init / dataloader / training runtime",
    )
    parser.add_argument(
        "--no_split_shuffle",
        action="store_true",
        help="disable shuffling before train/test split",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="dataloader workers",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="enable DataLoader pin_memory",
    )
    parser.add_argument(
        "--persistent_workers",
        action="store_true",
        help="enable persistent DataLoader workers when num_workers>0",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=1,
        help="DataLoader prefetch_factor when num_workers>0",
    )
    parser.add_argument(
        "--ddp_find_unused_parameters",
        action="store_true",
        help="enable DDP find_unused_parameters; disabled by default to reduce overhead",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="train dataloader batch size",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=8,
        help="test/eval dataloader batch size",
    )
    parser.add_argument(
        "--train_manifest",
        type=str,
        default="",
        help="optional fixed train path manifest (.txt), one absolute sample path per line",
    )
    parser.add_argument(
        "--test_manifest",
        type=str,
        default="",
        help="optional fixed test path manifest (.txt), one absolute sample path per line",
    )
    parser.add_argument(
        "--no_fast_hdf5_index",
        action="store_true",
        help="disable fast full-scan index mode (fast mode avoids opening every hdf5 file just to build split)",
    )
    parser.add_argument(
        "--thinkjepa_use_cache_ext",
        dest="thinkjepa_use_cache_ext",
        action="store_true",
        help="for predictor=thinkjepa, load VLM conditioning features from cache_dir by sample path",
    )
    parser.add_argument(
        "--no_thinkjepa_use_cache_ext",
        dest="thinkjepa_use_cache_ext",
        action="store_false",
        help="disable loading VLM cache ext for ThinkJEPA predictor",
    )
    parser.set_defaults(thinkjepa_use_cache_ext=True)
    parser.add_argument(
        "--thinkjepa_use_vlm_merge",
        dest="thinkjepa_use_vlm_merge",
        action="store_true",
        help="enable VLM FiLM merge in ThinkJEPA predictor",
    )
    parser.add_argument(
        "--no_thinkjepa_use_vlm_merge",
        dest="thinkjepa_use_vlm_merge",
        action="store_false",
        help="disable VLM FiLM merge (direct ViT conditioning baseline)",
    )
    parser.set_defaults(thinkjepa_use_vlm_merge=True)
    parser.add_argument(
        "--thinkjepa_direct_vit_condition",
        action="store_true",
        help="shortcut ablation: disable ThinkJEPA VLM merge + disable cache ext + force thinkjepa_vlm_source=none",
    )
    parser.add_argument(
        "--vlm_pad_old_to",
        type=int,
        default=480,
        help="pad/truncate token length for vlm_old when loading cache ext",
    )
    parser.add_argument(
        "--vlm_pad_new_to",
        type=int,
        default=15,
        help="pad/truncate token length for vlm_new when loading cache ext",
    )
    parser.add_argument(
        "--thinkjepa_vlm_old_dim",
        type=int,
        default=0,
        help="input dim of thinkjepa vlm_old projection (0=auto infer from cache/extras, fallback 3584)",
    )
    parser.add_argument(
        "--thinkjepa_vlm_new_dim",
        type=int,
        default=0,
        help="input dim of thinkjepa vlm_new projection (0=auto infer from cache/extras, fallback 3584)",
    )
    parser.add_argument(
        "--thinkjepa_vlm_source",
        type=str,
        default="both",
        choices=["both", "old", "new", "none"],
        help="which VLM cache source to use for ThinkJEPA conditioning",
    )
    parser.add_argument(
        "--thinkjepa_vlm_layer_selector",
        type=str,
        default="last",
        choices=["last", "mid", "index", "all"],
        help="which VLM layer(s) to use for ThinkJEPA conditioning",
    )
    parser.add_argument(
        "--thinkjepa_vlm_layer_index",
        type=int,
        default=-1,
        help="specific VLM layer index when thinkjepa_vlm_layer_selector=index",
    )
    parser.add_argument(
        "--thinkjepa_vlm_cond_mode",
        type=str,
        default="film",
        choices=["film", "crossattn", "adaln"],
        help="conditioning mechanism for ThinkJEPA VLM merge: film | crossattn | adaln",
    )
    parser.add_argument(
        "--thinkjepa_drop_thinking_tokens",
        action="store_true",
        help="drop Qwen thinking/deepstack token positions from vlm_new conditioning by token_ids",
    )
    parser.add_argument(
        "--thinkjepa_think_start_ids",
        type=str,
        default="151667",
        help="comma/space separated token ids indicating thinking span start (Qwen3-VL default: <think>=151667)",
    )
    parser.add_argument(
        "--thinkjepa_think_end_ids",
        type=str,
        default="151668",
        help="comma/space separated token ids indicating thinking span end (Qwen3-VL default: </think>=151668)",
    )
    parser.add_argument(
        "--thinkjepa_think_drop_ids",
        type=str,
        default="",
        help="comma/space separated token ids to always drop from vlm_new conditioning",
    )
    parser.add_argument(
        "--thinkjepa_think_token_pad_id",
        type=int,
        default=-1,
        help="pad token id used in cached token_ids",
    )
    parser.add_argument(
        "--thinkjepa_think_prefix_open",
        action="store_true",
        help="treat thinking span as already open at first generated token (useful for Qwen3-VL-*-Thinking chat template that pre-appends '<think>')",
    )
    parser.add_argument(
        "--thinkjepa_think_drop_prefix_len",
        type=int,
        default=0,
        help="always drop the first N token positions from vlm_new conditioning (position-based proxy when think markers are absent in cache)",
    )
    parser.add_argument(
        "--thinkjepa_think_drop_suffix_len",
        type=int,
        default=0,
        help="always drop the last N token positions from vlm_new conditioning",
    )
    parser.add_argument(
        "--thinkjepa_zero_dropped_think_tokens",
        dest="thinkjepa_zero_dropped_think_tokens",
        action="store_true",
        help="zero vlm_new features at dropped thinking token positions",
    )
    parser.add_argument(
        "--no_thinkjepa_zero_dropped_think_tokens",
        dest="thinkjepa_zero_dropped_think_tokens",
        action="store_false",
        help="do not zero vlm_new features at dropped thinking token positions (mask-only)",
    )
    parser.set_defaults(thinkjepa_zero_dropped_think_tokens=True)
    parser.add_argument(
        "--thinkjepa_verbose",
        action="store_true",
        help="print ThinkJEPA conditioning debug info",
    )
    parser.add_argument(
        "--zero_visual_input",
        action="store_true",
        help="zero out V-JEPA visual context/features before predictor and traj head; useful for VLM-only baselines",
    )

    # NEW: maximum number of visualized batches
    parser.add_argument(
        "--max_visual_batches",
        type=int,
        default=10,
        help="maximum number of test batches to visualize (prevents excessively long stitched videos)",
    )

    args = parser.parse_args()
    if not (0.0 < float(args.train_ratio) < 1.0):
        raise ValueError(f"--train_ratio must be in (0,1), got {args.train_ratio}")
    if int(args.temporal_stride) <= 0:
        raise ValueError(
            f"--temporal_stride must be positive, got {args.temporal_stride}"
        )
    if int(args.thinkjepa_think_drop_prefix_len) < 0:
        raise ValueError(
            f"--thinkjepa_think_drop_prefix_len must be >= 0, got {args.thinkjepa_think_drop_prefix_len}"
        )
    if int(args.thinkjepa_think_drop_suffix_len) < 0:
        raise ValueError(
            f"--thinkjepa_think_drop_suffix_len must be >= 0, got {args.thinkjepa_think_drop_suffix_len}"
        )

    if bool(getattr(args, "thinkjepa_direct_vit_condition", False)):
        args.thinkjepa_use_vlm_merge = False
        args.thinkjepa_use_cache_ext = False
        args.thinkjepa_vlm_source = "none"
    if bool(getattr(args, "thinkjepa_drop_thinking_tokens", False)):
        has_ids = any(
            str(getattr(args, k, "")).strip()
            for k in ("thinkjepa_think_start_ids", "thinkjepa_think_end_ids", "thinkjepa_think_drop_ids")
        )
        has_positional = int(getattr(args, "thinkjepa_think_drop_prefix_len", 0)) > 0 or int(
            getattr(args, "thinkjepa_think_drop_suffix_len", 0)
        ) > 0
        if not has_ids and not has_positional:
            raise ValueError(
                "--thinkjepa_drop_thinking_tokens requires at least one rule: "
                "token ids (--thinkjepa_think_start_ids/--thinkjepa_think_end_ids/--thinkjepa_think_drop_ids) "
                "or positional (--thinkjepa_think_drop_prefix_len/--thinkjepa_think_drop_suffix_len)"
            )

    # if getattr(args, "vjepa_predictor", False) and (args.predictor == "none"):
    #     args.predictor = "tiny"
    if args.use_npz_cache and args.backbone == "vjepa":
        # Using cached features usually implies skipping the online V-JEPA forward pass
        args.skip_vjepa = True

    main(args)
    shutdown_distributed_runtime(bool(getattr(args, "ddp", False)))
