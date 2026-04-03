#!/usr/bin/env python3

# ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model
# Copyright (c) 2024-2026 Northeastern University.
# Developed in NEU SMILE LAB by Haichao Zhang (https://zhanghaichao.xyz)
# and Yun Raymond Fu (https://www1.ece.neu.edu/~yunfu/).
# SPDX-style identifier: LicenseRef-ThinkJEPA-Attribution
# Original source: https://github.com/Hai-chao-Zhang/ThinkJEPA
# See the root LICENSE, NOTICE, CITATION.cff, and CITATION.bib for attribution and citation requirements.

"""
Research-grade overnight experiment scheduler for ThinkJEPA EgoDex main-suite runs.

What this script does:
1. Runs all requested experiments in a strict, fixed order.
2. Runs 3 seeds per experiment (serially).
3. Skips completed runs when --resume is enabled and continues incomplete runs in place.
4. Saves per-seed configs, logs, metrics, runtime, git hash, and visualization paths.
5. Aggregates mean/std across seeds.
6. Generates paper-style markdown summary tables.
7. Runs an autoregressive rollout evaluator for the 4->4 horizon experiments.

Assumptions:
- Training entrypoint is scripts/train.sh.
- EgoDex cache already contains vjepa_feats/imgs and Qwen VLM features.
- Current "VLM_only" baseline is implemented as zeroing visual V-JEPA inputs while
  keeping VLM conditioning active. This is a valid ablation under the current model.
- "remove_thinker_module" is mapped to removing VLM conditioning from ThinkJEPA, which is
  the closest train-time ablation available with the current codebase.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
VJEPA2_ROOT = Path(
    os.environ.get("VJEPA2_ROOT", str(REPO_ROOT / "external" / "vjepa2"))
).resolve()

from cache_train.hf_egodex import (
    DEFAULT_EGODEX_PART2_HF_DIR,
    configure_huggingface_cache_dirs,
    is_huggingface_cache_path,
    resolve_egodex_data_reference,
)

def _prepend_source_import_path(path: Path) -> None:
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)


# The training stack relies on top-level imports such as `predictor`, `src`, and
# `egodex.*` that are only resolvable when these source roots are on sys.path.
_prepend_source_import_path(REPO_ROOT / "cache_train")
_prepend_source_import_path(REPO_ROOT / "cache_train" / "egodex")
_prepend_source_import_path(VJEPA2_ROOT)
_prepend_source_import_path(VJEPA2_ROOT.parent)
_prepend_source_import_path(REPO_ROOT / "vjepa2")
_prepend_source_import_path(REPO_ROOT)


HORIZONS = (4, 8, 16, 32)
_RUNTIME_MODULES: Optional[Dict[str, Any]] = None
THINKJEPA_NAME = "ThinkJEPA"


def load_runtime_modules() -> Dict[str, Any]:
    global _RUNTIME_MODULES
    if _RUNTIME_MODULES is None:
        _RUNTIME_MODULES = {
            "train": importlib.import_module("cache_train.thinker_train"),
            "models": importlib.import_module("cache_train.models"),
            "thinkjepa": importlib.import_module("cache_train.thinker_predictor"),
            "tiny_pred": importlib.import_module("predictor"),
            "official_pred": importlib.import_module("vjepa2.src.models.predictor"),
        }
    return _RUNTIME_MODULES


def canonicalize_experiment_name(name: str) -> str:
    raw = str(name).strip()
    if raw.lower() == THINKJEPA_NAME.lower():
        return THINKJEPA_NAME
    return raw


def resolve_suite_cache_preload_policy(args: argparse.Namespace) -> bool:
    explicit = getattr(args, "preload_cache_to_memory", None)
    if explicit is not None:
        return bool(explicit)
    value = getattr(args, "cache_dir", None)
    return isinstance(value, str) and value and is_huggingface_cache_path(value)


@dataclass
class ExperimentSpec:
    name: str
    section: str
    env: Dict[str, str]
    rollout: bool = False
    notes: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ThinkJEPA EgoDex main-suite experiment scheduler")
    p.add_argument("--results_root", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--train_script", type=Path, default=REPO_ROOT / "scripts" / "train.sh")
    p.add_argument("--data_dir", type=str, default=DEFAULT_EGODEX_PART2_HF_DIR)
    p.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_EGODEX_PART2_HF_DIR,
    )
    p.add_argument(
        "--preload_cache_to_memory",
        dest="preload_cache_to_memory",
        action="store_true",
        help="eagerly load the full resolved dataset/cache into RAM for suite runtime paths",
    )
    p.add_argument(
        "--no_preload_cache_to_memory",
        dest="preload_cache_to_memory",
        action="store_false",
        help="disable eager RAM preload even when using Hugging Face cache paths",
    )
    p.set_defaults(preload_cache_to_memory=None)
    p.add_argument(
        "--train_manifest",
        type=str,
        default=os.environ.get("TRAIN_MANIFEST", ""),
    )
    p.add_argument(
        "--test_manifest",
        type=str,
        default=os.environ.get("TEST_MANIFEST", ""),
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--gpu_list", type=str, default="0,1,2,3")
    p.add_argument("--nproc_per_node", type=int, default=4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=str, default="1e-3")
    p.add_argument("--lr_pred", type=str, default="1e-4")
    p.add_argument(
        "--vlm_only_lr",
        type=str,
        default="",
        help="optional override for the VLM_only task-head learning rate",
    )
    p.add_argument(
        "--vlm_only_lr_pred",
        type=str,
        default="",
        help="optional override for the VLM_only predictor learning rate",
    )
    p.add_argument(
        "--vjepa_only_lr",
        type=str,
        default="",
        help="optional override for the VJEPA_only task-head learning rate",
    )
    p.add_argument(
        "--vjepa_only_lr_pred",
        type=str,
        default="",
        help="optional override for the VJEPA_only predictor learning rate",
    )
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--test_batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument(
        "--camera_mode",
        type=str,
        default=os.environ.get("CAMERA_MODE", "auto"),
        choices=["auto", "egodex", "egoexo"],
        help="camera source passed to thinker_train.py cache loader",
    )
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--prefetch_factor", type=int, default=1)
    p.add_argument("--ddp_find_unused_parameters", action="store_true")
    p.add_argument("--omp_threads", type=int, default=1)
    p.add_argument("--mkl_threads", type=int, default=1)
    p.add_argument("--openblas_threads", type=int, default=1)
    p.add_argument("--numexpr_threads", type=int, default=1)
    p.add_argument("--malloc_arena_max", type=int, default=2)
    p.add_argument(
        "--pytorch_cuda_alloc_conf",
        type=str,
        default="expandable_segments:True,max_split_size_mb:128",
    )
    p.add_argument("--max_visual_batches", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--skip_nonfinite_loss", action="store_true")
    p.add_argument("--skip_rollout", action="store_true")
    p.add_argument(
        "--model_names",
        nargs="*",
        default=[THINKJEPA_NAME, "VJEPA_only"],
        help="limit to exact experiment names; pass an empty list to run all models",
    )
    p.add_argument(
        "--thinkjepa_epochs",
        type=int,
        default=100,
        help="target epochs for ThinkJEPA experiments",
    )
    p.add_argument(
        "--sections",
        nargs="*",
        choices=["main", "ablation", "layer", "horizon_rollout"],
        default=["main", "ablation", "layer", "horizon_rollout"],
    )
    return p.parse_args()


def make_safe_experiment_dir_name(name: str) -> str:
    x = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    x = re.sub(r"_+", "_", x).strip("_")
    return x or "experiment"


def ensure_results_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_repository_commit_hash(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
            )
            .strip()
        )
    except Exception:
        return "UNKNOWN"


def write_yaml_summary(path: Path, payload: Dict[str, Any]) -> None:
    ensure_results_directory(path.parent)
    try:
        import yaml  # type: ignore

        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
        return
    except Exception:
        pass

    def _emit(obj: Any, indent: int = 0) -> List[str]:
        sp = "  " * indent
        if isinstance(obj, dict):
            lines: List[str] = []
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{sp}{k}:")
                    lines.extend(_emit(v, indent + 1))
                else:
                    lines.append(f"{sp}{k}: {json.dumps(v, ensure_ascii=False)}")
            return lines
        if isinstance(obj, list):
            lines = []
            for v in obj:
                if isinstance(v, (dict, list)):
                    lines.append(f"{sp}-")
                    lines.extend(_emit(v, indent + 1))
                else:
                    lines.append(f"{sp}- {json.dumps(v, ensure_ascii=False)}")
            return lines
        return [f"{sp}{json.dumps(obj, ensure_ascii=False)}"]

    path.write_text("\n".join(_emit(payload)) + "\n", encoding="utf-8")


def latest_video_under_directory(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    videos = sorted(root.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return videos[0] if videos else None


def symlink_or_copy_file(src: Path, dst: Path) -> None:
    ensure_results_directory(dst.parent)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        rel = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel)
    except Exception:
        shutil.copy2(src, dst)


def build_suite_experiments() -> Dict[str, List[ExperimentSpec]]:
    common_full = {
        "PREDICTOR": "thinkjepa",
        "BACKBONE": "vjepa",
        "JOINT_PRED": "1",
        "SKIP_VJEPA": "1",
        "THINKJEPA_VLM_SOURCE": "both",
        "THINKJEPA_VLM_LAYER_SELECTOR": "last",
        "ZERO_VISUAL_INPUT": "0",
    }
    return {
        "main": [
            ExperimentSpec(THINKJEPA_NAME, "main", dict(common_full)),
            ExperimentSpec(
                "VLM_only",
                "main",
                {**common_full, "ZERO_VISUAL_INPUT": "1"},
                notes="Visual V-JEPA inputs zeroed; VLM conditioning kept active.",
            ),
            ExperimentSpec(
                "VJEPA_only",
                "main",
                {
                    **common_full,
                    "THINKJEPA_VLM_SOURCE": "none",
                    "NO_THINKJEPA_CACHE_EXT": "1",
                },
                notes="Strict no-VLM baseline: disable cache-ext loading and drop VLM extras.",
            ),
        ],
        "ablation": [
            ExperimentSpec(
                "vlm_old+vjepa",
                "ablation",
                {**common_full, "THINKJEPA_VLM_SOURCE": "old"},
            ),
            ExperimentSpec(
                "vlm_old only",
                "ablation",
                {
                    **common_full,
                    "THINKJEPA_VLM_SOURCE": "old",
                    "ZERO_VISUAL_INPUT": "1",
                },
            ),
            ExperimentSpec(
                "vlm_new+vjepa",
                "ablation",
                {**common_full, "THINKJEPA_VLM_SOURCE": "new"},
            ),
            ExperimentSpec(
                "vlm_new only",
                "ablation",
                {
                    **common_full,
                    "THINKJEPA_VLM_SOURCE": "new",
                    "ZERO_VISUAL_INPUT": "1",
                },
            ),
            ExperimentSpec(
                "remove_thinker_module",
                "ablation",
                {**common_full, "NO_THINKJEPA_CACHE_EXT": "1"},
                notes="Mapped to removing VLM conditioning from ThinkJEPA in the current codebase.",
            ),
        ],
        "layer": [
            ExperimentSpec(
                "vlm_last_layer",
                "layer",
                {**common_full, "THINKJEPA_VLM_LAYER_SELECTOR": "last"},
            ),
            ExperimentSpec(
                "vlm_mid_layer",
                "layer",
                {**common_full, "THINKJEPA_VLM_LAYER_SELECTOR": "mid"},
            ),
        ],
        "horizon_rollout": [
            ExperimentSpec(
                THINKJEPA_NAME,
                "horizon_rollout",
                {
                    **common_full,
                    "PAST_T": "4",
                    "FUTURE_T": "4",
                },
                rollout=True,
            ),
            ExperimentSpec(
                "VLM_only",
                "horizon_rollout",
                {
                    **common_full,
                    "PAST_T": "4",
                    "FUTURE_T": "4",
                    "ZERO_VISUAL_INPUT": "1",
                },
                rollout=True,
                notes="Visual V-JEPA inputs zeroed; VLM conditioning kept active.",
            ),
            ExperimentSpec(
                "VJEPA_only",
                "horizon_rollout",
                {
                    **common_full,
                    "PAST_T": "4",
                    "FUTURE_T": "4",
                    "THINKJEPA_VLM_SOURCE": "none",
                    "NO_THINKJEPA_CACHE_EXT": "1",
                },
                rollout=True,
                notes="Strict no-VLM baseline: disable cache-ext loading and drop VLM extras.",
            ),
        ],
    }


def target_epochs_for_experiment(args: argparse.Namespace, spec: ExperimentSpec) -> int:
    if spec.name == THINKJEPA_NAME:
        return int(args.thinkjepa_epochs)
    return int(args.epochs)


def select_experiments_for_sections(
    args: argparse.Namespace,
    experiments_by_section: Dict[str, List[ExperimentSpec]],
    section: str,
) -> List[ExperimentSpec]:
    specs = list(experiments_by_section.get(section, []))
    model_names = [
        canonicalize_experiment_name(x)
        for x in (getattr(args, "model_names", []) or [])
        if str(x).strip()
    ]
    if not model_names:
        return specs
    selected = set(model_names)
    return [spec for spec in specs if spec.name in selected]


def build_training_environment(args: argparse.Namespace, spec: ExperimentSpec, seed: int, seed_dir: Path) -> Dict[str, str]:
    target_epochs = target_epochs_for_experiment(args, spec)
    lr = str(args.lr)
    lr_pred = str(args.lr_pred)
    if spec.name == "VLM_only":
        if str(getattr(args, "vlm_only_lr", "")).strip():
            lr = str(args.vlm_only_lr)
        if str(getattr(args, "vlm_only_lr_pred", "")).strip():
            lr_pred = str(args.vlm_only_lr_pred)
    elif spec.name == "VJEPA_only":
        if str(getattr(args, "vjepa_only_lr", "")).strip():
            lr = str(args.vjepa_only_lr)
        if str(getattr(args, "vjepa_only_lr_pred", "")).strip():
            lr_pred = str(args.vjepa_only_lr_pred)
    env = os.environ.copy()
    env.update(
        {
            "DATA_DIR": args.data_dir,
            "CACHE_DIR": args.cache_dir,
            "TRAIN_MANIFEST": args.train_manifest,
            "TEST_MANIFEST": args.test_manifest,
            "GPU_LIST": args.gpu_list,
            "NPROC_PER_NODE": str(args.nproc_per_node),
            "OUT_DIR": str(seed_dir),
            "OUT_ROOT": str(seed_dir.parent),
            "RUN_NAME": seed_dir.name,
            "RESULTS_MD": str(seed_dir / "test_results.md"),
            "LOG_FILE": str(seed_dir / "log.txt"),
            "OUTPUT_MP4": str(seed_dir / "vis"),
            "EPOCHS": str(target_epochs),
            "LR": lr,
            "LR_PRED": lr_pred,
            "NUM_WORKERS": str(args.num_workers),
            "TRAIN_BATCH_SIZE": str(args.train_batch_size),
            "TEST_BATCH_SIZE": str(args.test_batch_size),
            "PREFETCH_FACTOR": str(args.prefetch_factor),
            "MAX_VIS_BATCHES": str(args.max_visual_batches),
            "SEED": str(seed),
            "OMP_THREADS": str(args.omp_threads),
            "MKL_THREADS": str(args.mkl_threads),
            "OPENBLAS_THREADS": str(args.openblas_threads),
            "NUMEXPR_THREADS": str(args.numexpr_threads),
            "MALLOC_ARENA_MAX": str(args.malloc_arena_max),
            "PYTORCH_CUDA_ALLOC_CONF": str(args.pytorch_cuda_alloc_conf),
            "PAST_T": spec.env.get("PAST_T", "32"),
            "FUTURE_T": spec.env.get("FUTURE_T", "32"),
            "PREDICTOR": spec.env.get("PREDICTOR", "thinkjepa"),
            "BACKBONE": spec.env.get("BACKBONE", "vjepa"),
            "JOINT_PRED": spec.env.get("JOINT_PRED", "1"),
            "USE_NPZ_CACHE": spec.env.get("USE_NPZ_CACHE", "1"),
            "SKIP_VJEPA": spec.env.get("SKIP_VJEPA", "1"),
            "THINKJEPA_VLM_SOURCE": spec.env.get("THINKJEPA_VLM_SOURCE", "both"),
            "THINKJEPA_VLM_LAYER_SELECTOR": spec.env.get("THINKJEPA_VLM_LAYER_SELECTOR", "last"),
            "THINKJEPA_VLM_LAYER_INDEX": spec.env.get("THINKJEPA_VLM_LAYER_INDEX", "-1"),
            "CAMERA_MODE": str(getattr(args, "camera_mode", "auto")).lower(),
            "ZERO_VISUAL_INPUT": spec.env.get("ZERO_VISUAL_INPUT", "0"),
            "HF_HOME": os.environ.get("HF_HOME", ""),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", ""),
            "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
            "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE", ""),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE", ""),
            "PRELOAD_CACHE_TO_MEMORY": "1"
            if bool(getattr(args, "preload_cache_to_memory", False))
            else "0",
            "AUTO_RESUME": "1" if args.resume else "0",
        }
    )
    env["PIN_MEMORY"] = "1" if args.pin_memory else "0"
    env["PERSISTENT_WORKERS"] = "1" if args.persistent_workers else "0"
    env["DDP_FIND_UNUSED_PARAMETERS"] = "1" if args.ddp_find_unused_parameters else "0"
    if args.no_amp:
        env["NO_AMP"] = "1"
    if args.skip_nonfinite_loss:
        env["SKIP_NONFINITE_LOSS"] = "1"
    if spec.env.get("NO_THINKJEPA_CACHE_EXT", "0") == "1":
        env["NO_THINKJEPA_CACHE_EXT"] = "1"
    return env


def run_training_process(train_script: Path, env: Dict[str, str], cwd: Path) -> None:
    subprocess.run(["bash", str(train_script)], cwd=cwd, env=env, check=True)


def load_json_dict(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_dict(path: Path, payload: Dict[str, Any]) -> None:
    ensure_results_directory(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def is_training_run_complete(metrics_path: Path, expected_epochs: int) -> bool:
    if not metrics_path.exists():
        return False
    try:
        logs = load_json_dict(metrics_path)
    except Exception:
        return False
    epochs = logs.get("epochs", [])
    if not epochs:
        return False
    max_epoch = max(int(e.get("epoch", -1)) for e in epochs)
    return max_epoch >= int(expected_epochs)


def format_mean_and_std(values: List[float]) -> str:
    if not values:
        return "NA"
    mean = float(np.mean(values))
    std = float(np.std(values))
    return f"{mean:.4f} ± {std:.4f}"


def parse_best_epoch_metrics_from_file(metrics_path: Path) -> Dict[str, float]:
    logs = load_json_dict(metrics_path)
    epochs = logs.get("epochs", [])
    if not epochs:
        return {
            "ADE": float("nan"),
            "FDE": float("nan"),
            "Accuracy": float("nan"),
            "vjepa_feat_distance": float("nan"),
            "pred_loss": float("nan"),
            "latent_smooth_l1": float("nan"),
            "latent_cosine_distance": float("nan"),
            "best_epoch": 0.0,
        }
    best_epoch = int(logs.get("best", {}).get("epoch", epochs[-1]["epoch"]))
    row = None
    for e in epochs:
        if int(e.get("epoch", -1)) == best_epoch:
            row = e
            break
    if row is None:
        row = epochs[-1]
    return {
        "ADE": float(row["val_avg_dist"]),
        "FDE": float(row["val_final_dist"]),
        "Accuracy": float(row["val_acc"]),
        "vjepa_feat_distance": float(row.get("val_pred_latent_dist", float("nan"))),
        "pred_loss": float(row.get("val_pred_loss", float("nan"))),
        "latent_smooth_l1": float(
            row.get("val_pred_latent_smooth_l1", float("nan"))
        ),
        "latent_cosine_distance": float(
            row.get("val_pred_latent_cosine_distance", float("nan"))
        ),
        "best_epoch": float(best_epoch),
    }


def compute_trajectory_statistics(pred: torch.Tensor, target: torch.Tensor, thr: float = 0.05) -> Dict[str, float]:
    pred = pred.float()
    target = target.float()
    finite = torch.isfinite(pred).all(dim=-1) & torch.isfinite(target).all(dim=-1)
    err = torch.linalg.norm(
        torch.nan_to_num(pred) - torch.nan_to_num(target), dim=-1
    )  # [B,T,J]
    valid = finite.float()
    ade_per = (err * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1.0)
    fde_valid = valid[:, -1, :]
    fde_per = (err[:, -1, :] * fde_valid).sum(dim=1) / fde_valid.sum(dim=1).clamp_min(
        1.0
    )
    return {
        "ade_sum": float(ade_per.sum().item()),
        "fde_sum": float(fde_per.sum().item()),
        "sample_count": int(pred.shape[0]),
        "acc_correct": float(((err < thr) & finite).float().sum().item()),
        "acc_total": int(finite.sum().item()),
    }


def compute_latent_statistics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = pred.float()
    target = target.float()
    finite = torch.isfinite(pred).all(dim=-1) & torch.isfinite(target).all(dim=-1)
    pred = torch.nan_to_num(pred)
    target = torch.nan_to_num(target)
    dist = torch.linalg.norm(pred - target, dim=-1)  # [B,T,P]
    smooth_l1 = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
    cosine_dist = 1.0 - F.cosine_similarity(pred, target, dim=-1, eps=1e-8)
    valid = finite.float()
    return {
        "dist_sum": float((dist * valid).sum().item()),
        "smooth_l1_sum": float((smooth_l1 * valid).sum().item()),
        "cosine_dist_sum": float((cosine_dist * valid).sum().item()),
        "dist_total": int(finite.sum().item()),
        "invalid_total": int((~finite).sum().item()),
    }


def aggregate_rollout_statistics(stats: Dict[int, Dict[str, float]]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for h, s in stats.items():
        ade = s["ade_sum"] / max(s["sample_count"], 1)
        fde = s["fde_sum"] / max(s["sample_count"], 1)
        acc = s["acc_correct"] / max(s["acc_total"], 1)
        latent_dist = s["dist_sum"] / max(s["dist_total"], 1)
        latent_smooth_l1 = s["smooth_l1_sum"] / max(s["dist_total"], 1)
        latent_cosine_dist = s["cosine_dist_sum"] / max(s["dist_total"], 1)
        latent_invalid_ratio = s["invalid_total"] / max(
            s["dist_total"] + s["invalid_total"], 1
        )
        out[h] = {
            "ADE": ade,
            "FDE": fde,
            "Accuracy": acc,
            "vjepa_feat_distance": latent_dist,
            "latent_smooth_l1": latent_smooth_l1,
            "latent_cosine_distance": latent_cosine_dist,
            "latent_non_finite_ratio": latent_invalid_ratio,
        }
    return out


def initialize_rollout_curve_totals(max_h: int) -> Dict[str, np.ndarray]:
    return {
        "traj_err_sum": np.zeros(max_h, dtype=np.float64),
        "traj_err_count": np.zeros(max_h, dtype=np.float64),
        "traj_acc_correct": np.zeros(max_h, dtype=np.float64),
        "traj_acc_total": np.zeros(max_h, dtype=np.float64),
        "latent_l2_sum": np.zeros(max_h, dtype=np.float64),
        "latent_smooth_l1_sum": np.zeros(max_h, dtype=np.float64),
        "latent_cosine_sum": np.zeros(max_h, dtype=np.float64),
        "latent_valid_count": np.zeros(max_h, dtype=np.float64),
        "latent_total_count": np.zeros(max_h, dtype=np.float64),
        "latent_invalid_count": np.zeros(max_h, dtype=np.float64),
        "latent_collapse_count": np.zeros(max_h, dtype=np.float64),
        "latent_error_explosion_count": np.zeros(max_h, dtype=np.float64),
        "latent_unstable_count": np.zeros(max_h, dtype=np.float64),
    }


def compute_rollout_curve_statistics(
    pred_world_seq: torch.Tensor,
    gt_world_seq: torch.Tensor,
    pred_lat_seq: torch.Tensor,
    gt_lat_seq: torch.Tensor,
    *,
    thr: float = 0.05,
    collapse_ratio: float = 0.1,
    explosion_factor: float = 3.0,
) -> Dict[str, np.ndarray]:
    pred_world_seq = pred_world_seq.float()
    gt_world_seq = gt_world_seq.float()
    world_finite = torch.isfinite(pred_world_seq).all(dim=-1) & torch.isfinite(
        gt_world_seq
    ).all(dim=-1)
    world_err = torch.linalg.norm(
        torch.nan_to_num(pred_world_seq) - torch.nan_to_num(gt_world_seq), dim=-1
    )  # [B,T,J]
    world_valid = world_finite.float()

    pred_lat_seq = pred_lat_seq.float()
    gt_lat_seq = gt_lat_seq.float()
    lat_finite = torch.isfinite(pred_lat_seq).all(dim=-1) & torch.isfinite(
        gt_lat_seq
    ).all(dim=-1)  # [B,T,P]
    pred_lat_clean = torch.nan_to_num(pred_lat_seq)
    gt_lat_clean = torch.nan_to_num(gt_lat_seq)
    lat_l2 = torch.linalg.norm(pred_lat_clean - gt_lat_clean, dim=-1)
    lat_smooth_l1 = F.smooth_l1_loss(
        pred_lat_clean, gt_lat_clean, reduction="none"
    ).mean(dim=-1)
    lat_cosine = 1.0 - F.cosine_similarity(
        pred_lat_clean, gt_lat_clean, dim=-1, eps=1e-8
    )
    lat_valid = lat_finite.float()

    pred_std = pred_lat_clean.std(dim=-1, unbiased=False)
    gt_std = gt_lat_clean.std(dim=-1, unbiased=False)
    collapse_thr = torch.maximum(
        gt_std * collapse_ratio, torch.full_like(gt_std, 1e-6)
    )
    collapse_mask = (pred_std < collapse_thr) & lat_finite

    baseline_err = lat_l2[:, :1, :].clamp_min(1e-6)
    error_explosion_mask = (lat_l2 > (baseline_err * explosion_factor)) & lat_finite
    invalid_mask = ~lat_finite
    unstable_mask = invalid_mask | collapse_mask | error_explosion_mask

    return {
        "traj_err_sum": world_err.mul(world_valid).sum(dim=(0, 2)).cpu().numpy(),
        "traj_err_count": world_valid.sum(dim=(0, 2)).cpu().numpy(),
        "traj_acc_correct": ((world_err < thr) & world_finite)
        .float()
        .sum(dim=(0, 2))
        .cpu()
        .numpy(),
        "traj_acc_total": world_valid.sum(dim=(0, 2)).cpu().numpy(),
        "latent_l2_sum": lat_l2.mul(lat_valid).sum(dim=(0, 2)).cpu().numpy(),
        "latent_smooth_l1_sum": lat_smooth_l1.mul(lat_valid)
        .sum(dim=(0, 2))
        .cpu()
        .numpy(),
        "latent_cosine_sum": lat_cosine.mul(lat_valid).sum(dim=(0, 2)).cpu().numpy(),
        "latent_valid_count": lat_valid.sum(dim=(0, 2)).cpu().numpy(),
        "latent_total_count": np.full(
            lat_l2.shape[1], lat_l2.shape[0] * lat_l2.shape[2], dtype=np.float64
        ),
        "latent_invalid_count": invalid_mask.float().sum(dim=(0, 2)).cpu().numpy(),
        "latent_collapse_count": collapse_mask.float().sum(dim=(0, 2)).cpu().numpy(),
        "latent_error_explosion_count": error_explosion_mask.float()
        .sum(dim=(0, 2))
        .cpu()
        .numpy(),
        "latent_unstable_count": unstable_mask.float().sum(dim=(0, 2)).cpu().numpy(),
    }


def _normalize_curve_with_counts(sum_arr: np.ndarray, count_arr: np.ndarray) -> np.ndarray:
    out = np.full(sum_arr.shape, np.nan, dtype=np.float64)
    valid = count_arr > 0
    out[valid] = sum_arr[valid] / count_arr[valid]
    return out


def _build_prefix_curve(sum_arr: np.ndarray, count_arr: np.ndarray) -> np.ndarray:
    sum_cum = np.cumsum(sum_arr)
    count_cum = np.cumsum(count_arr)
    out = np.full(sum_arr.shape, np.nan, dtype=np.float64)
    valid = count_cum > 0
    out[valid] = sum_cum[valid] / count_cum[valid]
    return out


def _compute_curve_area(values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if not np.any(mask):
        return float("nan")
    xs = np.arange(1, len(values) + 1, dtype=np.float64)[mask]
    ys = values[mask]
    if ys.size == 1:
        return float(ys[0])
    trapz_fn = getattr(np, "trapezoid", np.trapz)
    return float(trapz_fn(ys, xs))


def _compute_curve_slope(values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    xs = np.arange(1, len(values) + 1, dtype=np.float64)[mask]
    ys = values[mask]
    return float(np.polyfit(xs, ys, 1)[0])


def aggregate_rollout_curve_statistics(curve_tot: Dict[str, np.ndarray]) -> Dict[str, Any]:
    traj_ade_step = _normalize_curve_with_counts(curve_tot["traj_err_sum"], curve_tot["traj_err_count"])
    traj_ade_prefix = _build_prefix_curve(
        curve_tot["traj_err_sum"], curve_tot["traj_err_count"]
    )
    traj_acc_prefix = _build_prefix_curve(
        curve_tot["traj_acc_correct"], curve_tot["traj_acc_total"]
    )
    latent_l2_step = _normalize_curve_with_counts(
        curve_tot["latent_l2_sum"], curve_tot["latent_valid_count"]
    )
    latent_l2_prefix = _build_prefix_curve(
        curve_tot["latent_l2_sum"], curve_tot["latent_valid_count"]
    )
    latent_s1_step = _normalize_curve_with_counts(
        curve_tot["latent_smooth_l1_sum"], curve_tot["latent_valid_count"]
    )
    latent_s1_prefix = _build_prefix_curve(
        curve_tot["latent_smooth_l1_sum"], curve_tot["latent_valid_count"]
    )
    latent_cos_step = _normalize_curve_with_counts(
        curve_tot["latent_cosine_sum"], curve_tot["latent_valid_count"]
    )
    latent_cos_prefix = _build_prefix_curve(
        curve_tot["latent_cosine_sum"], curve_tot["latent_valid_count"]
    )
    non_finite_ratio_step = _normalize_curve_with_counts(
        curve_tot["latent_invalid_count"], curve_tot["latent_total_count"]
    )
    collapse_ratio_step = _normalize_curve_with_counts(
        curve_tot["latent_collapse_count"], curve_tot["latent_total_count"]
    )
    error_explosion_ratio_step = _normalize_curve_with_counts(
        curve_tot["latent_error_explosion_count"], curve_tot["latent_total_count"]
    )
    unstable_ratio_step = _normalize_curve_with_counts(
        curve_tot["latent_unstable_count"], curve_tot["latent_total_count"]
    )

    stability = {
        "traj_ade_auc": _compute_curve_area(traj_ade_prefix),
        "traj_ade_slope": _compute_curve_slope(traj_ade_prefix),
        "latent_l2_auc": _compute_curve_area(latent_l2_prefix),
        "latent_l2_slope": _compute_curve_slope(latent_l2_prefix),
        "latent_smooth_l1_auc": _compute_curve_area(latent_s1_prefix),
        "latent_smooth_l1_slope": _compute_curve_slope(latent_s1_prefix),
        "latent_cosine_distance_auc": _compute_curve_area(latent_cos_prefix),
        "latent_cosine_distance_slope": _compute_curve_slope(latent_cos_prefix),
        "latent_drift_auc": _compute_curve_area(latent_l2_prefix),
        "latent_drift_slope": _compute_curve_slope(latent_l2_prefix),
        "latent_non_finite_ratio": float(
            curve_tot["latent_invalid_count"].sum()
            / max(curve_tot["latent_total_count"].sum(), 1.0)
        ),
        "collapsed_embedding_ratio": float(
            curve_tot["latent_collapse_count"].sum()
            / max(curve_tot["latent_total_count"].sum(), 1.0)
        ),
        "error_explosion_ratio": float(
            curve_tot["latent_error_explosion_count"].sum()
            / max(curve_tot["latent_total_count"].sum(), 1.0)
        ),
        "unstable_ratio": float(
            curve_tot["latent_unstable_count"].sum()
            / max(curve_tot["latent_total_count"].sum(), 1.0)
        ),
    }
    return {
        "steps": list(range(1, len(traj_ade_step) + 1)),
        "traj_ade_step": traj_ade_step.tolist(),
        "traj_ade_prefix": traj_ade_prefix.tolist(),
        "traj_accuracy_prefix": traj_acc_prefix.tolist(),
        "latent_l2_step": latent_l2_step.tolist(),
        "latent_l2_prefix": latent_l2_prefix.tolist(),
        "latent_smooth_l1_step": latent_s1_step.tolist(),
        "latent_smooth_l1_prefix": latent_s1_prefix.tolist(),
        "latent_cosine_distance_step": latent_cos_step.tolist(),
        "latent_cosine_distance_prefix": latent_cos_prefix.tolist(),
        "latent_non_finite_ratio_step": non_finite_ratio_step.tolist(),
        "collapsed_embedding_ratio_step": collapse_ratio_step.tolist(),
        "error_explosion_ratio_step": error_explosion_ratio_step.tolist(),
        "unstable_ratio_step": unstable_ratio_step.tolist(),
        "stability": stability,
    }


def _env_string_to_flag(env: Dict[str, str], key: str, default: str = "0") -> bool:
    return str(env.get(key, default)).lower() in {"1", "true", "yes", "on"}


def build_runtime_namespace(args: argparse.Namespace, env: Dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        preload_cache_to_memory=bool(getattr(args, "preload_cache_to_memory", False)),
        predictor=env.get("PREDICTOR", "thinkjepa"),
        backbone=env.get("BACKBONE", "vjepa"),
        trajmode="traj",
        past_T=int(env.get("PAST_T", "4")),
        future_T=int(env.get("FUTURE_T", "4")),
        skip_vjepa=_env_string_to_flag(env, "SKIP_VJEPA", "1"),
        thinkjepa_use_cache_ext=not _env_string_to_flag(env, "NO_THINKJEPA_CACHE_EXT", "0"),
        vlm_pad_old_to=480,
        vlm_pad_new_to=15,
        thinkjepa_vlm_old_dim=0,
        thinkjepa_vlm_new_dim=0,
        thinkjepa_vlm_source=env.get("THINKJEPA_VLM_SOURCE", "both"),
        thinkjepa_vlm_layer_selector=env.get("THINKJEPA_VLM_LAYER_SELECTOR", "last"),
        thinkjepa_vlm_layer_index=int(env.get("THINKJEPA_VLM_LAYER_INDEX", "-1")),
        zero_visual_input=_env_string_to_flag(env, "ZERO_VISUAL_INPUT", "0"),
        joint_pred=_env_string_to_flag(env, "JOINT_PRED", "1"),
        ref_mode="none",
        no_amp=_env_string_to_flag(env, "NO_AMP", "0"),
    )


def build_predictor_from_checkpoint(
    runtime_args: SimpleNamespace,
    predictor_state: Dict[str, Any],
    feats_shape: torch.Size,
    extras: Optional[Dict[str, Any]],
    device: torch.device,
) -> Optional[torch.nn.Module]:
    mods = load_runtime_modules()
    train_mod = mods["train"]
    thinkjepa_mod = mods["thinkjepa"]
    predictor_mod = mods["tiny_pred"]
    official_pred_mod = mods["official_pred"]
    predictor_name = str(runtime_args.predictor).lower()
    if predictor_name == "none" or predictor_state is None:
        return None

    mask_token_indices: List[int] = []
    for key in predictor_state.keys():
        if not key.startswith("mask_tokens."):
            continue
        parts = key.split(".")
        if len(parts) < 2:
            continue
        try:
            mask_token_indices.append(int(parts[1]))
        except ValueError:
            continue
    num_mask_tokens = max(mask_token_indices) + 1 if mask_token_indices else 1

    _, _, P, D = feats_shape
    total_frames = int(runtime_args.past_T) + int(runtime_args.future_T)
    predictor = None

    if predictor_name == "tiny":
        predictor = predictor_mod.PatchwiseAutoregressiveRolloutHead(
            in_dim=D, model_dim=384, depth=2, nhead=6, dropout=0.1
        )
    elif predictor_name == "official":
        predictor = official_pred_mod.VisionTransformerPredictor(
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
            norm_layer=partial(train_mod.nn.LayerNorm, eps=1e-6),
            init_std=0.02,
            uniform_power=False,
            use_mask_tokens=True,
            num_mask_tokens=num_mask_tokens,
            zero_init_mask_tokens=True,
            use_silu=False,
            wide_silu=True,
            use_activation_checkpointing=False,
            return_all_tokens=False,
            chop_last_n_tokens=0,
            use_rope=True,
        )
    elif predictor_name == "thinkjepa":
        old_dim = train_mod._feat_dim_from_extras(extras, "vlm_old")
        new_dim = train_mod._feat_dim_from_extras(extras, "vlm_new")
        old_dim = int(old_dim) if old_dim is not None else 2048
        new_dim = int(new_dim) if new_dim is not None else 2048
        predictor = thinkjepa_mod.CortexGuidedVideoPredictor(
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
            norm_layer=partial(train_mod.nn.LayerNorm, eps=1e-6),
            init_std=0.02,
            uniform_power=False,
            use_mask_tokens=True,
            num_mask_tokens=num_mask_tokens,
            zero_init_mask_tokens=True,
            use_silu=False,
            wide_silu=True,
            use_activation_checkpointing=False,
            return_all_tokens=False,
            chop_last_n_tokens=0,
            use_rope=True,
            vlm_old_dim=old_dim,
            vlm_new_dim=new_dim,
        )
    else:
        raise ValueError(f"Unsupported predictor for rollout: {predictor_name}")

    predictor.load_state_dict(predictor_state, strict=True)
    predictor.to(device)
    predictor.eval()
    return predictor


def predict_future_latent_rollout(
    predictor: torch.nn.Module,
    runtime_args: SimpleNamespace,
    context_feats: torch.Tensor,
    extras: Optional[Dict[str, Any]],
) -> torch.Tensor:
    train_mod = load_runtime_modules()["train"]
    predictor_name = str(runtime_args.predictor).lower()
    B, ctx_len, P, D = context_feats.shape
    future_t = int(runtime_args.future_T)

    if predictor_name == "tiny":
        out = predictor(context_feats)
        return out[:, -future_t:, ...].contiguous()

    if predictor_name in {"official", "thinkjepa"}:
        x_seq = train_mod._flatten_tp(context_feats)
        idx_ctx_1d = train_mod._make_tp_indices(P, 0, ctx_len)
        idx_tgt_1d = train_mod._make_tp_indices(P, ctx_len, ctx_len + future_t)
        masks_x = train_mod._repeat_index_for_batch(idx_ctx_1d.long(), B, x_seq.device)
        masks_y = train_mod._repeat_index_for_batch(idx_tgt_1d.long(), B, x_seq.device)
        x_ctxt = x_seq.gather(dim=1, index=masks_x.unsqueeze(-1).expand(-1, -1, D))
        if predictor_name == "official":
            y_future_seq = predictor(x_ctxt, masks_x, masks_y)
        else:
            ext = train_mod._build_thinkjepa_ext_from_extras(extras, runtime_args, x_seq.device)
            y_future_seq = predictor(x_ctxt, masks_x, masks_y, ext=ext)
        return y_future_seq.view(B, future_t, P, D).contiguous()

    raise ValueError(f"Unsupported predictor for rollout: {predictor_name}")


def evaluate_latent_rollout(
    args: argparse.Namespace,
    spec: ExperimentSpec,
    seed_dir: Path,
    env: Dict[str, str],
) -> Dict[str, Any]:
    mods = load_runtime_modules()
    train_mod = mods["train"]
    models_mod = mods["models"]
    out_path = seed_dir / "rollout_metrics.json"
    if args.resume and out_path.exists():
        cached = load_json_dict(out_path)
        if "curves" in cached and "stability" in cached:
            artifact_path = str(
                cached.get("artifacts", {}).get("rollout_curve_plot", "")
            ).strip()
            if artifact_path and Path(artifact_path).exists():
                return cached
            curve_plot = write_rollout_curve_figure(
                seed_dir / "rollout_curves.png",
                cached["curves"],
                title=f"{spec.name} seed={seed_dir.name}",
            )
            cached.setdefault("artifacts", {})["rollout_curve_plot"] = (
                str(curve_plot) if curve_plot is not None else ""
            )
            write_json_dict(out_path, cached)
            return cached

    runtime_args = build_runtime_namespace(args, env)
    ckpt_path = seed_dir / "ckpt_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing best checkpoint for rollout eval: {ckpt_path}")

    device_str = args.gpu_list.split(",")[0].strip()
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_str}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    train_loader, test_loader = train_mod.build_egodex_dataloaders(
        args.data_dir,
        train_mod.QUERY_TFS,
        if_return_path=True,
        train_batch=args.train_batch_size,
        test_batch=args.eval_batch_size,
        train_manifest=args.train_manifest,
        test_manifest=args.test_manifest,
        use_npz_cache=True,
        cache_dir=runtime_args.cache_dir,
        train_ratio=0.9,
        split_seed=42,
        num_workers=args.num_workers,
        fast_index_when_full_scan=True,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        preload_to_memory=bool(runtime_args.preload_cache_to_memory),
    )
    del train_loader

    cls_model = models_mod.TrajectoryReadoutMLP(
        downsample=bool(runtime_args.joint_pred)
    ).to(device)
    cls_model.load_state_dict(ckpt["cls_model"], strict=True)
    cls_model.eval()

    need_cache_index = bool(runtime_args.skip_vjepa) or (
        str(runtime_args.predictor).lower() == "thinkjepa" and bool(runtime_args.thinkjepa_use_cache_ext)
    )
    cache_index = (
        train_mod._build_vlm_cache_index(runtime_args.cache_dir)
        if need_cache_index
        else {}
    )
    preloaded_cache_archives = (
        train_mod._preload_npz_archives(runtime_args.cache_dir)
        if need_cache_index and bool(runtime_args.preload_cache_to_memory)
        else None
    )
    cache_path_cache: Dict[str, Optional[str]] = {}

    predictor = None
    rollout_tot = {
        h: {
            "ade_sum": 0.0,
            "fde_sum": 0.0,
            "sample_count": 0,
            "acc_correct": 0.0,
            "acc_total": 0,
            "dist_sum": 0.0,
            "smooth_l1_sum": 0.0,
            "cosine_dist_sum": 0.0,
            "dist_total": 0,
            "invalid_total": 0,
        }
        for h in HORIZONS
    }
    curve_tot = initialize_rollout_curve_totals(max(HORIZONS))

    with torch.no_grad():
        for batch in test_loader:
            if batch is None or len(batch) < 11:
                continue

            (
                xyz_cam,
                _R_cam,
                xyz_world,
                _R_world,
                _tfs_in_cam,
                _tfs,
                cam_ext,
                _cam_int,
                _img,
                _lang_instruct,
                _confs,
            ) = batch[:11]
            extras, paths = train_mod._parse_batch_extras_and_paths(batch)

            xyz_cam = xyz_cam.to(device, non_blocking=True)
            xyz_world = xyz_world.to(device, non_blocking=True)
            cam_ext = cam_ext.to(device, non_blocking=True)

            if (
                extras is not None
                and isinstance(extras, dict)
                and extras.get("vjepa_feats", None) is not None
            ):
                out_feats_full = extras["vjepa_feats"].to(device=device, dtype=torch.float32)
            else:
                out_feats_full = train_mod._try_load_cached_vjepa_feats(
                    paths=paths,
                    args=runtime_args,
                    device=device,
                    cache_index=cache_index,
                    path_cache=cache_path_cache,
                    preloaded_archives=preloaded_cache_archives,
                )
                if out_feats_full is None:
                    raise RuntimeError(
                        "Rollout evaluation requires cached vjepa_feats; online V-JEPA path is intentionally not used here."
                    )

            Tall = int(xyz_world.shape[1])
            needed_frames = int(runtime_args.past_T) + max(HORIZONS)
            if Tall < needed_frames:
                continue

            if str(runtime_args.predictor).lower() == "thinkjepa":
                extras = train_mod._ensure_thinkjepa_extras(
                    extras=extras,
                    paths=paths,
                    args=runtime_args,
                    device=device,
                    cache_index=cache_index,
                    path_cache=cache_path_cache,
                )

            feats_gt_full = out_feats_full.contiguous()
            feats_eval_full = (
                torch.zeros_like(feats_gt_full)
                if bool(runtime_args.zero_visual_input)
                else feats_gt_full
            )

            if predictor is None:
                predictor = build_predictor_from_checkpoint(
                    runtime_args=runtime_args,
                    predictor_state=ckpt.get("predictor"),
                    feats_shape=feats_gt_full.shape,
                    extras=extras,
                    device=device,
                )
                if predictor is None:
                    raise RuntimeError(
                        "Rollout evaluation expects a trained predictor checkpoint; got none."
                    )

            ctx = feats_eval_full[:, : runtime_args.past_T, ...].contiguous()
            pred_world_steps: List[torch.Tensor] = []
            gt_world_steps: List[torch.Tensor] = []
            pred_lat_steps: List[torch.Tensor] = []
            gt_lat_steps: List[torch.Tensor] = []

            n_steps = max(HORIZONS) // int(runtime_args.future_T)
            for step_idx in range(n_steps):
                gt_start = int(runtime_args.past_T) + step_idx * int(runtime_args.future_T)
                gt_end = gt_start + int(runtime_args.future_T)
                y_future = predict_future_latent_rollout(
                    predictor=predictor,
                    runtime_args=runtime_args,
                    context_feats=ctx,
                    extras=extras,
                )
                feats_task_in = y_future.detach()
                if bool(runtime_args.joint_pred):
                    feats_task_in = torch.cat([ctx, feats_task_in], dim=1).contiguous()

                xyz_cam_slice = xyz_cam[:, gt_start:gt_end, ...].contiguous()
                cam_ext_slice = cam_ext[:, gt_start:gt_end, ...].contiguous()
                right_ref_cam = xyz_cam_slice[..., train_mod.right_idx, :]
                left_ref_cam = xyz_cam_slice[..., train_mod.left_idx, :]

                pred_cam = (
                    train_mod.predict_trajectory_from_latents(cls_model, feats_task_in)
                    .view(feats_task_in.shape[0], int(runtime_args.future_T), xyz_world.shape[2], 3)
                    .contiguous()
                )
                if runtime_args.ref_mode == "perhand":
                    pred_cam = train_mod.add_ref_per_hand(pred_cam, right_ref_cam, left_ref_cam)
                else:
                    ref_offset_cam = None
                    if runtime_args.ref_mode == "right":
                        ref_offset_cam = right_ref_cam
                    elif runtime_args.ref_mode == "left":
                        ref_offset_cam = left_ref_cam
                    elif runtime_args.ref_mode == "both":
                        ref_offset_cam = right_ref_cam + left_ref_cam
                    elif runtime_args.ref_mode == "avg":
                        ref_offset_cam = 0.5 * (right_ref_cam + left_ref_cam)
                    if ref_offset_cam is not None:
                        pred_cam = pred_cam + ref_offset_cam.unsqueeze(2)

                pred_world = train_mod.cam_to_world_points(pred_cam, cam_ext_slice)
                gt_world = xyz_world[:, gt_start:gt_end, ...].contiguous()
                gt_lat = feats_gt_full[:, gt_start:gt_end, ...].contiguous()

                pred_world_steps.append(pred_world)
                gt_world_steps.append(gt_world)
                pred_lat_steps.append(y_future.detach())
                gt_lat_steps.append(gt_lat)
                ctx = y_future.detach()

            pred_world_seq = torch.cat(pred_world_steps, dim=1)[:, : max(HORIZONS), ...]
            gt_world_seq = torch.cat(gt_world_steps, dim=1)[:, : max(HORIZONS), ...]
            pred_lat_seq = torch.cat(pred_lat_steps, dim=1)[:, : max(HORIZONS), ...]
            gt_lat_seq = torch.cat(gt_lat_steps, dim=1)[:, : max(HORIZONS), ...]
            curve_stats = compute_rollout_curve_statistics(
                pred_world_seq=pred_world_seq,
                gt_world_seq=gt_world_seq,
                pred_lat_seq=pred_lat_seq,
                gt_lat_seq=gt_lat_seq,
            )
            for key, arr in curve_stats.items():
                curve_tot[key] += arr

            for h in HORIZONS:
                traj_stats = compute_trajectory_statistics(
                    pred_world_seq[:, :h, ...], gt_world_seq[:, :h, ...]
                )
                lat_stats = compute_latent_statistics(
                    pred_lat_seq[:, :h, ...], gt_lat_seq[:, :h, ...]
                )
                for k, v in traj_stats.items():
                    rollout_tot[h][k] += v
                for k, v in lat_stats.items():
                    rollout_tot[h][k] += v

    aggregated = {str(k): v for k, v in aggregate_rollout_statistics(rollout_tot).items()}
    curves = aggregate_rollout_curve_statistics(curve_tot)
    curve_plot = write_rollout_curve_figure(
        seed_dir / "rollout_curves.png", curves, title=f"{spec.name} seed={seed_dir.name}"
    )
    payload = {
        "horizons": aggregated,
        "curves": curves,
        "stability": curves["stability"],
        "artifacts": {
            "rollout_curve_plot": str(curve_plot) if curve_plot is not None else ""
        },
        "checkpoint": str(ckpt_path),
    }
    write_json_dict(out_path, payload)
    return payload


def aggregate_seed_statistics(seed_metrics: List[Dict[str, Any]], display_video: str) -> Dict[str, Any]:
    keys = [
        "ADE",
        "FDE",
        "Accuracy",
        "vjepa_feat_distance",
        "latent_smooth_l1",
        "latent_cosine_distance",
    ]
    agg = {"video": display_video, "num_seeds": len(seed_metrics), "seeds": seed_metrics}
    for k in keys:
        vals = [float(x[k]) for x in seed_metrics if x.get(k) is not None and not math.isnan(float(x[k]))]
        agg[k] = {
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals)) if vals else None,
        }
    return agg


def aggregate_rollout_seed_statistics(seed_metrics: List[Dict[str, Any]], display_video: str) -> Dict[str, Any]:
    agg: Dict[str, Any] = {"video": display_video, "num_seeds": len(seed_metrics), "seeds": seed_metrics}
    for h in HORIZONS:
        key = str(h)
        agg[key] = {}
        for metric in [
            "ADE",
            "FDE",
            "Accuracy",
            "vjepa_feat_distance",
            "latent_smooth_l1",
            "latent_cosine_distance",
            "latent_non_finite_ratio",
        ]:
            vals = [float(x["horizons"][key][metric]) for x in seed_metrics]
            agg[key][metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
    curve_keys = [
        "traj_ade_step",
        "traj_ade_prefix",
        "traj_accuracy_prefix",
        "latent_l2_step",
        "latent_l2_prefix",
        "latent_smooth_l1_step",
        "latent_smooth_l1_prefix",
        "latent_cosine_distance_step",
        "latent_cosine_distance_prefix",
        "latent_non_finite_ratio_step",
        "collapsed_embedding_ratio_step",
        "error_explosion_ratio_step",
        "unstable_ratio_step",
    ]
    agg["curves"] = {"steps": seed_metrics[0]["curves"]["steps"] if seed_metrics else []}
    for key in curve_keys:
        arr = np.asarray([m["curves"][key] for m in seed_metrics], dtype=np.float64)
        agg["curves"][key] = {
            "mean": np.nanmean(arr, axis=0).tolist(),
            "std": np.nanstd(arr, axis=0).tolist(),
        }
    stability_keys = [
        "traj_ade_auc",
        "traj_ade_slope",
        "latent_l2_auc",
        "latent_l2_slope",
        "latent_smooth_l1_auc",
        "latent_smooth_l1_slope",
        "latent_cosine_distance_auc",
        "latent_cosine_distance_slope",
        "latent_drift_auc",
        "latent_drift_slope",
        "latent_non_finite_ratio",
        "collapsed_embedding_ratio",
        "error_explosion_ratio",
        "unstable_ratio",
    ]
    agg["stability"] = {}
    for key in stability_keys:
        vals = [float(m["stability"][key]) for m in seed_metrics]
        agg["stability"][key] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
        }
    return agg


def path_relative_to_results_root(path: Optional[Path], results_root: Path) -> str:
    if path is None:
        return "NA"
    try:
        return os.path.relpath(path, start=results_root)
    except Exception:
        return str(path)


def write_experiment_text_log(path: Path, text: str) -> None:
    ensure_results_directory(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


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


def _extract_curve_series(curves: Dict[str, Any], key: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
    value = curves[key]
    if isinstance(value, dict):
        mean = np.asarray(value.get("mean", []), dtype=np.float64)
        std = np.asarray(value.get("std", []), dtype=np.float64)
        return mean, std
    return np.asarray(value, dtype=np.float64), None


def write_rollout_curve_figure(path: Path, curves: Dict[str, Any], title: str) -> Optional[Path]:
    plt = maybe_import_matplotlib_pyplot()
    steps = np.asarray(curves.get("steps", []), dtype=np.float64)
    if steps.size == 0:
        return None

    def plot_curve(ax, key: str, label: str):
        mean, std = _extract_curve_series(curves, key)
        if mean.size == 0:
            return
        ax.plot(steps[: mean.size], mean, label=label, linewidth=1.8)
        if std is not None and std.size == mean.size:
            lower = mean - std
            upper = mean + std
            ax.fill_between(steps[: mean.size], lower, upper, alpha=0.18)

    path = Path(path)
    ensure_results_directory(path.parent)
    if plt is not None:
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        plot_curve(axes[0], "traj_ade_prefix", "traj_ADE_prefix")
        axes[0].set_title(f"{title}: Error Accumulation")
        axes[0].set_ylabel("Trajectory ADE")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        plot_curve(axes[1], "latent_l2_prefix", "latent_L2_prefix")
        plot_curve(axes[1], "latent_smooth_l1_prefix", "latent_SmoothL1_prefix")
        plot_curve(axes[1], "latent_cosine_distance_prefix", "latent_Cosine_prefix")
        axes[1].set_ylabel("Latent Error")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plot_curve(axes[2], "latent_non_finite_ratio_step", "non_finite_ratio")
        plot_curve(axes[2], "collapsed_embedding_ratio_step", "collapsed_ratio")
        plot_curve(axes[2], "error_explosion_ratio_step", "error_explosion_ratio")
        plot_curve(axes[2], "unstable_ratio_step", "unstable_ratio")
        axes[2].set_xlabel("Rollout step k")
        axes[2].set_ylabel("Stability Ratio")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return path

    Image, ImageDraw = maybe_import_pil_modules()
    if Image is None or ImageDraw is None:
        return None

    def draw_panel(draw, box, title_text, series_specs, ylim=None):
        x0, y0, x1, y1 = box
        draw.rectangle(box, outline="#cfcfcf", width=1)
        draw.text((x0 + 6, y0 + 6), title_text, fill="#111111")
        plot_x0 = x0 + 56
        plot_y0 = y0 + 28
        plot_x1 = x1 - 12
        plot_y1 = y1 - 28
        draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill="#444444", width=1)
        draw.line((plot_x0, plot_y0, plot_x0, plot_y1), fill="#444444", width=1)

        values = []
        for _, key, _, _ in series_specs:
            mean, _ = _extract_curve_series(curves, key)
            values.extend([float(v) for v in mean if np.isfinite(v)])
        if not values:
            draw.text((plot_x0 + 12, plot_y0 + 12), "No finite values", fill="#666666")
            return
        if ylim is None:
            vmin = min(values)
            vmax = max(values)
        else:
            vmin, vmax = ylim
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1.0

        def project(xv, yv):
            if len(steps) == 1:
                xp = 0.5
            else:
                xp = (xv - steps[0]) / max(steps[-1] - steps[0], 1)
            yp = (float(yv) - vmin) / (vmax - vmin)
            return (
                plot_x0 + xp * (plot_x1 - plot_x0),
                plot_y1 - yp * (plot_y1 - plot_y0),
            )

        legend_x = plot_x0
        for label, key, color, show_band in series_specs:
            mean, std = _extract_curve_series(curves, key)
            draw.text((legend_x, y0 + 6), label, fill=color)
            legend_x += 140
            prev = None
            for x_val, y_val in zip(steps[: mean.size], mean):
                if not np.isfinite(y_val):
                    prev = None
                    continue
                pt = project(x_val, y_val)
                if prev is not None:
                    draw.line((prev[0], prev[1], pt[0], pt[1]), fill=color, width=2)
                prev = pt
            if show_band and std is not None and std.size == mean.size:
                for x_val, y_val, s_val in zip(steps[: mean.size], mean, std):
                    if not np.isfinite(y_val) or not np.isfinite(s_val):
                        continue
                    lower = max(vmin, y_val - s_val)
                    upper = min(vmax, y_val + s_val)
                    px, py_low = project(x_val, lower)
                    _, py_high = project(x_val, upper)
                    draw.line((px, py_low, px, py_high), fill=color, width=1)

        draw.text((x0 + 6, plot_y0 - 6), f"{vmax:.3f}", fill="#666666")
        draw.text((x0 + 6, plot_y1 - 6), f"{vmin:.3f}", fill="#666666")
        draw.text((plot_x0, plot_y1 + 4), str(int(steps[0])), fill="#666666")
        draw.text((plot_x1 - 22, plot_y1 + 4), str(int(steps[-1])), fill="#666666")

    img = Image.new("RGB", (1100, 1200), "white")
    draw = ImageDraw.Draw(img)
    draw_panel(
        draw,
        (20, 20, 1080, 390),
        f"{title}: Error Accumulation",
        [("traj_ADE_prefix", "traj_ade_prefix", "#1f77b4", True)],
    )
    draw_panel(
        draw,
        (20, 415, 1080, 785),
        "Latent Error",
        [
            ("latent_L2_prefix", "latent_l2_prefix", "#1f77b4", True),
            ("latent_SmoothL1_prefix", "latent_smooth_l1_prefix", "#d62728", True),
            (
                "latent_Cosine_prefix",
                "latent_cosine_distance_prefix",
                "#2ca02c",
                True,
            ),
        ],
    )
    draw_panel(
        draw,
        (20, 810, 1080, 1180),
        "Rollout Stability",
        [
            ("non_finite_ratio", "latent_non_finite_ratio_step", "#9467bd", True),
            (
                "collapsed_ratio",
                "collapsed_embedding_ratio_step",
                "#8c564b",
                True,
            ),
            (
                "error_explosion_ratio",
                "error_explosion_ratio_step",
                "#e377c2",
                True,
            ),
            ("unstable_ratio", "unstable_ratio_step", "#ff7f0e", True),
        ],
        ylim=(0.0, 1.0),
    )
    img.save(path)
    return path


def write_markdown_summary_table(results_root: Path, section: str, rows: List[Dict[str, Any]]) -> None:
    out_path = results_root / f"summary_{section}.md"
    lines: List[str] = []
    lines.append(f"# Summary: {section}")
    lines.append("")

    if section != "rollout":
        lines.append(
            "| Model | ADE | FDE | Accuracy | vjepa_feat_distance | latent_smooth_l1 | latent_cosine_distance | Video |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for row in rows:
            lines.append(
                f"| {row['Model']} | {row['ADE']} | {row['FDE']} | {row['Accuracy']} | "
                f"{row['vjepa_feat_distance']} | {row['latent_smooth_l1']} | "
                f"{row['latent_cosine_distance']} | {row['Video']} |"
            )
    else:
        lines.append("## ADE")
        lines.append("")
        lines.append("| Model | H=4 ADE | H=8 ADE | H=16 ADE | H=32 ADE | Video |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for row in rows:
            lines.append(
                f"| {row['Model']} | {row['H4_ADE']} | {row['H8_ADE']} | {row['H16_ADE']} | {row['H32_ADE']} | {row['Video']} |"
            )
        for metric in [
            "FDE",
            "Accuracy",
            "vjepa_feat_distance",
            "latent_smooth_l1",
            "latent_cosine_distance",
            "latent_non_finite_ratio",
        ]:
            lines.append("")
            lines.append(f"## {metric}")
            lines.append("")
            lines.append(
                f"| Model | H=4 {metric} | H=8 {metric} | H=16 {metric} | H=32 {metric} | Video |"
            )
            lines.append("|---|---:|---:|---:|---:|---|")
            for row in rows:
                lines.append(
                    f"| {row['Model']} | {row[f'H4_{metric}']} | {row[f'H8_{metric}']} | {row[f'H16_{metric}']} | {row[f'H32_{metric}']} | {row['Video']} |"
                )
        lines.append("")
        lines.append("## Stability")
        lines.append("")
        lines.append(
            "| Model | latent_l2_auc | latent_l2_slope | latent_drift_slope | unstable_ratio | error_explosion_ratio | collapsed_embedding_ratio | Video |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for row in rows:
            lines.append(
                f"| {row['Model']} | {row['latent_l2_auc']} | {row['latent_l2_slope']} | "
                f"{row['latent_drift_slope']} | {row['unstable_ratio']} | "
                f"{row['error_explosion_ratio']} | {row['collapsed_embedding_ratio']} | {row['Video']} |"
            )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_huggingface_cache_dirs()
    args = parse_args()
    args.data_dir = resolve_egodex_data_reference(str(args.data_dir))
    args.cache_dir = resolve_egodex_data_reference(str(args.cache_dir))
    args.preload_cache_to_memory = resolve_suite_cache_preload_policy(args)
    results_root = args.results_root.resolve()
    ensure_results_directory(results_root)
    git_hash = get_repository_commit_hash(REPO_ROOT)

    experiments_by_section = build_suite_experiments()

    total_start = time.time()
    summary_rows: Dict[str, List[Dict[str, Any]]] = {
        "main": [],
        "ablation": [],
        "layer": [],
        "rollout": [],
    }

    selected_specs: List[ExperimentSpec] = []
    for section in ["main", "ablation", "layer", "horizon_rollout"]:
        if section not in args.sections:
            continue
        selected_specs.extend(select_experiments_for_sections(args, experiments_by_section, section))

    total_experiments = len(selected_specs) * len(args.seeds)
    progress_idx = 0

    for section in ["main", "ablation", "layer", "horizon_rollout"]:
        if section not in args.sections:
            continue
        specs_for_section = select_experiments_for_sections(args, experiments_by_section, section)
        for spec in specs_for_section:
            exp_dir = results_root / section / make_safe_experiment_dir_name(spec.name)
            ensure_results_directory(exp_dir)
            target_epochs = target_epochs_for_experiment(args, spec)

            exp_config = {
                "name": spec.name,
                "section": section,
                "env": spec.env,
                "target_epochs": target_epochs,
                "rollout": spec.rollout,
                "notes": spec.notes,
                "seeds": args.seeds,
                "git_commit": git_hash,
                "data_dir": args.data_dir,
                "cache_dir": args.cache_dir,
                "train_manifest": args.train_manifest,
                "test_manifest": args.test_manifest,
            }
            write_yaml_summary(exp_dir / "config.yaml", exp_config)

            per_seed_results: List[Dict[str, Any]] = []
            representative_video: Optional[Path] = None
            exp_start = time.time()

            for seed in args.seeds:
                progress_idx += 1
                seed_dir = exp_dir / f"seed_{seed:04d}"
                metrics_path = seed_dir / "metrics.json"
                rollout_path = seed_dir / "rollout_metrics.json"
                ensure_results_directory(seed_dir)
                env = build_training_environment(args, spec, seed, seed_dir)

                seed_cfg = {
                    "seed": seed,
                    "experiment": spec.name,
                    "section": section,
                    "env": {k: env[k] for k in sorted(env) if k in {
                        "DATA_DIR","CACHE_DIR","TRAIN_MANIFEST","TEST_MANIFEST","GPU_LIST","NPROC_PER_NODE","OUT_DIR","RESULTS_MD","LOG_FILE","OUTPUT_MP4",
                        "EPOCHS","LR","LR_PRED","NUM_WORKERS","TRAIN_BATCH_SIZE","TEST_BATCH_SIZE","PREFETCH_FACTOR","MAX_VIS_BATCHES","SEED","PAST_T","FUTURE_T","PREDICTOR","BACKBONE","JOINT_PRED",
                        "SKIP_VJEPA","THINKJEPA_VLM_SOURCE","THINKJEPA_VLM_LAYER_SELECTOR","THINKJEPA_VLM_LAYER_INDEX","CAMERA_MODE","ZERO_VISUAL_INPUT","NO_THINKJEPA_CACHE_EXT",
                        "PIN_MEMORY","PERSISTENT_WORKERS","DDP_FIND_UNUSED_PARAMETERS","OMP_THREADS","MKL_THREADS","OPENBLAS_THREADS","NUMEXPR_THREADS","MALLOC_ARENA_MAX","PYTORCH_CUDA_ALLOC_CONF","AUTO_RESUME"
                    }},
                    "git_commit": git_hash,
                }
                write_yaml_summary(seed_dir / "config.yaml", seed_cfg)

                train_complete = is_training_run_complete(metrics_path, target_epochs)
                need_train = not (args.resume and train_complete)
                need_rollout = (
                    spec.rollout
                    and not args.skip_rollout
                    and (need_train or not (args.resume and rollout_path.exists()))
                )

                print(
                    f"[{progress_idx}/{total_experiments}] section={section} exp={spec.name} seed={seed} "
                    f"epochs={target_epochs} train={'yes' if need_train else 'skip'} "
                    f"rollout={'yes' if need_rollout else 'skip'}"
                )
                if args.resume and seed_dir.exists() and (not train_complete):
                    print(
                        f"[INFO] continuing incomplete seed dir in place: {seed_dir}"
                    )

                seed_start = time.time()
                if need_train:
                    run_training_process(args.train_script, env, REPO_ROOT)
                train_runtime = time.time() - seed_start

                video = latest_video_under_directory(seed_dir / "vis")
                if video is not None:
                    symlink_or_copy_file(video, seed_dir / "visualization.mp4")
                    if representative_video is None:
                        representative_video = seed_dir / "visualization.mp4"

                one_shot_metrics = parse_best_epoch_metrics_from_file(metrics_path)
                one_shot_metrics["seed"] = seed
                one_shot_metrics["train_runtime_sec"] = train_runtime
                one_shot_metrics["video"] = path_relative_to_results_root(
                    seed_dir / "visualization.mp4" if (seed_dir / "visualization.mp4").exists() else video,
                    results_root,
                )

                if spec.rollout and not args.skip_rollout:
                    rollout_metrics = evaluate_latent_rollout(args, spec, seed_dir, env)
                    seed_total_runtime = time.time() - seed_start
                    rollout_seed_payload = {
                        "seed": seed,
                        "train_runtime_sec": train_runtime,
                        "runtime_sec": seed_total_runtime,
                        "video": one_shot_metrics["video"],
                        "horizons": {
                            str(h): rollout_metrics["horizons"][str(h)] for h in HORIZONS
                        },
                        "curves": rollout_metrics["curves"],
                        "stability": rollout_metrics["stability"],
                        "artifacts": rollout_metrics.get("artifacts", {}),
                    }
                    per_seed_results.append(rollout_seed_payload)
                else:
                    one_shot_metrics["runtime_sec"] = time.time() - seed_start
                    per_seed_results.append(one_shot_metrics)

                write_experiment_text_log(
                    exp_dir / "log.txt",
                    f"seed={seed} train_runtime_sec={train_runtime:.1f} total_runtime_sec={(time.time() - seed_start):.1f} "
                    f"metrics_path={metrics_path} video={one_shot_metrics['video']}",
                )

            exp_runtime = time.time() - exp_start
            if representative_video is not None:
                symlink_or_copy_file(representative_video, exp_dir / "visualization.mp4")
            exp_video_rel = path_relative_to_results_root(
                exp_dir / "visualization.mp4" if (exp_dir / "visualization.mp4").exists() else representative_video,
                results_root,
            )

            if spec.rollout and not args.skip_rollout:
                aggregate = aggregate_rollout_seed_statistics(per_seed_results, exp_video_rel)
                aggregate["runtime_sec"] = exp_runtime
                aggregate["git_commit"] = git_hash
                agg_curve_plot = write_rollout_curve_figure(
                    exp_dir / "rollout_curves.png",
                    aggregate["curves"],
                    title=f"{spec.name} aggregate",
                )
                aggregate["artifacts"] = {
                    "rollout_curve_plot": (
                        path_relative_to_results_root(agg_curve_plot, results_root)
                        if agg_curve_plot is not None
                        else "NA"
                    )
                }
                write_json_dict(exp_dir / "metrics.json", aggregate)
                row = {
                    "Model": spec.name,
                    "H4_ADE": format_mean_and_std([float(x["horizons"]["4"]["ADE"]) for x in per_seed_results]),
                    "H8_ADE": format_mean_and_std([float(x["horizons"]["8"]["ADE"]) for x in per_seed_results]),
                    "H16_ADE": format_mean_and_std([float(x["horizons"]["16"]["ADE"]) for x in per_seed_results]),
                    "H32_ADE": format_mean_and_std([float(x["horizons"]["32"]["ADE"]) for x in per_seed_results]),
                    "H4_FDE": format_mean_and_std([float(x["horizons"]["4"]["FDE"]) for x in per_seed_results]),
                    "H8_FDE": format_mean_and_std([float(x["horizons"]["8"]["FDE"]) for x in per_seed_results]),
                    "H16_FDE": format_mean_and_std([float(x["horizons"]["16"]["FDE"]) for x in per_seed_results]),
                    "H32_FDE": format_mean_and_std([float(x["horizons"]["32"]["FDE"]) for x in per_seed_results]),
                    "H4_Accuracy": format_mean_and_std([float(x["horizons"]["4"]["Accuracy"]) for x in per_seed_results]),
                    "H8_Accuracy": format_mean_and_std([float(x["horizons"]["8"]["Accuracy"]) for x in per_seed_results]),
                    "H16_Accuracy": format_mean_and_std([float(x["horizons"]["16"]["Accuracy"]) for x in per_seed_results]),
                    "H32_Accuracy": format_mean_and_std([float(x["horizons"]["32"]["Accuracy"]) for x in per_seed_results]),
                    "H4_vjepa_feat_distance": format_mean_and_std([float(x["horizons"]["4"]["vjepa_feat_distance"]) for x in per_seed_results]),
                    "H8_vjepa_feat_distance": format_mean_and_std([float(x["horizons"]["8"]["vjepa_feat_distance"]) for x in per_seed_results]),
                    "H16_vjepa_feat_distance": format_mean_and_std([float(x["horizons"]["16"]["vjepa_feat_distance"]) for x in per_seed_results]),
                    "H32_vjepa_feat_distance": format_mean_and_std([float(x["horizons"]["32"]["vjepa_feat_distance"]) for x in per_seed_results]),
                    "H4_latent_smooth_l1": format_mean_and_std([float(x["horizons"]["4"]["latent_smooth_l1"]) for x in per_seed_results]),
                    "H8_latent_smooth_l1": format_mean_and_std([float(x["horizons"]["8"]["latent_smooth_l1"]) for x in per_seed_results]),
                    "H16_latent_smooth_l1": format_mean_and_std([float(x["horizons"]["16"]["latent_smooth_l1"]) for x in per_seed_results]),
                    "H32_latent_smooth_l1": format_mean_and_std([float(x["horizons"]["32"]["latent_smooth_l1"]) for x in per_seed_results]),
                    "H4_latent_cosine_distance": format_mean_and_std([float(x["horizons"]["4"]["latent_cosine_distance"]) for x in per_seed_results]),
                    "H8_latent_cosine_distance": format_mean_and_std([float(x["horizons"]["8"]["latent_cosine_distance"]) for x in per_seed_results]),
                    "H16_latent_cosine_distance": format_mean_and_std([float(x["horizons"]["16"]["latent_cosine_distance"]) for x in per_seed_results]),
                    "H32_latent_cosine_distance": format_mean_and_std([float(x["horizons"]["32"]["latent_cosine_distance"]) for x in per_seed_results]),
                    "H4_latent_non_finite_ratio": format_mean_and_std([float(x["horizons"]["4"]["latent_non_finite_ratio"]) for x in per_seed_results]),
                    "H8_latent_non_finite_ratio": format_mean_and_std([float(x["horizons"]["8"]["latent_non_finite_ratio"]) for x in per_seed_results]),
                    "H16_latent_non_finite_ratio": format_mean_and_std([float(x["horizons"]["16"]["latent_non_finite_ratio"]) for x in per_seed_results]),
                    "H32_latent_non_finite_ratio": format_mean_and_std([float(x["horizons"]["32"]["latent_non_finite_ratio"]) for x in per_seed_results]),
                    "latent_l2_auc": format_mean_and_std([float(x["stability"]["latent_l2_auc"]) for x in per_seed_results]),
                    "latent_l2_slope": format_mean_and_std([float(x["stability"]["latent_l2_slope"]) for x in per_seed_results]),
                    "latent_drift_slope": format_mean_and_std([float(x["stability"]["latent_drift_slope"]) for x in per_seed_results]),
                    "unstable_ratio": format_mean_and_std([float(x["stability"]["unstable_ratio"]) for x in per_seed_results]),
                    "error_explosion_ratio": format_mean_and_std([float(x["stability"]["error_explosion_ratio"]) for x in per_seed_results]),
                    "collapsed_embedding_ratio": format_mean_and_std([float(x["stability"]["collapsed_embedding_ratio"]) for x in per_seed_results]),
                    "Video": exp_video_rel,
                }
                summary_rows["rollout"].append(row)
            else:
                aggregate = aggregate_seed_statistics(per_seed_results, exp_video_rel)
                aggregate["runtime_sec"] = exp_runtime
                aggregate["git_commit"] = git_hash
                write_json_dict(exp_dir / "metrics.json", aggregate)
                row = {
                    "Model": spec.name,
                    "ADE": format_mean_and_std([float(x["ADE"]) for x in per_seed_results]),
                    "FDE": format_mean_and_std([float(x["FDE"]) for x in per_seed_results]),
                    "Accuracy": format_mean_and_std([float(x["Accuracy"]) for x in per_seed_results]),
                    "vjepa_feat_distance": format_mean_and_std(
                        [float(x["vjepa_feat_distance"]) for x in per_seed_results]
                    ),
                    "latent_smooth_l1": format_mean_and_std(
                        [float(x["latent_smooth_l1"]) for x in per_seed_results]
                    ),
                    "latent_cosine_distance": format_mean_and_std(
                        [float(x["latent_cosine_distance"]) for x in per_seed_results]
                    ),
                    "Video": exp_video_rel,
                }
                summary_rows[section].append(row)

            write_experiment_text_log(
                exp_dir / "log.txt",
                f"completed experiment={spec.name} runtime_sec={exp_runtime:.1f} git={git_hash}",
            )

    if "main" in args.sections:
        write_markdown_summary_table(results_root, "main", summary_rows["main"])
    if "ablation" in args.sections:
        write_markdown_summary_table(results_root, "ablation", summary_rows["ablation"])
    if "layer" in args.sections:
        write_markdown_summary_table(results_root, "layer", summary_rows["layer"])
    if "horizon_rollout" in args.sections:
        write_markdown_summary_table(results_root, "rollout", summary_rows["rollout"])

    total_runtime = time.time() - total_start
    write_json_dict(
        results_root / "scheduler_run.json",
        {
            "git_commit": git_hash,
            "results_root": str(results_root),
            "seeds": args.seeds,
            "sections": args.sections,
            "runtime_sec": total_runtime,
        },
    )
    print(f"[DONE] results_root={results_root}")
    print(f"[DONE] total_runtime_sec={total_runtime:.1f}")


if __name__ == "__main__":
    main()
