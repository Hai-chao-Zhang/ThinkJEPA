# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import gzip
import json
import logging
import math
import pickle
import pprint
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp

from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.data_manager import init_data

from src.models.attentive_pooler import (
    TrajRegressorHead,
    TrajTFhead,
    VJEPA2SOTAHandTrackingHead,
)
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduce, init_distributed
from src.utils.logging import AverageMeter, CSVLogger
from torch.nn.parallel import (
    DataParallel as DP,
    DistributedDataParallel,
    DistributedDataParallel as DDP,
)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def _worker_init_fn(worker_id):
    """
    DataLoader worker 初始化函数：限制每个 worker 的线程数，防止 fork 死锁

    CRITICAL: 必须在每个 worker 进程启动时调用，清理父进程的多线程库状态
    """
    import os
    import random

    import numpy as np
    import torch

    # 1. 限制 NumPy/BLAS 线程数（防止 fork 继承父进程线程池）
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # 2. 重置随机种子（保证可复现）
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    # 3. 清理 PyTorch 内部状态
    torch.set_num_threads(1)


def check_nan_detailed(tensor, name, batch_idx=0, head_idx=0):
    """Comprehensive NaN/Inf diagnostic for debugging"""
    if tensor is None:
        return False

    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()

    if has_nan or has_inf:
        print(f"\n{'='*80}")
        print(f"⚠️  NaN/Inf detected in {name}")
        print(f"   Batch: {batch_idx}, Head: {head_idx}")
        print(f"   Shape: {tensor.shape}, Dtype: {tensor.dtype}")
        print(f"   Has NaN: {has_nan}, Has Inf: {has_inf}")

        if has_nan:
            nan_mask = torch.isnan(tensor)
            nan_count = nan_mask.sum().item()
            print(f"   NaN count: {nan_count}/{tensor.numel()}")

            # Show which dimensions have NaN
            if tensor.ndim >= 2:
                nan_per_dim = nan_mask.float().sum(dim=tuple(range(tensor.ndim - 1)))
                print(f"   NaN per last dim: {nan_per_dim.tolist()}")

        if has_inf:
            inf_mask = torch.isinf(tensor)
            inf_count = inf_mask.sum().item()
            print(f"   Inf count: {inf_count}/{tensor.numel()}")

        # Show statistics of non-NaN values
        valid_mask = ~(torch.isnan(tensor) | torch.isinf(tensor))
        if valid_mask.any():
            valid_values = tensor[valid_mask]
            print(
                f"   Valid range: [{valid_values.min():.6f}, {valid_values.max():.6f}]"
            )
            print(
                f"   Valid mean: {valid_values.mean():.6f}, std: {valid_values.std():.6f}"
            )

        print(f"{'='*80}\n")

        # Print first few NaN locations
        if has_nan:
            nan_indices = torch.nonzero(torch.isnan(tensor))[:5]
            if len(nan_indices) > 0:
                print(f"First NaN locations: {nan_indices.tolist()}")

    return has_nan or has_inf


def set_deterministic(deterministic: bool):
    """切换确定性模式（用于 cache_encoder）"""
    torch.backends.cudnn.benchmark = not deterministic
    try:
        torch.use_deterministic_algorithms(deterministic)
    except Exception:
        pass
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)


def main(args_eval, resume_preempt=False):
    # ---------------- config ----------------
    val_only = args_eval.get("val_only", False)
    save_only = args_eval.get("save_only", False)  # True: 只跑 DataLoader 并保存统计
    cache_encoder = args_eval.get("cache_encoder", False)  # 新增：离线缓存 encoder 特征
    use_cached = args_eval.get("use_cached_features", False)  # 新增：从缓存读取
    if val_only:
        logger.info("VAL ONLY")
    if save_only:
        logger.info("SAVE-ONLY MODE (no model/optimizer)")
    if cache_encoder:
        logger.info("CACHE-ENCODER MODE (generate per-sample cache)")

    pretrain_folder = args_eval.get("folder", "./checkpoints")
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    num_workers = args_eval.get("num_workers", 12)

    # pretrain/model cfg
    args_pretrain = args_eval.get("model_kwargs", {})
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = args_eval.get("experiment", {})
    args_classifier = args_exp.get("classifier", {})
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)

    # metrics 配置（默认启用 ADE/FDE）
    args_metrics = args_exp.get("metrics", {})
    pos_offset = int(args_metrics.get("pos_offset", 0))
    pos_dims = int(args_metrics.get("pos_dims", 3))
    csv_log_ade_fde = bool(args_metrics.get("csv_log_ade_fde", True))  # 默认启用

    # data
    args_data = args_exp.get("data", {})
    dataset_type = args_data.get("dataset_type", "VideoDataset")
    train_data_path = [args_data.get("dataset_train")]
    val_data_path = [args_data.get("dataset_val")]
    resolution = args_data.get("resolution", 224)
    num_segments = args_data.get("num_segments", 1)
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 4)
    duration = args_data.get("clip_duration", None)
    num_views_per_segment = args_data.get("num_views_per_segment", 1)
    normalization = args_data.get("normalization", None)

    # optim
    args_opt = args_exp.get("optimization", {})
    batch_size = args_opt.get("batch_size", 8)
    num_epochs = args_opt.get("num_epochs", 1)
    use_bfloat16 = args_opt.get("use_bfloat16", False)
    opt_kwargs = [
        dict(
            ref_wd=kwargs.get("weight_decay"),
            final_wd=kwargs.get("final_weight_decay"),
            start_lr=kwargs.get("start_lr"),
            ref_lr=kwargs.get("lr"),
            final_lr=kwargs.get("final_lr"),
            warmup=kwargs.get("warmup"),
        )
        for kwargs in args_opt.get("multihead_kwargs", [{}])
    ]

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # io dirs
    folder = os.path.join(pretrain_folder, "video_classification_frozen/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    step_log_file = os.path.join(folder, f"steps_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")
    best_path = os.path.join(folder, "best.pt")

    if rank == 0:
        if csv_log_ade_fde:
            csv_logger = CSVLogger(
                log_file,
                ("%d", "epoch"),
                ("%.6f", "train_mae"),
                ("%.6f", "val_mae"),
                ("%.6f", "train_ade"),
                ("%.6f", "train_fde"),
                ("%.6f", "val_ade"),
                ("%.6f", "val_fde"),
            )
            step_logger = CSVLogger(
                step_log_file,
                ("%d", "step"),
                ("%.6f", "train_mae"),
                ("%.6f", "val_mae"),
                ("%.6f", "train_ade"),
                ("%.6f", "train_fde"),
                ("%.6f", "val_ade"),
                ("%.6f", "val_fde"),
            )
        else:
            csv_logger = CSVLogger(
                log_file, ("%d", "epoch"), ("%.6f", "train_mae"), ("%.6f", "val_mae")
            )
            step_logger = CSVLogger(
                step_log_file,
                ("%d", "step"),
                ("%.6f", "train_mae"),
                ("%.6f", "val_mae"),
            )
    else:
        csv_logger = None
        step_logger = None

    # ---- dataloaders ----
    cache_root = os.environ.get("VJEPA2_CACHE_ROOT", "/path/to/vjepa2/cache")

    if use_cached:
        # 从缓存读取，不需要 transforms/随机性
        logger.info(f"[USE_CACHED] Reading from cache: {cache_root}")

        # 从 args_data 读取 cache 相关配置（默认保留原始 dtype）
        cache_dtype_bf16 = args_data.get("cache_dtype_bf16", False)
        split_per_clip = args_data.get("split_per_clip", False)

        train_loader, train_sampler = make_cached_dataloader_auto(
            cache_dir=os.path.join(cache_root, "train"),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=False,
            expect_label=True,
            cache_dtype_bf16=cache_dtype_bf16,
            split_per_clip=split_per_clip,
            world_size=world_size,
            rank=rank,
        )
        val_loader, _ = make_cached_dataloader_auto(
            cache_dir=os.path.join(cache_root, "val"),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
            expect_label=True,
            cache_dtype_bf16=cache_dtype_bf16,
            split_per_clip=split_per_clip,
            world_size=world_size,
            rank=rank,
        )
    elif cache_encoder:
        # cache_encoder 模式：train/val 都用 eval transform（关闭随机性）
        set_deterministic(True)
        logger.info(
            "[CACHE_ENCODER] Deterministic mode ON; both train/val use eval transforms; shuffle=False"
        )
        train_loader, train_sampler = make_dataloader(
            dataset_type=dataset_type,
            root_path=train_data_path,
            img_size=resolution,
            frames_per_clip=frames_per_clip,
            frame_step=frame_step,
            eval_duration=duration,
            num_segments=num_segments,
            num_views_per_segment=1,
            allow_segment_overlap=True,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            training=False,  # 关键：train 也用 eval transform
            num_workers=num_workers,
            normalization=normalization,
            cache_save_mode=True,  # cache_encoder 模式，shuffle=False
        )
        val_loader, _ = make_dataloader(
            dataset_type=dataset_type,
            root_path=val_data_path,
            img_size=resolution,
            frames_per_clip=frames_per_clip,
            frame_step=frame_step,
            num_segments=num_segments,
            eval_duration=duration,
            num_views_per_segment=num_views_per_segment,
            allow_segment_overlap=True,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            training=False,  # eval
            num_workers=num_workers,
            normalization=normalization,
            cache_save_mode=True,  # cache_encoder 模式，shuffle=False
        )
    else:
        # 默认路径：train 用 train transform，val 用 eval transform
        set_deterministic(False)
        train_loader, train_sampler = make_dataloader(
            dataset_type=dataset_type,
            root_path=train_data_path,
            img_size=resolution,
            frames_per_clip=frames_per_clip,
            frame_step=frame_step,
            eval_duration=duration,
            num_segments=num_segments,
            num_views_per_segment=1,
            allow_segment_overlap=True,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            training=True,
            num_workers=num_workers,
            normalization=normalization,
        )
        val_loader, _ = make_dataloader(
            dataset_type=dataset_type,
            root_path=val_data_path,
            img_size=resolution,
            frames_per_clip=frames_per_clip,
            frame_step=frame_step,
            num_segments=num_segments,
            eval_duration=duration,
            num_views_per_segment=num_views_per_segment,
            allow_segment_overlap=True,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            training=False,
            num_workers=num_workers,
            normalization=normalization,
        )
    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")

    # ================= SAVE-ONLY 路径 =================
    if save_only:
        from pathlib import Path

        out_dir = Path(folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        train_out = str(out_dir / f"dl_only_train_rank{rank}.pkl.gz")
        val_out = str(out_dir / f"dl_only_val_rank{rank}.pkl.gz")

        logger.info(f"[SAVE-ONLY] saving train stats/examples to: {train_out}")
        _ = save_data_local_test(
            data_loader=train_loader,
            out_path=train_out,
            max_batches=None,
            save_examples=0,
        )

        logger.info(f"[SAVE-ONLY] saving val stats/examples to: {val_out}")
        _ = save_data_local_test(
            data_loader=val_loader,
            out_path=val_out,
            max_batches=None,
            save_examples=0,
        )

        logger.info("[SAVE-ONLY] finished. Exit.")
        return
    # =================================================

    # encoder / classifier
    encoder = None
    if not use_cached:
        encoder = init_module(
            module_name=module_name,
            frames_per_clip=frames_per_clip,
            resolution=resolution,
            checkpoint=checkpoint,
            model_kwargs=args_model,
            wrapper_kwargs=args_wrapper,
            device=device,
        )

    # ================= CACHE-ENCODER (PER-SAMPLE) =================
    if cache_encoder:
        # 从 args_data 读取 cache 相关配置（默认保留原始 dtype）
        cache_dtype_bf16 = args_data.get("cache_dtype_bf16", False)
        split_per_clip = args_data.get("split_per_clip", False)

        train_dir = os.path.join(cache_root, "train")
        val_dir = os.path.join(cache_root, "val")

        logger.info(f"[CACHE_ENCODER] start caching TRAIN -> {train_dir}")
        dtype_str = "bf16 (compressed)" if cache_dtype_bf16 else "fp32 (original)"
        logger.info(
            f"[CACHE_ENCODER] config: dtype={dtype_str}, split_per_clip={split_per_clip}"
        )
        cache_encoder_per_sample(
            encoder=encoder,
            data_loader=train_loader,
            device=device,
            out_root=train_dir,
            use_bfloat16=use_bfloat16,
            split_tag="train",
            split_per_clip=split_per_clip,
            cache_dtype_bf16=cache_dtype_bf16,
        )

        logger.info(f"[CACHE_ENCODER] start caching VAL -> {val_dir}")
        cache_encoder_per_sample(
            encoder=encoder,
            data_loader=val_loader,
            device=device,
            out_root=val_dir,
            use_bfloat16=use_bfloat16,
            split_tag="val",
            split_per_clip=split_per_clip,
            cache_dtype_bf16=cache_dtype_bf16,
        )

        logger.info("[CACHE_ENCODER] done. Exit.")
        return
    # =================================================

    # 推断 embed_dim：从 encoder 或缓存特征
    if use_cached:
        # 从缓存里取一个样本推断特征维度（最后一维）
        try:
            sample_feats, _, _, _ = next(iter(train_loader))  # [B, C, ..., D]
            inferred_embed_dim = int(sample_feats.shape[-1])
            logger.info(
                f"[USE_CACHED] Inferred embed_dim={inferred_embed_dim} from cached features"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to infer embed_dim from cached feats: {e}")
    else:
        inferred_embed_dim = int(encoder.embed_dim)
        logger.info(f"Using encoder.embed_dim={inferred_embed_dim}")

    # Choose classifier type from config or default to TrajRegressorHead
    classifier_type = args_classifier.get("type", "TrajTFhead")

    print(f"the classifier type is {classifier_type} \n ******************")

    if classifier_type == "TrajTFhead":
        classifiers = [
            TrajTFhead(
                embed_dim=inferred_embed_dim,
                frames_per_clip=frames_per_clip,
                num_heads=num_heads,
                depth=num_probe_blocks,
                use_activation_checkpointing=True,
            ).to(device)
            for _ in opt_kwargs
        ]
    elif classifier_type == "VJEPA2SOTAHandTrackingHead":

        classifiers = [
            VJEPA2SOTAHandTrackingHead(
                embed_dim=inferred_embed_dim,
                frames_per_clip=frames_per_clip,
                num_joints=args_classifier.get("num_joints", 1),
                out_mode=args_classifier.get("out_mode", "rigid6d"),
                pool_depth=num_probe_blocks,
                num_heads=num_heads,
                temporal_type=args_classifier.get("temporal_type", "gru"),
                temporal_layers=args_classifier.get("temporal_layers", 2),
                temporal_dim=args_classifier.get("temporal_dim", 384),
                use_activation_checkpointing=True,
            ).to(device)
            for _ in opt_kwargs
        ]
    else:
        # Default: TrajRegressorHead
        classifiers = [
            TrajRegressorHead(
                embed_dim=inferred_embed_dim,
                frames_per_clip=frames_per_clip,
                num_heads=num_heads,
                depth=num_probe_blocks,
                use_activation_checkpointing=True,
            ).to(device)
            for _ in opt_kwargs
        ]

    if world_size > 1:
        classifiers = [
            DistributedDataParallel(c, static_graph=True) for c in classifiers
        ]

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=classifiers,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    best_val_acc = float("+inf")  # MAE 越小越好
    global_step = 0

    start_epoch = 0
    if resume_checkpoint and os.path.exists(latest_path):
        classifiers, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            classifiers=classifiers,
            opt=optimizer,
            scaler=scaler,
            val_only=val_only,
        )
        for _ in range(start_epoch * ipe):
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]
        if os.path.exists(best_path):
            try:
                _best = robust_checkpoint_loader(best_path, map_location="cpu")
                if "best_acc" in _best:
                    best_val_acc = float(_best["best_acc"])
                    # Sanity check: 旧 checkpoint 可能是 accuracy (越大越好)
                    if best_val_acc > 1.0:  # 明显不是 MAE (通常 < 1)
                        logger.warning(
                            f"best_acc from checkpoint looks like 'higher-is-better' ({best_val_acc:.3f}); resetting to +inf."
                        )
                        best_val_acc = float("+inf")
                    else:
                        logger.info(
                            f"[RESUME] loaded best_acc={best_val_acc:.5f} (MAE)"
                        )
            except Exception:
                pass
        global_step = start_epoch * ipe

    # ---- TRAIN LOOP ----
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if val_only:
            train_acc = -1.0
        train_mae, train_ade, train_fde = float("nan"), float("nan"), float("nan")
        if not val_only:
            train_mae, train_ade, train_fde, global_step, best_val_acc = run_one_epoch(
                device=device,
                training=True,
                encoder=encoder,
                classifiers=classifiers,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                data_loader=train_loader,
                use_bfloat16=use_bfloat16,
                epoch=epoch,
                checkpoint_meta={
                    "path": latest_path,
                    "rank": rank,
                    "batch_size": batch_size,
                    "world_size": world_size,
                },
                save_every=100,
                save_on_first_iter=True,
                save_on_last_iter=False,
                test_every=500,
                val_loader=val_loader,
                step_logger=step_logger if rank == 0 else None,
                global_step=global_step,
                best_val_acc=best_val_acc,
                best_path=best_path,
                use_cached_features=use_cached,
                pos_offset=pos_offset,
                pos_dims=pos_dims,
                csv_log_ade_fde=csv_log_ade_fde,
            )

        # epoch 末尾完整验证
        val_mae, val_ade, val_fde = evaluate(
            device=device,
            encoder=encoder,
            classifiers=classifiers,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            use_cached_features=use_cached,
            pos_offset=pos_offset,
            pos_dims=pos_dims,
        )

        if csv_log_ade_fde:
            logger.info(
                "[%5d] train_mae: %.6f  val_mae: %.6f | train_ade: %.6f  train_fde: %.6f | val_ade: %.6f  val_fde: %.6f",
                epoch + 1,
                train_mae,
                val_mae,
                train_ade,
                train_fde,
                val_ade,
                val_fde,
            )
            if rank == 0:
                csv_logger.log(
                    epoch + 1,
                    train_mae,
                    val_mae,
                    train_ade,
                    train_fde,
                    val_ade,
                    val_fde,
                )
        else:
            logger.info(
                "[%5d] train_mae: %.6f  val_mae: %.6f",
                epoch + 1,
                train_mae,
                val_mae,
            )
            if rank == 0:
                csv_logger.log(epoch + 1, train_mae, val_mae)

        # 更新 best（MAE 越小越好）
        if val_mae < best_val_acc and rank == 0:
            best_val_acc = val_mae
            _save_best(best_path, classifiers, optimizer, scaler, epoch, best_val_acc)
            if csv_log_ade_fde:
                logger.info(
                    f"[BEST] epoch-end improved to MAE={best_val_acc:.5f}, ADE={val_ade:.5f}, FDE={val_fde:.5f}, saved -> {best_path}"
                )
            else:
                logger.info(
                    f"[BEST] epoch-end improved to {best_val_acc:.5f} (MAE), saved -> {best_path}"
                )

        if val_only:
            return


def _save_best(best_path, classifiers, optimizer, scaler, epoch, best_acc):
    """保存 best checkpoint（仅 rank0 调用）"""
    all_classifier_dicts = [c.state_dict() for c in classifiers]
    all_opt_dicts = [o.state_dict() for o in optimizer]
    save_dict = {
        "classifiers": all_classifier_dicts,
        "opt": all_opt_dicts,
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": int(epoch),
        "best_acc": float(best_acc),
    }
    torch.save(save_dict, best_path)


def _basename_noext(p: str) -> str:
    """获取不带扩展名的文件名"""
    from pathlib import Path

    return Path(p).stem


def _as_list_of_meta(meta_obj, B: int):
    """
    将 collate 后的 meta 统一成长度为 B 的 list[dict]：
      - 若已是 list[dict]，直接返回
      - 若是 dict of lists（默认 collate 行为），转回 list[dict]
      - 否则，返回 B 个空 dict
    """
    if isinstance(meta_obj, list) and (B == 0 or isinstance(meta_obj[0], dict)):
        return meta_obj
    if isinstance(meta_obj, dict):
        keys = list(meta_obj.keys())
        if not keys:
            return [dict() for _ in range(B)]
        L = len(meta_obj[keys[0]])
        out = []
        for i in range(L):
            out.append(
                {k: meta_obj[k][i] if i < len(meta_obj[k]) else None for k in keys}
            )
        return out
    return [dict() for _ in range(B)]


def _orig_name_from_meta(m: dict) -> str:
    """从 meta 字典中提取原始文件名"""
    from pathlib import Path

    if isinstance(m, dict):
        if m.get("orig_name"):
            return Path(m["orig_name"]).stem
        for k in ("mp4_path", "hdf5_path"):
            if m.get(k):
                try:
                    return Path(m[k]).stem
                except Exception:
                    pass
    return "sample"


def _torch_to_numpy_dtype(t: torch.Tensor, use_bf16: bool = False) -> np.ndarray:
    """
    将 torch.Tensor 转为 numpy 数组：
    - 当 use_bf16=True 且是浮点：bf16 位级压缩（int16/uint16 view 往返）
    - 否则：**不降精度**，直接保留原始 dtype（如 float32、int64）
    """
    if t.is_floating_point():
        if use_bf16:
            # bf16 压缩路径（仅显式开启时使用）
            # 确保连续内存再 view
            return (
                t.to(torch.bfloat16)
                .contiguous()
                .view(torch.int16)
                .cpu()
                .numpy()
                .view(np.uint16)
            )
        else:
            # 保持 float32/float64 原样，不降精度
            return t.cpu().numpy()
    else:
        # 整型等其他类型：保持原样（int64/int32 等）
        return t.cpu().numpy()


def _numpy_to_torch(arr: np.ndarray, is_bf16_view: bool = False) -> torch.Tensor:
    """
    将 numpy 数组恢复为 torch.Tensor：
    - 如果 is_bf16_view=True，则 uint16 → bfloat16
    - 否则直接转换
    """
    if is_bf16_view and arr.dtype == np.uint16:
        # 先把 numpy 的 uint16 重新解释为 int16，再交给 torch，再 view 回 bfloat16
        int16_view = arr.view(np.int16)
        return torch.from_numpy(int16_view).view(torch.bfloat16)
    else:
        return torch.from_numpy(arr)


@torch.no_grad()
def cache_encoder_per_sample(
    *,
    encoder,
    data_loader,
    device,
    out_root: str,
    use_bfloat16: bool = False,
    split_tag: str = "train",
    split_per_clip: bool = False,
    cache_dtype_bf16: bool = False,
):
    """
    逐样本保存为 .npz + index.json 格式（默认保留原始 dtype）：

    目录结构：
        out_root/
            samples/
                r{rank}/
                    <uid>.npz
                    ...
            index.json  # 列表，每条记录含 uid, npz, orig_name, itr, bi[, c], mp4_path, hdf5_path

    参数：
        - split_per_clip: False → 每样本 1 个 .npz（feats 形状 [C,...]）
                          True  → 每样本的每个 clip 各 1 个 .npz（feats 形状 [...]，在 uid/index 中加 _c{c}）
        - cache_dtype_bf16: False（默认）→ 保留原始 dtype（float32/int64 等）
                            True → 浮点降为 bf16（存为 uint16 view），整型保持原样

    .npz 内键：
        - feats: numpy 数组（默认 float32 或可选 bf16 压缩）
        - labels: numpy 数组（如存在）
        - clipidx_0, clipidx_1, ...: 每个 clip_indices 的数组

    index.json 记录：
        {
          "uid": "r00__<orig>__it000123_bi007[_c003]",
          "npz": "samples/r00/<same_uid>.npz",
          "orig_name": "...",
          "itr": 123,
          "bi": 7,
          "c": 3,  # 仅 split_per_clip=True 时存在
          "mp4_path": "...",
          "hdf5_path": "..."
        }
    """
    import torch.distributed as dist

    # 取得 rank/world_size（未初始化时退回单卡）
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    out_dir = Path(out_root)
    # 每个 rank 写到独立的子目录，避免 .npz 文件名冲突
    samples_dir = out_dir / "samples" / f"r{rank:02d}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    index_records = []

    # uid 前缀，包含 rank
    uid_prefix = f"r{rank:02d}__"

    # 分片索引路径（增量刷新）
    rank_index_path = out_dir / f"index.r{rank}.json"
    tmp_rank_index = out_dir / f"index.r{rank}.json.tmp"

    def _flush_index():
        """将当前 index_records 原子写入分片索引"""
        with open(tmp_rank_index, "w", encoding="utf-8") as f:
            json.dump(index_records, f, indent=2, ensure_ascii=False)
        tmp_rank_index.replace(rank_index_path)

    pbar = tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        desc=f"CACHE[{split_tag}][rank{rank}]->{out_dir}",
        dynamic_ncols=True,
    )

    try:
        for itr, data in pbar:
            clips = data[0]
            labels_field = data[1]
            clip_indices_field = data[2]
            meta_field = data[3] if len(data) >= 4 else None

            # 估计 batch size
            if (
                isinstance(labels_field, (list, tuple))
                and len(labels_field) > 0
                and torch.is_tensor(labels_field[0])
            ):
                B = int(labels_field[0].shape[0])
            elif torch.is_tensor(labels_field):
                B = int(labels_field.shape[0])
            else:
                try:
                    B = int(clips[0][0].shape[0])
                except Exception:
                    B = 0

            metas = _as_list_of_meta(meta_field, B)

            # with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=False):
                clips_dev = [
                    [dij.to(device, non_blocking=True) for dij in di] for di in clips
                ]
                clip_indices_dev = [
                    d.to(device, non_blocking=True) for d in clip_indices_field
                ]
                if isinstance(labels_field, (list, tuple)):
                    labels = torch.stack(labels_field, dim=1)  # [B, C, ...]
                else:
                    labels = labels_field

                feats = encoder(clips_dev, clip_indices_dev)
                feats = _stack_clip_features(feats)  # [B, C, ...]
                C = int(feats.shape[1])

            # CPU 化
            feats_cpu = feats.detach().cpu()
            if torch.is_tensor(labels):
                labels_cpu = labels.detach().cpu()
            else:
                labels_cpu = labels
            clip_indices_cpu = []
            for ci in clip_indices_field:
                if torch.is_tensor(ci):
                    clip_indices_cpu.append(ci.detach().cpu())
                else:
                    clip_indices_cpu.append(ci)

            # 逐样本落盘
            for bi in range(B):
                meta = metas[bi] if bi < len(metas) else {}
                orig_name = _orig_name_from_meta(meta)

                if split_per_clip:
                    # 按 clip 拆分：每个 clip 只保存当前 c 的数据
                    for c in range(C):
                        uid = (
                            f"{uid_prefix}{orig_name}__it{itr:06d}_bi{bi:03d}_c{c:03d}"
                        )
                        npz_rel = f"samples/r{rank:02d}/{uid}.npz"
                        npz_path = samples_dir / f"{uid}.npz"

                        # 构造 .npz 数据
                        npz_data = {
                            "feats": _torch_to_numpy_dtype(
                                feats_cpu[bi, c], use_bf16=cache_dtype_bf16
                            )
                        }

                        # labels: 只保存当前 clip 的
                        if torch.is_tensor(labels_cpu):
                            npz_data["labels"] = _torch_to_numpy_dtype(
                                labels_cpu[bi, c], use_bf16=cache_dtype_bf16
                            )
                        elif (
                            isinstance(labels_cpu, (list, tuple))
                            and len(labels_cpu) > c
                        ):
                            # 如果 labels 是 list[C,...]，取当前 c
                            label_c = labels_cpu[c]
                            if torch.is_tensor(label_c):
                                npz_data["labels"] = _torch_to_numpy_dtype(
                                    label_c[bi], use_bf16=cache_dtype_bf16
                                )

                        # clipidx: 只保存当前 clip 的索引
                        # clip_indices_cpu 通常是长度为 C 的 list，每个元素形状 [B,...]
                        if len(clip_indices_cpu) > c:
                            ci = clip_indices_cpu[c]
                            ci_val = ci[bi] if torch.is_tensor(ci) else ci
                            if torch.is_tensor(ci_val):
                                npz_data["clipidx"] = _torch_to_numpy_dtype(
                                    ci_val, use_bf16=False
                                )
                            else:
                                npz_data["clipidx"] = np.asarray(ci_val)

                        np.savez_compressed(npz_path, **npz_data)

                        # index 记录
                        rec = {
                            "uid": uid,
                            "npz": npz_rel,
                            "orig_name": orig_name,
                            "itr": int(itr),
                            "bi": int(bi),
                            "c": int(c),
                            "mp4_path": (
                                meta.get("mp4_path") if isinstance(meta, dict) else None
                            ),
                            "hdf5_path": (
                                meta.get("hdf5_path")
                                if isinstance(meta, dict)
                                else None
                            ),
                        }
                        index_records.append(rec)
                else:
                    # 不拆分：每样本 1 个 .npz（feats: [C,...]）
                    uid = f"{uid_prefix}{orig_name}__it{itr:06d}_bi{bi:03d}"
                    npz_rel = f"samples/r{rank:02d}/{uid}.npz"
                    npz_path = samples_dir / f"{uid}.npz"

                    npz_data = {
                        "feats": _torch_to_numpy_dtype(
                            feats_cpu[bi], use_bf16=cache_dtype_bf16
                        ),
                        "feats_dtype": "bf16" if cache_dtype_bf16 else "fp32",  # 元信息
                    }
                    if torch.is_tensor(labels_cpu):
                        npz_data["labels"] = _torch_to_numpy_dtype(
                            labels_cpu[bi], use_bf16=cache_dtype_bf16
                        )

                    # —— 关键：clipidx 齐次化 —— #
                    C = int(feats_cpu.shape[1])
                    ci_list = clip_indices_cpu  # list (可能为张量/ndarray)

                    def _to_tensor(x):
                        return x if torch.is_tensor(x) else torch.as_tensor(x)

                    if len(ci_list) == C:
                        # 标准情况：len == C，逐个保存
                        for idx_i, ci in enumerate(ci_list):
                            ci_val = ci[bi] if torch.is_tensor(ci) else ci
                            npz_data[f"clipidx_{idx_i}"] = _torch_to_numpy_dtype(
                                _to_tensor(ci_val), use_bf16=False
                            )

                    elif C == 1 and len(ci_list) > 1:
                        # 多段合并为一段
                        parts = []
                        for ci in ci_list:
                            ci_val = ci[bi] if torch.is_tensor(ci) else ci
                            parts.append(_to_tensor(ci_val))
                        merged = torch.cat(parts, dim=0)
                        npz_data["clipidx"] = _torch_to_numpy_dtype(
                            merged, use_bf16=False
                        )

                    elif C > 1 and len(ci_list) == 1:
                        # 单段复制成 C 段
                        ci_val = (
                            ci_list[0][bi]
                            if torch.is_tensor(ci_list[0])
                            else ci_list[0]
                        )
                        base = _to_tensor(ci_val)
                        for idx_i in range(C):
                            npz_data[f"clipidx_{idx_i}"] = _torch_to_numpy_dtype(
                                base, use_bf16=False
                            )

                    else:
                        # 无法判定的错配：不写 clipidx，读侧会兜底
                        if itr == 0 and bi == 0:
                            logging.getLogger(__name__).warning(
                                "[CACHE] clipidx count mismatch: feats C=%d, got %d clipidx parts; "
                                "skip saving clipidx for %s",
                                C,
                                len(ci_list),
                                uid,
                            )

                    np.savez_compressed(npz_path, **npz_data)

                    rec = {
                        "uid": uid,
                        "npz": npz_rel,
                        "orig_name": orig_name,
                        "itr": int(itr),
                        "bi": int(bi),
                        "mp4_path": (
                            meta.get("mp4_path") if isinstance(meta, dict) else None
                        ),
                        "hdf5_path": (
                            meta.get("hdf5_path") if isinstance(meta, dict) else None
                        ),
                    }
                    index_records.append(rec)

            # ---- 定期刷新索引（每 500 iter） ----
            if (itr + 1) % 500 == 0:
                _flush_index()
                logger.info(f"[CACHE][rank{rank}] Flushed index at iter {itr+1}")

    finally:
        # ---- 确保最终刷新索引（即使中途异常） ----
        _flush_index()

    # ---- 同步，只有 rank0 合并 ----
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if rank == 0:
        final = []
        for r in range(world_size):
            p = out_dir / f"index.r{r}.json"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    final.extend(json.load(f))
        # 原子写 index.json
        index_path = out_dir / "index.json"
        temp_index = out_dir / "index.json.tmp"
        with open(temp_index, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2, ensure_ascii=False)
        temp_index.replace(index_path)
        logger.info(f"[CACHE] merged {len(final)} records into {index_path}")
        # 可选：清理分片
        for r in range(world_size):
            pr = out_dir / f"index.r{r}.json"
            try:
                pr.unlink()
            except Exception:
                pass
    else:
        logger.info(
            f"[CACHE][rank{rank}] wrote shard with {len(index_records)} records"
        )


def save_data_local_test(
    data_loader,
    out_path: str,
    max_batches: int | None = None,
    save_examples: int = 0,
):
    """
    纯 DataLoader 消耗与可选的小样本落盘（带 tqdm 进度条）。
    - 不搬到 GPU，不调用模型，不做反传
    - 统计吞吐量（samples/s, frames/s）
    - 可选：将前几条样本的 (buffer, labels=traj['states'], clip_indices) 序列化保存
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    def _count_frames(x: torch.Tensor) -> int:
        if not torch.is_tensor(x):
            return 0
        if x.ndim >= 5:
            if 1 <= x.shape[2] <= 128:
                T = x.shape[2]
            elif 1 <= x.shape[1] <= 128:
                T = x.shape[1]
            else:
                T = 0
        else:
            T = 0
        B = x.shape[0] if x.ndim >= 1 else 1
        return int(B * max(0, T))

    stats = dict(
        num_batches=0,
        num_samples=0,
        num_frames=0,
        start_time=time.time(),
        per_batch_time=[],
    )
    saved_examples: List[dict[str, Any]] = []

    total_iters = len(data_loader)
    if max_batches is not None:
        total_iters = min(total_iters, max_batches)

    t0 = time.perf_counter()
    t_prev = t0

    with torch.no_grad():
        pbar = tqdm(
            enumerate(data_loader),
            total=total_iters,
            dynamic_ncols=True,
            smoothing=0.1,
            desc="DL-ONLY",
        )
        for itr, data in pbar:
            if max_batches is not None and itr >= max_batches:
                break

            clips = data[0]
            labels = data[1]
            clip_indices = data[2]

            if (
                isinstance(labels, (list, tuple))
                and len(labels) > 0
                and torch.is_tensor(labels[0])
            ):
                batch_size = int(labels[0].shape[0])
            elif torch.is_tensor(labels):
                batch_size = int(labels.shape[0])
            else:
                try:
                    batch_size = int(clips[0][0].shape[0])
                except Exception:
                    batch_size = 0

            stats["num_samples"] += batch_size
            stats["num_batches"] += 1

            frames = 0
            for time_views in clips:
                for v in time_views:
                    if torch.is_tensor(v):
                        _ = v.sum().item()
                        frames += _count_frames(v)
            stats["num_frames"] += frames

            if isinstance(labels, (list, tuple)):
                for lb in labels:
                    if torch.is_tensor(lb):
                        _ = lb.sum().item()
            elif torch.is_tensor(labels):
                _ = labels.sum().item()

            if isinstance(clip_indices, (list, tuple)):
                for ci in clip_indices:
                    if torch.is_tensor(ci):
                        _ = ci.sum().item()
                    elif isinstance(ci, np.ndarray):
                        _ = float(ci.sum())
            elif torch.is_tensor(clip_indices):
                _ = clip_indices.sum().item()

            if save_examples > 0 and batch_size > 0:
                Bsave = min(save_examples, batch_size)
                one_view = [
                    (tv[0].detach().cpu() if torch.is_tensor(tv[0]) else tv[0])
                    for tv in clips
                ]
                if (
                    isinstance(labels, (list, tuple))
                    and len(labels) > 0
                    and torch.is_tensor(labels[0])
                ):
                    states = [
                        labels[i][:Bsave].detach().cpu() for i in range(len(labels))
                    ]
                else:
                    states = (
                        [labels[:Bsave].detach().cpu()]
                        if torch.is_tensor(labels)
                        else []
                    )
                if isinstance(clip_indices, (list, tuple)) and len(clip_indices) > 0:
                    idx = [np.asarray(ci)[:Bsave] for ci in clip_indices]
                else:
                    idx = [np.asarray(clip_indices)[:Bsave]]
                saved_examples.append(
                    dict(itr=itr, buffers=one_view, states=states, clip_indices=idx)
                )

            now = time.perf_counter()
            stats["per_batch_time"].append(now - t_prev)
            elapsed = now - t0
            sps = stats["num_samples"] / max(elapsed, 1e-6)
            fps = stats["num_frames"] / max(elapsed, 1e-6)
            pbar.set_postfix(
                {
                    "batches": stats["num_batches"],
                    "samples": stats["num_samples"],
                    "frames": stats["num_frames"],
                    "samp/s": f"{sps:,.1f}",
                    "frames/s": f"{fps:,.1f}",
                }
            )
            t_prev = now

    total = time.perf_counter() - t0
    sps = stats["num_samples"] / max(total, 1e-6)
    fps = stats["num_frames"] / max(total, 1e-6)

    out_obj = dict(
        stats=stats | {"duration_sec": total, "samp_per_s": sps, "frames_per_s": fps},
        examples=saved_examples,
    )
    with gzip.open(out_path, "wb") as f:
        pickle.dump(out_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    return sps


# ---------------- CachedFeatureDataset and DataLoader ----------------


class CachedFeatureDataset(Dataset):
    """从缓存的 .pkl 文件中读取特征（旧格式，向后兼容）"""

    def __init__(self, cache_dir: str, expect_label: bool = True):
        from pathlib import Path

        self.cache_dir = Path(cache_dir)
        self.files = sorted([p for p in self.cache_dir.glob("*.pkl") if p.is_file()])
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .pkl found in {self.cache_dir}")
        self.expect_label = expect_label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        with open(fp, "rb") as f:
            obj = pickle.load(f)
        feats = obj["feats"]  # Tensor[C,...]
        labels = obj.get("labels", None)  # Tensor[C,...] / list / None
        clip_indices = obj.get("clip_indices", None)
        meta = dict(
            mp4_path=obj.get("mp4_path"),
            hdf5_path=obj.get("hdf5_path"),
            orig_name=obj.get("orig_name", fp.stem),
            source=str(fp),
        )
        if self.expect_label and labels is None:
            raise ValueError(f"{fp} missing 'labels' but expect_label=True")

        # 强制 labels 为 Tensor（如果不是，转换）
        if labels is not None and not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)  # 转换为 Tensor[C,...]

        return feats, labels, clip_indices, meta


class NPZCacheDataset(Dataset):
    """
    从 .npz + index.json 格式的缓存中读取特征（新格式）

    返回：
        feats: Tensor[C,...] 或 Tensor[...] （取决于 split_per_clip）
        labels: Tensor[C,...] 或 Tensor[...] 或 None
        clip_indices: list[Tensor] 或 None
        meta: dict 含 uid, orig_name, itr, bi, [c], mp4_path, hdf5_path
    """

    def __init__(
        self,
        cache_dir: str,
        expect_label: bool = True,
        cache_dtype_bf16: bool = False,
        split_per_clip: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        index_path = self.cache_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"index.json not found in {self.cache_dir}")

        with open(index_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        if len(self.records) == 0:
            raise ValueError(f"Empty index.json in {self.cache_dir}")

        self.expect_label = expect_label
        self.cache_dtype_bf16 = cache_dtype_bf16
        self.split_per_clip = split_per_clip

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        npz_rel = rec["npz"]
        npz_path = self.cache_dir / npz_rel

        with np.load(npz_path) as data:
            # feats
            feats_np = data["feats"]
            feats = _numpy_to_torch(
                feats_np, is_bf16_view=(feats_np.dtype == np.uint16)
            )
            feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            # labels
            labels = None
            if "labels" in data:
                labels_np = data["labels"]
                labels_np = np.nan_to_num(labels_np, nan=0.0, posinf=0.0, neginf=0.0)
                labels = _numpy_to_torch(
                    labels_np, is_bf16_view=(labels_np.dtype == np.uint16)
                )
            elif self.expect_label:
                raise ValueError(f"{npz_path} missing 'labels' but expect_label=True")

            # clip_indices: 优先读取 "clipidx"（per-clip 格式），否则回退到 "clipidx_*"
            clip_indices = None
            if "clipidx" in data:
                # Per-clip 格式：只有一个 clipidx
                ci_np = data["clipidx"]
                ci_t = _numpy_to_torch(ci_np, is_bf16_view=False)
                clip_indices = [ci_t]
            else:
                # Non-split 格式：多个 clipidx_0, clipidx_1, ...
                clipidx_keys = sorted(
                    [k for k in data.keys() if k.startswith("clipidx_")]
                )
                if clipidx_keys:
                    clip_indices = []
                    for k in clipidx_keys:
                        ci_np = data[k]
                        ci_t = _numpy_to_torch(ci_np, is_bf16_view=False)
                        clip_indices.append(ci_t)

        meta = {
            "uid": rec["uid"],
            "orig_name": rec["orig_name"],
            "itr": rec["itr"],
            "bi": rec["bi"],
            "mp4_path": rec.get("mp4_path"),
            "hdf5_path": rec.get("hdf5_path"),
            "source": str(npz_path),
        }
        if "c" in rec:
            meta["c"] = rec["c"]

        return feats, labels, clip_indices, meta


_WARNED_MISMATCH = False  # 模块级变量，避免刷屏


def _collate_cached(batch):
    """
    自定义 collate:
      feats/labels -> [B,C,...]
      clip_indices -> 长度 C 的 list，每个元素 [B,...]
    """
    global _WARNED_MISMATCH
    B = len(batch)
    feats_list, labels_list, clipidx_list, meta_list = zip(*batch)

    # feats -> [B,C,...]
    assert all(torch.is_tensor(x) for x in feats_list)
    C = int(feats_list[0].shape[0])
    assert all(x.shape[0] == C for x in feats_list), "All samples must have same C"
    feats = torch.stack(feats_list, dim=0)  # [B,C,...]

    # labels
    labels = (
        torch.stack(labels_list, dim=0)
        if torch.is_tensor(labels_list[0])
        else list(labels_list)
    )

    # clip_indices（容错处理）
    clip_indices = None
    if clipidx_list[0] is not None:
        item0 = clipidx_list[0]
        # 情况 A：标准，len == C
        if isinstance(item0, (list, tuple)) and len(item0) == C:
            clip_indices = []
            for c in range(C):
                elems_c = []
                for b in range(B):
                    ci = clipidx_list[b][c]
                    elems_c.append(ci if torch.is_tensor(ci) else torch.as_tensor(ci))
                clip_indices.append(torch.stack(elems_c, dim=0))  # [B,...]

        # 情况 B：feats 的 C=1，但 clipidx 有多段 -> 合并为一个
        elif C == 1 and isinstance(item0, (list, tuple)) and len(item0) > 1:
            merged = []
            for b in range(B):
                cis = [
                    ci if torch.is_tensor(ci) else torch.as_tensor(ci)
                    for ci in clipidx_list[b]
                ]
                merged.append(torch.cat(cis, dim=0))
            clip_indices = [torch.stack(merged, dim=0)]  # len=1

        # 情况 C：feats 的 C>1，但 clipidx 只有 1 段 -> 复用给每个 C
        elif C > 1 and (
            (isinstance(item0, (list, tuple)) and len(item0) == 1)
            or torch.is_tensor(item0)  # 直接给了一段
        ):
            # 取出"那一段"，复制 C 份
            def _one_ci(b):
                x = clipidx_list[b]
                if isinstance(x, (list, tuple)):
                    x = x[0]
                return x if torch.is_tensor(x) else torch.as_tensor(x)

            base = [_one_ci(b) for b in range(B)]  # [B,...]
            base = torch.stack(base, dim=0)  # [B,...]
            clip_indices = [base.clone() for _ in range(C)]  # len=C

        # 情况 D：其余错配 -> 直接置 None（heads/decoder 在 use_cached 下并不需要它）
        else:
            clip_indices = None
            if not _WARNED_MISMATCH:
                _WARNED_MISMATCH = True
                logging.getLogger(__name__).warning(
                    "clip_indices shape mismatch: feats C=%d, but got %s with len=%s; "
                    "set clip_indices=None for cached path.",
                    C,
                    type(item0),
                    (len(item0) if isinstance(item0, (list, tuple)) else "n/a"),
                )

    return feats, labels, clip_indices, list(meta_list)


def _collate_cached_perclip(batch):
    """
    per-clip 专用 collate：每个样本只有一个 clip（feats: [...]）
    扩展为 [B,1,...] 以兼容后续逻辑
    """
    B = len(batch)
    feats_list, labels_list, clipidx_list, meta_list = zip(*batch)

    # feats: [...] -> [B,1,...]
    feats = torch.stack([x.unsqueeze(0) for x in feats_list], dim=0)

    # labels
    if labels_list[0] is None:
        labels = None
    elif torch.is_tensor(labels_list[0]):
        labels = torch.stack([x.unsqueeze(0) for x in labels_list], dim=0)
    else:
        labels = list(labels_list)

    # clip_indices: 每个样本只有一个 clipidx
    clip_indices = None
    if clipidx_list[0] is not None:
        flat = []
        for item in clipidx_list:
            if isinstance(item, (list, tuple)):
                assert len(item) == 1, "per-clip file should carry ONE clipidx"
                flat.append(
                    item[0] if torch.is_tensor(item[0]) else torch.as_tensor(item[0])
                )
            else:
                flat.append(item if torch.is_tensor(item) else torch.as_tensor(item))
        clip_indices = [torch.stack(flat, dim=0)]  # 长度=1，每个 [B,...]

    return feats, labels, clip_indices, list(meta_list)


def make_cached_dataloader(
    cache_dir: str,
    batch_size: int,
    num_workers: int = 8,
    shuffle: bool = False,
    drop_last: bool = False,
    expect_label: bool = True,
    world_size: int = 1,
    rank: int = 0,
):
    """创建从旧 .pkl 缓存读取的 DataLoader（向后兼容）"""
    import multiprocessing as mp

    ds = CachedFeatureDataset(cache_dir, expect_label=expect_label)
    sampler = (
        DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=shuffle)
        if world_size > 1
        else None
    )

    # CRITICAL FIX: spawn + persistent_workers + prefetch
    mp_context = mp.get_context("spawn") if num_workers > 0 else None
    pin_dev = "cuda" if torch.cuda.is_available() else None

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        pin_memory_device=pin_dev,  # ✅ 指定 pin 到哪个设备
        drop_last=drop_last,
        collate_fn=_collate_cached,
        multiprocessing_context=mp_context,  # ✅ spawn
        persistent_workers=(num_workers > 0),  # ✅ 不重启 worker
        prefetch_factor=2 if num_workers > 0 else None,  # ✅ 预取
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,  # ✅ 清理状态
    )
    return loader, sampler


def make_npz_cached_dataloader(
    cache_dir: str,
    batch_size: int,
    num_workers: int = 8,
    shuffle: bool = False,
    drop_last: bool = False,
    expect_label: bool = True,
    cache_dtype_bf16: bool = False,
    split_per_clip: bool = False,
    world_size: int = 1,
    rank: int = 0,
):
    """
    创建从新 .npz + index.json 格式缓存读取的 DataLoader

    参数：
        - cache_dir: 缓存目录（含 index.json 和 samples/）
        - cache_dtype_bf16: 是否用 bf16 存储（读取时需恢复）
        - split_per_clip: 是否按 clip 拆分存储（影响 feats 维度）
        - world_size: 分布式训练的总进程数
        - rank: 当前进程的 rank

    返回：
        (loader, sampler) 以兼容调用点
    """
    import multiprocessing as mp

    ds = NPZCacheDataset(
        cache_dir=cache_dir,
        expect_label=expect_label,
        cache_dtype_bf16=cache_dtype_bf16,
        split_per_clip=split_per_clip,
    )
    # 根据 split_per_clip 选择对应的 collate 函数
    collate_fn = _collate_cached_perclip if split_per_clip else _collate_cached
    sampler = (
        DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=shuffle)
        if world_size > 1
        else None
    )
    # CRITICAL FIX: spawn + persistent_workers + prefetch
    mp_context = mp.get_context("spawn") if num_workers > 0 else None

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_fn,
        multiprocessing_context=mp_context,  # ✅ spawn
        persistent_workers=(num_workers > 0),  # ✅ 不重启 worker
        prefetch_factor=2 if num_workers > 0 else None,  # ✅ 预取
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,  # ✅ 清理状态
    )
    return loader, sampler


def make_cached_dataloader_auto(
    cache_dir: str,
    batch_size: int,
    num_workers: int = 8,
    shuffle: bool = False,
    drop_last: bool = False,
    expect_label: bool = True,
    cache_dtype_bf16: bool = False,
    split_per_clip: bool = False,
    world_size: int = 1,
    rank: int = 0,
):
    """
    自动检测缓存格式并创建对应的 DataLoader：
    - 如果存在 index.json → 使用 NPZCacheDataset（新格式）
    - 否则 → 使用 CachedFeatureDataset（旧 .pkl 格式）

    返回：
        (loader, sampler) 以兼容调用点
    """
    cache_path = Path(cache_dir)
    index_path = cache_path / "index.json"

    if index_path.exists():
        logger.info(f"[AUTO] Detected index.json → using NPZ format: {cache_dir}")
        return make_npz_cached_dataloader(
            cache_dir=cache_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            expect_label=expect_label,
            cache_dtype_bf16=cache_dtype_bf16,
            split_per_clip=split_per_clip,
            world_size=world_size,
            rank=rank,
        )
    else:
        logger.info(
            f"[AUTO] No index.json found → using legacy PKL format: {cache_dir}"
        )
        return make_cached_dataloader(
            cache_dir=cache_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            expect_label=expect_label,
            world_size=world_size,
            rank=rank,
        )


# ---------------- New helpers for vectorized decoder ----------------


def _ade_fde_meters(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    pos_offset: int = 0,
    pos_dims: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    基于 MSE 的位移误差 → 开方得到米，进而得到 ADE/FDE（单位: 米）
    pred/target: [B, C, T, D]
    pos_offset/pos_dims 确定最后一维中 (x,y[,z]) 的切片。
    返回: (ade_mean_m, fde_mean_m) —— 均为当前 rank 的标量张量（后续会 AllReduce）

    CRITICAL: 添加 finite 掩码防止 NaN 扩散
    """
    if pred.ndim != 3 or target.ndim != 3:
        raise ValueError(
            f"Expected pred/target to be [B,C,T,D], got {pred.shape} / {target.shape}"
        )

    D = pred.shape[-1]
    if not (0 <= pos_offset < D) or not (1 <= pos_dims <= D - pos_offset):
        raise ValueError(
            f"Invalid (pos_offset={pos_offset}, pos_dims={pos_dims}) for D={D}"
        )

    pos_slice = slice(pos_offset, pos_offset + pos_dims)
    pred_pos = pred[..., pos_slice].float()  # [B,C,T,pos_dims]
    targ_pos = target[..., pos_slice].float()  # [B,C,T,pos_dims]
    if pred_pos.shape != targ_pos.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")

    # # CRITICAL: Finite mask - 只计算有效值
    # pred_finite = torch.isfinite(pred_pos).all(dim=-1)  # [B,C,T]
    # targ_finite = torch.isfinite(targ_pos).all(dim=-1)  # [B,C,T]
    # finite_mask = pred_finite & targ_finite  # [B,C,T]

    # if not finite_mask.any():
    #     # 全是 NaN/Inf，返回零（避免 NaN 传播）
    #     device = pred.device
    #     return torch.zeros((), device=device), torch.zeros((), device=device)

    final_dist = torch.norm(
        pred_pos[:, -1, :, :] - targ_pos[:, -1, :, :], dim=-1
    )  # euclidean distances
    fde = final_dist.mean(dim=1)  # average over num keypoints -> tensor of shape (B,)
    avg_dist = torch.norm(pred_pos - targ_pos, dim=-1)  # euclidean distances
    ade = avg_dist.mean(
        dim=(1, 2)
    )  # average over T and num keypoints -> tensor of shape (B,)

    # # 按位置维求平方误差和（m^2）
    # # sq_err = (pred_pos - targ_pos).pow(2).sum(dim=-1)  # [B,C,T]
    # # 按位置维求平方误差和（m^2）
    # sq_err = (pred_pos - targ_pos).pow(2).sum(dim=-1).clamp_min_(0)  # [B,C,T]
    # # 开方得到每步位移误差（m）
    # dist = sq_err.sqrt()  # [B,C,T]

    # # 只对有效值计算均值
    # # ade = dist[finite_mask].mean()  # 标量
    # # fde = dist[..., -1][fde_mask].mean()
    # ade = dist.mean()  # 标量
    # fde = dist[..., -1].mean()

    # # FDE: 只取最后一步的有效值
    # fde_mask = finite_mask[..., -1]  # [B,C]
    # if fde_mask.any():
    #     fde = dist[..., -1][fde_mask].mean()
    # else:
    #     fde = torch.zeros((), device=pred.device)

    return ade, fde


def _stack_clip_features(enc_out: Any) -> torch.Tensor:
    """
    将 encoder 的输出统一为 [B, C, ...] 的 Tensor：
    - 若 enc_out 是 list/tuple，长度 = C，每个元素形状 [B, ...]，则 stack 到 dim=1；
    - 若 enc_out 已是 Tensor：
        * [B, ...] -> 在 dim=1 插入 C=1；
        * [B, C, ...] -> 原样返回。
    """
    if isinstance(enc_out, (list, tuple)):
        assert len(enc_out) > 0, "Empty encoder outputs"
        assert all(
            torch.is_tensor(x) for x in enc_out
        ), "Each clip output must be Tensor"
        base_shape = enc_out[0].shape
        assert all(
            x.shape[0] == base_shape[0] and x.shape[1:] == base_shape[1:]
            for x in enc_out
        ), "All clip outputs must share [B,*] shape"
        return torch.stack(enc_out, dim=1)  # [B, C, ...]
    elif torch.is_tensor(enc_out):
        if enc_out.ndim >= 3:  # [B, C, ...] or more
            return enc_out
        elif enc_out.ndim == 2:  # [B, D] -> [B, 1, D]
            return enc_out.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported encoder tensor shape: {enc_out.shape}")
    else:
        raise TypeError(f"Unsupported encoder output type: {type(enc_out)}")


def _apply_heads_vectorized(
    classifiers: List[torch.nn.Module], feats_BC: torch.Tensor, initp=None
) -> List[torch.Tensor]:
    """
    对合并维度后的特征做一次性前向。
    feats_BC: [B*C, ...] （原始为 [B,C,...] 展平）
    返回：长度 H 的 list，每个元素形状 [B*C, ...]
    """
    return [head(initp, feats_BC) for head in classifiers]


def evaluate(
    device,
    encoder,
    classifiers,
    data_loader,
    use_bfloat16,
    use_cached_features: bool = False,
    pos_offset: int = 0,
    pos_dims: int = 3,
) -> tuple[float, float, float]:
    """
    评估 1 个 epoch（MAE 越低越好）
    返回: (mae_min, ade_min, fde_min) —— 取多 head 的最小值
    """
    for c in classifiers:
        c.train(mode=False)

    # criterion = torch.nn.L1Loss(reduction="mean")  # 直接用 MAE 计算
    criterion_pos = torch.nn.L1Loss(reduction="mean")  # Position-only MAE
    top1_meters = [AverageMeter() for _ in classifiers]
    ade_meters = [AverageMeter() for _ in classifiers]
    fde_meters = [AverageMeter() for _ in classifiers]
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16

    with torch.no_grad():
        for itr, data in enumerate(data_loader):
            if use_cached_features:
                # === 从缓存读取：data = (feats[B,C,...], labels[B,C,...], clip_indices, meta) ===
                feats = data[0].to(device, non_blocking=True).float()  # [B,C,...] FP32
                labels_in = data[1]
                if torch.is_tensor(labels_in):
                    labels = labels_in.to(device, non_blocking=True).float()
                elif isinstance(labels_in, (list, tuple)):
                    labels = torch.stack(list(labels_in), dim=1).to(device).float()
                else:
                    raise TypeError(
                        f"Unexpected labels type in cached path: {type(labels_in)}"
                    )
                B, C = feats.shape[:2]
            else:
                # 原路径：从 encoder 计算 (NO autocast for FP32)
                clips = [
                    [dij.to(device, non_blocking=True) for dij in di] for di in data[0]
                ]
                clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
                labels = torch.stack(data[1], dim=1).to(device).float()  # [B, C, T, 6]

                # encoder 前向（冻结，FP32）
                with torch.no_grad():
                    feats = encoder(
                        clips, clip_indices
                    )  # list of [B,...] (len=C) 或 Tensor
                    feats = _stack_clip_features(feats).float()  # [B, C, ...] FP32
                B, C = feats.shape[:2]

            # 多 head 一次性前向（NO autocast）
            feats_BC = feats.reshape(B * C, *feats.shape[2:])  # 合并 B×C
            outs_BC = _apply_heads_vectorized(
                classifiers, feats_BC, labels[..., 0:1]
            )  # list of [B*C, ...]
            # 还原 [B, C, ...]
            outs = [o.reshape(B, *o.shape[1:]) for o in outs_BC]

            # 对齐 clip 数（一般等于 C；此处保险）
            safe_C = min(labels.shape[1], C)
            labels_safe = labels[:, :safe_C]  # [B, safe_C, T, 6]

            # 向量化 MAE + ADE/FDE：对每个 head，直接在 [B, safe_C, ...] 上做 reduction
            for h, (t1m, ade_m, fde_m, pred) in enumerate(
                zip(top1_meters, ade_meters, fde_meters, outs)
            ):
                pred_safe = pred[:, :safe_C]  # [B, safe_C, T, 6]

                # NaN diagnostic (每 50 iter 降频)
                if itr % 50 == 0:
                    check_nan_detailed(pred_safe, f"eval_pred_head{h}", itr, h)
                    check_nan_detailed(labels_safe, f"eval_labels", itr, h)

                # Position-only MAE with finite mask (CRITICAL FIX)
                pred_pos = pred_safe[..., pos_offset : pos_offset + pos_dims].float()
                targ_pos = labels_safe[..., pos_offset : pos_offset + pos_dims].float()

                # Finite mask
                pred_finite = torch.isfinite(pred_pos)
                targ_finite = torch.isfinite(targ_pos)
                finite_mask = pred_finite & targ_finite

                if finite_mask.any():
                    mae = criterion_pos(pred_pos[finite_mask], targ_pos[finite_mask])
                else:
                    mae = torch.zeros((), device=pred_pos.device)

                mae = float(AllReduce.apply(mae))
                t1m.update(mae)

                # ADE / FDE (already position-only with finite mask)
                ade, fde = _ade_fde_meters(
                    pred_safe, labels_safe, pos_offset=pos_offset, pos_dims=pos_dims
                )
                ade = float(AllReduce.apply(ade))
                fde = float(AllReduce.apply(fde))
                ade_m.update(ade)
                fde_m.update(fde)

    mae_vals = np.array([t1m.avg for t1m in top1_meters])
    ade_vals = np.array([m.avg for m in ade_meters])
    fde_vals = np.array([m.avg for m in fde_meters])

    mae_min = mae_vals.min() if len(mae_vals) else float("nan")
    ade_min = ade_vals.min() if len(ade_vals) else float("nan")
    fde_min = fde_vals.min() if len(fde_vals) else float("nan")

    return mae_min, ade_min, fde_min


def run_one_epoch(
    device,
    training,
    encoder,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    *,
    epoch: Optional[int] = None,
    # —— 保存策略，仅在 training=True 时启用 ——
    checkpoint_meta: Optional[
        dict
    ] = None,  # {"path": str, "rank": int, "batch_size": int, "world_size": int}
    save_every: Optional[int] = None,  # 间隔保存（步）
    save_on_first_iter: bool = False,  # 第 1 个 iter 后保存
    save_on_last_iter: bool = False,  # 最后一个 iter 保存
    # —— 在线验证配置（仅 training=True 时启用） ——
    test_every: Optional[int] = None,  # 每 N step 做一次验证
    val_loader=None,
    step_logger: Optional[CSVLogger] = None,
    global_step: int = 0,
    best_val_acc: float = float("+inf"),  # MAE 越小越好，默认 +inf
    best_path: Optional[str] = None,
    use_cached_features: bool = False,  # 新增：从缓存读取
    pos_offset: int = 0,
    pos_dims: int = 3,
    csv_log_ade_fde: bool = False,
) -> Tuple[float, float, float, int, float]:
    """
    返回：(train_acc_like, new_global_step, new_best_val_acc)
    - 训练/评测均支持；评测时会用 no_grad；
    - 解码部分改为向量化：合并 (B×C) 一次前向，多 head 并行；
    """
    for c in classifiers:
        c.train(mode=training)

    # Position-only loss + MAE
    criterion_pos = torch.nn.L1Loss(reduction="mean")

    top1_meters = [AverageMeter() for _ in classifiers]
    ade_meters = [AverageMeter() for _ in classifiers]
    fde_meters = [AverageMeter() for _ in classifiers]

    dataset_len = len(data_loader)
    _agg_top1 = None
    _agg_ade = None
    _agg_fde = None

    # CRITICAL: Force FP32, disable mixed precision
    amp_enabled = False
    amp_dtype = torch.float32

    # 时间统计
    dataloader_time = 0.0
    encoder_time = 0.0
    decoder_time = 0.0
    total_iterations = 0

    def _maybe_save(itr_idx: int, is_first_iter: bool, is_last_iter: bool):
        if not training or checkpoint_meta is None:
            return
        rank = int(checkpoint_meta.get("rank", 0))
        if rank != 0:
            return

        do_save = False
        if save_on_first_iter and is_first_iter:
            do_save = True
        if (
            (save_every is not None)
            and (save_every > 0)
            and ((itr_idx + 1) % save_every == 0)
        ):
            do_save = True
        if save_on_last_iter and is_last_iter:
            do_save = True
        if not do_save:
            return

        all_classifier_dicts = [c.state_dict() for c in classifiers]
        all_opt_dicts = [o.state_dict() for o in optimizer]
        save_dict = {
            "classifiers": all_classifier_dicts,
            "opt": all_opt_dicts,
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": int(epoch if epoch is not None else -1),
            "iter": int(itr_idx + 1),
            "batch_size": int(checkpoint_meta.get("batch_size", -1)),
            "world_size": int(checkpoint_meta.get("world_size", -1)),
        }
        save_path = checkpoint_meta["path"]
        torch.save(save_dict, save_path)
        logger.info(f"[CKPT] Saved at iter {itr_idx+1} -> {save_path}")

    for itr, data in enumerate(data_loader):
        if training:
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        if use_cached_features:
            # === 从缓存读取：data = (feats[B,C,...], labels[B,C,...], clip_indices, meta) ===
            t0 = time.time()
            feats = data[0].to(device, non_blocking=True)  # [B,C,...]
            labels_in = data[1]
            if torch.is_tensor(labels_in):
                labels = labels_in.to(device, non_blocking=True)  # [B,C,...]
            elif isinstance(labels_in, (list, tuple)):
                labels = torch.stack(list(labels_in), dim=1).to(device)  # [B,C,...]
            else:
                raise TypeError(
                    f"Unexpected labels type in cached path: {type(labels_in)}"
                )
            B = labels.size(0)  # 16
            labels = labels.reshape(
                B, -1, 6
            )  # [16, 32, 6]，顺序是 [0,*,*] 后接 [1,*,*]
            dataloader_time += time.time() - t0

            # 跳过 encoder（feats 已经是缓存的）
            t1 = time.time()
            t2 = t1  # encoder_time += 0
            B, C = feats.shape[:2]

            # === Decoder / Heads ===
            # with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=False):
                feats_BC = feats.reshape(B * C, *feats.shape[2:])  # 合并 B×C
                if training:
                    outs_BC = _apply_heads_vectorized(
                        classifiers, feats_BC, labels[..., 0:1]
                    )
                else:
                    with torch.no_grad():
                        outs_BC = _apply_heads_vectorized(
                            classifiers, feats_BC, labels[..., 0:1]
                        )
                outs = [
                    o.reshape(B, *o.shape[1:]) for o in outs_BC
                ]  # list of [B, C, ...]
            t3 = time.time()
        else:
            # === 原路径：从 encoder 计算 ===
            # === DataLoader 时间（CPU 处理 + H2D 传输）===
            t0 = time.time()
            # with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=False):
                clips = [
                    [dij.to(device, non_blocking=True) for dij in di] for di in data[0]
                ]
                clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
                labels = torch.stack(data[1], dim=1).to(device)  # [B, C, T, 6]
            dataloader_time += time.time() - t0

            # === Encoder 前向（冻结，只对 encoder no_grad）===
            t1 = time.time()
            # with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=False):
                # 仅 encoder 放 no_grad，提早结束
                with torch.no_grad():
                    feats = encoder(clips, clip_indices)  # list of [B,...] 或 Tensor
                    feats = _stack_clip_features(feats)  # [B, C, ...]
            B, C = feats.shape[:2]
            t2 = time.time()  # encoder 计时到此为止

            # === Decoder / Heads（需建图，不能 no_grad）===
            # with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=False):
                feats_BC = feats.reshape(B * C, *feats.shape[2:])  # 合并 B×C
                if training:
                    outs_BC = _apply_heads_vectorized(
                        classifiers, feats_BC, labels[..., 0:1]
                    )  # 这里要有 grad
                else:
                    with torch.no_grad():
                        outs_BC = _apply_heads_vectorized(
                            classifiers, feats_BC, labels[..., 0:1]
                        )
                outs = [
                    o.reshape(B, *o.shape[1:]) for o in outs_BC
                ]  # list of [B, C, ...]
            t3 = time.time()

        encoder_time += t2 - t1
        decoder_time += t3 - t2

        # === 按 (B*C) 前向多个 head ===
        # outs_BC = _apply_heads_vectorized(classifiers, feats_BC, labels[...,0:1])
        # outs = [o.reshape(B, *o.shape[1:]) for o in outs_BC]     # [B, C, ...]
        # safe_C = min(labels.shape[1], C)
        # labels_safe = labels[:, :safe_C]
        # outs = [o[:, :safe_C] for o in outs]

        # === 计算 MAE + ADE/FDE（日志展示用）===
        with torch.no_grad():
            for h, (t1m, ade_m, fde_m, pred) in enumerate(
                zip(top1_meters, ade_meters, fde_meters, outs)
            ):
                # # NaN diagnostic (每 50 iter 降频)
                # if itr % 50 == 0:
                #     check_nan_detailed(pred, f"train_pred_head{h}", itr, h)
                #     check_nan_detailed(labels_safe, "train_labels", itr, h)

                # Position-only MAE with finite mask (CRITICAL FIX)
                pred_pos = pred[..., pos_offset : pos_offset + pos_dims].float()
                targ_pos = labels[..., pos_offset : pos_offset + pos_dims].float()

                # Finite mask
                # pred_finite = torch.isfinite(pred_pos)
                # targ_finite = torch.isfinite(targ_pos)
                # finite_mask = pred_finite & targ_finite
                mae = criterion_pos(pred_pos, targ_pos)

                # if finite_mask.any():
                #     mae = criterion_pos(pred_pos[finite_mask], targ_pos[finite_mask])
                # else:
                #     mae = torch.zeros((), device=pred_pos.device)

                mae = float(AllReduce.apply(mae))
                t1m.update(mae)

                # ADE / FDE (already position-only)
                ade, fde = _ade_fde_meters(
                    pred, labels, pos_offset=pos_offset, pos_dims=pos_dims
                )
                ade = float(AllReduce.apply(ade))
                fde = float(AllReduce.apply(fde))
                ade_m.update(ade)
                fde_m.update(fde)

        # 仅用于控制台速览（第一个 head）
        ade_console = ade_meters[0].avg
        fde_console = fde_meters[0].avg

        # === 训练反传（CRITICAL: Position-only + finite mask）===
        if training:
            # Position-only loss with finite mask
            losses = []
            for pred in outs:
                pred_pos = pred[..., pos_offset : pos_offset + pos_dims].float()
                targ_pos = labels[..., pos_offset : pos_offset + pos_dims].float()

                # Finite mask: only valid values contribute to loss
                # pred_finite = torch.isfinite(pred_pos)
                # targ_finite = torch.isfinite(targ_pos)
                # finite_mask = pred_finite & targ_finite

                # if finite_mask.any():
                # MSE on position only
                # diff = pred_pos - targ_pos
                loss_i = criterion_pos(pred_pos, targ_pos)
                # else:
                #     # All NaN/Inf: zero loss (skip this head)
                #     loss_i = torch.zeros(
                #         (), device=pred_pos.device, dtype=pred_pos.dtype
                #     )

                losses.append(loss_i)

            loss = torch.stack(losses).mean()
            loss.backward()
            # Check loss before backward
            # if torch.isfinite(loss):
            #     loss.backward()
            #     for o in optimizer:
            #         o.step()
            # else:
            #     logger.warning(f"⚠️  Loss is NaN/Inf at iter {itr}, skipping backward")

            for o in optimizer:
                o.zero_grad()

        _agg_top1 = np.array([t1m.avg for t1m in top1_meters])
        total_iterations += 1

        if itr % 10 == 0:
            avg_dataloader = dataloader_time / max(1, total_iterations)
            avg_encoder = encoder_time / max(1, total_iterations)
            avg_decoder = decoder_time / max(1, total_iterations)

            mem_mb = (
                torch.cuda.max_memory_allocated() / 1024.0**2
                if torch.cuda.is_available()
                else 0.0
            )
            if csv_log_ade_fde:
                logger.info(
                    "[%5d / %5d] mae(max/mean/min)= %.6f / %.6f / %.6f | ADE= %.6f m | FDE= %.6f m [mem: %.2f MB] "
                    "[time] dataloader: %.3fms, encoder: %.3fms, decoder: %.3fms",
                    itr,
                    dataset_len,
                    _agg_top1.max(),
                    _agg_top1.mean(),
                    _agg_top1.min(),
                    ade_console,
                    fde_console,
                    mem_mb,
                    avg_dataloader * 1000,
                    avg_encoder * 1000,
                    avg_decoder * 1000,
                )
            else:
                logger.info(
                    "[%5d / %5d] mae(max/mean/min)= %.6f / %.6f / %.6f [mem: %.2f MB] "
                    "[time] dataloader: %.3fms, encoder: %.3fms, decoder: %.3fms",
                    itr,
                    dataset_len,
                    _agg_top1.max(),
                    _agg_top1.mean(),
                    _agg_top1.min(),
                    mem_mb,
                    avg_dataloader * 1000,
                    avg_encoder * 1000,
                    avg_decoder * 1000,
                )

        # —— 迭代内保存（仅训练 + rank0）——
        _maybe_save(
            itr_idx=itr,
            is_first_iter=(itr == 0),
            is_last_iter=(itr == dataset_len - 1),
        )

        # —— 在线验证（仅训练）——
        if (
            training
            and (test_every is not None)
            and (test_every > 0)
            and (val_loader is not None)
        ):
            global_step += 1
            if (global_step % test_every) == 0:
                val_mae, val_ade, val_fde = evaluate(
                    device=device,
                    encoder=encoder,
                    classifiers=classifiers,
                    data_loader=val_loader,
                    use_bfloat16=use_bfloat16,
                    use_cached_features=use_cached_features,
                    pos_offset=pos_offset,
                    pos_dims=pos_dims,
                )
                if step_logger is not None:
                    train_mae_curr = (
                        float(_agg_top1.min())
                        if _agg_top1 is not None
                        else float("nan")
                    )
                    if csv_log_ade_fde:
                        # 用当前 meters 的均值（与 evaluate 口径一致）
                        curr_ade_vals = (
                            np.array([m.avg for m in ade_meters])
                            if len(ade_meters)
                            else np.array([])
                        )
                        curr_fde_vals = (
                            np.array([m.avg for m in fde_meters])
                            if len(fde_meters)
                            else np.array([])
                        )
                        train_ade_curr = (
                            float(curr_ade_vals.min())
                            if curr_ade_vals.size
                            else float("nan")
                        )
                        train_fde_curr = (
                            float(curr_fde_vals.min())
                            if curr_fde_vals.size
                            else float("nan")
                        )
                        step_logger.log(
                            global_step,
                            train_mae_curr,
                            float(val_mae),
                            train_ade_curr,
                            train_fde_curr,
                            float(val_ade),
                            float(val_fde),
                        )
                    else:
                        step_logger.log(global_step, train_mae_curr, float(val_mae))
                if (
                    (val_mae < best_val_acc)  # MAE 越小越好
                    and (best_path is not None)
                    and (checkpoint_meta is not None)
                ):
                    if int(checkpoint_meta.get("rank", 0)) == 0:
                        best_val_acc = float(val_mae)
                        _save_best(
                            best_path,
                            classifiers,
                            optimizer,
                            scaler,
                            epoch if epoch is not None else -1,
                            best_val_acc,
                        )
                        logger.info(
                            f"[BEST] step {global_step} improved to {best_val_acc:.5f} (MAE), saved -> {best_path}"
                        )
        else:
            if training:
                global_step += 1

    # 在 epoch 结束时打印时间统计总结
    logger.info(
        f"[TIMING SUMMARY] Total - DataLoader: {dataloader_time:.3f}s, "
        f"Encoder: {encoder_time:.3f}s, Decoder: {decoder_time:.3f}s, "
        f"Avg per iter - DataLoader: {dataloader_time/max(1,total_iterations)*1000:.3f}ms, "
        f"Encoder: {encoder_time/max(1,total_iterations)*1000:.3f}ms, "
        f"Decoder: {decoder_time/max(1,total_iterations)*1000:.3f}ms"
    )

    _agg_ade = np.array([m.avg for m in ade_meters])
    _agg_fde = np.array([m.avg for m in fde_meters])

    if _agg_top1 is None:
        return float("nan"), float("nan"), float("nan"), global_step, best_val_acc

    mae_min = float(_agg_top1.min())
    ade_min = float(_agg_ade.min()) if len(_agg_ade) else float("nan")
    fde_min = float(_agg_fde.min()) if len(_agg_fde) else float("nan")

    return mae_min, ade_min, fde_min, global_step, best_val_acc


def _normalize_state_dict(sd: dict) -> dict:
    if sd and all(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module.") :]: v for k, v in sd.items()}
    return sd


def _get_inner_module(m):
    return m.module if isinstance(m, (DDP, DP)) else m


def load_checkpoint(device, r_path, classifiers, opt, scaler, val_only=False):
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    logger.info(f"read-path: {r_path}")

    pretrained_list = checkpoint["classifiers"]
    pretrained_list = [_normalize_state_dict(sd) for sd in pretrained_list]

    msgs = []
    for i, (c, pd) in enumerate(zip(classifiers, pretrained_list)):
        target = _get_inner_module(c)
        msg = target.load_state_dict(pd, strict=True)
        msgs.append(msg)
        if msg.missing_keys or msg.unexpected_keys:
            logger.warning(
                f"[Classifier {i}] missing: {msg.missing_keys}, unexpected: {msg.unexpected_keys}"
            )

    if val_only:
        logger.info(f"loaded pretrained classifier with msgs: {msgs}")
        return classifiers, opt, scaler, 0

    epoch = checkpoint.get("epoch", 0)
    logger.info(f"loaded pretrained classifier from epoch {epoch} with msgs: {msgs}")

    if (
        "opt" in checkpoint
        and checkpoint["opt"] is not None
        and len(opt) == len(checkpoint["opt"])
    ):
        for o, pd in zip(opt, checkpoint["opt"]):
            o.load_state_dict(pd)

    # 恢复 scaler：兼容新旧格式
    if scaler is not None and checkpoint.get("scaler") is not None:
        try:
            scaler.load_state_dict(checkpoint["scaler"])
            logger.info("loaded scaler state")
        except Exception as e:
            logger.warning(f"failed to load scaler state (ignored): {e}")

    logger.info(f"loaded optimizers/scalers from epoch {epoch}")
    return classifiers, opt, scaler, epoch


def load_pretrained(encoder, pretrained, checkpoint_key="target_encoder"):
    logger.info(f"Loading pretrained model from {pretrained}")
    checkpoint = robust_checkpoint_loader(pretrained, map_location="cpu")
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint["encoder"]

    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {
        k.replace("backbone.", ""): v for k, v in pretrained_dict.items()
    }
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f"key '{k}' could not be found in loaded state dict")
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f"{pretrained_dict[k].shape} | {v.shape}")
            logger.info(
                f"key '{k}' is of different shape in model and loaded state dict"
            )
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained model with msg: {msg}")
    logger.info(
        f"loaded pretrained encoder from epoch: {checkpoint.get('epoch', -1)}\n path: {pretrained}"
    )
    del checkpoint
    return encoder


DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type="VideoDataset",
    img_size=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    subset_file=None,
    normalization=None,
    cache_save_mode=False,
):
    if normalization is None:
        normalization = DEFAULT_NORMALIZATION

    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=img_size,
        normalize=normalization,
    )

    data_loader, data_sampler = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_segments,
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        drop_last=False,
        subset_file=subset_file,
        cache_save_mode=cache_save_mode,
    )
    return data_loader, data_sampler


def init_opt(
    classifiers, iterations_per_epoch, opt_kwargs, num_epochs, use_bfloat16=False
):
    optimizers, schedulers, wd_schedulers = [], [], []
    for c, kwargs in zip(classifiers, opt_kwargs):
        # 提供合理默认值，避免 config 漏填时 TypeError
        w = kwargs or {}
        warmup = float(w.get("warmup", 0.0))
        start_lr = float(w.get("start_lr", w.get("lr", 1e-4)))
        ref_lr = float(w.get("lr", w.get("ref_lr", 1e-3)))
        final_lr = float(w.get("final_lr", ref_lr * 0.01))
        ref_wd = float(w.get("weight_decay", 0.05))
        final_wd = float(w.get("final_weight_decay", ref_wd))

        param_groups = [
            {
                "params": (p for n, p in c.named_parameters()),
                "mc_warmup_steps": int(warmup * iterations_per_epoch),
                "mc_start_lr": start_lr,
                "mc_ref_lr": ref_lr,
                "mc_final_lr": final_lr,
                "mc_ref_wd": ref_wd,
                "mc_final_wd": final_wd,
            }
        ]
        logger.info("Using AdamW")
        optimizers += [torch.optim.AdamW(param_groups)]
        schedulers += [
            WarmupCosineLRSchedule(
                optimizers[-1], T_max=int(num_epochs * iterations_per_epoch)
            )
        ]
        wd_schedulers += [
            CosineWDSchedule(
                optimizers[-1], T_max=int(num_epochs * iterations_per_epoch)
            )
        ]
    # 所有 optimizer 共用一个 scaler
    scaler = torch.cuda.amp.GradScaler(enabled=not use_bfloat16)
    return optimizers, scaler, schedulers, wd_schedulers


class WarmupCosineLRSchedule(object):
    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            ref_lr = group.get("mc_ref_lr")
            final_lr = group.get("mc_final_lr")
            start_lr = group.get("mc_start_lr")
            warmup_steps = group.get("mc_warmup_steps")
            # 防御性 clamp：避免 warmup_steps >= T_max 时除零
            T_max = max(1, self.T_max - warmup_steps)
            if self._step < warmup_steps:
                progress = float(self._step) / float(max(1, warmup_steps))
                new_lr = start_lr + progress * (ref_lr - start_lr)
            else:
                progress = float(self._step - warmup_steps) / float(T_max)
                new_lr = max(
                    final_lr,
                    final_lr
                    + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
                )
            group["lr"] = new_lr


class CosineWDSchedule(object):
    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = max(1, int(T_max))  # 防 0
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        for group in self.optimizer.param_groups:
            ref_wd = group.get("mc_ref_wd")
            final_wd = group.get("mc_final_wd")
            new_wd = final_wd + (ref_wd - final_wd) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
            if final_wd <= ref_wd:
                new_wd = max(final_wd, new_wd)
            else:
                new_wd = min(final_wd, new_wd)
            group["weight_decay"] = new_wd
