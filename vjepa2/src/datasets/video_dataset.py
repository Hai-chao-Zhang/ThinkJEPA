# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import pathlib
import warnings
from logging import getLogger
from typing import List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import torch
import torchvision
from decord import cpu, VideoReader
from scipy.spatial.transform import Rotation

from src.datasets.utils.dataloader import (
    ConcatIndices,
    MonitoredDataset,
    NondeterministicDataLoader,
)
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


def make_videodataset(
    data_paths,
    batch_size,
    frames_per_clip=8,
    dataset_fpcs=None,
    frame_step=4,
    duration=None,
    fps=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    log_dir=None,
    *,
    resolve_paths: bool = False,  # NEW: 默认关闭（不做 realpath/拼绝对路径）
    check_exists: bool = False,  # NEW: 默认关闭（不做 os.path.exists）
    cache_save_mode: bool = False,  # NEW: cache save 模式（shuffle=False）
):
    dataset = VideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        duration=duration,
        fps=fps,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        shared_transform=shared_transform,
        transform=transform,
        resolve_paths=resolve_paths,
        check_exists=check_exists,
    )

    log_dir = pathlib.Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        # Worker ID will replace '%w'
        resource_log_filename = log_dir / f"resource_file_{rank}_%w.csv"
        dataset = MonitoredDataset(
            dataset=dataset,
            log_filename=str(resource_log_filename),
            log_interval=10.0,
            monitor_interval=5.0,
        )

    logger.info("VideoDataset dataset created")

    # 根据 cache_save_mode 决定 shuffle 行为
    # cache_save_mode=True 时，shuffle=False（保证缓存顺序一致）
    # cache_save_mode=False 时，shuffle=True（正常训练打乱）
    should_shuffle = not cache_save_mode

    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=should_shuffle
        )
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=should_shuffle
        )

    if deterministic:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
    else:
        data_loader = NondeterministicDataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
    logger.info("VideoDataset unsupervised data loader created")

    return dataset, data_loader, dist_sampler


class VideoDataset(torch.utils.data.Dataset):
    """Video classification dataset."""

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        frames_per_clip=16,
        fps=None,
        dataset_fpcs=None,
        frame_step=4,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,  # duration in seconds
        *,
        resolve_paths: bool = False,  # NEW: 是否把相对路径拼成绝对路径（不默认 realpath）
        check_exists: bool = False,  # NEW: 是否做 os.path.exists 过滤
    ):
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration
        self.fps = fps
        self.resolve_paths = bool(resolve_paths)
        self.check_exists = bool(check_exists)

        # ---- 默认关节与置信度阈值（防止属性不存在）----
        joints = ["rightHand"]
        conf_threshold = 0.0

        if sum([v is not None for v in (fps, duration, frame_step)]) != 1:
            raise ValueError(
                f"Must specify exactly one of either {fps=}, {duration=}, or {frame_step=}."
            )

        if isinstance(data_paths, str):
            data_paths = [data_paths]

        if dataset_fpcs is None:
            self.dataset_fpcs = [frames_per_clip for _ in data_paths]
        else:
            if len(dataset_fpcs) != len(data_paths):
                raise ValueError(
                    "Frames per clip not properly specified for NFS data paths"
                )
            self.dataset_fpcs = dataset_fpcs

        if VideoReader is None:
            raise ImportError(
                'Unable to import "decord" which is required to read videos.'
            )

        # --------- Load video paths and labels (FAST by default) ----------
        samples, labels = [], []
        self.num_samples_per_dataset = []

        for data_path in data_paths:
            if data_path.endswith(".csv"):
                # 更快：C 引擎 + 明确逗号 + 只读两列（4:h5, 5:mp4）
                try:
                    df = pd.read_csv(
                        data_path,
                        engine="c",
                        sep=",",
                        header=None,
                        usecols=[4, 5],
                        dtype="string",
                        on_bad_lines="skip",
                        low_memory=False,
                    ).rename(columns={4: "h5", 5: "mp4"})
                except Exception:
                    # 回退：自动分隔，但依然只保留两列
                    df_raw = pd.read_csv(
                        data_path, engine="python", sep=None, header=None, dtype=str
                    )
                    if df_raw.shape[1] == 1:
                        df_raw = df_raw[0].str.split(",", expand=True)
                    df_raw = df_raw.dropna(axis=1, how="all")
                    if df_raw.shape[1] >= 6:
                        df = df_raw.iloc[:, [4, 5]]
                        df.columns = ["h5", "mp4"]
                    else:
                        # 没法解析，跳过
                        logger.warning(f"CSV has insufficient columns: {data_path}")
                        self.num_samples_per_dataset.append(0)
                        continue

                # 可能第一行是表头，简单丢弃
                if len(df) and (
                    (df.iloc[0].astype(str) == "split").any()
                    or (df.iloc[0].astype(str) == "hdf5_path").any()
                    or (df.iloc[0].astype(str) == "mp4_path").any()
                ):
                    df = df.iloc[1:].reset_index(drop=True)

                # 清理
                df = df.dropna(subset=["h5", "mp4"])
                df["h5"] = df["h5"].str.strip()
                df["mp4"] = df["mp4"].str.strip()

                # 扩展名过滤（忽略大小写）
                h5l, mp4l = df["h5"].str.lower(), df["mp4"].str.lower()
                mask_ext = (mp4l.str.endswith(".mp4", na=False)) & (
                    h5l.str.endswith(".h5", na=False)
                    | h5l.str.endswith(".hdf5", na=False)
                )
                if not mask_ext.all():
                    df = df[mask_ext]

                if not len(df):
                    self.num_samples_per_dataset.append(0)
                    continue

                # 路径解析开关（默认不做任何 realpath/绝对化）
                if self.resolve_paths:
                    csv_dir = os.path.dirname(os.path.realpath(data_path))

                    # 尽量只做「相对 -> 基于 csv_dir 的绝对」拼接，不做 realpath
                    def _join_if_rel(p: str) -> str:
                        p = str(p)
                        if os.path.isabs(p) or (
                            len(p) > 1
                            and p[1] == ":"
                            and (p[2:3] == "\\" or p[2:3] == "/")
                        ):
                            return p  # 已经是绝对路径（兼容 Windows）
                        return os.path.join(csv_dir, p)

                    df["h5"] = df["h5"].map(_join_if_rel)
                    df["mp4"] = df["mp4"].map(_join_if_rel)
                    # 若你**强制**要规范化，再手动开：会很慢
                    # df["h5"]  = df["h5"].map(os.path.realpath)
                    # df["mp4"] = df["mp4"].map(os.path.realpath)

                # 存在性检查（默认关闭，避免远端 IO 慢）
                if self.check_exists:
                    from concurrent.futures import ThreadPoolExecutor

                    def _exists_many(paths):
                        paths = list(paths)
                        with ThreadPoolExecutor(max_workers=os.cpu_count() or 16) as ex:
                            return np.fromiter(
                                ex.map(os.path.exists, paths),
                                dtype=bool,
                                count=len(paths),
                            )

                    e_mp4 = _exists_many(df["mp4"].tolist())
                    e_h5 = _exists_many(df["h5"].tolist())
                    df = df[e_mp4 & e_h5]

                samples += df["mp4"].astype(str).tolist()
                labels += df["h5"].astype(str).tolist()
                self.num_samples_per_dataset.append(len(df))

            elif data_path.endswith(".npy"):
                arr = np.load(data_path, allow_pickle=True)
                data = list(map(lambda x: repr(x)[1:-1], arr))
                samples += data
                labels += [0] * len(data)
                self.num_samples_per_dataset.append(len(data))

        self.per_dataset_indices = ConcatIndices(self.num_samples_per_dataset)

        # [Optional] Weights for weighted sampler
        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [dw / ns] * ns

        self.samples = samples
        self.labels = labels

        # ---- 实例级兜底，若外部未设置则使用默认 ----
        self.joints = getattr(self, "joints", ["rightHand"])
        self.conf_threshold = getattr(self, "conf_threshold", 0.0)

    def __getitem__(self, index):
        sample = self.samples[index]
        loaded_sample = False
        # Keep trying to load videos until you find a valid sample
        while not loaded_sample:
            if not isinstance(sample, str):
                logger.warning("Invalid sample.")
            else:
                if sample.split(".")[-1].lower() in ("jpg", "png", "jpeg"):
                    loaded_sample = self.get_item_image(index)
                else:
                    loaded_sample = self.get_item_video(index)

            if not loaded_sample:
                index = np.random.randint(self.__len__())
                sample = self.samples[index]

        return loaded_sample

    def get_item_video(self, index):
        sample = self.samples[index]  # mp4 path
        dataset_idx, _ = self.per_dataset_indices[index]
        frames_per_clip = self.dataset_fpcs[dataset_idx]

        # 读视频帧与对应的全局帧索引（clip_indices 是一个长度为 self.num_clips 的 list）
        buffer, clip_indices = self.loadvideo_decord(
            sample, frames_per_clip
        )  # buffer: [T H W 3]
        loaded_video = len(buffer) > 0
        if not loaded_video:
            return

        # label 现在是 hdf5 路径
        h5_path = self.labels[index]
        if not isinstance(h5_path, str) or (
            self.check_exists and not os.path.exists(h5_path)
        ):
            logger.warning(f"HDF5 not found for sample idx={index}: {h5_path}")
            return

        # 将各 clip 的索引拼为视频级索引
        if isinstance(clip_indices, (list, tuple)) and len(clip_indices) > 0:
            flat_indices = np.concatenate(
                [np.asarray(ci, dtype=np.int64) for ci in clip_indices], axis=0
            )
        else:
            flat_indices = np.asarray(clip_indices, dtype=np.int64)

        vlen = int(flat_indices.max()) + 1

        # ---- 从 HDF5 读取姿态并与视频索引对齐 ----
        try:
            with h5py.File(h5_path, "r") as h5:
                poses_all = self._load_joint_poses(
                    h5, self.joints, getattr(self, "conf_threshold", 0.0)
                )  # (T_all, 6*K)
        except Exception as e:
            logger.exception(f"Failed to read trajectory from HDF5: {h5_path} ({e})")
            return

        T_all = poses_all.shape[0]
        if vlen - 1 > 0:
            indices_h5 = np.clip(
                np.round(
                    flat_indices.astype(np.float64) / (vlen - 1) * (T_all - 1)
                ).astype(np.int64),
                0,
                T_all - 1,
            )
        else:
            indices_h5 = np.zeros_like(flat_indices, dtype=np.int64)

        states_all = poses_all[indices_h5].astype(np.float32, copy=False)  # (T, 6*K)
        actions_all = self.poses_to_diffs(states_all)  # (T-1, 6*K)
        extrinsics_all = np.zeros((states_all.shape[0], 6), dtype=np.float32)
        indices_all = flat_indices.astype(np.int64, copy=False)

        # ---- 按 clip 切分（保证和视频切分一致）----
        def split_by_counts(arr, counts):
            out, s = [], 0
            for c in counts:
                e = s + c
                out.append(arr[s:e])
                s = e
            return out

        if isinstance(clip_indices, (list, tuple)):
            per_clip_counts = [len(ci) for ci in clip_indices]
        else:
            per_clip_counts = [len(buffer)]  # 兼容单段

        states_clips = split_by_counts(states_all, per_clip_counts)
        extri_clips = split_by_counts(extrinsics_all, per_clip_counts)
        idx_clips = split_by_counts(indices_all, per_clip_counts)

        actions_clips = []
        for sc in states_clips:
            if len(sc) >= 2:
                actions_clips.append(self.poses_to_diffs(sc))
            else:
                actions_clips.append(
                    np.zeros((0, sc.shape[1] if sc.ndim == 2 else 0), dtype=np.float32)
                )

        # ---- 视频增强：先全局共享，再按 clip 单独 transform ----
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        fpc = frames_per_clip
        nc = self.num_clips

        def split_into_clips(video):
            return [video[i * fpc : (i + 1) * fpc] for i in range(nc)]

        buffer = split_into_clips(buffer)

        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]

        # ---- 统一转 tensor，作为 label 字典返回（与原三元组返回保持兼容）----
        def _to_tensor(x, dtype=None):
            if torch.is_tensor(x):
                return x.to(dtype=dtype) if dtype is not None else x
            t = torch.from_numpy(x)
            return t.to(dtype=dtype) if dtype is not None else t

        traj = {
            "actions": [_to_tensor(a, torch.float32) for a in actions_clips],
            "states": [_to_tensor(s, torch.float32) for s in states_clips],
            "extrinsics": [_to_tensor(e, torch.float32) for e in extri_clips],
            "indices": [_to_tensor(i, torch.int64) for i in idx_clips],
        }

        from pathlib import Path

        # 避免隐式 realpath：根据开关决定
        mp4_meta = os.path.realpath(sample) if self.resolve_paths else sample
        h5_meta = (
            os.path.realpath(h5_path)
            if (self.resolve_paths and h5_path is not None)
            else h5_path
        )

        meta = {
            "mp4_path": mp4_meta,
            "hdf5_path": h5_meta,
            "orig_name": Path(sample).stem,
        }

        return buffer, traj["states"], clip_indices, meta

    def get_item_image(self, index):
        sample = self.samples[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        fpc = self.dataset_fpcs[dataset_idx]

        try:
            image_tensor = torchvision.io.read_image(
                path=sample, mode=torchvision.io.ImageReadMode.RGB
            )
        except Exception:
            return
        label = self.labels[index]
        clip_indices = [np.arange(start=0, stop=fpc, dtype=np.int32)]

        # Expanding the input image [3, H, W] ==> [T, 3, H, W]
        buffer = image_tensor.unsqueeze(dim=0).repeat((fpc, 1, 1, 1))
        buffer = buffer.permute((0, 2, 3, 1))  # [T, 3, H, W] ==> [T H W 3]

        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)

        if self.transform is not None:
            buffer = [self.transform(buffer)]

        from pathlib import Path

        mp4_meta = os.path.realpath(sample) if self.resolve_paths else sample
        meta = {
            "mp4_path": mp4_meta,
            "hdf5_path": None,
            "orig_name": Path(sample).stem,
        }

        return buffer, label, clip_indices, meta

    def loadvideo_decord(self, sample, fpc):
        """Load video content using Decord"""

        fname = sample
        if self.check_exists and not os.path.exists(fname):
            warnings.warn(f"video path not found {fname=}")
            return [], None

        # 如果不做 exists 检查，直接尝试打开；失败就跳过
        try:
            _fsize = os.path.getsize(fname)
        except Exception:
            warnings.warn(f"video not accessible {fname=}")
            return [], None

        if _fsize > self.filter_long_videos:
            warnings.warn(f"skipping long video of size {_fsize=} (bytes)")
            return [], None

        try:
            vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
        except Exception:
            return [], None

        fstp = self.frame_step
        if self.duration is not None or self.fps is not None:
            try:
                video_fps = math.ceil(vr.get_avg_fps())
            except Exception as e:
                logger.warning(e)
                video_fps = 30
            if self.duration is not None:
                assert self.fps is None
                fstp = int(self.duration * video_fps / fpc)
            else:
                assert self.duration is None
                fstp = max(1, video_fps // self.fps)

        assert fstp is not None and fstp > 0
        clip_len = int(fpc * fstp)

        if self.filter_short_videos and len(vr) < clip_len:
            warnings.warn(f"skipping video of length {len(vr)}")
            return [], None

        vr.seek(0)  # Go to start of video before sampling frames

        # Partition video into equal sized segments and sample each clip
        partition_len = len(vr) // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):
            if partition_len > clip_len:
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
                indices = indices + i * partition_len
            else:
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate(
                        (
                            indices,
                            np.ones(fpc - partition_len // fstp) * partition_len,
                        )
                    )
                    indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                    indices = indices + i * partition_len
                else:
                    sample_len = min(clip_len, len(vr)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate(
                        (
                            indices,
                            np.ones(fpc - sample_len // fstp) * sample_len,
                        )
                    )
                    indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
                    clip_step = 0
                    if len(vr) > clip_len:
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        buffer = vr.get_batch(all_indices).asnumpy()
        return buffer, clip_indices

    # ---------- helpers for trajectory ----------
    @staticmethod
    def _transform_to_pose(M: np.ndarray) -> np.ndarray:
        """4x4 -> (6,) pose [x,y,z, rx,ry,rz] (Euler XYZ, radians)."""
        t = M[:3, 3]
        R = M[:3, :3]
        eul = Rotation.from_matrix(R).as_euler("xyz", degrees=False)
        out = np.empty((6,), dtype=np.float32)
        out[:3] = t.astype(np.float32, copy=False)
        out[3:] = eul.astype(np.float32, copy=False)
        return out

    @staticmethod
    def poses_to_diffs(poses: np.ndarray) -> np.ndarray:
        """poses: (T, 6*K) -> diffs: (T-1, 6*K)"""
        T, D = poses.shape
        assert D % 6 == 0, "poses should be concatenated [xyz(3), euler(3)] per joint"
        K = D // 6
        xyz = poses[:, : 3 * K].reshape(T, K, 3)
        eul = poses[:, 3 * K :].reshape(T, K, 3)
        d_xyz = (xyz[1:] - xyz[:-1]).astype(np.float32, copy=False)
        d_eul = (eul[1:] - eul[:-1]).astype(np.float32, copy=False)
        diffs = np.concatenate([d_xyz, d_eul], axis=2).reshape(T - 1, 6 * K)
        return diffs.astype(np.float32, copy=False)

    def _load_joint_poses(
        self, h5: h5py.File, joints: List[str], conf_th: float = 0.0
    ) -> np.ndarray:
        """Load per-joint (T,6) poses from HDF5 transforms, concat over joints -> (T, 6*K)."""
        T = None
        pose_list = []
        for j in joints:
            if "transforms" not in h5 or j not in h5["transforms"]:
                raise KeyError(f"Joint not found in HDF5: transforms/{j}")
            M = np.asarray(h5["transforms"][j])  # (T,4,4)
            if T is None:
                T = M.shape[0]
            poses_j = np.stack(
                [self._transform_to_pose(M[t]) for t in range(T)], axis=0
            )  # (T,6)
            if conf_th > 0 and "confidences" in h5 and j in h5["confidences"]:
                c = np.asarray(h5["confidences"][j]).astype(np.float32)  # (T,)
                mask = c < conf_th
                poses_j[mask] = np.nan  # 保留形状，用 NaN 屏蔽
            pose_list.append(poses_j.astype(np.float32, copy=False))
        poses = np.concatenate(pose_list, axis=1).astype(
            np.float32, copy=False
        )  # (T, 6*K)
        return poses

    def __len__(self):
        return len(self.samples)
