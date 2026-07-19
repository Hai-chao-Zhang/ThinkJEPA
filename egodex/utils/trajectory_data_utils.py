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
For licensing see accompanying LICENSE.txt file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

import fnmatch
import os

import h5py
import numpy as np
from tqdm import tqdm


def index_episode_files(dataset_path, thread=None):
    # find all hdf5 files
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_path):
        for filename in fnmatch.filter(files, "*.hdf5"):
            hdf5_files.append(os.path.join(root, filename))
    print(f"Found {len(hdf5_files)} hdf5 files")

    # get lengths of all hdf5 files
    valid_hdf5_files = []
    all_episode_len = []
    for idx, dataset_path in enumerate(
        tqdm(hdf5_files, desc="iterating dataset_path to get all episode lengths...")
    ):
        try:
            with h5py.File(dataset_path, "r") as root:
                action = root["/transforms/leftHand"][()]
            valid_hdf5_files.append(dataset_path)
            all_episode_len.append(int(len(action)))
        except Exception as e:
            print(f"Error loading {dataset_path}: {e}")

        if thread is not None and idx > thread:
            break

    return valid_hdf5_files, all_episode_len
def project_transforms_to_camera_frame(tfs, cam_ext):
    """
    tfs: a set of transforms in the world frame, shape N x 4 x 4
    cam_ext: camera extrinsics in the world frame, shape 4 x 4
    """
    return np.linalg.inv(cam_ext)[None] @ tfs
